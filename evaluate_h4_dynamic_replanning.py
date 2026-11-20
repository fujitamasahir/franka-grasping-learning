"""Sequential, resumable frozen-H4 dynamic-noise study; no model training."""
import argparse
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from zoneinfo import ZoneInfo

import numpy as np
from action_chunking import BCActionChunkController, load_bc_policy, sha256
from evaluate_h4_adaptive_replanning import estimate_sigma
from evaluate_h4_observation_noise import make_noise_schedule, apply_observation_noise, _schedule_hash, region, NOISE_SEED
from evaluate_h4_replanning import exclusive_run_lock
from franka_autonomous_env import FrankaAutonomousEnv
from h4_dynamic_controller import RollingReplanningRule, advance_at_boundary

PROTOCOL = Path('bc_results/h4_dynamic_replanning_protocol.json')
PROTOCOL_SHA = PROTOCOL.with_suffix('.sha256')
OUT = Path('bc_results/h4_dynamic_replanning')
MODELS = (42, 123, 456)
PROFILES = ('clean', 'noisy', 'clean_to_noisy', 'noisy_to_clean')
CONTROLLERS = ('R1', 'R4', 'OneShot', 'Online')
DECISIONS = tuple(range(20, 81, 4))
FIELDS = ['seed','model_seed','profile','controller','initial_cube_x','initial_cube_y','initial_cube_z','region',
          'success','steps','return','first_close_step','first_lift_step','closed_fraction','bc_inference_calls',
          'bc_total_inference_ms','bc_mean_latency_ms','episode_wall_seconds','steps_R1','steps_R4',
          'decision_count','switch_R4_to_R1','switch_R1_to_R4','final_interval','terminal_chunk_index',
          'decisions_json','windows_json','prefix_actions_sha256','prefix_observations_sha256',
          'example_trace_json','noise_generation_seed','base_noise_sha256','applied_noise_sha256',
          'checkpoint_sha256','protocol_sha256']


def protocol():
    digest = PROTOCOL_SHA.read_text().split()[0]
    if sha256(PROTOCOL) != digest:
        raise RuntimeError('Protocol checksum mismatch')
    return json.loads(PROTOCOL.read_text()), digest


def dynamic_noise(seed, profile, max_steps=160):
    if profile not in PROFILES:
        raise ValueError('Unknown noise profile')
    base, _ = make_noise_schedule(seed, 1., NOISE_SEED, max_steps)
    before, after = {'clean': (0.,0.), 'noisy': (.01,.01),
                     'clean_to_noisy': (0.,.01), 'noisy_to_clean': (.01,0.)}[profile]
    sigma = np.full(max_steps+1, before, dtype=np.float32)
    sigma[40:] = after
    return base, (base*sigma[:,None]).astype(np.float32), sigma


def window_diagnostic(step, observed_xy, gt_xyz, sigma):
    """Evaluator-only diagnostics; never an input to the rule."""
    a = np.asarray(gt_xyz[step-20:step], dtype=np.float64)
    diffs = a[:,None,:] - a[None,:,:]
    xy_diameter = float(np.linalg.norm(diffs[:,:,:2], axis=2).max())
    xyz_diameter = float(np.linalg.norm(diffs, axis=2).max())
    window_sigma = np.asarray(sigma[step-20:step], dtype=float)
    clean_variance_sigma = estimate_sigma(a[:,:2])
    return {'step':step, 'window_start':step-20, 'window_end':step-1,
            'estimated_sigma_m':estimate_sigma(observed_xy[step-20:step]),
            'injected_rms_sigma_m':float(np.sqrt(np.mean(window_sigma**2))),
            'mixed_window':bool(np.min(window_sigma) != np.max(window_sigma)),
            'post_change_only':step-20 >= 40,
            'gt_xy_diameter_m':xy_diameter, 'gt_xyz_diameter_m':xyz_diameter,
            'gt_xy_from_first_m':float(np.linalg.norm(a[:,:2]-a[0,:2], axis=1).max()),
            'gt_position_variance_sigma_m':clean_variance_sigma}


def rollout(model, ck, model_seed, profile, controller_name, seed, ckhash, ph, example=False):
    env = FrankaAutonomousEnv(position_range=.20)
    try:
        env.max_move=float(ck['max_move']); env.max_time=float(ck['max_time'])
        if not np.isclose(env.control_dt, float(ck['control_dt'])):
            raise ValueError('Frozen timestep mismatch')
        base, applied, sigma = dynamic_noise(seed, profile, int(round(env.max_time/env.control_dt)))
        ctrl = BCActionChunkController(model, ck['obs_mean'].numpy(), ck['obs_std'].numpy(), 4,
                                       replan_interval=1 if controller_name=='R1' else 4)
        rule = RollingReplanningRule(controller_name=='Online') if controller_name in ('OneShot','Online') else None
        obs,_=env.reset(seed=int(seed)); initial=env.data.xpos[env.cube_id].copy()
        noisy=apply_observation_noise(obs, applied[0]); observed=[]; gt=[]; prefix_actions=[];prefix_obs=[]
        windows=[];trace=[];steps=0;total=0.;closed=0;close='';lift='';interval_steps={1:0,4:0}
        started=time.perf_counter();ctrl.reset(noisy)
        while True:
            observed.append(noisy[3:5].copy());gt.append(env.data.xpos[env.cube_id].copy())
            if rule is not None:rule.observe(noisy[3:5])
            action=ctrl.planned_action.copy()
            if not np.isfinite(action).all():raise ValueError('Nonfinite action')
            if steps<20:prefix_actions.append(action.copy());prefix_obs.append(noisy.copy())
            if example:
                trace.append({'step':steps,'interval':ctrl.replan_interval,'chunk_index':ctrl.chunk_index,
                              'action':action.tolist(),'noisy_cube_xy':noisy[3:5].tolist(),
                              'gt_cube_xyz':gt[-1].tolist()})
            if action[3]<0 and close=='':close=steps
            closed+=int(action[3]<0);interval_steps[ctrl.replan_interval]+=1
            nxt,reward,terminated,truncated,info=env.step(action);steps+=1;total+=reward
            if info['lift_height']>=.10 and lift=='':lift=steps-1
            # Include every fully observed 20-sample audit window, even if this
            # action ended the episode. No policy decision/inference at terminal.
            if steps in DECISIONS:
                windows.append(window_diagnostic(steps,observed,gt,sigma))
            if terminated or truncated:break
            if steps>=len(applied):raise RuntimeError('Noise schedule exhausted')
            noisy=apply_observation_noise(nxt,applied[steps])
            event=rule.decide() if rule is not None else None
            advance_at_boundary(ctrl,noisy,event)
        lat=np.asarray(ctrl.inference_latencies,dtype=float);events=rule.events if rule is not None else []
        return dict(zip(FIELDS, [
            int(seed),model_seed,profile,controller_name,*map(float,initial),region(*initial[:2]),
            int(bool(info['is_success'])),steps,total,close,lift,closed/steps,ctrl.inference_calls,
            float(lat.sum()*1000),float(lat.mean()*1000),time.perf_counter()-started,
            interval_steps[1],interval_steps[4],len(events),
            sum(e['before']==4 and e['after']==1 for e in events),
            sum(e['before']==1 and e['after']==4 for e in events),ctrl.replan_interval,ctrl.chunk_index,
            json.dumps(events,separators=(',',':'),allow_nan=False),
            json.dumps(windows,separators=(',',':'),allow_nan=False),
            _schedule_hash(prefix_actions),_schedule_hash(prefix_obs),
            json.dumps(trace,separators=(',',':'),allow_nan=False),NOISE_SEED,_schedule_hash(base),_schedule_hash(applied),ckhash,ph]))
    finally:
        env.close()


def read_rows(path):
    with Path(path).open(newline='') as stream:
        reader=csv.DictReader(stream)
        if reader.fieldnames!=FIELDS:raise ValueError(f'CSV schema mismatch: {path}')
        rows=list(reader)
        if any(None in r or any(v is None for v in r.values()) for r in rows):
            raise ValueError('Truncated or extra-field row; preserving evidence')
        return rows


def validate_rows(rows, seeds, model, profile, controller, ckhash, ph):
    expected=list(seeds);got=[]
    for r in rows:
        seed=int(r['seed']);got.append(seed)
        if (int(r['model_seed'])!=model or r['profile']!=profile or r['controller']!=controller
            or int(r['noise_generation_seed'])!=NOISE_SEED or r['checkpoint_sha256']!=ckhash or r['protocol_sha256']!=ph):
            raise ValueError('Row provenance/configuration mismatch')
        steps=int(r['steps']);calls=int(r['bc_inference_calls'])
        if int(r['success']) not in (0,1) or not 1<=steps<=160 or not 1<=calls<=steps:
            raise ValueError('Invalid outcome/counters')
        if int(r['steps_R1'])+int(r['steps_R4'])!=steps:raise ValueError('Interval exposure mismatch')
        numbers=[float(r[k]) for k in ('initial_cube_x','initial_cube_y','initial_cube_z','return','closed_fraction','bc_total_inference_ms','bc_mean_latency_ms','episode_wall_seconds')]
        if not np.isfinite(numbers).all() or not 0<=numbers[4]<=1 or min(numbers[5:])<0 or numbers[-1]<=0:
            raise ValueError('Nonfinite/invalid measurement')
        if not np.isclose(numbers[5]/calls,numbers[6],rtol=1e-10,atol=1e-10):raise ValueError('Inference time accounting mismatch')
        if r['region']!=region(numbers[0],numbers[1]):raise ValueError('Initial region mismatch')
        base,applied,_=dynamic_noise(seed,profile)
        if r['base_noise_sha256']!=_schedule_hash(base) or r['applied_noise_sha256']!=_schedule_hash(applied):
            raise ValueError('Absolute-step noise hash mismatch')
        for k in ('prefix_actions_sha256','prefix_observations_sha256'):
            if len(r[k])!=64 or any(ch not in '0123456789abcdef' for ch in r[k]):raise ValueError('Bad prefix hash')
        events=json.loads(r['decisions_json']);windows=json.loads(r['windows_json']);json.loads(r['example_trace_json'])
        required=([t for t in DECISIONS if t<steps] if controller=='Online' else
                  [20] if controller=='OneShot' and steps>20 else [])
        if [e['step'] for e in events]!=required:raise ValueError('Incorrect controller decision timing')
        if [w['step'] for w in windows]!=[t for t in DECISIONS if t<=steps]:raise ValueError('Missing audit window')
        for w in windows:
            t=w['step']
            if w['window_start']!=t-20 or w['window_end']!=t-1:raise ValueError('Incorrect audit window indices')
            if not all(np.isfinite(float(w[k])) and float(w[k])>=0 for k in ('estimated_sigma_m','injected_rms_sigma_m','gt_xy_diameter_m','gt_xyz_diameter_m','gt_position_variance_sigma_m','gt_xy_from_first_m')):
                raise ValueError('Invalid audit diagnostics')
        interval=1 if controller=='R1' else 4
        exposure={1:0,4:0};last=0
        for e in events:
            t=e['step'];estimate=float(e['estimated_sigma_m'])
            if not np.isfinite(estimate) or estimate<0:raise ValueError('Bad estimate')
            after=(4 if estimate<=.0075 else 1) if t==20 else (1 if interval==4 and estimate>.0085 else 4 if interval==1 and estimate<.0065 else interval)
            if e!={'step':t,'window_start':t-20,'window_end':t-1,'estimated_sigma_m':estimate,'before':interval,'after':after,'switched':interval!=after}:
                raise ValueError('Decision rule/window mismatch')
            if not np.isclose(estimate,next(w['estimated_sigma_m'] for w in windows if w['step']==t),rtol=0,atol=1e-15):raise ValueError('Decision estimate differs from observed window diagnostic')
            exposure[interval]+=t-last;last=t;interval=after
        exposure[interval]+=steps-last
        if interval!=int(r['final_interval']) or (steps-last-1)%interval!=int(r['terminal_chunk_index']):raise ValueError('Terminal chunk mismatch')
        if any(exposure[i]!=int(r[f'steps_R{i}']) for i in (1,4)):raise ValueError('Incorrect R1/R4 action counts')
        expected_calls=exposure[1]+(exposure[4]+3)//4
        if calls!=expected_calls:raise ValueError('Unexpected inference count or terminal inference')
        if int(r['decision_count'])!=len(events):raise ValueError('Decision count mismatch')
        for before,after in ((4,1),(1,4)):
            if int(r[f'switch_R{before}_to_R{after}'])!=sum(e['before']==before and e['after']==after for e in events):raise ValueError('Switch count mismatch')
    # A resumable file is a complete prefix, not merely an unordered subset.
    if got!=expected[:len(got)] or len(got)>len(expected):raise ValueError('Duplicate, missing, unordered or unexpected seed; preserve output')
    return set(got)


def check_model(model_seed, p):
    path=Path(p['frozen_models'][str(model_seed)]['path']);digest=sha256(path)
    if digest!=p['frozen_models'][str(model_seed)]['sha256']:raise RuntimeError('Frozen checkpoint changed')
    model,ck=load_bc_policy(path)
    if int(ck['horizon'])!=4 or int(ck['seed'])!=model_seed:raise ValueError('Unexpected frozen architecture/seed')
    for k,v in {'max_move':.15,'max_time':16.,'control_dt':.1}.items():
        if not np.isclose(float(ck[k]),v):raise ValueError('Frozen environment config mismatch')
    if ck['obs_mean'].shape!=(12,) or ck['obs_std'].shape!=(12,) or not np.isfinite(ck['obs_mean'].numpy()).all() or not (ck['obs_std'].numpy()>0).all():raise ValueError('Invalid normalization')
    return model,ck,digest


def require_gate(phase, ph):
    if phase=='smoke':return
    prerequisite='smoke' if phase=='calibration' else 'calibration'
    path=OUT/f'{prerequisite}_integrity_audit.json'
    a=json.loads(path.read_text())
    if not a.get('gate_passed') or a['protocol_sha256']!=ph:raise RuntimeError(f'{prerequisite} integrity/stationarity gate failed; no {phase}')
    for cell in a['cells']:
        if sha256(OUT/cell['file'])!=cell['sha256']:raise RuntimeError('Prerequisite evidence changed')


def run(a):
    p,ph=protocol();require_gate(a.phase,ph)
    # Committed preregistration required, even for smoke; no new study data yet.
    tracked=subprocess.check_output(['git','show',f'HEAD:{PROTOCOL}'])
    if hashlib.sha256(tracked).hexdigest()!=ph:raise RuntimeError('Protocol is not committed')
    models=(a.model_seed,) if a.model_seed is not None else MODELS
    lo,hi=p['evaluation'][a.phase];seeds=range(lo,hi+1)
    for m in models:
        model,ck,ckhash=check_model(m,p)
        for profile in PROFILES:
            for controller in CONTROLLERS:
                if (OUT/'.STOP_AFTER_CELL').exists():print('STOP_AFTER_CELL honored',flush=True);return
                tag=f'{a.phase}_H4_seed{m}_{controller}_{profile}'
                complete=OUT/f'{tag}.csv';part=OUT/f'{tag}.partial.csv';meta=OUT/f'{tag}.json'
                cfg={'phase':a.phase,'model_seed':m,'controller':controller,'profile':profile,'seed_range':[lo,hi],
                     'checkpoint_sha256':ckhash,'protocol_sha256':ph,'noise_generation_seed':NOISE_SEED,'schema':FIELDS}
                if complete.exists() and part.exists():raise ValueError('Both complete/partial files present')
                if meta.exists():
                    if json.loads(meta.read_text())!=cfg:raise ValueError('Metadata mismatch')
                elif complete.exists() or part.exists():raise ValueError('Existing output without metadata')
                else:
                    with meta.open('x') as f:f.write(json.dumps(cfg,indent=2,sort_keys=True)+'\n')
                path=complete if complete.exists() else part
                rows=read_rows(path) if path.exists() else []
                done=validate_rows(rows,seeds,m,profile,controller,ckhash,ph)
                if complete.exists():
                    if len(done)!=len(seeds):raise ValueError('Incomplete completed output')
                    print(f'Already complete {tag}',flush=True);continue
                for seed in seeds:
                    if seed in done:continue
                    if datetime.now(ZoneInfo('America/Denver')).date().isoformat()>'2026-11-14':
                        print('Hard research cutoff reached; outputs preserved',flush=True);return
                    row=rollout(model,ck,m,profile,controller,seed,ckhash,ph,example=seed==lo)
                    new=not part.exists()
                    with part.open('a',newline='') as f:
                        w=csv.DictWriter(f,fieldnames=FIELDS,lineterminator='\n')
                        if new:w.writeheader()
                        w.writerow(row);f.flush();os.fsync(f.fileno())
                    if (seed-lo+1)%20==0:print(f'{tag}: {seed-lo+1}/{len(seeds)}',flush=True)
                validate_rows(read_rows(part),seeds,m,profile,controller,ckhash,ph)
                if complete.exists():raise FileExistsError(complete)
                part.rename(complete)
                if sha256(Path(p['frozen_models'][str(m)]['path']))!=ckhash:raise RuntimeError('Checkpoint mutation')
                print(f'Completed {tag}',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=('smoke','calibration','final'),required=True)
    parser.add_argument('--model-seed',type=int,choices=MODELS)
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    with exclusive_run_lock(OUT/'.study_run.lock'):run(args)


if __name__=='__main__':main()

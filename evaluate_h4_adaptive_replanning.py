"""Frozen-H4 uncertainty-aware episode-level R4-to-(R1/R4) evaluator."""
import argparse, csv, hashlib, json, os, time
from pathlib import Path

import numpy as np

from action_chunking import BCActionChunkController, load_bc_policy, sha256
from evaluate_h4_observation_noise import (make_noise_schedule, apply_observation_noise,
    _schedule_hash, region, NOISE_SEED)
from evaluate_h4_replanning import exclusive_run_lock
from franka_autonomous_env import FrankaAutonomousEnv

PROTOCOL=Path('bc_results/h4_adaptive_replanning_protocol.json')
PROTOCOL_SHA=Path('bc_results/h4_adaptive_replanning_protocol.sha256')
OUT=Path('bc_results/h4_adaptive_replanning')
FIELDS=['seed','model_seed','controller','threshold_m','noise_sigma_m','noise_generation_seed','success','steps','return',
 'cube_x','cube_y','region','first_close_step','first_lift_step','closed_fraction','bc_inference_calls','bc_total_inference_ms',
 'bc_mean_latency_ms','episode_wall_seconds','estimated_sigma_m','chosen_interval','decision_made','steps_before_decision',
 'stationarity_gt_max_displacement_m','terminal_chunk_index','base_noise_sha256','applied_noise_sha256','checkpoint_sha256','protocol_sha256']
RANGES={'smoke':[332000,332019],'smoke2':[333000,333019],'calibration':[330000,330299],'final':[331000,331999]}

def estimate_sigma(observations):
    a=np.asarray(observations,dtype=np.float64)
    if a.shape!=(20,2) or not np.isfinite(a).all(): raise ValueError('Estimator requires 20 finite XY observations')
    return float(np.sqrt((np.var(a[:,0],ddof=1)+np.var(a[:,1],ddof=1))/2.0))

def rollout(model,ck,model_seed,controller_name,sigma,seed,threshold,checkpoint_hash,protocol_hash,trace=False):
    env=FrankaAutonomousEnv(position_range=.20);env.max_move=float(ck['max_move']);env.max_time=float(ck['max_time'])
    max_steps=int(round(env.max_time/env.control_dt));base,applied=make_noise_schedule(seed,sigma,NOISE_SEED,max_steps)
    ctrl=BCActionChunkController(model,ck['obs_mean'].numpy(),ck['obs_std'].numpy(),int(ck['horizon']),replan_interval=1 if controller_name=='R1' else 4)
    obs,_=env.reset(seed=int(seed)); obs_noisy=apply_observation_noise(obs,applied[0]);start=time.perf_counter()
    ctrl.reset(obs_noisy);observed_xy=[[float(obs_noisy[3]),float(obs_noisy[4])]]
    gt_xy=[[float(env.data.xpos[env.cube_id][0]),float(env.data.xpos[env.cube_id][1])]]
    total_return=0.;steps=0;closes=[];lift='';closed=0;estimated=None
    chosen=None if controller_name in ('Adaptive','WarmupR1') else (1 if controller_name=='R1' else 4);decision=False
    while True:
        action=ctrl.planned_action.copy()
        if not np.isfinite(action).all(): raise ValueError('Non-finite action')
        if action[3]<0 and not closes: closes.append(steps)
        closed+=int(action[3]<0)
        nxt,reward,terminated,truncated,info=env.step(action);total_return+=reward;steps+=1
        if info['lift_height']>=.10 and not lift: lift=steps-1
        if terminated or truncated: break
        if steps>=len(applied): raise RuntimeError('Noise schedule exhausted')
        next_noisy=apply_observation_noise(nxt,applied[steps])
        gt_xy.append([float(env.data.xpos[env.cube_id][0]),float(env.data.xpos[env.cube_id][1])])
        if controller_name in ('Adaptive','WarmupR1') and not decision and steps<20:
            observed_xy.append([float(next_noisy[3]),float(next_noisy[4])])
        # Decide at the t=20 chunk boundary, before controller.advance can infer the next plan.
        if controller_name in ('Adaptive','WarmupR1') and not decision and steps==20:
            if len(observed_xy)!=20: raise RuntimeError(f'Expected 20 estimator observations, got {len(observed_xy)}')
            estimated=estimate_sigma(observed_xy)
            chosen=(1 if controller_name=='WarmupR1' else (4 if estimated<=threshold else 1));decision=True
            ctrl.replan_interval=chosen
        ctrl.advance(next_noisy)
    elapsed=time.perf_counter()-start;lat=np.asarray(ctrl.inference_latencies,dtype=float);x,y=float(info['cube_x']),float(info['cube_y'])
    gt=np.asarray(gt_xy[:20],float);motion=float(np.max(np.linalg.norm(gt-gt[0],axis=1))) if len(gt) else 0.
    row={'seed':int(seed),'model_seed':model_seed,'controller':controller_name,'threshold_m':'' if threshold is None else threshold,'noise_sigma_m':sigma,
      'noise_generation_seed':NOISE_SEED,'success':int(bool(info['is_success'])),'steps':steps,'return':total_return,'cube_x':x,'cube_y':y,'region':region(x,y),
      'first_close_step':closes[0] if closes else '','first_lift_step':lift,'closed_fraction':closed/steps,'bc_inference_calls':ctrl.inference_calls,
      'bc_total_inference_ms':float(lat.sum()*1000),'bc_mean_latency_ms':float(lat.mean()*1000),'episode_wall_seconds':elapsed,
      'estimated_sigma_m':'' if estimated is None else estimated,'chosen_interval':'' if chosen is None else chosen,'decision_made':int(decision),
      'steps_before_decision':min(steps,20),'stationarity_gt_max_displacement_m':motion,'terminal_chunk_index':ctrl.chunk_index,
      'base_noise_sha256':_schedule_hash(base),'applied_noise_sha256':_schedule_hash(applied),'checkpoint_sha256':checkpoint_hash,'protocol_sha256':protocol_hash}
    env.close();return row

def validate_rows(rows,seeds,m,r,controller,threshold,sigma,ckhash,prothash):
    got=[]
    for x in rows:
        got.append(int(x['seed']))
        if int(x['model_seed'])!=m or x['controller']!=controller or not np.isclose(float(x['noise_sigma_m']),sigma,rtol=0,atol=1e-12): raise ValueError('Cell mismatch')
        if x['checkpoint_sha256']!=ckhash or x['protocol_sha256']!=prothash: raise ValueError('Provenance mismatch')
        if (x['threshold_m']=='' and threshold is not None) or (x['threshold_m']!='' and threshold is not None and not np.isclose(float(x['threshold_m']),threshold)):raise ValueError('Threshold mismatch')
        if int(x['success']) not in (0,1) or len(x['base_noise_sha256'])!=64 or len(x['applied_noise_sha256'])!=64:raise ValueError('Malformed row')
    if len(got)!=len(set(got)) or got!=sorted(got) or not set(got)<=set(seeds): raise ValueError('Duplicate, unordered, or unexpected seed')
    return set(got)

def run(a):
    if not PROTOCOL.exists() or not PROTOCOL_SHA.exists(): raise FileNotFoundError('Preregistered protocol and SHA required')
    ph=PROTOCOL_SHA.read_text().split()[0]
    if sha256(PROTOCOL)!=ph: raise RuntimeError('Protocol hash mismatch')
    protocol=json.loads(PROTOCOL.read_text());m=a.model_seed;ckpath=Path(f'bc_models/action_chunking/H4_seed{m}.pt');ckhash=sha256(ckpath)
    if ckhash!=protocol['frozen_models'][str(m)]['sha256']:raise RuntimeError('Frozen checkpoint mismatch')
    model,ck=load_bc_policy(ckpath)
    if int(ck['horizon'])!=4 or int(ck['seed'])!=m:raise RuntimeError('Unexpected frozen model architecture/seed')
    if a.phase=='calibration': lo,hi=protocol['calibration']['seed_range_inclusive']
    else: lo,hi=protocol['evaluation'][f'{a.phase}_seed_range_inclusive']
    seeds=range(lo,hi+1)
    sigmas=(0,5,10) if a.phase=='smoke' else ((a.noise_mm,) if a.noise_mm is not None else (0,5,10))
    controllers=('R1','R4','Adaptive','WarmupR1') if a.phase in ('smoke','smoke2') else (('R1','R4','WarmupR1') if a.phase=='calibration' else ('R1','R4','Adaptive'))
    for sigma_mm in sigmas:
      for controller in controllers:
        threshold=a.threshold if controller=='Adaptive' else None
        if controller=='Adaptive' and threshold is None: raise ValueError('Adaptive cells require a frozen/candidate threshold')
        tag=f'{a.phase}_H4_seed{m}_{controller}_noise{sigma_mm}mm'+(f'_thr{threshold*1000:g}mm' if a.phase=='calibration' and threshold is not None else '')
        final=OUT/f'{tag}.csv';part=OUT/f'{tag}.partial.csv';meta=OUT/f'{tag}.json';cfg={'phase':a.phase,'model_seed':m,'controller':controller,'threshold_m':threshold,'noise_sigma_m':sigma_mm/1000,'seed_range_inclusive':[lo,hi],'checkpoint_sha256':ckhash,'protocol_sha256':ph,'noise_generation_seed':NOISE_SEED}
        if final.exists():
            if json.loads(meta.read_text())!=cfg:raise ValueError(f'Existing final metadata mismatch: {final}')
            with final.open(newline='') as f: rows=list(csv.DictReader(f))
            validate_rows(rows,seeds,m,None,controller,threshold,sigma_mm/1000,ckhash,ph)
            if len(rows)!=len(seeds):raise ValueError('Final file incomplete')
            print(f'Already complete {final}',flush=True);continue
        if not meta.exists():meta.write_text(json.dumps(cfg,indent=2,sort_keys=True)+'\n')
        elif json.loads(meta.read_text())!=cfg:raise ValueError(f'Partial metadata mismatch: {meta}')
        rows=[]
        if part.exists():
            with part.open(newline='') as f: rows=list(csv.DictReader(f))
        done=validate_rows(rows,seeds,m,None,controller,threshold,sigma_mm/1000,ckhash,ph) if rows else set();new=not part.exists()
        for i,s in enumerate(seeds,1):
            if s in done:continue
            row=rollout(model,ck,m,controller,sigma_mm/1000,s,threshold,ckhash,ph)
            with part.open('a',newline='',encoding='utf-8') as f:
                w=csv.DictWriter(f,fieldnames=FIELDS,lineterminator='\n');
                if new:w.writeheader();new=False
                w.writerow(row);f.flush();os.fsync(f.fileno())
            if i%20==0 or i==len(seeds):print(f'{tag}: {i}/{len(seeds)}',flush=True)
        with part.open(newline='') as f: rows=list(csv.DictReader(f))
        validate_rows(rows,seeds,m,None,controller,threshold,sigma_mm/1000,ckhash,ph)
        if len(rows)!=len(seeds):raise RuntimeError('Cell incomplete')
        part.replace(final);print(f'Completed {final}',flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=('smoke','smoke2','calibration','final'),required=True);p.add_argument('--model-seed',type=int,choices=(42,123,456),required=True);p.add_argument('--noise-mm',type=int,choices=(0,5,10));p.add_argument('--threshold',type=float);a=p.parse_args();OUT.mkdir(parents=True,exist_ok=True)
 with exclusive_run_lock(OUT/'.study_run.lock'):run(a)
if __name__=='__main__':main()

"""Paired synthetic XY observation-noise evaluation for frozen H4 BC policies."""
import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from action_chunking import BCActionChunkController, load_bc_policy, sha256
from evaluate_h4_replanning import exclusive_run_lock
from franka_autonomous_env import FrankaAutonomousEnv


PROTOCOL=Path('bc_results/h4_observation_noise_protocol.json')
PROTOCOL_SHA=Path('bc_results/h4_observation_noise_protocol.sha256')
OUT=Path('bc_results/h4_observation_noise')
NOISE_SEED=20261106
FIELDS=['seed','model_seed','replan_interval','noise_sigma_m','noise_generation_seed','success','steps','return',
        'cube_x','cube_y','region','first_close_step','first_lift_step','closed_fraction','bc_inference_calls',
        'bc_total_inference_ms','bc_mean_latency_ms','episode_wall_seconds','terminal_chunk_index',
        'base_noise_sha256','applied_noise_sha256','checkpoint_sha256','protocol_sha256']


def make_noise_schedule(evaluation_seed, noise_sigma_m, noise_seed=NOISE_SEED, max_control_steps=160):
    """Precompute common standard-normal XY errors indexed by absolute control step."""
    rng=np.random.default_rng(np.random.SeedSequence([int(noise_seed),int(evaluation_seed)]))
    base=rng.standard_normal((max_control_steps+1,2)).astype(np.float32)
    applied=(base*np.float32(noise_sigma_m)).astype(np.float32)
    return base,applied


def _schedule_hash(values):
    canonical=np.asarray(values,dtype='<f4',order='C')
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def apply_observation_noise(observation, xy_noise):
    """Perturb cube XY and matching relative XY only; never alter simulator state."""
    noisy=np.asarray(observation,dtype=np.float32).copy()
    xy=np.asarray(xy_noise,dtype=np.float32)
    if noisy.shape!=(12,) or xy.shape!=(2,) or not np.isfinite(xy).all():
        raise ValueError('Expected a finite 12D observation and finite 2D XY noise')
    noisy[[3,6]]+=xy[0]
    noisy[[4,7]]+=xy[1]
    return noisy


def region(x,y): return ('E' if x>=.55 else 'W')+('N' if y>=0 else 'S')


def phase_seed_range(evaluation,phase):
    if phase in evaluation:return evaluation[phase]
    key=f'{phase}_seed_range_inclusive'
    if key not in evaluation:raise KeyError(f'Missing preregistered seed range for phase {phase}')
    return evaluation[key]


def read_rows(path):
    with path.open(newline='',encoding='utf-8') as f:
        reader=csv.DictReader(f)
        if reader.fieldnames!=FIELDS: raise ValueError(f'Unexpected CSV schema in {path}')
        return list(reader)


def validate_rows(rows,seeds,model_seed,interval,sigma,checkpoint_hash,protocol_hash):
    expected=list(map(int,seeds));got=[]
    for r in rows:
        seed=int(r['seed']);got.append(seed)
        if (int(r['model_seed'])!=model_seed or int(r['replan_interval'])!=interval
                or not np.isclose(float(r['noise_sigma_m']),sigma,rtol=0,atol=1e-12)
                or int(r['noise_generation_seed'])!=NOISE_SEED):
            raise ValueError('Existing row has different model, interval, or noise configuration')
        if r['checkpoint_sha256']!=checkpoint_hash or r['protocol_sha256']!=protocol_hash:
            raise ValueError('Existing row checkpoint or protocol hash mismatch')
        if int(r['success']) not in (0,1) or int(r['steps'])<1 or int(r['bc_inference_calls'])<1:
            raise ValueError('Invalid result or controller counters in existing row')
        nums=[float(r[k]) for k in ('return','cube_x','cube_y','closed_fraction','bc_total_inference_ms','bc_mean_latency_ms','episode_wall_seconds')]
        if not np.isfinite(nums).all() or not 0<=nums[3]<=1 or nums[-1]<=0:
            raise ValueError('Non-finite/invalid measurement in existing row')
        if len(r['base_noise_sha256'])!=64 or len(r['applied_noise_sha256'])!=64:
            raise ValueError('Invalid noise schedule digest')
    if len(got)!=len(set(got)): raise ValueError('Duplicate seed in existing output; preserving file and stopping')
    if got!=sorted(got) or not set(got)<=set(expected): raise ValueError('Unexpected or out-of-order seed; preserving file and stopping')
    return set(got)


def append_row(path,row,new_file):
    with path.open('a',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=FIELDS,lineterminator='\n')
        if new_file:w.writeheader()
        w.writerow(row);f.flush();os.fsync(f.fileno())


def rollout(model,checkpoint,model_seed,interval,sigma,evaluation_seed,checkpoint_hash,protocol_hash,trace=False):
    env=FrankaAutonomousEnv(position_range=.20)
    env.max_move=float(checkpoint['max_move']);env.max_time=float(checkpoint['max_time'])
    if not np.isclose(env.control_dt,float(checkpoint['control_dt'])):
        raise ValueError('Environment timestep differs from frozen checkpoint')
    max_steps=int(round(env.max_time/env.control_dt))
    base_noise,applied_noise=make_noise_schedule(evaluation_seed,sigma,NOISE_SEED,max_steps)
    controller=BCActionChunkController(model,checkpoint['obs_mean'].numpy(),checkpoint['obs_std'].numpy(),
                                      int(checkpoint['horizon']),replan_interval=interval)
    obs,_=env.reset(seed=int(evaluation_seed));start=time.perf_counter()
    controller.reset(apply_observation_noise(obs,applied_noise[0]))
    total_return=0.;steps=0;closes=[];lift_step='';closed=0;trace_rows=[]
    while True:
        action=controller.planned_action.copy()
        if not np.isfinite(action).all():raise ValueError('Non-finite policy action')
        if action[3]<0 and not closes:closes.append(steps)
        closed+=int(action[3]<0)
        if trace:
            trace_rows.append({'step':steps,'chunk_index':controller.chunk_index,'planned_action':action.tolist(),
                'hand_xyz':env.data.site_xpos[env.site_id].copy().tolist(),'cube_xyz':env.data.xpos[env.cube_id].copy().tolist()})
        nxt,reward,terminated,truncated,info=env.step(action)
        total_return+=reward;steps+=1
        if info['lift_height']>=.10 and not lift_step:lift_step=steps-1
        if terminated or truncated:break
        if steps>=len(applied_noise):raise RuntimeError('Episode exceeded precomputed noise schedule')
        # Schedule index is the absolute environment control step, never call count.
        controller.advance(apply_observation_noise(nxt,applied_noise[steps]))
    elapsed=time.perf_counter()-start
    lat=np.asarray(controller.inference_latencies,dtype=float)
    x,y=float(info['cube_x']),float(info['cube_y'])
    row={'seed':int(evaluation_seed),'model_seed':model_seed,'replan_interval':interval,'noise_sigma_m':sigma,
        'noise_generation_seed':NOISE_SEED,'success':int(bool(info['is_success'])),'steps':steps,'return':total_return,
        'cube_x':x,'cube_y':y,'region':region(x,y),'first_close_step':closes[0] if closes else '',
        'first_lift_step':lift_step,'closed_fraction':closed/steps,'bc_inference_calls':controller.inference_calls,
        'bc_total_inference_ms':float(lat.sum()*1000),'bc_mean_latency_ms':float(lat.mean()*1000),
        'episode_wall_seconds':elapsed,'terminal_chunk_index':controller.chunk_index,
        'base_noise_sha256':_schedule_hash(base_noise),'applied_noise_sha256':_schedule_hash(applied_noise),
        'checkpoint_sha256':checkpoint_hash,'protocol_sha256':protocol_hash}
    env.close()
    return row,trace_rows


def run(a):
    if not PROTOCOL.is_file() or not PROTOCOL_SHA.is_file():raise FileNotFoundError('Committed noise protocol and SHA are required')
    protocol_hash=PROTOCOL_SHA.read_text().split()[0]
    if sha256(PROTOCOL)!=protocol_hash:raise RuntimeError('Noise protocol SHA mismatch')
    protocol=json.loads(PROTOCOL.read_text())
    model_path=Path(f'bc_models/action_chunking/H4_seed{a.model_seed}.pt');checkpoint_hash=sha256(model_path)
    model_prov=protocol['frozen_models'][str(a.model_seed)]
    if str(model_path)!=model_prov['path'] or checkpoint_hash!=model_prov['sha256']:
        raise RuntimeError('Frozen checkpoint hash differs from preregistration')
    model,ck=load_bc_policy(model_path)
    if int(ck['horizon'])!=4 or int(ck['seed'])!=a.model_seed:raise RuntimeError('Unexpected checkpoint architecture/seed')
    sigma=a.noise_mm/1000.0
    seed_range=phase_seed_range(protocol['evaluation'],a.phase)
    seeds=range(seed_range[0],seed_range[1]+1)
    tag=f'{a.phase}_H4_seed{a.model_seed}_R{a.interval}_noise{a.noise_mm}mm'
    final=OUT/f'{tag}.csv';partial=OUT/f'{tag}.partial.csv';meta=OUT/f'{tag}.json'
    config={'phase':a.phase,'model_seed':a.model_seed,'interval':a.interval,'noise_sigma_m':sigma,
        'noise_generation_seed':NOISE_SEED,'seed_range_inclusive':list(seed_range),'checkpoint_path':str(model_path),
        'checkpoint_sha256':checkpoint_hash,'protocol_sha256':protocol_hash,'horizon':4,
        'environment':protocol['environment'],'noise_algorithm':protocol['noise']['algorithm']}
    if final.exists():
        if not meta.is_file() or json.loads(meta.read_text())!=config:raise ValueError('Complete output metadata mismatch')
        rows=read_rows(final);validate_rows(rows,seeds,a.model_seed,a.interval,sigma,checkpoint_hash,protocol_hash)
        if len(rows)!=len(seeds):raise ValueError('Existing final output is incomplete; preserving it')
        print(f'Validated complete existing output: {final}',flush=True);return
    if meta.exists():
        if json.loads(meta.read_text())!=config:raise ValueError('Partial output metadata mismatch; preserving files')
    elif partial.exists():raise FileExistsError('Partial output has no metadata; preserving evidence')
    else:
        OUT.mkdir(parents=True,exist_ok=True);meta.write_text(json.dumps(config,indent=2,sort_keys=True)+'\n')
    rows=read_rows(partial) if partial.exists() else []
    done=validate_rows(rows,seeds,a.model_seed,a.interval,sigma,checkpoint_hash,protocol_hash)
    new_file=not partial.exists()
    for n,seed in enumerate(seeds,1):
        if seed in done:continue
        row,_=rollout(model,ck,a.model_seed,a.interval,sigma,seed,checkpoint_hash,protocol_hash)
        append_row(partial,row,new_file);new_file=False;done.add(seed)
        if n%100==0 or n==len(seeds):print(f'{tag}: {n}/{len(seeds)} verified/written',flush=True)
    rows=read_rows(partial);validate_rows(rows,seeds,a.model_seed,a.interval,sigma,checkpoint_hash,protocol_hash)
    if len(rows)!=len(seeds):raise RuntimeError('Final row count does not match preregistration')
    if final.exists():raise FileExistsError(f'Refusing to overwrite {final}')
    partial.replace(final);print(f'Completed {final}: {len(rows)} episodes',flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--phase',choices=('smoke','final'),required=True)
    p.add_argument('--model-seed',type=int,choices=(42,123,456),required=True)
    p.add_argument('--interval',type=int,choices=(1,4),required=True)
    p.add_argument('--noise-mm',type=int,choices=(0,5,10),required=True)
    a=p.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    with exclusive_run_lock(OUT/'.study_run.lock'):
        run(a)


if __name__=='__main__':main()

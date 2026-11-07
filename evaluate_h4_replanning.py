"""Run preregistered frozen-H4 runtime replanning ablations."""
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
from franka_autonomous_env import FrankaAutonomousEnv


PROTOCOL = Path('bc_results/h4_replanning_protocol.json')
PROTOCOL_SHA = Path('bc_results/h4_replanning_protocol.sha256')
OUT = Path('bc_results/h4_replanning')
FIELDS = ['seed','model_seed','replan_interval','success','steps','return','cube_x','cube_y','region',
          'first_close_step','first_lift_step','closed_fraction','bc_inference_calls',
          'bc_total_inference_ms','bc_mean_latency_ms','episode_wall_seconds','terminal_chunk_index',
          'checkpoint_sha256','protocol_sha256']


def region(x, y):
    return ('E' if x >= .55 else 'W') + ('N' if y >= 0 else 'S')


def read_rows(path):
    with path.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != FIELDS:
            raise ValueError(f'Unexpected CSV schema in {path}: {reader.fieldnames}')
        return list(reader)


def validate_rows(rows, seeds, model_seed, interval, checkpoint_hash, protocol_hash):
    expected = list(map(int, seeds))
    got = []
    for row in rows:
        seed = int(row['seed']); got.append(seed)
        if int(row['model_seed']) != model_seed or int(row['replan_interval']) != interval:
            raise ValueError('Partial/final CSV contains a row for a different experiment configuration')
        if row['checkpoint_sha256'] != checkpoint_hash or row['protocol_sha256'] != protocol_hash:
            raise ValueError('Checkpoint or preregistration hash differs in existing CSV row')
        if int(row['success']) not in (0, 1) or int(row['steps']) < 1 or int(row['bc_inference_calls']) < 1:
            raise ValueError('Invalid outcome or controller counters in existing CSV row')
        if int(row['bc_inference_calls']) > int(row['steps']):
            raise ValueError('Inference calls exceed control steps')
        vals = [float(row[k]) for k in ('return','cube_x','cube_y','closed_fraction','bc_total_inference_ms','bc_mean_latency_ms','episode_wall_seconds')]
        if not np.isfinite(vals).all() or not (0 <= vals[3] <= 1) or vals[6] <= 0:
            raise ValueError('Non-finite or invalid numeric value in existing CSV row')
    if len(got) != len(set(got)):
        raise ValueError('Duplicate evaluation seed in existing CSV; preserving output and stopping')
    if got != sorted(got) or not set(got) <= set(expected):
        raise ValueError('Unexpected or out-of-order seed in existing CSV; preserving output and stopping')
    return set(got)


def append_row(path, row, new_file):
    with path.open('a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, lineterminator='\n')
        if new_file:
            writer.writeheader()
        writer.writerow(row)
        f.flush(); os.fsync(f.fileno())


def rollout(model, checkpoint, model_seed, interval, seed, checkpoint_hash, protocol_hash, trace=False):
    env = FrankaAutonomousEnv(position_range=.20)
    env.max_move = float(checkpoint['max_move'])
    env.max_time = float(checkpoint['max_time'])
    if not np.isclose(env.control_dt, float(checkpoint['control_dt'])):
        raise ValueError('Environment control timestep differs from frozen checkpoint config')
    controller = BCActionChunkController(model, checkpoint['obs_mean'].numpy(), checkpoint['obs_std'].numpy(),
                                        int(checkpoint['horizon']), replan_interval=interval)
    obs, _ = env.reset(seed=int(seed)); start = time.perf_counter(); controller.reset(obs)
    total_return = 0.; steps = 0; closes = []; lift_step = ''
    closed = 0; trace_rows = []
    while True:
        action = controller.planned_action.copy()
        if action.shape != (4,) or not np.isfinite(action).all():
            raise ValueError('Controller emitted invalid action')
        if action[3] < 0 and not closes:
            closes.append(steps)
        closed += int(action[3] < 0)
        if trace:
            trace_rows.append({'step':steps,'planned_action':action.tolist(),'chunk_index':controller.chunk_index,
                               'hand_xyz':env.data.site_xpos[env.site_id].copy().tolist(),
                               'cube_xyz':env.data.xpos[env.cube_id].copy().tolist()})
        nxt, reward, terminated, truncated, info = env.step(action)
        total_return += reward; steps += 1
        if info['lift_height'] >= .10 and not lift_step:
            lift_step = steps - 1
        # Do not infer after a terminal/truncated transition.
        if terminated or truncated:
            break
        controller.advance(nxt)
    elapsed = time.perf_counter() - start
    latencies = np.asarray(controller.inference_latencies, dtype=float)
    x, y = float(info['cube_x']), float(info['cube_y'])
    row = {'seed':int(seed),'model_seed':model_seed,'replan_interval':interval,'success':int(bool(info['is_success'])),'steps':steps,
           'return':total_return,'cube_x':x,'cube_y':y,'region':region(x,y),'first_close_step':closes[0] if closes else '',
           'first_lift_step':lift_step,'closed_fraction':closed/steps,'bc_inference_calls':controller.inference_calls,
           'bc_total_inference_ms':float(latencies.sum()*1000),'bc_mean_latency_ms':float(latencies.mean()*1000),
           'episode_wall_seconds':elapsed,'terminal_chunk_index':controller.chunk_index,
           'checkpoint_sha256':checkpoint_hash,'protocol_sha256':protocol_hash}
    env.close()
    return row, trace_rows


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model-seed',type=int,choices=(42,123,456),required=True)
    p.add_argument('--interval',type=int,choices=(1,2,4),required=True)
    p.add_argument('--phase',choices=('smoke','final'),required=True)
    a=p.parse_args()
    if not PROTOCOL.exists() or not PROTOCOL_SHA.exists():
        raise FileNotFoundError('Preregistered protocol and SHA sidecar are required')
    expected_protocol_hash=PROTOCOL_SHA.read_text().split()[0]
    actual_protocol_hash=sha256(PROTOCOL)
    if actual_protocol_hash != expected_protocol_hash:
        raise RuntimeError('Preregistration SHA mismatch; refusing evaluation')
    protocol=json.loads(PROTOCOL.read_text())
    model_path=Path(f'bc_models/action_chunking/H4_seed{a.model_seed}.pt')
    checkpoint_hash=sha256(model_path)
    expected_model=protocol['frozen_models'][str(a.model_seed)]
    if str(model_path) != expected_model['path'] or checkpoint_hash != expected_model['sha256']:
        raise RuntimeError('Frozen checkpoint provenance mismatch')
    model, ck=load_bc_policy(model_path)
    if int(ck['horizon']) != 4 or int(ck['seed']) != a.model_seed:
        raise RuntimeError('Checkpoint architecture or seed does not match preregistration')
    for key in ('max_move','max_time','control_dt'):
        if not np.isclose(float(ck[key]),float(protocol['environment'][key])):
            raise RuntimeError(f'Checkpoint environment mismatch: {key}')
    # position_range in a BC checkpoint is training provenance (±0.12 m),
    # while this experiment deliberately evaluates at the preregistered ±0.20 m.
    if not np.isclose(float(ck['position_range']),0.12):
        raise RuntimeError('Unexpected training position range in frozen checkpoint')
    seeds_cfg=protocol['evaluation'][a.phase]
    seeds=range(seeds_cfg[0],seeds_cfg[1]+1)
    tag=f'{a.phase}_H4_seed{a.model_seed}_R{a.interval}'
    final=OUT/f'{tag}.csv'; partial=OUT/f'{tag}.partial.csv'; meta=OUT/f'{tag}.json'
    OUT.mkdir(parents=True,exist_ok=True)
    config={'phase':a.phase,'model_seed':a.model_seed,'interval':a.interval,'seed_range_inclusive':list(seeds_cfg),
            'checkpoint_path':str(model_path),'checkpoint_sha256':checkpoint_hash,'protocol_sha256':actual_protocol_hash,
            'horizon':4,'environment':protocol['environment']}
    if final.exists():
        rows=read_rows(final)
        validate_rows(rows,seeds,a.model_seed,a.interval,checkpoint_hash,actual_protocol_hash)
        if len(rows)==len(seeds):
            print(f'Validated complete existing file: {final}',flush=True); return
        raise FileExistsError(f'Incomplete final result exists; preserving evidence: {final}')
    if meta.exists():
        if json.loads(meta.read_text()) != config:
            raise ValueError(f'Existing run metadata differs; preserving evidence: {meta}')
    elif partial.exists():
        raise FileExistsError(f'Partial CSV has no run metadata; preserving evidence: {partial}')
    else:
        meta.write_text(json.dumps(config,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    rows=read_rows(partial) if partial.exists() else []
    done=validate_rows(rows,seeds,a.model_seed,a.interval,checkpoint_hash,actual_protocol_hash)
    new_file=not partial.exists()
    for i,seed in enumerate(seeds,1):
        if seed in done: continue
        row,_=rollout(model,ck,a.model_seed,a.interval,seed,checkpoint_hash,actual_protocol_hash)
        append_row(partial,row,new_file);new_file=False;done.add(seed)
        if i%100==0 or i==len(seeds):
            print(f'{tag}: {i}/{len(seeds)} rows verified/written',flush=True)
    rows=read_rows(partial)
    validate_rows(rows,seeds,a.model_seed,a.interval,checkpoint_hash,actual_protocol_hash)
    if len(rows)!=len(seeds): raise RuntimeError('Completed row count differs from preregistered range')
    if final.exists(): raise FileExistsError(f'Refusing to overwrite {final}')
    partial.replace(final)
    print(f'Completed {final}: {len(rows)} episodes',flush=True)


if __name__=='__main__': main()

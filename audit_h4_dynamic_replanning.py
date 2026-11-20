"""Count/provenance/pairing and preregistered stationarity gate, no success rates."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from action_chunking import sha256
from evaluate_h4_dynamic_replanning import (OUT, MODELS, PROFILES, CONTROLLERS, DECISIONS,
    protocol, read_rows, validate_rows, dynamic_noise, window_diagnostic)
from franka_autonomous_env import FrankaAutonomousEnv


def audit(phase, allow_incomplete=False):
    p,ph=protocol();lo,hi=p['evaluation'][phase];seeds=list(range(lo,hi+1))
    positions={};env=FrankaAutonomousEnv(position_range=.20)
    try:
        for seed in seeds:
            env.reset(seed=seed);positions[seed]=env.data.xpos[env.cube_id].copy()
    finally:env.close()
    cells=[];all_rows={};complete=0;partial=[];base_by={};applied_by={}
    maxima={c:0. for c in CONTROLLERS};bad_windows=[];bad_episodes=set();window_count=0
    for m in MODELS:
        ckhash=sha256(p['frozen_models'][str(m)]['path'])
        if ckhash!=p['frozen_models'][str(m)]['sha256']:raise RuntimeError('Model changed')
        for profile in PROFILES:
            for ctrl in CONTROLLERS:
                stem=f'{phase}_H4_seed{m}_{ctrl}_{profile}'
                full=OUT/f'{stem}.csv';part=OUT/f'{stem}.partial.csv';meta=OUT/f'{stem}.json'
                if full.exists() and part.exists():raise ValueError('Overlapping outputs')
                path=full if full.exists() else part if part.exists() else None
                if path is None:
                    cells.append({'file':full.name,'rows':0,'complete':False});continue
                metadata=json.loads(meta.read_text())
                expected={'phase':phase,'model_seed':m,'controller':ctrl,'profile':profile,'seed_range':[lo,hi],
                          'checkpoint_sha256':ckhash,'protocol_sha256':ph,'noise_generation_seed':p['noise']['generation_seed']}
                if any(metadata[k]!=v for k,v in expected.items()):raise ValueError('Cell metadata mismatch')
                rows=read_rows(path);validate_rows(rows,seeds,m,profile,ctrl,ckhash,ph)
                if full.exists() and len(rows)!=len(seeds):raise ValueError('Truncated completed file')
                if full.exists():complete+=1
                else:partial.append(path.name)
                all_rows[m,profile,ctrl]={int(r['seed']):r for r in rows}
                cells.append({'file':path.name,'rows':len(rows),'complete':full.exists(),'sha256':sha256(path)})
                for r in rows:
                    seed=int(r['seed']);pos=np.array([float(r[k]) for k in ('initial_cube_x','initial_cube_y','initial_cube_z')])
                    if not np.array_equal(pos,positions[seed]):raise ValueError('Initial position mismatch')
                    if seed in base_by and r['base_noise_sha256']!=base_by[seed]:raise ValueError('Base-noise mismatch')
                    base_by[seed]=r['base_noise_sha256'];key=(seed,profile)
                    if key in applied_by and r['applied_noise_sha256']!=applied_by[key]:raise ValueError('Applied-noise mismatch')
                    applied_by[key]=r['applied_noise_sha256']
                    events=json.loads(r['decisions_json']);event_steps={e['step'] for e in events}
                    for w in json.loads(r['windows_json']):
                        maxima[ctrl]=max(maxima[ctrl],w['gt_xy_diameter_m']);window_count+=1
                        required=(ctrl=='Online' and w['step'] in event_steps) or (ctrl=='OneShot' and w['step']==20 and 20 in event_steps)
                        if required and w['gt_xy_diameter_m']>p['stationarity_gate']['threshold_m']:
                            bad_windows.append({'model_seed':m,'profile':profile,'controller':ctrl,'seed':seed,**w})
                            bad_episodes.add((m,profile,ctrl,seed))
                    if seed==lo:
                        trace=json.loads(r['example_trace_json'])
                        if [t['step'] for t in trace]!=list(range(int(r['steps']))):raise ValueError('Example trace steps missing')
                        for t in trace:
                            expected_interval=1 if ctrl=='R1' else 4;last=0
                            for e in events:
                                if e['step']<=t['step']:expected_interval=e['after'];last=e['step']
                            if t['interval']!=expected_interval or t['chunk_index']!=(t['step']-last)%expected_interval:
                                raise ValueError('Trace chunk timing/gripper alignment mismatch')
                        noisy=[t['noisy_cube_xy'] for t in trace];gt=[t['gt_cube_xyz'] for t in trace]
                        _,_,sigma=dynamic_noise(seed,profile)
                        for w in json.loads(r['windows_json']):
                            recomputed=window_diagnostic(w['step'],noisy,gt,sigma)
                            if recomputed!=w:raise ValueError('Trace windows differ from stored diagnostics')
    prefix_checks=0
    for m in MODELS:
        for profile in PROFILES:
            paired=[all_rows.get((m,profile,c),{}) for c in ('R4','OneShot','Online')]
            for seed in set.intersection(*(set(r) for r in paired)):
                for key in ('prefix_actions_sha256','prefix_observations_sha256'):
                    if len({r[seed][key] for r in paired})!=1:raise ValueError('First20 trajectory mismatch')
                prefix_checks+=1
    done=complete==48 and not partial
    if not done and not allow_incomplete:raise RuntimeError(f'Incomplete {phase}: {complete}/48 cells')
    legacy_path=OUT/'legacy_equivalence_audit.json';legacy=json.loads(legacy_path.read_text()) if legacy_path.exists() else {}
    legacy_passed=legacy.get('passed',False) and legacy.get('protocol_sha256')==ph
    result={'phase':phase,'protocol_sha256':ph,'complete_cells':complete,'expected_cells':48,
            'total_rows':sum(c['rows'] for c in cells),'expected_rows':48*len(seeds),'all_cells_complete':done,
            'software_integrity_passed':done,'legacy_equivalence_passed':legacy_passed,
            'stationarity_passed':not bad_windows,'gate_passed':done and legacy_passed and not bad_windows,
            'stationarity_threshold_m':p['stationarity_gate']['threshold_m'],
            'maximum_xy_window_diameter_by_controller_m':maxima,'audited_windows':window_count,
            'violating_required_windows':len(bad_windows),'affected_required_episodes':len(bad_episodes),
            'violations':bad_windows,'first20_prefix_pairing_checks':prefix_checks,
            'initial_position_map_sha256':hashlib.sha256(np.array([positions[s] for s in seeds],dtype='<f8').tobytes()).hexdigest(),
            'cells':cells,'partial_files':partial,'no_success_statistics_computed':True}
    path=OUT/(f'{phase}_integrity_audit.json' if done else f'{phase}_progress_{complete}cells.json')
    with path.open('x') as f:f.write(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:result[k] for k in ('phase','complete_cells','total_rows','software_integrity_passed','legacy_equivalence_passed','stationarity_passed','violating_required_windows','affected_required_episodes','gate_passed')},indent=2))
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=('smoke','calibration','final'),required=True)
    parser.add_argument('--allow-incomplete',action='store_true');a=parser.parse_args();result=audit(a.phase,a.allow_incomplete)
    if result['all_cells_complete'] and not result['gate_passed']:raise SystemExit(2)


if __name__=='__main__':main()

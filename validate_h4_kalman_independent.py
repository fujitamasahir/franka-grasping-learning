"""Independent estimator-only Final: generate/audit/analyze, never auto-run."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np
from action_chunking import sha256
import run_h4_motion_estimator_stage1 as stage
import analyze_h4_motion_estimator_stage1 as analysis
from evaluate_h4_replanning import exclusive_run_lock
from kalman_noise_estimator import kalman_noise

PROTOCOL=Path('bc_results/h4_kalman_independent_protocol.json')
OUT=Path('bc_results/h4_kalman_independent_validation')
ESTIMATORS={**stage.ESTIMATORS,'kalman':lambda xy:kalman_noise(xy)['sigma_m']}


@contextmanager
def reuse(output):
    """Redirect only this process's legacy functions; never edit legacy files."""
    previous=(stage.OUT,stage.ESTIMATORS,analysis.ESTIMATORS)
    stage.OUT=Path(output);stage.ESTIMATORS=ESTIMATORS;analysis.ESTIMATORS=ESTIMATORS
    try:yield
    finally:stage.OUT,stage.ESTIMATORS,analysis.ESTIMATORS=previous


def load_protocol():
    p=json.loads(PROTOCOL.read_text());ph=sha256(PROTOCOL)
    if ph!=PROTOCOL.with_suffix('.sha256').read_text().split()[0]:raise ValueError('Protocol hash mismatch')
    committed=subprocess.check_output(['git','show',f'HEAD:{PROTOCOL}'])
    if hashlib.sha256(committed).hexdigest()!=ph:raise ValueError('Preregistration must be committed before execution')
    if p['seeds']!=list(range(352000,352020)) or p['noise_seed']!=20261109:raise ValueError('Independent seed specification changed')
    old=json.loads(Path('bc_results/h4_motion_estimator_protocol.json').read_text())
    if p['pass_criteria']!=old['pass_criteria']:raise ValueError('Stage1 criteria changed')
    for filename,digest in p['frozen_source_sha256'].items():
        if sha256(filename)!=digest:raise ValueError(f'Frozen implementation changed: {filename}')
    for spec in p['frozen_models'].values():
        if sha256(spec['path'])!=spec['sha256']:raise ValueError('Checkpoint changed')
    return p,ph


def expected_keys(p):
    return {(m,s,seed) for m in p['models'] for s in p['motion_sources'] for seed in p['seeds']}


def write_json(path,value):
    payload=json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n'
    with path.open('x') as stream:
        stream.write(payload);stream.flush();os.fsync(stream.fileno())


def generate(p,ph,output):
    if any((output/n).exists() for n in ('diagnostic_sequences.jsonl','diagnostic_sequences.partial.jsonl','dataset_manifest.json')):
        raise FileExistsError('Preserve existing complete/partial dataset; generation refuses overwrite')
    with reuse(output):
        stage.collect(p,ph)  # Existing clean-policy physical collector, new seeds/output.
        partial=output/'diagnostic_sequences.partial.jsonl'
        with partial.open('x') as stream:
            for m,source,seed in sorted(expected_keys(p)):
                a=json.loads((output/f'trajectory_{m}_{source}_{seed}.json').read_text())
                for profile in p['profiles']:
                    record=stage.evaluate_sequence(a['clean_observed_xy'],a['true_xyz'],profile,seed,p,
                                                   {'model_seed':m,'motion_source':source})
                    record['protocol_sha256']=ph
                    stream.write(json.dumps(record,separators=(',',':'),allow_nan=False)+'\n')
                    stream.flush();os.fsync(stream.fileno())
        partial.rename(output/'diagnostic_sequences.jsonl')
    paths=sorted(output.glob('trajectory_*.json'))+[output/'diagnostic_sequences.jsonl']
    write_json(output/'dataset_manifest.json',{'protocol_sha256':ph,'noise_seed':p['noise_seed'],
        'files_sha256':{x.name:sha256(x) for x in paths}})


def audit(p,ph,output):
    """Reconstruct all observations, estimates and diagnostic labels; no tuning."""
    manifest=json.loads((output/'dataset_manifest.json').read_text())
    if manifest['protocol_sha256']!=ph or manifest['noise_seed']!=p['noise_seed']:raise ValueError('Manifest provenance mismatch')
    trajectories={};expected=expected_keys(p)
    for m,source,seed in sorted(expected):
        path=output/f'trajectory_{m}_{source}_{seed}.json';a=json.loads(path.read_text())
        if (a['model_seed'],a['motion_source'],a['seed'])!=(m,source,seed):raise ValueError('Trajectory identity mismatch')
        if a['protocol_sha256']!=ph or a['checkpoint_sha256']!=p['frozen_models'][str(m)]['sha256']:raise ValueError('Trajectory provenance mismatch')
        n=len(a['true_xyz'])
        if not 40<=n<=160 or len(a['actions'])!=n or len(a['clean_observed_xy'])!=n:raise ValueError('Unexpected trajectory lengths')
        if any(k in a for k in ('success','return','reward')):raise ValueError('Success labels prohibited')
        if not all(np.isfinite(np.asarray(a[k],float)).all() for k in ('true_xyz','clean_observed_xy','actions')):raise ValueError('Nonfinite trajectory')
        trajectories[m,source,seed]=a
    # Reconstruct initial physical positions from the independent reset indices.
    initial={};env=stage.FrankaAutonomousEnv(position_range=.20)
    try:
        for seed in p['seeds']:
            env.reset(seed=seed);initial[seed]=env.data.xpos[env.cube_id].copy()
    finally:env.close()
    for (_,_,seed),trajectory in trajectories.items():
        if not np.array_equal(np.asarray(trajectory['true_xyz'][0]),initial[seed]):raise ValueError('Reset provenance mismatch: possible reused trajectory')
    paths={f'trajectory_{m}_{source}_{seed}.json' for m,source,seed in expected}|{'diagnostic_sequences.jsonl'}
    if set(manifest['files_sha256'])!=paths or {x.name for x in output.glob('trajectory_*.json')}!=paths-{'diagnostic_sequences.jsonl'}:raise ValueError('Unexpected/missing dataset files')
    for name,digest in manifest['files_sha256'].items():
        if sha256(output/name)!=digest:raise ValueError('Dataset bytes changed')
    sequences=[];seen=set();windows=0
    with reuse(output),(output/'diagnostic_sequences.jsonl').open() as stream:
        for line in stream:
            a=json.loads(line);i=a['identity'];key=(i['model_seed'],i['motion_source'],a['seed']);tag=key+(a['profile'],)
            if key not in expected or a['profile'] not in p['profiles'] or tag in seen:raise ValueError('Duplicate or foreign sequence')
            seen.add(tag);traj=trajectories[key]
            rebuilt=stage.evaluate_sequence(traj['clean_observed_xy'],traj['true_xyz'],a['profile'],a['seed'],p,i)
            rebuilt['protocol_sha256']=ph
            if a!=rebuilt:raise ValueError('Observation/noise/window/estimate/diagnostic reconstruction mismatch')
            windows+=len(a['windows']);sequences.append(a)
    if len(seen)!=480:raise ValueError('Incomplete480-sequence independent Final')
    # Existing development records have disjoint reset/noise indices; a new seed
    # alone cannot prove physical acquisition. Hash manifest and clean collector
    # provenance are retained, and reconstructed noisy streams exclude replay RNG reuse.
    old=set(subprocess.check_output(['git','ls-tree','-r','--name-only',p['base_commit']],text=True).splitlines())
    changed=set(subprocess.check_output(['git','diff',p['base_commit'],'--name-only'],text=True).splitlines())
    if old&changed:raise ValueError('Historical tracked artifacts changed')
    result={'passed':True,'protocol_sha256':ph,'trajectories':120,'sequences':480,'estimator_windows':windows,
            'seeds':p['seeds'],'noise_seed':p['noise_seed'],'development_seed_overlap':False,
            'model_hashes_verified':True,'historical_tracked_files_unchanged':len(old),
            'reconstructed_all_observations_estimates_and_labels':True,
            'dataset_manifest_sha256':sha256(output/'dataset_manifest.json')}
    return result,sequences


def gates(metrics,delays,rows,estimator,criteria):
    m=metrics[estimator];checks={}
    checks['coverage']=all(m[phase][s]['n']>=100 for phase in ('stationary','moving') for s in ('0','10'))
    def le(value,limit):return value is not None and value<=limit
    def ge(value,limit):return value is not None and value>=limit
    for sigma in ('0','10'):
        r=m['moving'][sigma]
        checks[f'moving_MAE_{sigma}mm']=le(r['MAE_mm'],criteria['moving_MAE_mm_max_each_noise'])
        checks[f'moving_classification_{sigma}mm']=ge(r['classification_accuracy'],criteria['moving_classification_accuracy_min_each_noise'])
        checks[f'moving_wrong_interval_{sigma}mm']=le(r['wrong_interval_fraction'],criteria['moving_false_selected_interval_fraction_max_each_noise'])
        checks[f'moving_wrong_switch_{sigma}mm']=le(r['wrong_direction_actual_switches_per_decision'],criteria['moving_wrong_direction_actual_switch_rate_per_decision_max_each_noise'])
        checks[f'stationary_MAE_{sigma}mm']=le(m['stationary'][sigma]['MAE_mm'],criteria['stationary_MAE_mm_max_each_noise'])
    def errors(name):return [abs(r['error']) for r in rows if r['estimator']==name and r['motion']=='moving' and not r['mixed']]
    baseline=errors('position');candidate=errors(estimator)
    reduction=1-float(np.mean(candidate))/float(np.mean(baseline)) if baseline and candidate and np.mean(baseline)>0 else None
    checks['relative_moving_MAE_reduction']=ge(reduction,criteria['moving_MAE_relative_reduction_vs_position_min'])
    for profile,d in delays[estimator].items():
        checks[f'detection_p90_{profile}']=le(d['p90_delay_steps'],criteria['dynamic_switch_detection_p90_steps_max_each_direction'])
        checks[f'detection_misses_{profile}']=le(d['miss_fraction'],criteria['dynamic_switch_miss_fraction_max_each_direction'])
    return {'all_passed':all(checks.values()),'checks':{k:'PASS' if v else 'FAIL' for k,v in checks.items()},'moving_MAE_relative_reduction':reduction}


def summarize(p,ph,sequences):
    rows=[w for s in sequences for w in s['windows']]
    with reuse(OUT):
        metrics=analysis.summarize(rows);delays=analysis.detection(sequences)
        per_seed={str(seed):{'metrics':analysis.summarize([r for r in rows if r['seed']==seed]),
                            'detection':analysis.detection([s for s in sequences if s['seed']==seed])} for seed in p['seeds']}
    checks={e:gates(metrics,delays,rows,e,p['pass_criteria']) for e in ESTIMATORS}
    mixed={}
    for e in ESTIMATORS:
        mixed[e]={}
        for profile in ('clean_to_noisy','noisy_to_clean'):
            rr=[r for r in rows if r['estimator']==e and r['mixed'] and r['profile']==profile]
            mixed[e][profile]={'n':len(rr),'MAE_vs_window_RMS_mm':float(np.mean([abs(r['error'])*1000 for r in rr])) if rr else None}
    # Seed-equal summaries and paired bootstrap: one shared resample per seed
    # jointly retains model/source/profile/estimator correspondence.
    rng=np.random.default_rng(p['bootstrap_seed']);indices=rng.integers(0,20,(p['bootstrap_replicates'],20));seed_equal={};bootstrap={}
    for e in ESTIMATORS:
        seed_equal[e]={};bootstrap[e]={}
        for sigma in ('0','10'):
            vals=[per_seed[str(seed)]['metrics'][e]['moving'][sigma] for seed in p['seeds']]
            seed_equal[e][sigma]={k:float(np.mean([v[k] for v in vals])) if all(v[k] is not None for v in vals) else None for k in vals[0] if k!='n'}
            a=[per_seed[str(seed)]['metrics'][e]['moving'][sigma]['MAE_mm'] for seed in p['seeds']]
            b=[per_seed[str(seed)]['metrics']['position']['moving'][sigma]['MAE_mm'] for seed in p['seeds']]
            if all(v is not None for v in a+b):
                diff=np.asarray(a)-np.asarray(b)
                bootstrap[e][sigma]={'MAE_difference_mm':float(diff.mean()),'paired_seed_bootstrap_95pct_CI_mm':np.quantile(diff[indices].mean(axis=1),[.025,.975]).tolist()}
            else:bootstrap[e][sigma]=None
    return {'protocol_sha256':ph,'kalman_accepted':checks['kalman']['all_passed'],'criteria_by_estimator':checks,
            'metrics':metrics,'detection':delays,'mixed_windows':mixed,'per_seed':per_seed,
            'seed_equal_moving_metrics':seed_equal,'paired_seed_bootstrap':bootstrap,
            'independent_unit':'20 reset/noise indices shared across models, motion sources and profiles; conditional on3 policies',
            'trajectory_count':120,'sequence_count':480,'no_grasp_success_evaluation':True,
            'parameters_changed_after_Final':False,'stage2_automatically_authorized':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('generate','audit','analyze','summary'));a=parser.parse_args()
    if a.command=='summary':
        s=json.loads((OUT/'summary.json').read_text());print(json.dumps({k:s[k] for k in ('kalman_accepted','criteria_by_estimator','metrics')},indent=2));return
    p,ph=load_protocol()
    if a.command=='generate':
        OUT.mkdir(exist_ok=True)
        with exclusive_run_lock(OUT/'.independent.lock'):generate(p,ph,OUT)
        print('Generated120 new trajectories and480 estimator diagnostic sequences; no analysis performed');return
    # An audit/analysis never writes the old Stage1 output location.
    with exclusive_run_lock(OUT/'.independent.lock'):
        result,sequences=audit(p,ph,OUT)
        if a.command=='audit':write_json(OUT/'integrity_audit.json',result);print('Independent dataset audit passed');return
        # Persist a fresh audit beside analysis; existing audits are verified.
        path=OUT/'integrity_audit.json'
        if path.exists():
            if json.loads(path.read_text())!=result:raise ValueError('Audit evidence changed')
        else:write_json(path,result)
        summary=summarize(p,ph,sequences);summary['integrity_audit_sha256']=sha256(path)
        write_json(OUT/'summary.json',summary)
        print(json.dumps({'kalman_accepted':summary['kalman_accepted'],'checks':summary['criteria_by_estimator']['kalman']['checks']},indent=2))


if __name__=='__main__':main()

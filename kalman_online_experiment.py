"""Frozen Kalman closed-loop experiment. CLI only; import never runs MuJoCo."""
import argparse
from contextlib import contextmanager
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
from action_chunking import sha256
import evaluate_h4_dynamic_replanning as legacy
from evaluate_h4_replanning import exclusive_run_lock
from evaluate_h4_observation_noise import make_noise_schedule, _schedule_hash
from h4_kalman_online_rule import KalmanReplanningRule
from kalman_noise_estimator import kalman_noise
from analyze_h4_replanning import exact_mcnemar,holm

PROTOCOL=Path('bc_results/h4_kalman_online_protocol.json')
OUT=Path('bc_results/h4_kalman_online_dynamic')
MODELS=(42,123,456);PROFILES=('clean_to_noisy','noisy_to_clean');CONTROLLERS=('R1','R4','OneShot','Online')
ORIGINAL_WINDOW=legacy.window_diagnostic


def write_json(path,value):
    payload=json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n'
    if path.exists():
        if path.read_text()!=payload:raise FileExistsError(path)
        return
    with path.open('x') as stream:stream.write(payload);stream.flush();os.fsync(stream.fileno())


def protocol():
    p=json.loads(PROTOCOL.read_text());ph=sha256(PROTOCOL)
    if ph!=PROTOCOL.with_suffix('.sha256').read_text().split()[0]:raise ValueError('Protocol changed')
    if hashlib.sha256(subprocess.check_output(['git','show',f'HEAD:{PROTOCOL}'])).hexdigest()!=ph:raise ValueError('Preregistration not committed')
    for name,digest in {**p['frozen_source_sha256'],**p['independent_validation_evidence'],**p['prior_protected_untracked_files']}.items():
        if sha256(name)!=digest:raise ValueError(f'Frozen source/evidence changed: {name}')
    for spec in p['frozen_models'].values():
        if sha256(spec['path'])!=spec['sha256']:raise ValueError('Checkpoint changed')
    return p,ph


def dynamic_noise(seed,profile,max_steps=160):
    if profile not in PROFILES:raise ValueError('Dynamic profiles only')
    base,_=make_noise_schedule(seed,1.,20261110,max_steps)
    before,after=(0.,.01) if profile=='clean_to_noisy' else (.01,0.)
    sigma=np.full(max_steps+1,before,dtype=np.float32);sigma[40:]=after
    return base,(base*sigma[:,None]).astype(np.float32),sigma


def window_diagnostic(step,observed,truth,sigma):
    result=ORIGINAL_WINDOW(step,observed,truth,sigma)
    result['estimated_sigma_m']=kalman_noise(observed[step-20:step])['sigma_m']
    return result


@contextmanager
def reuse():
    previous=(legacy.NOISE_SEED,legacy.dynamic_noise,legacy.RollingReplanningRule,legacy.window_diagnostic)
    legacy.NOISE_SEED=20261110;legacy.dynamic_noise=dynamic_noise
    legacy.RollingReplanningRule=KalmanReplanningRule;legacy.window_diagnostic=window_diagnostic
    try:yield
    finally:legacy.NOISE_SEED,legacy.dynamic_noise,legacy.RollingReplanningRule,legacy.window_diagnostic=previous


def cutoff():return datetime.now(ZoneInfo('America/Denver')).date().isoformat()>'2026-11-14'


def cells(phase,p):
    lo,hi=p['evaluation'][phase]
    for m in MODELS:
        for profile in PROFILES:
            for controller in CONTROLLERS:
                stem=f'{phase}_H4_seed{m}_{controller}_{profile}'
                cfg={'phase':phase,'model_seed':m,'profile':profile,'controller':controller,'seed_range':[lo,hi],
                     'checkpoint_sha256':p['frozen_models'][str(m)]['sha256'],'protocol_sha256':sha256(PROTOCOL),
                     'noise_generation_seed':20261110,'runner_sha256':sha256(__file__),'schema':legacy.FIELDS}
                yield stem,cfg


def verify_saved_audit(phase,p,ph):
    a=json.loads((OUT/f'{phase}_audit.json').read_text())
    if a.get('phase')!=phase or a.get('protocol_sha256')!=ph or not a.get('gate_passed') or a.get('complete_cells')!=24:raise RuntimeError(f'Complete passing {phase} audit required')
    if a.get('runner_sha256')!=sha256(__file__) or len(a.get('cells',[]))!=24:raise RuntimeError('Audit runner/cell count mismatch')
    expected_names={stem for stem,_ in cells(phase,p)}
    if {c['cell'] for c in a['cells']}!=expected_names or not all(c['complete'] for c in a['cells']):raise RuntimeError('Missing audited cells')
    expected=240 if phase=='smoke' else 24000
    if a.get('total_rows')!=expected:raise RuntimeError('Wrong audit count')
    for cell in a['cells']:
        for name,digest in cell['files_sha256'].items():
            if sha256(OUT/name)!=digest:raise ValueError('Audited evidence changed')
    return a


def run(phase,p,ph):
    if phase=='final':
        smoke=verify_saved_audit('smoke',p,ph)
        estimate=smoke.get('estimated_final_wall_hours')
        if estimate is not None:
            written=0
            with reuse():
                for stem,cfg in cells(phase,p):
                    for suffix in ('.csv','.partial.csv'):
                        path=OUT/f'{stem}{suffix}'
                        if path.exists():
                            rows=legacy.read_rows(path);lo,hi=cfg['seed_range']
                            legacy.validate_rows(rows,range(lo,hi+1),cfg['model_seed'],cfg['profile'],cfg['controller'],cfg['checkpoint_sha256'],ph);written+=len(rows)
            deadline=datetime.fromisoformat(p['research_cutoff'])
            remaining_hours=(deadline-datetime.now(ZoneInfo('America/Denver'))).total_seconds()/3600
            if estimate*max(0,24000-written)/24000>remaining_hours:
                print(json.dumps({'stopped':'runtime forecast exceeds hard deadline; evidence preserved','remaining_episodes':24000-written}));return
    start=time.perf_counter();new_episodes=0;reason='complete'
    try:
        model_seed=None;model=ck=None
        with reuse():
            for stem,cfg in cells(phase,p):
                if (OUT/'.STOP_AFTER_CELL').exists():reason='logistical_cell_boundary_pause';return
                m=cfg['model_seed'];profile=cfg['profile'];controller=cfg['controller']
                if model_seed!=m:
                    model,ck,ckhash=legacy.check_model(m,p);model_seed=m
                full=OUT/f'{stem}.csv';partial=OUT/f'{stem}.partial.csv';meta=OUT/f'{stem}.json'
                if full.exists() and partial.exists():raise ValueError('Overlapping complete/partial outputs')
                if meta.exists():
                    if json.loads(meta.read_text())!=cfg:raise ValueError('Configuration mismatch')
                elif full.exists() or partial.exists():raise ValueError('Missing output metadata')
                else:write_json(meta,cfg)
                seeds=range(cfg['seed_range'][0],cfg['seed_range'][1]+1)
                path=full if full.exists() else partial
                rows=legacy.read_rows(path) if path.exists() else []
                done=legacy.validate_rows(rows,seeds,m,profile,controller,ckhash,ph)
                if full.exists():
                    if len(done)!=len(seeds):raise ValueError('Truncated complete cell')
                    continue
                for seed in seeds:
                    if seed in done:continue
                    if cutoff():reason='hard_deadline';return
                    row=legacy.rollout(model,ck,m,profile,controller,seed,ckhash,ph,example=seed==seeds.start)
                    new=not partial.exists()
                    with partial.open('a',newline='') as stream:
                        writer=csv.DictWriter(stream,fieldnames=legacy.FIELDS,lineterminator='\n')
                        if new:writer.writeheader()
                        writer.writerow(row);stream.flush();os.fsync(stream.fileno())
                    new_episodes+=1
                legacy.validate_rows(legacy.read_rows(partial),seeds,m,profile,controller,ckhash,ph)
                if full.exists():raise FileExistsError(full)
                partial.rename(full)
                if sha256(p['frozen_models'][str(m)]['path'])!=ckhash:raise ValueError('Checkpoint changed')
                # No outcome or per-episode log output.
    except BaseException:
        reason='interrupted_or_integrity_error';raise
    finally:
        record={'phase':phase,'new_episodes':new_episodes,'elapsed_wall_seconds':time.perf_counter()-start,'reason':reason,'protocol_sha256':ph}
        with (OUT/'runtime_journal.jsonl').open('a') as stream:
            stream.write(json.dumps(record)+'\n');stream.flush();os.fsync(stream.fileno())
        print(json.dumps(record))


def audit(phase,p,ph,allow_incomplete=False):
    if phase=='final':verify_saved_audit('smoke',p,ph)
    lo,hi=p['evaluation'][phase];seeds=range(lo,hi+1);positions={}
    env=legacy.FrankaAutonomousEnv(position_range=.20)
    try:
        for seed in seeds:env.reset(seed=seed);positions[seed]=env.data.xpos[env.cube_id].copy()
    finally:env.close()
    records=[];paired={};total=0;complete=0;prefix_checks=0
    with reuse():
        for stem,cfg in cells(phase,p):
            full=OUT/f'{stem}.csv';partial=OUT/f'{stem}.partial.csv';meta=OUT/f'{stem}.json'
            if full.exists() and partial.exists():raise ValueError('Overlapping outputs')
            path=full if full.exists() else partial if partial.exists() else None
            if path is None:records.append({'cell':stem,'rows':0,'complete':False,'files_sha256':{}});continue
            if json.loads(meta.read_text())!=cfg:raise ValueError('Metadata mismatch')
            rows=legacy.read_rows(path);m=cfg['model_seed'];profile=cfg['profile'];controller=cfg['controller']
            legacy.validate_rows(rows,seeds,m,profile,controller,cfg['checkpoint_sha256'],ph)
            if full.exists() and len(rows)!=len(seeds):raise ValueError('Truncated complete cell')
            complete+=int(full.exists());total+=len(rows)
            records.append({'cell':stem,'rows':len(rows),'complete':full.exists(),'files_sha256':{x.name:sha256(x) for x in (path,meta)}})
            for r in rows:
                seed=int(r['seed']);pos=np.array([float(r[k]) for k in ('initial_cube_x','initial_cube_y','initial_cube_z')])
                if not np.array_equal(pos,positions[seed]):raise ValueError('Seed/initial position mismatch')
                key=(m,profile,seed);paired.setdefault(key,{})[controller]=r
                if seed==lo:
                    trace=json.loads(r['example_trace_json']);events=json.loads(r['decisions_json'])
                    if [x['step'] for x in trace]!=list(range(int(r['steps']))):raise ValueError('Missing example trace')
                    rule=KalmanReplanningRule(controller=='Online') if controller in ('OneShot','Online') else None
                    reconstructed=[];last=0;interval=1 if controller=='R1' else 4
                    for t in trace:
                        if rule:
                            event=rule.decide()
                            if event:reconstructed.append(event);last=event['step'];interval=event['after']
                            rule.observe(t['noisy_cube_xy'])
                        if t['interval']!=interval or t['chunk_index']!=(t['step']-last)%interval:raise ValueError('Invalid fresh chunk timing')
                        if len(t['action'])!=4 or not np.isfinite(t['action']).all() or t['action'][3] not in (-1,1):raise ValueError('Invalid/gripper action')
                    if reconstructed!=events:raise ValueError('Observation-only rule reconstruction mismatch')
                    noisy=[t['noisy_cube_xy'] for t in trace];truth=[t['gt_cube_xyz'] for t in trace];_,_,sigma=dynamic_noise(seed,profile)
                    for w in json.loads(r['windows_json']):
                        if window_diagnostic(w['step'],noisy,truth,sigma)!=w:raise ValueError('Window reconstruction mismatch')
    for key,group in paired.items():
        if len({r['base_noise_sha256'] for r in group.values()})!=1 or len({r['applied_noise_sha256'] for r in group.values()})!=1:raise ValueError('Policy noise pairing mismatch')
        adaptive=[group[c] for c in ('R4','OneShot','Online') if c in group]
        if len(adaptive)==3:
            if any(len({r[k] for r in adaptive})!=1 for k in ('prefix_actions_sha256','prefix_observations_sha256')):raise ValueError('First20 prefix mismatch')
            prefix_checks+=1
    passed=complete==24 and total==24*len(seeds)
    if not passed and not allow_incomplete:raise RuntimeError(f'Incomplete {phase}: {complete}/24 cells; outcomes not summarized')
    journal=[json.loads(x) for x in (OUT/'runtime_journal.jsonl').read_text().splitlines()] if (OUT/'runtime_journal.jsonl').exists() else []
    smoke=[x for x in journal if x['phase']=='smoke' and x['protocol_sha256']==ph]
    count=sum(x['new_episodes'] for x in smoke);duration=sum(x['elapsed_wall_seconds'] for x in smoke)
    result={'phase':phase,'protocol_sha256':ph,'runner_sha256':sha256(__file__),'gate_passed':passed,'complete_cells':complete,'total_rows':total,
            'expected_rows':24*len(seeds),'cells':records,'prefix_pairing_checks':prefix_checks,
            'software_integrity_passed':passed,'finite_estimates_and_decision_timing_verified':True,
            'stationarity_not_an_acceptance_gate':True,'no_success_statistics_computed':True,
            'smoke_runtime_seconds':duration,'smoke_timed_episodes':count,
            'estimated_final_wall_hours':duration/count*24000/3600 if count else None,
            'timing_note':'Extrapolated active execution time including model load/validation; hardware/load/audit overhead may differ.'}
    path=OUT/(f'{phase}_audit.json' if passed else f'{phase}_progress_{total}rows.json')
    write_json(path,result)
    print(json.dumps({k:result[k] for k in ('phase','gate_passed','complete_cells','total_rows','estimated_final_wall_hours')}))
    return result


def paired_ci(differences,indices):return (100*np.quantile(differences[indices].mean(axis=1),[.025,.975])).tolist()


def summarize(data,p):
    models=p['evaluation']['model_seeds'];profiles=p['evaluation']['profiles']
    keys=[(m,q) for m in models for q in profiles];n=len(next(iter(data.values())))
    bootstrap=p['statistics']['bootstrap_replicates'];rng=np.random.default_rng(p['statistics']['bootstrap_seed'])
    # Bounded ~80MB at10000x1000; batches keep indices under2MB.
    success={k:np.array([int(r['success']) for r in rows],float) for k,rows in data.items()}
    primary_d=np.mean([success[m,q,'Online']-success[m,q,'OneShot'] for m,q in keys],axis=0)
    distributions=[]
    for start in range(0,bootstrap,200):
        ix=rng.integers(0,n,(min(200,bootstrap-start),n));distributions.extend(primary_d[ix].mean(axis=1))
    ci=(100*np.quantile(distributions,[.025,.975])).tolist();effect=100*float(primary_d.mean())
    comparisons=[];cell_metrics=[];switching=[]
    for m,q in keys:
        online=data[m,q,'Online'];a=success[m,q,'Online']
        for baseline in ('OneShot','R1','R4'):
            rows=data[m,q,baseline];b=success[m,q,baseline]
            beneficial=int(((b==0)&(a==1)).sum());harmful=int(((b==1)&(a==0)).sum())
            if int(a.sum()-b.sum())!=beneficial-harmful:raise AssertionError('Paired arithmetic')
            mc=exact_mcnemar(dict(enumerate(rows)),dict(enumerate(online)))
            comparisons.append({'model_seed':m,'profile':q,'baseline':baseline,'difference_pp':100*float((a-b).mean()),
                'beneficial':beneficial,'harmful':harmful,'both_success':int(((a==1)&(b==1)).sum()),'both_failure':int(((a==0)&(b==0)).sum()),**mc})
        for controller in CONTROLLERS:
            rows=data[m,q,controller];events=[json.loads(r['decisions_json']) for r in rows]
            delays=[];misses=0;already=0;target=1 if q=='clean_to_noisy' else 4
            for ee in events:
                current=4
                for e in ee:
                    if e['step']<=40:current=e['after']
                already+=current==target
                found=[e['step']-40 for e in ee if e['step']>=40 and e['switched'] and e['after']==target]
                if found:delays.append(found[0])
                else:misses+=1
            switching.append({'model_seed':m,'profile':q,'controller':controller,'switch_R4_to_R1':sum(int(r['switch_R4_to_R1']) for r in rows),'switch_R1_to_R4':sum(int(r['switch_R1_to_R4']) for r in rows),'already_target_at40':already if controller in ('OneShot','Online') else None,'actual_post_change_switch_misses':misses if controller in ('OneShot','Online') else None,'median_actual_switch_delay_steps':float(np.median(delays)) if delays else None,'max_actual_switch_delay_steps':max(delays) if delays else None})
            metric={'model_seed':m,'profile':q,'controller':controller,'episodes':n,'success_rate_pct':100*float(success[m,q,controller].mean())}
            for field in ('bc_inference_calls','bc_total_inference_ms','episode_wall_seconds','steps','steps_R1','steps_R4','decision_count'):
                metric['mean_'+field]=float(np.mean([float(r[field]) for r in rows]))
            cell_metrics.append(metric)
    for baseline in ('OneShot','R1','R4'):
        selected=[r for r in comparisons if r['baseline']==baseline]
        for row,value in zip(selected,holm([r['exact_two_sided_p'] for r in selected])):row['holm_adjusted_p']=float(value)
    effects=[100*float(np.mean([success[m,q,'Online']-success[m,q,'OneShot'] for q in profiles])) for m in models]
    return {'primary_online_minus_oneshot_pp':effect,'paired_bootstrap_95pct_CI_pp':ci,
            'clear_improvement':effect>0 and ci[0]>0,'interpretation':'Conditional on three frozen policies; positive mean with CI containing0 does not establish clear improvement.',
            'primary_effect_by_model_pp':dict(zip(map(str,models),effects)),
            'seed_descriptives':{'mean':float(np.mean(effects)),'median':float(np.median(effects)),'sample_sd':float(np.std(effects,ddof=1)),'min':min(effects),'max':max(effects),'positive_models':sum(v>0 for v in effects)},
            'paired_comparisons':comparisons,'cell_metrics':cell_metrics,'switching':switching,'timing_is_descriptive':True}


def analyze(p,ph):
    verify_saved_audit('final',p,ph)
    data={}
    with reuse():
        for stem,cfg in cells('final',p):
            rows=legacy.read_rows(OUT/f'{stem}.csv');m=cfg['model_seed'];q=cfg['profile'];c=cfg['controller'];lo,hi=cfg['seed_range']
            legacy.validate_rows(rows,range(lo,hi+1),m,q,c,cfg['checkpoint_sha256'],ph)
            if len(rows)!=1000:raise ValueError('Incomplete Final cell')
            data[m,q,c]=rows
    result=summarize(data,p);result['protocol_sha256']=ph;result['final_audit_sha256']=sha256(OUT/'final_audit.json')
    write_json(OUT/'final_summary.json',result)
    for name,rows in (('final_cells.csv',result['cell_metrics']),('final_pairs.csv',result['paired_comparisons'])):
        path=OUT/name
        if path.exists():raise FileExistsError(path)
        with path.open('x',newline='') as stream:
            w=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    print(json.dumps({k:result[k] for k in ('primary_online_minus_oneshot_pp','paired_bootstrap_95pct_CI_pp','clear_improvement')}))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('run','audit','analyze'))
    parser.add_argument('--phase',choices=('smoke','final'));parser.add_argument('--allow-incomplete',action='store_true');a=parser.parse_args()
    if a.command in ('run','audit') and a.phase is None:parser.error('--phase required')
    p,ph=protocol()
    if a.command=='run':OUT.mkdir(exist_ok=True)
    with exclusive_run_lock(OUT/'.study_run.lock'):
        if a.command=='run':run(a.phase,p,ph)
        elif a.command=='audit':audit(a.phase,p,ph,a.allow_incomplete)
        else:analyze(p,ph)


if __name__=='__main__':main()

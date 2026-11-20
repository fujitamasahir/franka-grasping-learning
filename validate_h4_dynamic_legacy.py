"""Regress constant profiles against unchanged One-Shot/R1/R4 legacy code."""
import json
import numpy as np
from action_chunking import sha256
from evaluate_h4_adaptive_replanning import rollout as legacy_rollout
from evaluate_h4_dynamic_replanning import OUT, protocol, check_model, read_rows, rollout


def main():
    p,ph=protocol();checks=[]
    exact=('success','steps','first_close_step','first_lift_step','bc_inference_calls','terminal_chunk_index','base_noise_sha256','applied_noise_sha256')
    numeric=('return','closed_fraction')
    for m in (42,123,456):
        model,ck,digest=check_model(m,p)
        for profile,sigma in (('clean',0.),('noisy',.01)):
            for controller in ('R1','R4','OneShot'):
                path=OUT/f'smoke_H4_seed{m}_{controller}_{profile}.csv'
                rows=read_rows(path)
                for seed in p['evaluation']['legacy_regression_seeds']:
                    new=next(r for r in rows if int(r['seed'])==seed)
                    old=legacy_rollout(model,ck,m,'Adaptive' if controller=='OneShot' else controller,sigma,seed,
                                       .0075 if controller=='OneShot' else None,digest,ph)
                    for key in exact:
                        if str(old[key])!=new[key]:raise AssertionError((m,profile,controller,seed,key))
                    for key in numeric:
                        if not np.isclose(float(old[key]),float(new[key]),rtol=1e-9,atol=1e-9):raise AssertionError(key)
                    if controller=='OneShot':
                        event=json.loads(new['decisions_json'])[0]
                        if event['after']!=old['chosen_interval'] or not np.isclose(event['estimated_sigma_m'],old['estimated_sigma_m'],rtol=0,atol=1e-15):raise AssertionError('OneShot decision mismatch')
                    checks.append({'model_seed':m,'profile':profile,'controller':controller,'seed':seed,'passed':True})
        # Independent repeated Online rollout for one predetermined seed/profile.
        reference=read_rows(OUT/f'smoke_H4_seed{m}_Online_clean_to_noisy.csv')[0]
        again=rollout(model,ck,m,'clean_to_noisy','Online',342000,digest,ph,True)
        timing={'episode_wall_seconds','bc_mean_latency_ms','bc_total_inference_ms'}
        for key in reference:
            if key in timing:continue
            if str(again[key])!=reference[key]:raise AssertionError(('determinism',m,key))
        print(f'Legacy equivalence and Online determinism passed: model {m}',flush=True)
    result={'passed':True,'protocol_sha256':ph,'legacy_regression_checks':checks,'online_deterministic_reproductions':3,
            'source_legacy_file_sha256':sha256('evaluate_h4_adaptive_replanning.py'),'floating_tolerance':{'rtol':1e-9,'atol':1e-9},'timing_excluded':True}
    with (OUT/'legacy_equivalence_audit.json').open('x') as f:f.write(json.dumps(result,indent=2,sort_keys=True)+'\n')


if __name__=='__main__':main()

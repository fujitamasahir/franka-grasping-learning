"""Reconstruct estimates/noise/window labels, verify independence and provenance."""
import json,subprocess
from pathlib import Path
import numpy as np
from action_chunking import sha256
from run_h4_motion_estimator_stage1 import OUT,PROTOCOL,ESTIMATORS,load_protocol


def main():
    p,ph=load_protocol();base=p['base_commit']
    old=set(subprocess.check_output(['git','ls-tree','-r','--name-only',base],text=True).splitlines())
    changes=set(subprocess.check_output(['git','diff',base,'--name-only'],text=True).splitlines())
    assert not old.intersection(changes),'Historical modification'
    paths=sorted(OUT.glob('trajectory_*.json'));assert len(paths)==120
    trajectories={}
    for path in paths:
        a=json.loads(path.read_text());key=(a['model_seed'],a['motion_source'],a['seed'])
        assert key not in trajectories and a['protocol_sha256']==ph
        assert a['checkpoint_sha256']==p['frozen_models'][str(key[0])]['sha256']
        assert not any(k in a for k in ('success','return','reward'))
        assert 20<=len(a['true_xyz'])<=160 and len(a['actions'])==len(a['true_xyz'])==len(a['clean_observed_xy'])
        trajectories[key]=a
    expected={(m,s,seed) for m in p['models'] for s in p['motion_sources'] for seed in p['seeds']};assert set(trajectories)==expected
    checks=0;seen=set();paired={};counter=0
    for line in (OUT/'diagnostic_sequences.jsonl').open():
        a=json.loads(line);ident=a['identity'];key=(ident['model_seed'],ident['motion_source'],a['seed']);assert a['protocol_sha256']==ph
        tag=key+(a['profile'],);assert tag not in seen;seen.add(tag)
        t=trajectories[key];profile=a['profile'];before,after=p['profiles'][profile]
        sig=np.full(len(t['true_xyz']),before,dtype=np.float64);sig[40:]=after
        rng=np.random.default_rng(np.random.SeedSequence([p['noise_seed'],a['seed']]))
        z=rng.standard_normal((len(sig),2)).astype(np.float32)
        obs=(np.asarray(t['clean_observed_xy'],np.float32)+z*sig.astype(np.float32)[:,None]).astype(np.float32)
        np.testing.assert_array_equal(obs,np.asarray(a['observed_xy'],np.float32))
        assert not any(k in a for k in ('success','return','reward'))
        for e,fn in ESTIMATORS.items():
            rows=[r for r in a['windows'] if r['estimator']==e]
            assert [r['step'] for r in rows]==list(range(20,len(obs)+1,4))
            for r in rows:
                end=r['step'];assert (r['window_start'],r['window_end'])==(end-20,end-1)
                assert fn(obs[end-20:end])==r['sigma_hat'];checks+=1
                xyz=np.asarray(t['true_xyz'][end-20:end]);diam=np.linalg.norm(xyz[:,None,:2]-xyz[None,:,:2],axis=2).max()
                assert diam==r['true_xy_diameter'];assert r['motion']==('moving' if diam>.001 else 'stationary')
                assert np.isclose(r['target_sigma'],np.sqrt(np.mean(sig[end-20:end]**2)))
        counter+=1
    assert len(seen)==480
    for spec in p['frozen_models'].values():assert sha256(spec['path'])==spec['sha256']
    summary=json.loads((OUT/'stage1_summary.json').read_text());assert summary['stage1_passed']==all(summary['checks'].values())
    data={'passed':True,'protocol_sha256':ph,'unique_actual_trajectories':120,'unique_replay_sequences':counter,'reconstructed_observation_only_estimates':checks,'historical_files_unchanged':len(old),'model_hashes_unchanged':True,'success_labels_not_recorded':True,'stage2_gate':summary['stage1_passed'],'files_sha256':{x.name:sha256(x) for x in sorted(OUT.iterdir()) if x.is_file() and x.name!='stage1_integrity_audit.json'}}
    with (OUT/'stage1_integrity_audit.json').open('x') as f:f.write(json.dumps(data,indent=2,sort_keys=True)+'\n')
    print({k:v for k,v in data.items() if k!='files_sha256'})


if __name__=='__main__':main()

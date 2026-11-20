"""New estimator-only motion dataset; no success labels or adaptive control."""
import json,os,hashlib,subprocess
from pathlib import Path
import numpy as np
from action_chunking import BCActionChunkController,load_bc_policy,sha256
from evaluate_h4_replanning import exclusive_run_lock
from franka_autonomous_env import FrankaAutonomousEnv
from motion_aware_noise_estimator import position_variance,second_difference_mad

OUT=Path('bc_results/h4_motion_estimator');PROTOCOL=Path('bc_results/h4_motion_estimator_protocol.json')
ESTIMATORS={'position':position_variance,'motion_MAD':second_difference_mad}


def load_protocol():
    p=json.loads(PROTOCOL.read_text());ph=sha256(PROTOCOL)
    assert ph==PROTOCOL.with_suffix('.sha256').read_text().split()[0]
    assert hashlib.sha256(subprocess.check_output(['git','show',f'HEAD:{PROTOCOL}'])).hexdigest()==ph
    return p,ph


def collect(p,ph):
    for m in p['models']:
        spec=p['frozen_models'][str(m)];assert sha256(spec['path'])==spec['sha256']
        model,ck=load_bc_policy(spec['path']);assert ck['horizon']==4
        for source in p['motion_sources']:
            for seed in p['seeds']:
                path=OUT/f'trajectory_{m}_{source}_{seed}.json'
                if path.exists():
                    a=json.loads(path.read_text());assert a['protocol_sha256']==ph and a['seed']==seed
                    assert a['checkpoint_sha256']==spec['sha256'];continue
                env=FrankaAutonomousEnv(position_range=.20);env.max_move=float(ck['max_move']);env.max_time=float(ck['max_time'])
                try:
                    obs,_=env.reset(seed=seed);ctrl=BCActionChunkController(model,ck['obs_mean'].numpy(),ck['obs_std'].numpy(),4,int(source[1:]));ctrl.reset(obs)
                    xyz=[];xy=[];actions=[]
                    while True:
                        xyz.append(env.data.xpos[env.cube_id].copy().tolist());xy.append(obs[3:5].copy().tolist());action=ctrl.planned_action.copy();actions.append(action.tolist())
                        obs,_,terminated,truncated,_=env.step(action)
                        if terminated or truncated:break
                        ctrl.advance(obs)
                    a={'model_seed':m,'motion_source':source,'seed':seed,'protocol_sha256':ph,'checkpoint_sha256':spec['sha256'],'true_xyz':xyz,'clean_observed_xy':xy,'actions':actions}
                    with path.open('x') as f:json.dump(a,f,separators=(',',':'),allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
                finally:env.close()
            print('Collected',m,source,flush=True)
        assert sha256(spec['path'])==spec['sha256']


def evaluate_sequence(xy,xyz,profile,seed,p,identity):
    a,b=p['profiles'][profile];sigma=np.full(len(xy),a,dtype=np.float64);sigma[40:]=b
    rng=np.random.default_rng(np.random.SeedSequence([p['noise_seed'],int(seed)]))
    standard=rng.standard_normal((len(xy),2)).astype(np.float32)
    noisy=(np.asarray(xy,dtype=np.float32)+standard*sigma.astype(np.float32)[:,None]).astype(np.float32)
    estimates=[];intervals={k:4 for k in ESTIMATORS}
    # Estimate only from noisy arrays first. Truth diagnostics occur afterward.
    for t in range(20,len(noisy)+1,4):
        for name,fn in ESTIMATORS.items():
            value=fn(noisy[t-20:t]);before=intervals[name]
            after=(4 if value<=.0075 else 1) if t==20 else (1 if before==4 and value>.0085 else 4 if before==1 and value<.0065 else before)
            intervals[name]=after
            estimates.append({'estimator':name,'step':t,'sigma_hat':value,'before':before,'after':after,'switched':before!=after})
    for e in estimates:
        t=e['step'];truth=np.asarray(xyz[t-20:t]);diff=truth[:,None,:]-truth[None,:,:]
        diameter=float(np.linalg.norm(diff[:,:,:2],axis=2).max());win_sigma=sigma[t-20:t]
        mixed=bool(np.ptp(win_sigma)>0);target=float(np.sqrt(np.mean(win_sigma**2)))
        e.update(identity);e.update({'profile':profile,'seed':seed,'window_start':t-20,'window_end':t-1,'true_xy_diameter':diameter,'motion':'moving' if diameter>.001 else 'stationary','true_xyz_diameter':float(np.linalg.norm(diff,axis=2).max()),'true_acceleration_scale':float(np.sqrt(np.mean(np.diff(truth[:,:2],n=2,axis=0)**2))),'target_sigma':target,'mixed':mixed,'post_change_only':t>=60,'error':e['sigma_hat']-target,'classification_correct':None if mixed else (e['sigma_hat']>.0075)==(target>.0075),'wrong_interval':None if mixed else e['after']!=(1 if target>.0075 else 4),'wrong_switch':None if mixed else e['switched'] and e['after']!=(1 if target>.0075 else 4)})
    return {'identity':identity,'seed':seed,'profile':profile,'observed_xy':noisy.tolist(),'base_noise_sha256':hashlib.sha256(standard.tobytes()).hexdigest(),'windows':estimates}


def evaluate(p,ph):
    out=OUT/'diagnostic_sequences.jsonl'
    if out.exists():raise FileExistsError(out)
    with out.open('x') as f:
        for path in sorted(OUT.glob('trajectory_*.json')):
            a=json.loads(path.read_text())
            for profile in p['profiles']:
                r=evaluate_sequence(a['clean_observed_xy'],a['true_xyz'],profile,a['seed'],p,{k:a[k] for k in ('model_seed','motion_source')});r['protocol_sha256']=ph
                f.write(json.dumps(r,separators=(',',':'),allow_nan=False)+'\n')
    stress=OUT/'acceleration_stress.jsonl'
    with stress.open('x') as f:
        t=np.arange(160,dtype=float);signals={'static':t*0,'velocity':t*.001,'acceleration':t*t*.00002,'oscillation':.015*np.sin(2*np.pi*t/4),'impact':np.where(t>=70,.03,0.)}
        for name,x in signals.items():
            xy=np.column_stack([x,np.zeros_like(x)]);xyz=np.column_stack([xy,np.zeros_like(x)])
            for rep in range(20):
                for profile in p['profiles']:
                    r=evaluate_sequence(xy,xyz,profile,351000+rep,p,{'synthetic_motion':name});r['protocol_sha256']=ph
                    f.write(json.dumps(r,separators=(',',':'),allow_nan=False)+'\n')


def main():
    p,ph=load_protocol();OUT.mkdir(exist_ok=True)
    with exclusive_run_lock(OUT/'.stage1.lock'):
        collect(p,ph);evaluate(p,ph)


if __name__=='__main__':main()

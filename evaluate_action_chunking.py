"""Deterministic, resumable episode rollouts for action-chunked BC/SAC."""
import argparse,csv,os,time
import json,hashlib
from pathlib import Path
import numpy as np
from stable_baselines3 import SAC
from franka_autonomous_env import FrankaAutonomousEnv
from action_chunking import load_bc_policy,BCActionChunkController
from franka_chunk_aware_residual_env import FrankaChunkAwareResidualEnv

OUT=Path('bc_results/action_chunking_eval')
FIELDS=['seed','success','steps','return','cube_x','cube_y','region','first_close_step','closed_fraction',
        'bc_inference_calls','bc_mean_latency_ms','episode_wall_seconds','sac_inference_calls','sac_mean_latency_ms',
        'mean_residual_norm_m','saturation_rate','intervention_rate','first_lift_step','terminal_chunk_index']

def _rows(path):
    if not path.exists():return []
    with path.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def _save(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True);fresh=not path.exists()
    with path.open('a',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=FIELDS,lineterminator='\n')
        if fresh:w.writeheader()
        w.writerows(rows);f.flush();os.fsync(f.fileno())
def _region(x,y):return ('E' if x>=.55 else 'W')+('N' if y>=0 else 'S')

def evaluate_bc(bc_path,seeds,tag):
    OUT.mkdir(parents=True,exist_ok=True);final=OUT/f'{tag}.csv';partial=OUT/f'{tag}.partial.csv';seeds=list(seeds)
    if final.exists():
        rows=_rows(final)
        if len(rows)==len(seeds):return rows
        if partial.exists():raise FileExistsError(f'Both final and partial outputs exist: {final}')
        final.replace(partial)
    path=partial
    done={int(r['seed']) for r in _rows(path)};model,ck=load_bc_policy(bc_path);h=int(ck['horizon'])
    controller=BCActionChunkController(model,ck['obs_mean'].numpy(),ck['obs_std'].numpy(),h)
    env=FrankaAutonomousEnv(position_range=.20)
    # Match the demonstration/action normalization stored in the frozen BC checkpoint.
    env.max_move=float(ck['max_move']);env.max_time=float(ck['max_time'])
    out=[]
    for seed in seeds:
        if seed in done:continue
        obs,_=env.reset(seed=int(seed));controller.reset(obs);start=time.perf_counter();ret=0.;steps=0;closes=[];liftstep='';traj=[];closed=0
        while True:
            act=controller.planned_action.copy();traj.append({'step':steps,'planned_xyz':act[:3].tolist(),'gripper':float(act[3]),'chunk_index':controller.chunk_index,
                'hand_xyz':env.data.site_xpos[env.site_id].copy().tolist(),'cube_xyz':env.data.xpos[env.cube_id].copy().tolist()})
            if act[3]<0 and not closes:closes.append(steps)
            closed+=int(act[3]<0)
            nxt,r,term,trunc,info=env.step(act);ret+=r;steps+=1
            if info['lift_height']>=.10 and not liftstep:liftstep=steps-1
            if term or trunc:break
            controller.advance(nxt)
        elapsed=time.perf_counter()-start;x=float(info['cube_x']);y=float(info['cube_y']);success=bool(info['is_success'])
        row={'seed':int(seed),'success':int(success),'steps':steps,'return':ret,'cube_x':x,'cube_y':y,'region':_region(x,y),
             'first_close_step':closes[0] if closes else '', 'closed_fraction':closed/steps,
             'bc_inference_calls':controller.inference_calls,'bc_mean_latency_ms':1000*float(np.mean(controller.inference_latencies)),'episode_wall_seconds':elapsed,
             'sac_inference_calls':0,'sac_mean_latency_ms':'','mean_residual_norm_m':0.,'saturation_rate':0.,'intervention_rate':0.,
             'first_lift_step':liftstep,'terminal_chunk_index':controller.chunk_index}
        out.append(row);_save(path,[row])
        if int(seed)==290042 and tag=='bc_final_H4_seed42':save_trace(tag,'BC',traj)
    env.close()
    if final.exists():raise FileExistsError(f'Refusing to overwrite complete result {final}')
    partial.replace(final)
    return _rows(final)

def save_trace(tag,method,steps):
    import json
    p=OUT/f'trajectory_{tag}_{method}.json'
    if p.exists():return
    p.write_text(json.dumps({'label':'selected illustrative example; not representative','steps':steps},indent=2)+'\n')

def evaluate_residual(model,bc_path,seeds,tag):
    OUT.mkdir(parents=True,exist_ok=True);final=OUT/f'{tag}.csv';partial=OUT/f'{tag}.partial.csv';seeds=list(seeds)
    if final.exists():
        rows=_rows(final)
        if len(rows)==len(seeds):return rows
        if partial.exists():raise FileExistsError(f'Both final and partial outputs exist: {final}')
        final.replace(partial)
    path=partial
    done={int(r['seed']) for r in _rows(path)}
    e=FrankaChunkAwareResidualEnv(bc_path,.20,.005,'uniform',.12,.90,'shaped',.005)
    out=[]
    for seed in seeds:
        if seed in done:continue
        obs,_=e.reset(seed=int(seed));start=time.perf_counter();ret=0.;steps=0;res=[];sat=0;active=0;closed=0;close='';liftstep='';trace=[];sac_lat=[]
        while True:
            t=time.perf_counter();action,_=model.predict(obs,deterministic=True);sac_lat.append(time.perf_counter()-t)
            obs,r,term,trunc,info=e.step(action);ret+=r;steps+=1
            res.append(info['residual_norm']);sat+=int(info['action_saturated']);active+=int(info['residual_action_norm']>1e-6)
            closed+=int(info['executed_action'][3]<0)
            if info['executed_action'][3]<0 and not close:close=steps-1
            if info['lift_height']>=.10 and not liftstep:liftstep=steps-1
            trace.append({'step':steps-1,'bc_xyz':info['bc_action'][:3].tolist(),'executed_xyz':info['executed_action'][:3].tolist(),
                          'residual_m':info['applied_residual_xyz_m'].tolist(),'chunk_index':info['chunk_index'],
                          'hand_xyz':e.env.data.site_xpos[e.env.site_id].copy().tolist(),'cube_xyz':e.env.data.xpos[e.env.cube_id].copy().tolist()})
            if term or trunc:break
        x=float(info['cube_x']);y=float(info['cube_y']);elapsed=time.perf_counter()-start
        row={'seed':int(seed),'success':int(bool(info['is_success'])),'steps':steps,'return':ret,'cube_x':x,'cube_y':y,'region':_region(x,y),
             'first_close_step':close,'closed_fraction':closed/steps,'bc_inference_calls':e.bc_inference_calls,
             'bc_mean_latency_ms':1000*e.total_bc_latency/max(e.bc_inference_calls,1),'episode_wall_seconds':elapsed,
             'sac_inference_calls':steps,'sac_mean_latency_ms':1000*float(np.mean(sac_lat)),'mean_residual_norm_m':float(np.mean(res)),
             'saturation_rate':sat/steps,'intervention_rate':active/steps,'first_lift_step':liftstep,'terminal_chunk_index':e.controller.chunk_index}
        out.append(row);_save(path,[row])
        if int(seed)==290042 and tag=='sac_final_fixed30k_H4_bc42_sac1103':save_trace(tag,'Residual_SAC',trace)
    e.close()
    if final.exists():raise FileExistsError(f'Refusing to overwrite complete result {final}')
    partial.replace(final)
    return _rows(final)

def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['bc-validation','sac-validation','bc-final','sac-final']);p.add_argument('--arch',choices=['H1','H4']);p.add_argument('--bc-seed',type=int);p.add_argument('--sac-seed',type=int);p.add_argument('--step',type=int);p.add_argument('--model')
    a=p.parse_args();archs=[a.arch] if a.arch else ['H1','H4'];bcs=[a.bc_seed] if a.bc_seed else [42,123,456]
    if a.phase in ('bc-validation','bc-final'):
        seeds=range(270000,270500) if a.phase=='bc-validation' else range(290000,292000)
        for h in archs:
            for b in bcs:evaluate_bc(f'bc_models/action_chunking/{h}_seed{b}.pt',seeds,f'bc_{a.phase.split("-")[1]}_{h}_seed{b}')
    elif a.phase=='sac-validation':
        raise RuntimeError('Use train_action_chunk_sac.py milestone validation or provide explicit checkpoint orchestration')
    else:
        manifest=Path('bc_results/action_chunking_checkpoint_selections.json');side=manifest.with_suffix('.sha256')
        if not manifest.is_file() or not side.is_file():raise FileNotFoundError('Freeze all validation selections before touching Final positions')
        expected=side.read_text().split()[0];actual=hashlib.sha256(manifest.read_bytes()).hexdigest()
        if expected!=actual:raise RuntimeError('Selection manifest SHA256 mismatch')
        data=json.loads(manifest.read_text())
        if not data.get('frozen_before_final_evaluation') or len(data.get('selections',[]))!=18:raise RuntimeError('Invalid/incomplete frozen selection manifest')
        seeds=range(290000,292000)
        for item in data['selections']:
            if a.arch and item['architecture']!=a.arch:continue
            if a.bc_seed and item['bc_seed']!=a.bc_seed:continue
            if a.sac_seed and item['sac_seed']!=a.sac_seed:continue
            arch=item['architecture'];bc=item['bc_seed'];sac=item['sac_seed'];bcpath=f'bc_models/action_chunking/{arch}_seed{bc}.pt'
            fixed=Path(f'bc_results/action_chunking_sac/{arch}_bc{bc}_sac{sac}/checkpoint_30000.zip')
            candidates=[('fixed30k',fixed)]
            chosen=Path(item['model_path'])
            if item['selected_step']!=30000:candidates.append((f'selected{item["selected_step"]}',chosen))
            for label,mp in candidates:
                expected=item['checkpoint_hashes'][str(int(mp.stem.split('_')[-1]))]
                if sha256(mp) != expected:
                    raise RuntimeError(f'Frozen model hash mismatch: {mp}')
                model=SAC.load(str(mp),device='cpu')
                tag=f'sac_final_{label}_{arch}_bc{bc}_sac{sac}'
                evaluate_residual(model,bcpath,seeds,tag)
if __name__=='__main__':main()

"""Train matched H1/H4 state-conditioned behavior-cloning policies."""
import copy, csv, json, random, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from action_chunking import ActionChunkBC, compute_chunk_loss, build_windows, load_demonstrations, sha256

OUT=Path('bc_results/action_chunking_bc_runs'); MODELS=Path('bc_models/action_chunking')
SEEDS=(42,123,456)

def write_history(path, rows):
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)

def main():
    obs,actions,ids,protocol,audit=load_demonstrations(); OUT.mkdir(parents=True,exist_ok=True); MODELS.mkdir(parents=True,exist_ok=True)
    tr_ids=audit['split']['training_episode_ids']; va_ids=audit['split']['validation_episode_ids']
    xtr,ytr,owntr=build_windows(obs,actions,ids,tr_ids,1)
    xva,yva,ownva=build_windows(obs,actions,ids,va_ids,1)
    # H1/H4 share a single frozen split and identical start observations.
    if len(xtr)!=audit['split']['training_windows'] or len(xva)!=audit['split']['validation_windows']:
        raise AssertionError('Observed window counts differ from preregistration')
    all_tr=obs[np.isin(ids,tr_ids)]
    mean=all_tr.mean(0).astype(np.float32); std=np.maximum(all_tr.std(0),1e-4).astype(np.float32)
    if len(xtr)==0:raise RuntimeError('No training windows')
    settings={'max_epochs':100,'batch_size':256,'learning_rate':0.0003,'patience':15,'minimum_improvement':1e-6,'optimizer':'Adam','split_seed':20261106}
    torch.set_num_threads(4)
    for horizon in (1,4):
      x_train,y_train,_=build_windows(obs,actions,ids,tr_ids,horizon)
      x_val,y_val,_=build_windows(obs,actions,ids,va_ids,horizon)
      for seed in SEEDS:
        name=f'H{horizon}_seed{seed}'; run=OUT/name; model_path=MODELS/f'{name}.pt'
        cfg={**settings,'horizon':horizon,'seed':seed,'model_path':str(model_path),'protocol_sha256':protocol_hash(),'dataset_sources':protocol['dataset']['sha256'],'training_episodes':270,'validation_episodes':30,'training_windows':len(x_train),'validation_windows':len(x_val),'training_targets':len(y_train)*horizon,'validation_targets':len(y_val)*horizon,'parameter_count':sum(p.numel() for p in ActionChunkBC(horizon).parameters())}
        run.mkdir(parents=True,exist_ok=True); cfg_path=run/'config.json'; hist_path=run/'history.csv'
        if model_path.exists():
            if not cfg_path.exists() or json.loads(cfg_path.read_text())!=cfg:raise RuntimeError(f'Existing BC run mismatch: {run}')
            print(f'Skipping completed {name}',flush=True);continue
        if cfg_path.exists() or hist_path.exists():raise RuntimeError(f'Incomplete BC run {run}; preserve and inspect before retry')
        cfg_path.write_text(json.dumps(cfg,indent=2)+'\n')
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        model=ActionChunkBC(horizon);optimizer=torch.optim.Adam(model.parameters(),lr=settings['learning_rate'])
        xt=torch.from_numpy(((x_train-mean)/std).astype(np.float32)); yt=torch.from_numpy(y_train.astype(np.float32))
        xv=torch.from_numpy(((x_val-mean)/std).astype(np.float32)); yv=torch.from_numpy(y_val.astype(np.float32))
        train_loader=DataLoader(TensorDataset(xt,yt),batch_size=256,shuffle=True,generator=torch.Generator().manual_seed(seed),num_workers=0)
        val_loader=DataLoader(TensorDataset(xv,yv),batch_size=1024,shuffle=False,num_workers=0)
        best=float('inf'); best_state=None;best_epoch=0;patience=0;history=[]; start=time.perf_counter()
        for epoch in range(1,101):
            model.train(); total=0.; xyz_total=0.; grip_total=0.
            for xb,yb in train_loader:
                optimizer.zero_grad(set_to_none=True); loss,xl,gl=compute_chunk_loss(model,xb,yb); loss.backward();optimizer.step()
                total+=float(loss.detach())*len(xb);xyz_total+=float(xl.detach())*len(xb);grip_total+=float(gl.detach())*len(xb)
            model.eval();vtotal=vxyz=vgrip=0.
            with torch.inference_mode():
                for xb,yb in val_loader:
                    loss,xl,gl=compute_chunk_loss(model,xb,yb);vtotal+=float(loss)*len(xb);vxyz+=float(xl)*len(xb);vgrip+=float(gl)*len(xb)
            tr=total/len(xt);va=vtotal/len(xv)
            history.append({'epoch':epoch,'train_loss':tr,'train_xyz_loss':xyz_total/len(xt),'train_gripper_loss':grip_total/len(xt),'validation_loss':va,'validation_xyz_loss':vxyz/len(xv),'validation_gripper_loss':vgrip/len(xv)})
            if va<best-1e-6:best=va;best_epoch=epoch;best_state=copy.deepcopy(model.state_dict());patience=0
            else:patience+=1
            if epoch==1 or epoch%10==0:print(f'{name} epoch {epoch}: train={tr:.5f} val={va:.5f}',flush=True)
            if patience>=15:break
        model.load_state_dict(best_state);elapsed=time.perf_counter()-start
        checkpoint={'model_state_dict':model.state_dict(),'obs_mean':torch.from_numpy(mean),'obs_std':torch.from_numpy(std),'horizon':horizon,'seed':seed,'best_epoch':best_epoch,'best_validation_loss':best,'max_move':0.15,'max_time':16.0,'control_dt':0.1,'position_range':0.12,'parameter_count':cfg['parameter_count'],'training_seconds':elapsed,'training_episode_ids':tr_ids,'validation_episode_ids':va_ids,'protocol_sha256':cfg['protocol_sha256'],'dataset_sha256':cfg['dataset_sources'],'training_windows':len(x_train),'training_action_targets':len(y_train)*horizon,'validation_windows':len(x_val),'validation_action_targets':len(y_val)*horizon,'normalization_episode_ids':tr_ids,'settings':settings}
        temp=model_path.with_suffix('.pt.tmp');torch.save(checkpoint,temp);temp.replace(model_path);write_history(hist_path,history)
        print(f'Completed {name}: epoch={best_epoch} val={best:.6f} parameters={cfg["parameter_count"]} seconds={elapsed:.1f}',flush=True)
def protocol_hash():return sha256('bc_results/action_chunking_protocol.json')
if __name__=='__main__':main()

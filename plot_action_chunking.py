"""Create the eight preregistered English SVG/PNG figures."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

A=Path('bc_results/action_chunking_analysis');E=Path('bc_results/action_chunking_eval');OUT=Path('bc_results/action_chunking_figures')
plt.rcParams.update({'font.size':10,'axes.titlesize':12,'figure.dpi':120,'savefig.dpi':180})
def rows(p):
    with Path(p).open(newline='') as f:return list(csv.DictReader(f))
def save(fig,name):
    OUT.mkdir(parents=True,exist_ok=True);fig.tight_layout();fig.savefig(OUT/f'{name}.png',bbox_inches='tight');fig.savefig(OUT/f'{name}.svg',bbox_inches='tight');plt.close(fig)
def fval(x):return float(x)

def main():
    bc=rows(A/'bc_final_metrics.csv');cells=rows(A/'final_cell_metrics.csv');gains=rows(A/'difference_in_gains.csv')
    curves=rows(A/'checkpoint_learning_curves.csv');regions=rows(A/'regional_rescue_regression.csv')
    # 1. BC H1/H4 success by matched training seed.
    fig,ax=plt.subplots(figsize=(7,4));x=np.arange(3);w=.36
    for j,h in enumerate(('H1','H4')):
        rr=[r for r in bc if r['architecture']==h];ax.bar(x+(j-.5)*w,[100*fval(r['success_rate']) for r in rr],w,label=h)
    ax.set_xticks(x,[r['bc_seed'] for r in [z for z in bc if z['architecture']=='H1']]);ax.set_ylabel('Success (%)');ax.set_xlabel('BC training seed');ax.set_title('Single-step vs action-chunked BC');ax.legend();ax.set_ylim(0,100);save(fig,'01_bc_success')
    # 2. Matched baseline and residual-SAC absolute final success.
    fig,ax=plt.subplots(figsize=(7,4));labels=[];bv=[];sv=[]
    for h in ('H1','H4'):
      bmean=np.mean([fval(r['success_rate']) for r in bc if r['architecture']==h])*100
      smean=np.mean([fval(r['success_rate']) for r in cells if r['architecture']==h and r['checkpoint']=='selected'])*100
      labels.append(h);bv.append(bmean);sv.append(smean)
    x=np.arange(2);ax.bar(x-.18,bv,.36,label='BC only');ax.bar(x+.18,sv,.36,label='BC + selected SAC');ax.set_xticks(x,labels);ax.set_ylim(0,100);ax.set_ylabel('Success (%)');ax.set_title('Final success with and without residual SAC');ax.legend();save(fig,'02_bc_vs_residual_sac')
    # 3. Difference-in-gains cell heatmaps.
    fig,axs=plt.subplots(1,2,figsize=(9,4),sharey=True)
    for ax,label in zip(axs,('fixed30k','selected')):
      mat=np.full((3,3),np.nan)
      for r in gains:
        if r['checkpoint']==label:mat[(42,123,456).index(int(r['bc_seed'])),(1103,1117,1123).index(int(r['sac_seed']))]=fval(r['difference_in_gains_pp'])
      im=ax.imshow(mat,cmap='coolwarm',vmin=-np.nanmax(abs(mat)),vmax=np.nanmax(abs(mat)))
      ax.set_xticks(range(3),[1103,1117,1123]);ax.set_yticks(range(3),[42,123,456]);ax.set_xlabel('SAC training seed');ax.set_title(label)
      for i in range(3):
       for j in range(3):ax.text(j,i,f'{mat[i,j]:.1f}',ha='center',va='center')
    axs[0].set_ylabel('BC training seed');fig.colorbar(im,ax=axs.ravel().tolist(),label='H4 minus H1 gain (percentage points)');fig.suptitle('Difference-in-gains across crossed training seeds');save(fig,'03_difference_in_gains')
    # 4. Paired rescues and regressions.
    fig,ax=plt.subplots(figsize=(6,5))
    for h,c in (('H1','#3973ac'),('H4','#e07a24')):
      for label,m in (('fixed30k','o'),('selected','x')):
        rr=[r for r in cells if r['architecture']==h and r['checkpoint']==label]
        ax.scatter([int(r['rescue']) for r in rr],[int(r['regression']) for r in rr],c=c,marker=m,label=f'{h} {label}',alpha=.8)
    top=max(max(int(r['rescue']) for r in cells),max(int(r['regression']) for r in cells));ax.plot([0,top],[0,top],'k--',lw=1);ax.set_xlabel('Rescue (BC fail → SAC success)');ax.set_ylabel('Regression (BC success → SAC fail)');ax.set_title('Paired outcomes by seed cell');ax.legend(fontsize=8);save(fig,'04_rescue_vs_regression')
    # 5. Validation learning curves at the three checkpoints.
    fig,ax=plt.subplots(figsize=(7,4));steps=(10000,20000,30000)
    for h,c in (('H1','#3973ac'),('H4','#e07a24')):
      means=[];sds=[]
      for step in steps:
        vals=[float(r['success_rate'])*100 for r in curves if r['architecture']==h and int(r['timesteps'])==step];means.append(np.mean(vals));sds.append(np.std(vals,ddof=1))
      ax.errorbar(np.array(steps)/1000,means,yerr=sds,marker='o',capsize=3,label=h,color=c)
    ax.set_xticks([10,20,30]);ax.set_xlabel('SAC training steps (thousands)');ax.set_ylabel('Validation success (%)');ax.set_title('SAC checkpoint success on shared validation positions');ax.legend();save(fig,'05_checkpoint_success')
    # 6. BC inference call frequency and latency.
    fig,axs=plt.subplots(1,2,figsize=(9,4));
    for j,h in enumerate(('H1','H4')):
      r=[x for x in bc if x['architecture']==h];axs[0].bar(j,np.mean([fval(x['mean_bc_calls']) for x in r]),color=('#3973ac' if h=='H1' else '#e07a24'));axs[1].bar(j,np.mean([fval(x['mean_bc_latency_ms']) for x in r]),color=('#3973ac' if h=='H1' else '#e07a24'))
    axs[0].set_xticks([0,1],['H1','H4']);axs[0].set_ylabel('BC inference calls / episode');axs[0].set_title('Replanning frequency');axs[1].set_xticks([0,1],['H1','H4']);axs[1].set_ylabel('Mean BC inference latency (ms)');axs[1].set_title('Per-call latency');fig.suptitle('Action chunking reduces BC replanning frequency');save(fig,'06_bc_inference_cost')
    # 7. Regional rescue/regression totals averaged over all paired runs.
    fig,ax=plt.subplots(figsize=(8,4));regs=('EN','ES','WN','WS');xx=np.arange(4);w=.34
    for j,(h,c) in enumerate((('H1','#3973ac'),('H4','#e07a24'))):
      rr=[r for r in regions if r['architecture']==h and r['checkpoint']=='selected']
      rescue=[sum(int(r['rescue']) for r in rr if r['region']==q) for q in regs];reg=[sum(int(r['regression']) for r in rr if r['region']==q) for q in regs]
      ax.bar(xx+(j-.5)*w,rescue,w,color=c,alpha=.85,label=f'{h} rescue');ax.bar(xx+(j-.5)*w,[-x for x in reg],w,color=c,alpha=.35,label=f'{h} regression')
    ax.axhline(0,color='black',lw=.8);ax.set_xticks(xx,regs);ax.set_ylabel('Paired counts (regressions below zero)');ax.set_title('Regional rescue and regression counts (selected SAC)');ax.legend(ncol=2,fontsize=8);save(fig,'07_regional_outcomes')
    # 8. Selected example trajectory; use hand paths with cube location annotation.
    bp=E/'trajectory_bc_final_H4_seed42_BC.json';rp=E/'trajectory_sac_final_fixed30k_H4_bc42_sac1103_Residual_SAC.json'
    if bp.exists() and rp.exists():
      b=json.loads(bp.read_text())['steps'];r=json.loads(rp.read_text())['steps'];fig,ax=plt.subplots(figsize=(6,5))
      for seq,label,color in ((b,'H4 BC open-loop chunks','#777777'),(r,'H4 + stepwise SAC residual','#d62728')):
        xy=np.array([x['hand_xyz'][:2] for x in seq]);ax.plot(xy[:,0],xy[:,1],'-o',ms=2,label=label,color=color,alpha=.8)
      cube=np.array(b[0]['cube_xyz']);ax.scatter([cube[0]],[cube[1]],marker='s',s=60,c='green',label='Cube (same seed 290042)');ax.set_xlabel('Hand x (m)');ax.set_ylabel('Hand y (m)');ax.set_title('Selected illustrative trajectory only; not representative');ax.legend(fontsize=8);save(fig,'08_selected_trajectory')
    print(f'Saved SVG and PNG figures under {OUT}',flush=True)
if __name__=='__main__':main()

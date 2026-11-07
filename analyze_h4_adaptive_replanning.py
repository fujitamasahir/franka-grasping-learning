"""Preregistered paired statistics, summaries, and six figures for complete Final."""
import csv,json,math
from pathlib import Path
import numpy as np
from action_chunking import sha256
from analyze_h4_replanning import exact_mcnemar,holm
from evaluate_h4_adaptive_replanning import OUT,FIELDS

MODELS=(42,123,456);NOISE=(0,5,10);CONTROLLERS=('R1','R4','Adaptive');SEEDS=range(331000,332000);B=10000;BOOT_SEED=20261107

def read_all():
 audit=json.loads((OUT/'final_integrity_audit.json').read_text())
 if not audit.get('final_evaluation_complete') or audit.get('complete_cells')!=27:raise RuntimeError('Full Final integrity audit required')
 frozen=json.loads((OUT/'frozen_threshold.json').read_text());threshold=frozen['selected_threshold_m']
 cells={}
 for m in MODELS:
  for c in CONTROLLERS:
   for n in NOISE:
    p=OUT/f'final_H4_seed{m}_{c}_noise{n}mm.csv'
    with p.open(newline='') as f:
     r=csv.DictReader(f)
     if r.fieldnames!=FIELDS:raise ValueError(f'Schema mismatch: {p}')
     rows=list(r)
    if [int(x['seed']) for x in rows]!=list(SEEDS):raise ValueError(f'Seed set mismatch: {p}')
    if c=='Adaptive' and any(not math.isclose(float(x['threshold_m']),threshold,abs_tol=1e-12) for x in rows):raise ValueError('Final threshold mismatch')
    cells[m,c,n]={int(x['seed']):x for x in rows}
 return cells,frozen,audit

def paired_ci(a,b,seed):
 d=b.astype(np.float64)-a.astype(np.float64);rng=np.random.default_rng(seed);vals=[]
 for st in range(0,B,250):
  k=min(250,B-st);ix=rng.integers(0,len(d),(k,len(d)));vals.extend(d[ix].mean(axis=1).tolist())
 return [100*float(x) for x in np.quantile(vals,[.025,.975])]

def overall_boot(success,baseline):
 rng=np.random.default_rng(BOOT_SEED);vals=[]
 for st in range(0,B,200):
  k=min(200,B-st);diff=[];ix=rng.integers(0,1000,(k,1000))
  for mi,m in enumerate(MODELS):
   for n in NOISE:
    diff.append((success[m,'Adaptive',n][ix].mean(axis=1)-success[m,baseline,n][ix].mean(axis=1)))
  vals.extend(np.mean(np.stack(diff,axis=1),axis=1).tolist())
 return [100*float(x) for x in np.quantile(vals,[.025,.975])]

def transitions(x,y):
 b=int(np.sum((x==0)&(y==1)));h=int(np.sum((x==1)&(y==0)))
 return {'beneficial':b,'harmful':h,'net':b-h}

def main():
 cells,frozen,audit=read_all();success={};cell_rows=[];comparisons=[];estimates={};switch={};conditional={};regions={}
 for k,rows in cells.items():
  success[k]=np.asarray([int(rows[s]['success']) for s in SEEDS],np.int8)
  m,c,n=k;rr=list(rows.values());calls=np.asarray([int(x['bc_inference_calls']) for x in rr]);tm=np.asarray([float(x['bc_total_inference_ms']) for x in rr])
  cell_rows.append({'model_seed':m,'controller':c,'noise_mm':n,'successes':int(success[k].sum()),'episodes':1000,'success_rate_pct':100*float(success[k].mean()),
   'mean_inference_calls':float(calls.mean()),'mean_inference_latency_ms_per_call':float(tm.sum()/calls.sum()),'mean_total_inference_ms':float(tm.mean()),
   'mean_episode_wall_seconds':float(np.mean([float(x['episode_wall_seconds']) for x in rr])),'mean_steps':float(np.mean([int(x['steps']) for x in rr]))})
  if c=='Adaptive':
   aa=list(rows.values());ee=np.asarray([float(x['estimated_sigma_m']) for x in aa if x['estimated_sigma_m']!='']);dd=np.asarray([int(x['decision_made']) for x in aa]);chosen=[int(x['chosen_interval']) for x in aa if x['chosen_interval']!='']
   nearest=[min(NOISE,key=lambda q:abs(v-q/1000)) for v in ee]
   estimates[str(n)]={'n_estimates':int(len(ee)),'mean_m':float(ee.mean()) if len(ee) else None,'median_m':float(np.median(ee)) if len(ee) else None,'sd_m':float(ee.std(ddof=1)) if len(ee)>1 else None,'min_m':float(ee.min()) if len(ee) else None,'max_m':float(ee.max()) if len(ee) else None,'nearest_nominal_sigma_bucket_accuracy_pct':100*float(np.mean(np.asarray(nearest)==n)) if len(ee) else None,'episodes_without_decision':int(1000-dd.sum())}
   switch[str(n)]={'n_decisions':int(dd.sum()),'R1':sum(x==1 for x in chosen),'R4':sum(x==4 for x in chosen),'R1_pct':100*sum(x==1 for x in chosen)/max(1,len(chosen)),'R4_pct':100*sum(x==4 for x in chosen)/max(1,len(chosen))}
   decided=[x for x in aa if x['chosen_interval']!='']
   conditional[str(n)]={str(i):{'n':sum(int(x['chosen_interval'])==i for x in decided),'successes':sum(int(x['success']) for x in decided if x['chosen_interval']==str(i)),'success_rate_pct':100*np.mean([int(x['success']) for x in decided if x['chosen_interval']==str(i)]) if any(x['chosen_interval']==str(i) for x in decided) else None} for i in (1,4)}
 # Determine calibration-selected fixed baseline, never from Final.
 baseline=frozen['calibration_selected_fixed_baseline']; all_compare=('R1','R4')
 overall={c:100*float(np.mean([success[m,c,n].mean() for m in MODELS for n in NOISE])) for c in CONTROLLERS}
 for c in all_compare:
  for m in MODELS:
   for n in NOISE:
    rows0=cells[m,c,n];rows1=cells[m,'Adaptive',n];mc=exact_mcnemar(rows0,rows1);tr=transitions(success[m,c,n],success[m,'Adaptive',n])
    if int(success[m,'Adaptive',n].sum()-success[m,c,n].sum())!=tr['net']:raise AssertionError('Paired rescue/harm arithmetic mismatch')
    comparisons.append({'comparison':f'Adaptive_vs_{c}','model_seed':m,'noise_mm':n,'difference_pp':100*float((success[m,'Adaptive',n]-success[m,c,n]).mean()),
     'paired_bootstrap_95pct_CI_pp':paired_ci(success[m,c,n],success[m,'Adaptive',n],BOOT_SEED+m+n+(0 if c=='R1' else 100)),**tr,**mc})
 for c in all_compare:
  idx=[i for i,x in enumerate(comparisons) if x['comparison']==f'Adaptive_vs_{c}'];p=holm([comparisons[i]['exact_two_sided_p'] for i in idx])
  for i,v in zip(idx,p):comparisons[i]['holm_adjusted_p_within_9_tests']=float(v)
 potentially_unfavorable={}
 for m in MODELS:
  potentially_unfavorable[str(m)]={}
  for n in NOISE:
   count=sum(cells[m,'Adaptive',n][s]['chosen_interval']=='1' and int(cells[m,'Adaptive',n][s]['success'])==0 and int(cells[m,'R4',n][s]['success'])==1 for s in SEEDS)
   potentially_unfavorable[str(m)][str(n)]={'selected_R1_failures_where_paired_R4_succeeded':int(count),'interpretation':'counterfactual decision diagnostic; fixed R4 shares the R4 warm-up, but selection is not randomized'}
 diffs={c:overall['Adaptive']-overall[c] for c in all_compare};overall_ci={c:overall_boot(success,c) for c in all_compare}
 # The third primary comparison is the Calibration-selected fixed policy, selected without Final.
 paired_overall={'Adaptive_vs_R1':{'difference_pp':diffs['R1'],'bootstrap_95pct_CI_pp':overall_ci['R1']},
  'Adaptive_vs_R4':{'difference_pp':diffs['R4'],'bootstrap_95pct_CI_pp':overall_ci['R4']},
  f"Adaptive_vs_calibration_best_{baseline}":{'difference_pp':diffs[baseline],'bootstrap_95pct_CI_pp':overall_ci[baseline]}}
 by_model={};seed_diffs={'R1':[],'R4':[]}
 for m in MODELS:
  by_model[str(m)]={c:100*float(np.mean([success[m,c,n].mean() for n in NOISE])) for c in CONTROLLERS}
  for c in ('R1','R4'):
   d=by_model[str(m)]['Adaptive']-by_model[str(m)][c];seed_diffs[c].append(d)
 seed_level={}
 for c,vals in seed_diffs.items():
  seed_level[f'Adaptive_minus_{c}_pp']={'values_by_model_seed':dict(zip(map(str,MODELS),vals)),'mean':float(np.mean(vals)),'median':float(np.median(vals)),'sample_sd':float(np.std(vals,ddof=1)),'min':float(np.min(vals)),'max':float(np.max(vals)),'positive_model_seeds':sum(v>0 for v in vals)}
 # Region success summaries; regional inferential claims are exploratory only.
 for m in MODELS:
  regions[str(m)]={}
  for reg in ('EN','ES','WN','WS'):
   regions[str(m)][reg]={}
   for n in NOISE:
    keys=[s for s in SEEDS if cells[m,'Adaptive',n][s]['region']==reg]
    regions[str(m)][reg][str(n)]={}
    for c in CONTROLLERS:
     vals=[int(cells[m,c,n][s]['success']) for s in keys]
     regions[str(m)][reg][str(n)][c]={'n':len(vals),'successes':sum(vals),'success_rate_pct':100*np.mean(vals) if vals else None}
    regions[str(m)][reg][str(n)]['adaptive_vs_fixed_R1']=transitions(np.asarray([int(cells[m,'R1',n][s]['success']) for s in keys]),np.asarray([int(cells[m,'Adaptive',n][s]['success']) for s in keys]))
    regions[str(m)][reg][str(n)]['adaptive_vs_fixed_R4']=transitions(np.asarray([int(cells[m,'R4',n][s]['success']) for s in keys]),np.asarray([int(cells[m,'Adaptive',n][s]['success']) for s in keys]))
 summary={'protocol_sha256':Path('bc_results/h4_adaptive_replanning_protocol.sha256').read_text().split()[0],'final_integrity_audit_sha256':sha256(OUT/'final_integrity_audit.json'),
  'analysis':{'episodes_per_cell':1000,'model_seeds':list(MODELS),'noise_mm':list(NOISE),'controllers':list(CONTROLLERS),'paired_bootstrap_replicates':B,'bootstrap_seed':BOOT_SEED,'equal_weighting':'nine model_seed x noise cells','trained_policy_n':3,'unit':'paired evaluation seed within frozen model'},
  'overall_equal_weighted_success_rate_pct':overall,'adaptive_minus_fixed_overall':paired_overall,'calibration_selected_fixed_baseline':baseline,
  'equal_weighted_success_by_model_seed':by_model,'training_seed_level_descriptives':seed_level,
  'cell_summary':cell_rows,'paired_comparisons':comparisons,'estimated_sigma_by_true_noise_mm':estimates,'switching_by_true_noise_mm':switch,'conditional_success_by_chosen_interval':conditional,'potentially_unfavorable_R1_decisions':potentially_unfavorable,'regions':regions,
  'limitation':'Position bootstrap uncertainty is conditional on the three frozen BC policies and paired positions; it is not population-level uncertainty over arbitrary future trained policies.'}
 for name,obj in [('final_summary.json',summary)]:
  p=OUT/name
  if p.exists():raise FileExistsError(p)
  p.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')
 with (OUT/'final_cell_summary.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(cell_rows[0]));w.writeheader();w.writerows(cell_rows)
 with (OUT/'final_paired_comparisons.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(comparisons[0]));w.writeheader();w.writerows(comparisons)
 figures(summary)
 print(json.dumps({'overall':overall,'adaptive_minus_fixed':paired_overall},indent=2))

def figures(s):
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
 cdata=s['cell_summary'];colors={'R1':'#4477AA','R4':'#EE6677','Adaptive':'#228833'}
 def save(fig,name):
  for ext in ('svg','png'):fig.savefig(OUT/f'{name}.{ext}',bbox_inches='tight')
  plt.close(fig)
 # 1. Success/noise, three seed panels.
 fig,axs=plt.subplots(1,3,figsize=(12,4),sharey=True)
 for ax,m in zip(axs,MODELS):
  for ctrl in CONTROLLERS:
   ys=[next(x['success_rate_pct'] for x in cdata if x['model_seed']==m and x['controller']==ctrl and x['noise_mm']==n) for n in NOISE]
   ax.plot(NOISE,ys,marker='o',label=ctrl,color=colors[ctrl])
  ax.set_title(f'BC seed {m}');ax.set_xticks(NOISE);ax.set_xlabel('Observation noise σ (mm)');ax.grid(alpha=.25)
 axs[0].set_ylabel('Grasp success (%)');axs[-1].legend(frameon=False);fig.tight_layout();save(fig,'01_success_vs_noise')
 # 2. Adaptive improvement per seed/noise vs fixed baselines.
 fig,axs=plt.subplots(1,2,figsize=(11,4),sharey=True)
 for ax,base in zip(axs,('R1','R4')):
  x=np.arange(9);labels=[];vals=[]
  for m in MODELS:
   for n in NOISE:
    a=next(x['success_rate_pct'] for x in cdata if x['model_seed']==m and x['controller']=='Adaptive' and x['noise_mm']==n)
    b=next(x['success_rate_pct'] for x in cdata if x['model_seed']==m and x['controller']==base and x['noise_mm']==n)
    vals.append(a-b);labels.append(f'{m}\n{n}')
  ax.bar(x,vals,color=['#4477AA' if v>=0 else '#CC6677' for v in vals]);ax.axhline(0,color='black',lw=.8);ax.set_xticks(x,labels);ax.set_title(f'Adaptive − {base}');ax.set_xlabel('BC seed / noise mm');ax.grid(axis='y',alpha=.2)
 axs[0].set_ylabel('Success difference (pp)');fig.tight_layout();save(fig,'02_adaptive_improvement')
 # 3. Estimated vs true noise distributions.
 fig,ax=plt.subplots(figsize=(7,4.5));groups=[];labels=[]
 for n in NOISE:
  vals=[]
  for m in MODELS:
   p=OUT/f'final_H4_seed{m}_Adaptive_noise{n}mm.csv'
   vals.extend([1000*float(r['estimated_sigma_m']) for r in csv.DictReader(open(p)) if r['estimated_sigma_m']!=''])
  groups.append(vals);labels.append(str(n))
 ax.boxplot(groups,tick_labels=labels,showfliers=False);ax.plot([1,2,3],NOISE,'kx--',label='True σ');ax.set_xlabel('True observation noise σ (mm)');ax.set_ylabel('Estimated sigma_hat (mm)');ax.grid(axis='y',alpha=.2);ax.legend(frameon=False);save(fig,'03_estimated_vs_actual_noise')
 # 4. Switching rate.
 fig,ax=plt.subplots(figsize=(6.5,4.5));x=np.arange(3);r1=[s['switching_by_true_noise_mm'][str(n)]['R1_pct'] for n in NOISE];r4=[s['switching_by_true_noise_mm'][str(n)]['R4_pct'] for n in NOISE]
 ax.bar(x-.18,r1,.36,label='Select R1',color='#CC6677');ax.bar(x+.18,r4,.36,label='Select R4',color='#228833');ax.set_xticks(x,NOISE);ax.set_xlabel('True observation noise σ (mm)');ax.set_ylabel('Adaptive decisions (%)');ax.legend(frameon=False);ax.set_ylim(0,100);ax.grid(axis='y',alpha=.2);save(fig,'04_switch_frequency')
 # 5. Calls vs performance.
 fig,ax=plt.subplots(figsize=(7,4.5))
 for ctrl in CONTROLLERS:
  vals=[x for x in cdata if x['controller']==ctrl];ax.scatter([x['mean_inference_calls'] for x in vals],[x['success_rate_pct'] for x in vals],label=ctrl,color=colors[ctrl],s=45)
 ax.set_xlabel('Mean BC inference calls / episode');ax.set_ylabel('Success (%)');ax.grid(alpha=.25);ax.legend(frameon=False);save(fig,'05_inference_cost_vs_success')
 # 6. Paired beneficial/harmful counts.
 fig,axs=plt.subplots(1,2,figsize=(10,4),sharey=True)
 for ax,base in zip(axs,('R1','R4')):
  ix=np.arange(9);labs=[];benef=[];harm=[]
  for row in s['paired_comparisons']:
   if row['comparison']!=f'Adaptive_vs_{base}':continue
   labs.append(f"{row['model_seed']}\n{row['noise_mm']}");benef.append(row['beneficial']);harm.append(row['harmful'])
  w=.36;ax.bar(ix-w/2,benef,w,label=f'{base} fail → Adaptive success',color='#228833');ax.bar(ix+w/2,harm,w,label=f'{base} success → Adaptive fail',color='#CC6677');ax.set_xticks(ix,labs);ax.set_title(f'Adaptive vs {base}');ax.grid(axis='y',alpha=.2);ax.set_xlabel('BC seed / noise mm')
 axs[0].set_ylabel('Paired positions');axs[-1].legend(frameon=False,fontsize=8);fig.tight_layout();save(fig,'06_beneficial_harmful_transitions')

if __name__=='__main__':main()

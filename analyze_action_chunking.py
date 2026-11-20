"""Summarize the preregistered crossed H1/H4 × BC/SAC-seed evaluation."""
import csv,json,math
from pathlib import Path
from collections import defaultdict
import numpy as np
from action_chunking import sha256,verify_protocol

ROOT=Path('bc_results/action_chunking_eval');OUT=Path('bc_results/action_chunking_analysis')
BCS=(42,123,456);SACS=(1103,1117,1123)

def read(path):
    with Path(path).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def exact_p(a,b):
    n=a+b
    if not n:return 1.
    return min(1.,2.*sum(math.comb(n,k) for k in range(min(a,b)+1))/(2**n))
def holm(rows):
    ordered=sorted(rows,key=lambda x:x['p_raw']);m=len(rows);running=0.
    for rank,r in enumerate(ordered):
        running=max(running,min(1.,(m-rank)*r['p_raw']));r['p_holm']=running
def mean_sd(values):return (float(np.mean(values)),float(np.std(values,ddof=1)) if len(values)>1 else 0.)
def pair(baseline,policy):
    b={int(x['seed']):x for x in baseline};p={int(x['seed']):x for x in policy}
    if set(b)!=set(p):raise ValueError('Paired position sets differ')
    rescue=reg=0
    for s in b:
        x=int(b[s]['success']);y=int(p[s]['success']);rescue+=int(not x and y);reg+=int(x and not y)
    return b,p,rescue,reg
def file_bc(phase,h,b):return read(ROOT/f'bc_{phase}_{h}_seed{b}.csv')
def file_sac(label,h,b,s):return read(ROOT/f'sac_final_{label}_{h}_bc{b}_sac{s}.csv')

def main():
    protocol,_=verify_protocol();OUT.mkdir(parents=True,exist_ok=True)
    # Final rows: BC only and final policies.
    bc={ (h,b):file_bc('final',h,b) for h in ('H1','H4') for b in BCS }
    candidates=[]
    manifest=json.loads(Path('bc_results/action_chunking_checkpoint_selections.json').read_text())
    selected={ (x['architecture'],x['bc_seed'],x['sac_seed']):x['selected_step'] for x in manifest['selections'] }
    final={}
    for h in ('H1','H4'):
      for b in BCS:
       for s in SACS:
        label_files=[('fixed30k','fixed30k')]
        if selected[(h,b,s)]!=30000:label_files.append(('selected',f'selected{selected[(h,b,s)]}'))
        else:label_files.append(('selected','fixed30k'))
        for label,file_label in label_files:
            rows=file_sac(file_label,h,b,s);baseline=bc[(h,b)];bb,pp,res,reg=pair(baseline,rows)
            successes=sum(int(x['success']) for x in rows);n=len(rows)
            fail=sum(1-int(x['success']) for x in baseline);bsucc=n-fail
            out={'architecture':h,'bc_seed':b,'sac_seed':s,'checkpoint':label,'selected_step':selected[(h,b,s)],
             'episodes':n,'successes':successes,'success_rate':successes/n,'bc_successes':bsucc,'bc_success_rate':bsucc/n,
             'gain_pp':100*(successes-bsucc)/n,'rescue':res,'regression':reg,'net_improvement':res-reg,
             'regression_rate_given_bc_success':reg/bsucc if bsucc else 0.,'rescue_rate_given_bc_failure':res/fail if fail else 0.,
             'always_on_sac_calls':sum(int(x['sac_inference_calls']) for x in rows),
             'sac_activation_rate':sum(int(x['sac_inference_calls']) for x in rows)/sum(int(x['steps']) for x in rows),
             'mean_bc_calls':float(np.mean([int(x['bc_inference_calls']) for x in rows])),
             'mean_bc_latency_ms':float(np.mean([float(x['bc_mean_latency_ms']) for x in rows])),
             'mean_sac_latency_ms':float(np.mean([float(x['sac_mean_latency_ms']) for x in rows])),
             'mean_episode_seconds':float(np.mean([float(x['episode_wall_seconds']) for x in rows])),
             'mean_first_close_step':float(np.mean([float(x['first_close_step']) for x in rows if x['first_close_step']!=''])) if any(x['first_close_step']!='' for x in rows) else '',
             'mean_first_lift_step':float(np.mean([float(x['first_lift_step']) for x in rows if x['first_lift_step']!=''])) if any(x['first_lift_step']!='' for x in rows) else '',
             'mean_closed_fraction':float(np.mean([float(x['closed_fraction']) for x in rows])),
             'mean_first_close_chunk_phase':float(np.mean([int(x['first_close_step'])% (1 if h=='H1' else 4) for x in rows if x['first_close_step']!=''])) if any(x['first_close_step']!='' for x in rows) else '',
             'mean_first_lift_chunk_phase':float(np.mean([int(x['first_lift_step'])% (1 if h=='H1' else 4) for x in rows if x['first_lift_step']!=''])) if any(x['first_lift_step']!='' for x in rows) else '',
             'mean_residual_norm_m':float(np.mean([float(x['mean_residual_norm_m']) for x in rows])),
             'saturation_rate':float(np.mean([float(x['saturation_rate']) for x in rows])),
             'intervention_rate':float(np.mean([float(x['intervention_rate']) for x in rows]))}
            final[(h,b,s,label)]=(out,bb,pp);candidates.append(out)
    # Write per-cell paired outcomes.
    write_csv(OUT/'final_cell_metrics.csv',candidates)
    # BC architecture summaries and seed-level metrics.
    bc_rows=[]
    for h in ('H1','H4'):
      for b in BCS:
       rows=bc[(h,b)];succ=sum(int(x['success']) for x in rows);n=len(rows)
       bc_rows.append({'architecture':h,'bc_seed':b,'episodes':n,'successes':succ,'success_rate':succ/n,
          'mean_steps':float(np.mean([int(x['steps']) for x in rows])),
          'mean_first_close_step':float(np.mean([int(x['first_close_step']) for x in rows if x['first_close_step']!=''])) if any(x['first_close_step']!='' for x in rows) else '',
          'mean_closed_fraction':float(np.mean([float(x['closed_fraction']) for x in rows])),
          'failed_region_counts':json.dumps({q:sum(int(x['region']==q and int(x['success'])==0) for x in rows) for q in ('EN','ES','WN','WS')},sort_keys=True),
          'mean_bc_calls':float(np.mean([int(x['bc_inference_calls']) for x in rows])),
          'mean_bc_latency_ms':float(np.mean([float(x['bc_mean_latency_ms']) for x in rows])),
          'mean_episode_seconds':float(np.mean([float(x['episode_wall_seconds']) for x in rows]))})
    write_csv(OUT/'bc_final_metrics.csv',bc_rows)
    # Paired H4-H1 BC comparison on common final positions per corresponding BC seed.
    arch_tests=[]
    for b in BCS:
        h1,h4,r,g=pair(bc[('H1',b)],bc[('H4',b)])
        arch_tests.append({'bc_seed':b,'h1_successes':sum(int(x['success']) for x in h1.values()),
          'h4_successes':sum(int(x['success']) for x in h4.values()),'rescue_h1_to_h4':r,'regression_h4_to_h1':g,'p_raw':exact_p(r,g)})
    holm(arch_tests);write_csv(OUT/'h1_vs_h4_bc_mcnemar.csv',arch_tests)
    # Paired SAC-vs-BC McNemar tests, Holm within each architecture/checkpoint family.
    tests=[]
    for c in candidates:
        h,b,s,label=c['architecture'],c['bc_seed'],c['sac_seed'],c['checkpoint']
        tests.append({'architecture':h,'bc_seed':b,'sac_seed':s,'checkpoint':label,'rescue':c['rescue'],'regression':c['regression'],
                      'p_raw':exact_p(c['rescue'],c['regression'])})
    for h in ('H1','H4'):
      for label in sorted({r['checkpoint'] for r in tests}):
        grp=[r for r in tests if r['architecture']==h and r['checkpoint']==label]
        if grp:holm(grp)
    write_csv(OUT/'paired_mcnemar.csv',tests)
    # Primary Difference-in-Gains for each same BC/SAC seed pair and checkpoint.
    gains=[]
    labels=('fixed30k','selected')
    for label in labels:
      for b in BCS:
       for s in SACS:
        h1=next(c for c in candidates if c['architecture']=='H1' and c['bc_seed']==b and c['sac_seed']==s and c['checkpoint']==label)
        h4=next(c for c in candidates if c['architecture']=='H4' and c['bc_seed']==b and c['sac_seed']==s and c['checkpoint']==label)
        gains.append({'checkpoint':label,'bc_seed':b,'sac_seed':s,'gain_h1_pp':h1['gain_pp'],'gain_h4_pp':h4['gain_pp'],
                      'difference_in_gains_pp':h4['gain_pp']-h1['gain_pp']})
    write_csv(OUT/'difference_in_gains.csv',gains)
    # Crossed hierarchical bootstrap: independently resample BC and SAC seed indices,
    # then form their Cartesian grid. Positions remain paired inside each cell.
    rng=np.random.default_rng(8675309);boot={lab:[] for lab in labels}
    for _ in range(10000):
        bs=rng.choice(BCS,size=3,replace=True);ss=rng.choice(SACS,size=3,replace=True)
        for lab in labels:
            vals=[]
            for b in bs:
             for s in ss:
              a=next(x['difference_in_gains_pp'] for x in gains if x['checkpoint']==lab and x['bc_seed']==b and x['sac_seed']==s)
              vals.append(a)
            boot[lab].append(float(np.mean(vals)))
    summary=[]
    for lab,vals in boot.items():
        points=[x['difference_in_gains_pp'] for x in gains if x['checkpoint']==lab]
        summary.append({'checkpoint':lab,'mean_difference_in_gains_pp':float(np.mean(points)),
          'sd_across_9_crossed_cells_pp':float(np.std(points,ddof=1)),
          'hierarchical_bootstrap_95_low_pp':float(np.percentile(vals,2.5)),
          'hierarchical_bootstrap_95_high_pp':float(np.percentile(vals,97.5)),
          'n_bc_training_seeds':3,'n_sac_training_seeds':3,'bootstrap_replicates':10000})
    write_csv(OUT/'difference_in_gains_bootstrap.csv',summary)
    # Separate crossed-factor descriptive means for per SAC-seed and per BC-seed variation.
    variability=[]
    for h in ('H1','H4'):
      for label in labels:
       for factor,ids in (('bc_seed',BCS),('sac_seed',SACS)):
        for ident in ids:
            vals=[r['gain_pp'] for r in candidates if r['architecture']==h and r['checkpoint']==label and r[factor]==ident]
            if vals:variability.append({'architecture':h,'checkpoint':label,'factor':factor,'seed':ident,'mean_gain_pp':float(np.mean(vals)),
                'sd_over_crossed_other_factor_pp':float(np.std(vals,ddof=1)) if len(vals)>1 else 0.,'cells':len(vals)})
    write_csv(OUT/'seed_variability.csv',variability)
    # Four-region paired rescue/regression tables.
    regions=[]
    for h,b,s,label in final:
        _,base,pol=final[(h,b,s,label)]
        for region in ('EN','ES','WN','WS'):
            seeds=[k for k in base if base[k]['region']==region]
            r=sum(int(base[k]['success'])==0 and int(pol[k]['success'])==1 for k in seeds)
            g=sum(int(base[k]['success'])==1 and int(pol[k]['success'])==0 for k in seeds)
            regions.append({'architecture':h,'bc_seed':b,'sac_seed':s,'checkpoint':label,'region':region,'episodes':len(seeds),
                'bc_successes':sum(int(base[k]['success']) for k in seeds),'sac_successes':sum(int(pol[k]['success']) for k in seeds),
                'bc_failures':sum(1-int(base[k]['success']) for k in seeds),'sac_failures':sum(1-int(pol[k]['success']) for k in seeds),
                'rescue':r,'regression':g,'net_improvement':r-g})
    write_csv(OUT/'regional_rescue_regression.csv',regions)
    # Checkpoint selection effect and learning curve summaries.
    curves=[];selection=[]
    for row in manifest['selections']:
        h,b,s=row['architecture'],row['bc_seed'],row['sac_seed'];run=Path('bc_results/action_chunking_sac')/f'{h}_bc{b}_sac{s}'
        for v in read(run/'validation.csv'):curves.append({'architecture':h,'bc_seed':b,'sac_seed':s,**v})
        f=next(x for x in candidates if x['architecture']==h and x['bc_seed']==b and x['sac_seed']==s and x['checkpoint']=='fixed30k')
        chosen=next(x for x in candidates if x['architecture']==h and x['bc_seed']==b and x['sac_seed']==s and x['checkpoint']=='selected')
        selection.append({'architecture':h,'bc_seed':b,'sac_seed':s,'selected_step':row['selected_step'],
            'validation_success_rate':row['selection_success_rate'],'final_fixed30k_success_rate':f['success_rate'],
            'final_selected_success_rate':chosen['success_rate'],'final_selection_minus_fixed_pp':100*(chosen['success_rate']-f['success_rate']),
            'fixed30k_rescue':f['rescue'],'fixed30k_regression':f['regression'],
            'selected_rescue':chosen['rescue'],'selected_regression':chosen['regression']})
    write_csv(OUT/'checkpoint_learning_curves.csv',curves);write_csv(OUT/'checkpoint_selection_effect.csv',selection)
    summary_json={'protocol_sha256':sha256('bc_results/action_chunking_protocol.json'),'selection_manifest_sha256':sha256('bc_results/action_chunking_checkpoint_selections.json'),
      'bc':bc_rows,'final_cell_metrics':candidates,'difference_in_gains_bootstrap':summary,'limits':'Three crossed BC seeds and three SAC seeds; shared positions are paired outcomes, not independent training replications.'}
    (OUT/'summary.json').write_text(json.dumps(summary_json,indent=2)+'\n')
    print(f'Wrote action-chunking analysis to {OUT}',flush=True)

def write_csv(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
if __name__=='__main__':main()

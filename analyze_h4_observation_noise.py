"""Preregistered paired analysis and figures for frozen H4 observation noise."""
import csv, json
from pathlib import Path

import numpy as np

from action_chunking import sha256
from analyze_h4_replanning import exact_mcnemar, holm, paired_bootstrap_ci, mean_sd
from evaluate_h4_observation_noise import OUT, FIELDS, NOISE_SEED

MODELS=(42,123,456); INTERVALS=(1,4); SIGMAS=(0,5,10); ANALYSIS_SEED=20261107
BOOTSTRAPS=10000; SEEDS=range(321000,322000); REGIONS=('EN','ES','WN','WS')

def load_rows():
    audit=json.loads((OUT/'final_integrity_audit.json').read_text())
    if not audit.get('final_evaluation_complete') or audit['total_final_episodes']!=18000:
        raise RuntimeError('Final integrity audit absent or incomplete')
    rows={}
    for m in MODELS:
        for r in INTERVALS:
            for n in SIGMAS:
                p=OUT/f'final_H4_seed{m}_R{r}_noise{n}mm.csv'
                with p.open(newline='',encoding='utf-8') as f:
                    rd=csv.DictReader(f)
                    if rd.fieldnames!=FIELDS: raise ValueError(f'Bad schema {p}')
                    data={int(x['seed']):x for x in rd}
                if list(data)!=list(SEEDS): raise ValueError(f'Bad or missing seed set {p}')
                rows[m,r,n]=data
    return rows,audit

def outcome(rows,m,r,n): return np.asarray([int(rows[m,r,n][s]['success']) for s in SEEDS],dtype=np.int8)

def gap_ci(r1,r4,seed):
    d=r4-r1; rng=np.random.default_rng(seed); means=[]
    for start in range(0,BOOTSTRAPS,250):
        k=min(250,BOOTSTRAPS-start); idx=rng.integers(0,len(d),(k,len(d)))
        means.extend(d[idx].mean(axis=1).tolist())
    return [100*float(x) for x in np.quantile(means,[.025,.975])]

def transitions(a,b):
    beneficial=int(np.sum((a==0)&(b==1))); harmful=int(np.sum((a==1)&(b==0)))
    return {'beneficial':beneficial,'harmful':harmful,'net':beneficial-harmful}

def main():
    rows,audit=load_rows(); cells=[]; primary=[]; noisy_tests=[]; interactions={}; regions={}
    for m in MODELS:
        sm={}; outcomes={(r,n):outcome(rows,m,r,n) for r in INTERVALS for n in SIGMAS}
        for r in INTERVALS:
            for n in SIGMAS:
                arr=outcomes[r,n]; source=rows[m,r,n].values(); src=list(source)
                calls=np.asarray([int(x['bc_inference_calls']) for x in src]); ms=np.asarray([float(x['bc_total_inference_ms']) for x in src])
                steps=np.asarray([int(x['steps']) for x in src]); succsteps=[int(x['steps']) for x in src if int(x['success'])]
                close=[int(x['first_close_step']) for x in src if x['first_close_step']!='']
                lift=[int(x['first_lift_step']) for x in src if x['first_lift_step']!='']
                sm[r,n]={'successes':int(arr.sum()),'episodes':len(arr),'success_rate_pct':100*float(arr.mean()),
                    'mean_inference_calls':float(calls.mean()),'mean_inference_latency_ms_per_call':float(ms.sum()/calls.sum()),
                    'mean_total_inference_ms':float(ms.mean()),'mean_episode_wall_seconds':float(np.mean([float(x['episode_wall_seconds']) for x in src])),
                    'mean_control_steps':float(steps.mean()),'mean_successful_episode_steps':float(np.mean(succsteps)) if succsteps else None,
                    'mean_first_close_step':float(np.mean(close)) if close else None,'mean_first_lift_step':float(np.mean(lift)) if lift else None}
                cells.append({'model_seed':m,'interval':r,'noise_mm':n,**sm[r,n]})
            for n in (5,10):
                t=transitions(outcomes[r,0],outcomes[r,n]); p=exact_mcnemar(rows[m,r,0],rows[m,r,n])['exact_two_sided_p']
                noisy_tests.append({'model_seed':m,'interval':r,'noise_mm':n,'success_degradation_pp':sm[r,n]['success_rate_pct']-sm[r,0]['success_rate_pct'],**t,'exact_p':p})
        for n in SIGMAS:
            a,b=outcomes[1,n],outcomes[4,n]; trans=transitions(a,b)
            comp=exact_mcnemar(rows[m,1,n],rows[m,4,n]); ci=gap_ci(a,b,ANALYSIS_SEED+m+n)
            g=100*float((b-a).mean()); item={'model_seed':m,'noise_mm':n,'R4_minus_R1_gap_pp':g,'bootstrap_95pct_CI_pp':ci,**trans,**comp}
            primary.append(item)
        for n in (5,10):
            d=(outcomes[4,n]-outcomes[1,n])-(outcomes[4,0]-outcomes[1,0])
            rng=np.random.default_rng(ANALYSIS_SEED+m+n); means=[]
            # Joint row resampling preserves all six condition outcomes per position.
            joint=np.stack([outcomes[r,s] for r in INTERVALS for s in SIGMAS],axis=1)
            for start in range(0,BOOTSTRAPS,250):
                k=min(250,BOOTSTRAPS-start); ix=rng.integers(0,len(joint),(k,len(joint)))
                z=joint[ix]; means.extend((z[:,:,5 if n==10 else 4].mean(axis=1)-z[:,:,3 if n==10 else 2].mean(axis=1)-z[:,:,1].mean(axis=1)+z[:,:,0].mean(axis=1)).tolist())
            # Explicit paired interaction is computed from aligned outcome vectors.
            vals=[]
            for ix_start in range(0,BOOTSTRAPS,250):
                k=min(250,BOOTSTRAPS-ix_start); ix=rng.integers(0,len(joint),(k,len(joint))); z=joint[ix]
                clean_gap=z[:,:,3].mean(axis=1)-z[:,:,0].mean(axis=1)
                noisy_gap=z[:,:,5 if n==10 else 4].mean(axis=1)-z[:,:,2 if n==10 else 1].mean(axis=1)
                vals.extend((noisy_gap-clean_gap).tolist())
            interactions[m,n]={'interaction_pp':100*float(d.mean()),'bootstrap_95pct_CI_pp':[100*float(v) for v in np.quantile(vals,[.025,.975])]}
        regions[m]={}
        for region in REGIONS:
            rr={}
            for n in SIGMAS:
                by={}
                for r in INTERVALS:
                    subset=[x for x in rows[m,r,n].values() if x['region']==region]
                    by[f'R{r}']={'n':len(subset),'successes':sum(int(x['success']) for x in subset),'success_rate_pct':100*np.mean([int(x['success']) for x in subset]) if subset else None}
                keys=[s for s in SEEDS if rows[m,1,n][s]['region']==region]
                rr[str(n)]={**by,**transitions(np.array([int(rows[m,1,n][s]['success']) for s in keys]),np.array([int(rows[m,4,n][s]['success']) for s in keys]))}
            regions[m][region]=rr
    # Holm primary families: three model-specific McNemar tests at each noise level.
    for n in SIGMAS:
        items=[x for x in primary if x['noise_mm']==n]
        for x,p in zip(items,holm([y['exact_two_sided_p'] for y in items])): x['holm_adjusted_p_across_3_models_at_sigma']=float(p)
    for x,p in zip(noisy_tests,holm([y['exact_p'] for y in noisy_tests])): x['holm_adjusted_p_across_12_noisy_clean_tests']=float(p)
    seed_gaps={str(m):[x['R4_minus_R1_gap_pp'] for x in primary if x['model_seed']==m] for m in MODELS}
    summary={'protocol_sha256':audit['protocol_sha256'],'final_integrity_audit_sha256':sha256(OUT/'final_integrity_audit.json'),
      'analysis':{'positions_per_cell':1000,'model_seeds':list(MODELS),'intervals':list(INTERVALS),'noise_mm':list(SIGMAS),'bootstrap_replicates':BOOTSTRAPS,'bootstrap_seed':ANALYSIS_SEED,'paired_unit':'evaluation seed / shared initial position within frozen model','trained_policy_n':3,'noise_seed':NOISE_SEED},
      'primary_R4_minus_R1':primary,'noise_interactions':{f'{m}_{n}mm':v for (m,n),v in interactions.items()},'noisy_vs_clean':noisy_tests,
      'cells':cells,'regions':regions,'seed_level_gap_descriptives_by_noise':{str(n):mean_sd([x['R4_minus_R1_gap_pp'] for x in primary if x['noise_mm']==n]) for n in SIGMAS},
      'model_seed_gap_vectors_pp':seed_gaps,'scope_note':'Evaluation positions are paired and do not constitute independent trained policies; only three frozen training seeds.'}
    for name,obj in [('summary.json',summary)]:
        p=OUT/name
        if p.exists():raise FileExistsError(p)
        p.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')
    for name,data in [('cell_summary.csv',cells),('paired_comparisons.csv',primary),('noisy_vs_clean.csv',noisy_tests)]:
        p=OUT/name
        if p.exists():raise FileExistsError(p)
        with p.open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    make_figures(summary,interactions)
    print(f'Wrote summary for {len(cells)} cells; primary paired comparisons={len(primary)}, noisy-clean={len(noisy_tests)}')

def savefig(fig,name):
    for ext in ('svg','png'): fig.savefig(OUT/f'{name}.{ext}',bbox_inches='tight')
    import matplotlib.pyplot as plt; plt.close(fig)

def make_figures(s,interactions):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    colors={42:'#4477AA',123:'#EE6677',456:'#228833'}; noises=np.array([0,5,10])
    # 1: success rate vs noise, with R1/R4 by frozen BC seed.
    fig,axs=plt.subplots(1,3,figsize=(12,3.8),sharey=True)
    for ax,m in zip(axs,MODELS):
        for r,marker in ((1,'o'),(4,'s')):
            vals=[next(x['success_rate_pct'] for x in s['cells'] if x['model_seed']==m and x['interval']==r and x['noise_mm']==n) for n in SIGMAS]
            ax.plot(noises,vals,marker=marker,label=f'R{r}')
        ax.set_title(f'Frozen BC seed {m}');ax.set_xlabel('XY observation noise σ (mm)');ax.set_xticks(noises);ax.grid(alpha=.25)
    axs[0].set_ylabel('Grasp success (%)');axs[-1].legend(frameon=False);fig.tight_layout();savefig(fig,'01_success_by_noise')
    # 2: R4-R1 gap across noise.
    fig,ax=plt.subplots(figsize=(7,4.5))
    for m in MODELS:
        x=[next(y for y in s['primary_R4_minus_R1'] if y['model_seed']==m and y['noise_mm']==n) for n in SIGMAS]
        ax.errorbar(noises,[y['R4_minus_R1_gap_pp'] for y in x],yerr=[[y['R4_minus_R1_gap_pp']-y['bootstrap_95pct_CI_pp'][0] for y in x],[y['bootstrap_95pct_CI_pp'][1]-y['R4_minus_R1_gap_pp'] for y in x]],marker='o',capsize=3,label=f'BC seed {m}',color=colors[m])
    ax.axhline(0,color='black',lw=1);ax.set_xlabel('XY observation noise σ (mm)');ax.set_ylabel('R4 − R1 success gap (pp)');ax.set_xticks(noises);ax.grid(alpha=.25);ax.legend(frameon=False);savefig(fig,'02_gap_by_noise')
    # 3: paired noise interaction with bootstrap intervals.
    fig,ax=plt.subplots(figsize=(7,4.5));x=np.arange(6);labels=[];vals=[];lo=[];hi=[]
    for m in MODELS:
        for n in (5,10):
            z=interactions[m,n];labels.append(f'{m}\n{n} mm');vals.append(z['interaction_pp']);lo.append(z['interaction_pp']-z['bootstrap_95pct_CI_pp'][0]);hi.append(z['bootstrap_95pct_CI_pp'][1]-z['interaction_pp'])
    ax.errorbar(x,vals,yerr=[lo,hi],fmt='o',capsize=4,color='#4477AA');ax.axhline(0,color='black',lw=1);ax.set_xticks(x,labels);ax.set_ylabel('Noise interaction in R4 − R1 gap (pp)');ax.grid(axis='y',alpha=.25);savefig(fig,'03_noise_interaction')
    # 4: beneficial/harmful transitions by model and noise.
    fig,axs=plt.subplots(1,3,figsize=(12,4),sharey=True)
    for ax,m in zip(axs,MODELS):
        x=np.arange(3);w=.35;benef=[];harm=[]
        for n in SIGMAS:
            z=next(y for y in s['primary_R4_minus_R1'] if y['model_seed']==m and y['noise_mm']==n);benef.append(z['beneficial']);harm.append(z['harmful'])
        ax.bar(x-w/2,benef,w,label='R1 fail → R4 success',color='#228833');ax.bar(x+w/2,harm,w,label='R1 success → R4 fail',color='#CC6677');ax.set_xticks(x,[str(n) for n in SIGMAS]);ax.set_xlabel('Noise σ (mm)');ax.set_title(f'BC seed {m}');ax.grid(axis='y',alpha=.2)
    axs[0].set_ylabel('Paired positions');axs[-1].legend(frameon=False,fontsize=8);fig.tight_layout();savefig(fig,'04_paired_transitions')
    # 5: regional R4-R1 gap by noise, four facets.
    fig,axs=plt.subplots(2,2,figsize=(9,7),sharex=True)
    for ax,reg in zip(axs.flat,REGIONS):
        for m in MODELS:
            vals=[]
            for n in SIGMAS:
                z=s['regions'][m][reg][str(n)]; vals.append(100*(z['R4']['successes']/z['R4']['n']-z['R1']['successes']/z['R1']['n']) if z['R4']['n'] and z['R1']['n'] else np.nan)
            ax.plot(noises,vals,marker='o',label=f'BC seed {m}',color=colors[m])
        ax.axhline(0,color='black',lw=.8);ax.set_title(reg);ax.set_xticks(noises);ax.grid(alpha=.2)
    axs[1,0].set_xlabel('XY observation noise σ (mm)');axs[1,1].set_xlabel('XY observation noise σ (mm)');axs[0,0].set_ylabel('R4 − R1 success gap (pp)');axs[1,0].set_ylabel('R4 − R1 success gap (pp)');axs[0,0].legend(frameon=False,fontsize=8);fig.tight_layout();savefig(fig,'05_regional_gap')

if __name__=='__main__': main()

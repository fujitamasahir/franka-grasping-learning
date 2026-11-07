"""Paired analysis and publication-ready plots for the H4 replanning ablation."""
import csv
import json
import math
from pathlib import Path

import numpy as np

from evaluate_h4_replanning import FIELDS, OUT, PROTOCOL, PROTOCOL_SHA, validate_rows
from action_chunking import sha256


MODEL_SEEDS=(42,123,456)
INTERVALS=(1,2,4)
ANALYSIS_SEED=20261106
BOOTSTRAPS=10000


def read_config(model_seed, interval):
    p=OUT/f'final_H4_seed{model_seed}_R{interval}.csv'
    if not p.is_file(): raise FileNotFoundError(p)
    with p.open(newline='',encoding='utf-8') as f:
        reader=csv.DictReader(f)
        if reader.fieldnames!=FIELDS: raise ValueError(f'Unexpected schema in {p}')
        rows=list(reader)
    seeds=range(310000,312000)
    model_path=Path(f'bc_models/action_chunking/H4_seed{model_seed}.pt')
    digest=sha256(model_path); protocol_hash=PROTOCOL_SHA.read_text().split()[0]
    validate_rows(rows,seeds,model_seed,interval,digest,protocol_hash)
    if len(rows)!=2000: raise ValueError(f'Expected 2000 rows in {p}')
    meta=json.loads((OUT/f'final_H4_seed{model_seed}_R{interval}.json').read_text())
    if meta['checkpoint_sha256']!=digest or meta['protocol_sha256']!=protocol_hash:
        raise ValueError(f'Metadata provenance mismatch for {p}')
    return {int(r['seed']):r for r in rows}


def exact_mcnemar(baseline, comparison):
    b=sum(int(baseline[s]['success'])==1 and int(comparison[s]['success'])==0 for s in baseline)
    c=sum(int(baseline[s]['success'])==0 and int(comparison[s]['success'])==1 for s in baseline)
    n=b+c
    p=1.0 if n==0 else min(1.0,2.0*sum(math.comb(n,k) for k in range(min(b,c)+1))/(2**n))
    return {'baseline_success_comparison_failure':b,'baseline_failure_comparison_success':c,
            'discordant_pairs':n,'exact_two_sided_p':p}


def holm(pvalues):
    order=np.argsort(pvalues); adjusted=[1.0]*len(pvalues); running=0.0
    for rank,index in enumerate(order):
        value=min(1.0,(len(pvalues)-rank)*float(pvalues[index]))
        running=max(running,value);adjusted[index]=running
    return adjusted


def paired_bootstrap_ci(baseline, comparison, seed):
    keys=sorted(baseline)
    d=np.asarray([int(comparison[s]['success'])-int(baseline[s]['success']) for s in keys],dtype=np.int8)
    rng=np.random.default_rng(seed)
    # Bounded batches avoid allocating a 10000 x 2000 index matrix.
    means=[]
    for start in range(0,BOOTSTRAPS,250):
        n=min(250,BOOTSTRAPS-start)
        idx=rng.integers(0,len(d),size=(n,len(d)))
        means.extend(d[idx].mean(axis=1).tolist())
    lo,hi=np.quantile(means,[.025,.975])
    return [100*float(lo),100*float(hi)]


def compare(baseline, comparison, comparison_interval, model_seed):
    base_success=sum(int(r['success']) for r in baseline.values())
    comp_success=sum(int(r['success']) for r in comparison.values())
    paired=exact_mcnemar(baseline,comparison)
    rescue=paired['baseline_failure_comparison_success'];regression=paired['baseline_success_comparison_failure']
    if comp_success-base_success != rescue-regression:
        raise AssertionError('Paired rescue/regression arithmetic inconsistent')
    return {'baseline_interval':1,'comparison_interval':comparison_interval,'baseline_successes':base_success,
            'comparison_successes':comp_success,'baseline_success_rate_pct':100*base_success/len(baseline),
            'comparison_success_rate_pct':100*comp_success/len(comparison),
            'difference_pp':100*(comp_success-base_success)/len(baseline),'rescue':rescue,'regression':regression,
            'net_improvement':rescue-regression,'paired_difference_95pct_bootstrap_ci_pp':paired_bootstrap_ci(baseline,comparison,ANALYSIS_SEED+model_seed+comparison_interval),
            **paired}


def mean_sd(values):
    arr=np.asarray(values,dtype=float)
    return {'mean':float(arr.mean()),'median':float(np.median(arr)),'sample_sd':float(arr.std(ddof=1)),
            'min':float(arr.min()),'max':float(arr.max())}


def mean_or_none(values):
    return float(np.mean(values)) if values else None


def summarize(rows):
    outputs={}
    for model_seed in MODEL_SEEDS:
        baseline=rows[(model_seed,1)]
        cell={}
        prior_path=Path(f'bc_results/action_chunking_eval/bc_final_H4_seed{model_seed}.csv')
        with prior_path.open(newline='',encoding='utf-8') as f: prior=list(csv.DictReader(f))
        prior_success=sum(int(r['success']) for r in prior)
        for interval in INTERVALS:
            current=rows[(model_seed,interval)]
            successes=sum(int(r['success']) for r in current.values())
            cell[f'R{interval}']={'successes':successes,'episodes':len(current),'success_rate_pct':100*successes/len(current),
                'mean_inference_calls':float(np.mean([int(r['bc_inference_calls']) for r in current.values()])),
                'mean_inference_latency_ms_per_call':float(1000*sum(float(r['bc_total_inference_ms']) for r in current.values())/
                    sum(int(r['bc_inference_calls']) for r in current.values())),
                'mean_total_inference_ms':float(np.mean([float(r['bc_total_inference_ms']) for r in current.values()])),
                'mean_episode_wall_seconds':float(np.mean([float(r['episode_wall_seconds']) for r in current.values()])),
                'mean_control_steps':float(np.mean([int(r['steps']) for r in current.values()])),
                'mean_successful_episode_steps':mean_or_none([int(r['steps']) for r in current.values() if int(r['success'])]),
                'mean_first_close_step':mean_or_none([int(r['first_close_step']) for r in current.values() if r['first_close_step']!='']),
                'mean_first_lift_step':mean_or_none([int(r['first_lift_step']) for r in current.values() if r['first_lift_step']!=''])}
        cell['previous_H4_final_reference']={'seed_range_inclusive':[290000,291999],'successes':prior_success,
            'success_rate_pct':100*prior_success/len(prior),'new_R4_minus_previous_pp':cell['R4']['success_rate_pct']-100*prior_success/len(prior)}
        cell['paired_comparisons']={}
        for interval in (2,4): cell['paired_comparisons'][f'R{interval}_vs_R1']=compare(baseline,rows[(model_seed,interval)],interval,model_seed)
        cell['R4_vs_R2_descriptive']=compare(rows[(model_seed,2)],rows[(model_seed,4)],4,model_seed)
        # Replace the comparator's labels to avoid implying a comparison to R1.
        cell['R4_vs_R2_descriptive']['baseline_interval']=2
        outputs[str(model_seed)]=cell
    for family in ('R2_vs_R1','R4_vs_R1'):
        p=[outputs[str(s)]['paired_comparisons'][family]['exact_two_sided_p'] for s in MODEL_SEEDS]
        adj=holm(p)
        for seed,padj in zip(MODEL_SEEDS,adj):outputs[str(seed)]['paired_comparisons'][family]['holm_adjusted_p_across_3_models']=float(padj)
    primary=[outputs[str(s)]['paired_comparisons']['R4_vs_R1']['difference_pp'] for s in MODEL_SEEDS]
    seed_summary=mean_sd(primary);seed_summary['directional_consistency_positive']=sum(x>0 for x in primary)
    seed_summary['directional_consistency_nonnegative']=sum(x>=0 for x in primary)
    regions={}
    for model_seed in MODEL_SEEDS:
        regions[str(model_seed)]={}
        for interval in INTERVALS:
            current=rows[(model_seed,interval)];regions[str(model_seed)][f'R{interval}']={}
            for reg in ('EN','ES','WN','WS'):
                subset=[r for r in current.values() if r['region']==reg]
                regions[str(model_seed)][f'R{interval}'][reg]={'n':len(subset),'successes':sum(int(r['success']) for r in subset),
                    'success_rate_pct':100*sum(int(r['success']) for r in subset)/len(subset) if subset else None}
        base=current_by_region=rows[(model_seed,1)]
        regions[str(model_seed)]['paired_transitions_vs_R1']={}
        for interval in (2,4):
            other=rows[(model_seed,interval)]; regions[str(model_seed)]['paired_transitions_vs_R1'][f'R{interval}']={}
            for reg in ('EN','ES','WN','WS'):
                keys=[s for s,r in base.items() if r['region']==reg]
                rescue=sum(int(base[s]['success'])==0 and int(other[s]['success'])==1 for s in keys)
                regression=sum(int(base[s]['success'])==1 and int(other[s]['success'])==0 for s in keys)
                regions[str(model_seed)]['paired_transitions_vs_R1'][f'R{interval}'][reg]={'n':len(keys),'rescue':rescue,'regression':regression,'net':rescue-regression}
    return {'protocol_sha256':PROTOCOL_SHA.read_text().split()[0],'protocol_file_sha256':sha256(PROTOCOL),
            'analysis':{'positions_per_cell':2000,'model_seeds':list(MODEL_SEEDS),'intervals':list(INTERVALS),
                'bootstrap_replicates':BOOTSTRAPS,'bootstrap_seed_base':ANALYSIS_SEED,'position_level_inference_conditional_on_fixed_models':True},
            'per_model':outputs,'seed_level_primary_R4_minus_R1_pp':seed_summary,'regions':regions}


def make_figures(rows, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150})
    colors={42:'#4477AA',123:'#EE6677',456:'#228833'}
    # 1. Success by interval, model-seed lines.
    fig,ax=plt.subplots(figsize=(7.2,4.5))
    for s in MODEL_SEEDS:
        vals=[summary['per_model'][str(s)][f'R{i}']['success_rate_pct'] for i in INTERVALS]
        ax.plot(['R1','R2','R4'],vals,marker='o',linewidth=2,color=colors[s],label=f'BC seed {s}')
    ax.set_ylabel('Grasp success (%)');ax.set_xlabel('Runtime replanning interval (steps)');ax.set_ylim(70,100);ax.grid(axis='y',alpha=.25);ax.legend(frameon=False)
    savefig(fig,'success_by_interval')
    # 2. Paired transitions R4 relative to R1.
    fig,ax=plt.subplots(figsize=(7.2,4.5));x=np.arange(3);w=.34
    resc=[summary['per_model'][str(s)]['paired_comparisons']['R4_vs_R1']['rescue'] for s in MODEL_SEEDS]
    regs=[summary['per_model'][str(s)]['paired_comparisons']['R4_vs_R1']['regression'] for s in MODEL_SEEDS]
    ax.bar(x-w/2,resc,w,label='R1 failure → R4 success',color='#228833');ax.bar(x+w/2,regs,w,label='R1 success → R4 failure',color='#CC6677')
    ax.set_xticks(x,[str(s) for s in MODEL_SEEDS]);ax.set_xlabel('Frozen BC training seed');ax.set_ylabel('Paired position count');ax.legend(frameon=False);ax.grid(axis='y',alpha=.25)
    savefig(fig,'paired_transitions_R4_vs_R1')
    # 3. Inference calls vs success.
    fig,ax=plt.subplots(figsize=(7.2,4.5));markers={1:'o',2:'s',4:'^'}
    for s in MODEL_SEEDS:
        for i in INTERVALS:
            c=summary['per_model'][str(s)][f'R{i}'];ax.scatter(c['mean_inference_calls'],c['success_rate_pct'],color=colors[s],marker=markers[i],s=65)
            ax.annotate(f'{s}/R{i}',(c['mean_inference_calls'],c['success_rate_pct']),xytext=(4,3),textcoords='offset points',fontsize=8)
    ax.set_xlabel('Mean BC inference calls per episode');ax.set_ylabel('Grasp success (%)');ax.grid(alpha=.25)
    savefig(fig,'inference_calls_vs_success')
    # 4. Wall-clock runtime decomposition.
    fig,ax=plt.subplots(figsize=(8,4.8));x=np.arange(9);labels=[];infer=[];other=[]
    for s in MODEL_SEEDS:
        for i in INTERVALS:
            c=summary['per_model'][str(s)][f'R{i}'];labels.append(f'{s}\nR{i}')
            infer.append(c['mean_total_inference_ms']/1000);other.append(max(0,c['mean_episode_wall_seconds']-c['mean_total_inference_ms']/1000))
    ax.bar(x,infer,label='BC inference time',color='#4477AA');ax.bar(x,other,bottom=infer,label='Other measured episode time',color='#BBBBBB')
    ax.set_xticks(x,labels);ax.set_ylabel('Mean episode time (s)');ax.set_xlabel('BC seed / runtime interval');ax.legend(frameon=False);ax.grid(axis='y',alpha=.25)
    savefig(fig,'episode_runtime')
    # 5. Regional rates with region sample counts.
    fig,axes=plt.subplots(2,2,figsize=(9,6.5),sharey=True)
    for ax,reg in zip(axes.flat,('EN','ES','WN','WS')):
        for s in MODEL_SEEDS:
            vals=[];ns=[]
            for i in INTERVALS:
                r=summary['regions'][str(s)][f'R{i}'][reg];vals.append(r['success_rate_pct']);ns.append(r['n'])
            ax.plot(['R1','R2','R4'],vals,marker='o',color=colors[s],label=f'{s} (n={ns[0]})')
        ax.set_title(reg);ax.set_xlabel('Replanning interval');ax.grid(alpha=.25)
    axes[0,0].set_ylabel('Success (%)');axes[1,0].set_ylabel('Success (%)');axes[0,0].legend(frameon=False,fontsize=8)
    fig.suptitle('Regional grasp success (n shown per model and region)');fig.tight_layout()
    savefig(fig,'regional_success')


def savefig(fig, stem):
    for ext in ('svg','png'):
        fig.savefig(OUT/f'{stem}.{ext}',bbox_inches='tight')
    import matplotlib.pyplot as plt
    plt.close(fig)


def main():
    protocol=json.loads(PROTOCOL.read_text())
    expected=PROTOCOL_SHA.read_text().split()[0]
    if sha256(PROTOCOL)!=expected:raise RuntimeError('Preregistration SHA mismatch')
    rows={(s,i):read_config(s,i) for s in MODEL_SEEDS for i in INTERVALS}
    seed_sets=[set(rows[(s,i)]) for s in MODEL_SEEDS for i in INTERVALS]
    if any(x!=seed_sets[0] for x in seed_sets[1:]):raise ValueError('Final seed sets differ across configurations')
    for seed in seed_sets[0]:
        pos={(rows[(s,i)][seed]['cube_x'],rows[(s,i)][seed]['cube_y']) for s in MODEL_SEEDS for i in INTERVALS}
        if len(pos)!=1:raise ValueError(f'Initial positions differ across configurations for evaluation seed {seed}')
    summary=summarize(rows);OUT.mkdir(parents=True,exist_ok=True)
    out=OUT/'summary.json'
    if out.exists():raise FileExistsError(f'Refusing to overwrite {out}')
    out.write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    make_figures(rows,summary)
    print(f'Wrote {out} and five SVG/PNG figure pairs')


if __name__=='__main__':main()

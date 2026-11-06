"""Analyze preregistered fixed-30k and validation-selected replication results."""
import csv, json, math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_sac_stability import exact_mcnemar, holm_adjust

OUT=Path('bc_results'); FIG=OUT/'sac_regularization_replication_figures'
SEEDS=(1009,1013,1019,1021,1031,1033,1039,1049,1051,1061)
ARMS=('A','B'); REGIONS=('ES','WS')
FINAL=range(260000,262000)

def read(p):
    with Path(p).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def write(p, rows):
    p=Path(p)
    if p.exists(): raise FileExistsError(p)
    if not rows: raise ValueError(p)
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def paired(path):
    rows=read(path); assert len(rows)==4000, (path,len(rows))
    bc={int(r['seed']):r for r in rows if r['policy']=='bc_only'}
    sac={int(r['seed']):r for r in rows if r['policy']=='bc_plus_sac'}
    assert set(bc)==set(FINAL)==set(sac)
    for k in FINAL:
        assert (bc[k]['cube_x'],bc[k]['cube_y'])==(sac[k]['cube_x'],sac[k]['cube_y'])
    return bc,sac
def reg(r):
    x,y=float(r['cube_x'])-.55,float(r['cube_y'])
    if abs(x)<=.12 and abs(y)<=.12:return 'Inside ±12 cm'
    return ('E' if x>=0 else 'W')+('N' if y>=0 else 'S')
def stats_for(bc,sac):
    n=len(bc); bs=sum(int(x['success']) for x in bc.values()); ss=sum(int(x['success']) for x in sac.values())
    resc=regress=act=0; steps=[]; residual=[]; sat=[]
    for k,b in bc.items():
        s=sac[k]; a=int(b['success']); z=int(s['success']); resc+=not a and z; regress+=a and not z
        act+=int(float(s['mean_residual_norm_mm'])>0)
        if z:steps.append(int(s['steps']))
        residual.append(float(s['mean_residual_norm_mm'])); sat.append(float(s['action_saturation_frequency']))
    return dict(episodes=n,bc_successes=bs,bc_success_rate=bs/n,successes=ss,success_rate=ss/n,
      rescue_count=int(resc),regression_count=int(regress),net_improvement=int(resc-regress),
      success_delta_vs_bc_pp=100*(ss-bs)/n,mean_successful_steps=float(np.mean(steps)),
      sac_episodes=int(act),sac_activation_rate=act/n,mean_residual_norm_mm=float(np.mean(residual)),
      action_saturation_frequency=float(np.mean(sat)))
def region_counts(bc,sac,region):
    ids=[k for k in bc if reg(bc[k])==region]
    bsucc=sum(int(bc[k]['success']) for k in ids); bfail=len(ids)-bsucc
    resc=sum(int(not int(bc[k]['success']) and int(sac[k]['success'])) for k in ids)
    regr=sum(int(int(bc[k]['success']) and not int(sac[k]['success'])) for k in ids)
    return dict(region=region,episodes=len(ids),bc_successes=bsucc,bc_failures=bfail,
       sac_successes=sum(int(sac[k]['success']) for k in ids),rescues=resc,regressions=regr,
       rescue_rate=resc/bfail if bfail else '',regression_rate=regr/bsucc if bsucc else '')
def binom_sign(vals):
    q=[v for v in vals if abs(v)>1e-12]; n=len(q)
    if not n:return 1.0
    tail=sum(math.comb(n,k) for k in range(min(sum(v>0 for v in q),sum(v<0 for v in q))+1))
    return min(1.0,2*tail/(2**n))
def boot(vals,rng):
    a=np.asarray(vals,float); samples=rng.choice(a,(10000,len(a)),replace=True).mean(axis=1)
    return float(a.mean()),float(np.quantile(samples,.025)),float(np.quantile(samples,.975))
def summarize_effects(rows,metric,rng,comparison):
    vals=[float(r[metric]) for r in rows]
    mean,lo,hi=boot(vals,rng)
    favorable_sign=-1 if 'regression' in metric else 1
    return dict(comparison=comparison,metric=metric,n_training_seeds=len(vals),mean=float(np.mean(vals)),median=float(np.median(vals)),
      sample_sd=float(np.std(vals,ddof=1)),minimum=float(np.min(vals)),maximum=float(np.max(vals)),
      favorable=int(sum(favorable_sign*v>0 for v in vals)),unfavorable=int(sum(favorable_sign*v<0 for v in vals)),tied=int(sum(v==0 for v in vals)),
      bootstrap_mean=mean,bootstrap_95ci_low=lo,bootstrap_95ci_high=hi,exact_two_sided_sign_p=binom_sign(vals))
def main():
    FIG.mkdir(exist_ok=True)
    freeze=json.loads((OUT/'sac_regularization_replication_checkpoint_freeze.json').read_text())
    selected={(x['sac_seed'],x['arm']):int(x['selected_step']) for x in freeze['selections']}
    datasets={}; seedrows=[]; regional=[]; tests=[]
    for comparison in ('fixed30k','selected'):
      for seed in SEEDS:
       for arm in ARMS:
        step=30000 if comparison=='fixed30k' else selected[seed,arm]
        path=OUT/f'sac_replication_seed{seed}_arm{arm}_step{step}_final2000.csv'
        bc,sac=paired(path); datasets[comparison,seed,arm]=(bc,sac)
        m=stats_for(bc,sac)
        seedrows.append(dict(comparison=comparison,sac_seed=seed,arm=arm,penalty_coefficient=.005 if arm=='A' else .02,checkpoint_step=step,**m))
        for region in ('Inside ±12 cm','EN','ES','WN','WS'):
          regional.append(dict(comparison=comparison,sac_seed=seed,arm=arm,checkpoint_step=step,**region_counts(bc,sac,region)))
      # pair A/B within training seed and compare each model against BC
      for seed in SEEDS:
       ba,aa=datasets[comparison,seed,'A']; bb,ab=datasets[comparison,seed,'B']
       # direct A-vs-B overall plus BC-conditioned regression/rescue discordances
       for endpoint in ('overall','regression','rescue'):
        win=loss=0
        for k in FINAL:
          eligible=(endpoint=='overall' or (endpoint=='regression' and int(ba[k]['success'])==1) or (endpoint=='rescue' and int(ba[k]['success'])==0))
          if eligible:
            av,bv=int(aa[k]['success']),int(ab[k]['success'])
            # fewer regressions is favorable when A failed/B succeeded on BC successes; more rescues likewise
            if endpoint=='regression':win+=int(av==0 and bv==1);loss+=int(av==1 and bv==0)
            else:win+=int(av==0 and bv==1);loss+=int(av==1 and bv==0)
        tests.append(dict(comparison=comparison,sac_seed=seed,arm='B_vs_A',endpoint=endpoint,wins=win,losses=loss,p_raw=exact_mcnemar(win,loss)))
      # Regional A/B difference and exact McNemar tests among eligible baseline cases
      for region in REGIONS:
       for endpoint in ('regression','rescue'):
        for seed in SEEDS:
         ba,aa=datasets[comparison,seed,'A']; bb,ab=datasets[comparison,seed,'B']; wins=losses=0
         for k in FINAL:
          if reg(ba[k])!=region:continue
          if endpoint=='regression' and int(ba[k]['success'])==1:
            wins+=int(int(aa[k]['success'])==0 and int(ab[k]['success'])==1)
            losses+=int(int(aa[k]['success'])==1 and int(ab[k]['success'])==0)
          elif endpoint=='rescue' and int(ba[k]['success'])==0:
            wins+=int(int(aa[k]['success'])==0 and int(ab[k]['success'])==1)
            losses+=int(int(aa[k]['success'])==1 and int(ab[k]['success'])==0)
         tests.append(dict(comparison=comparison,sac_seed=seed,arm='B_vs_A',endpoint=f'{region}_{endpoint}',wins=wins,losses=losses,p_raw=exact_mcnemar(wins,losses)))
    # Holm correction per preregistered comparison family
    families={}
    for i,t in enumerate(tests):
      fam=(t['comparison'],t['endpoint']) if t['endpoint'] in ('overall','regression','rescue') else (t['comparison'],'regional_ES_WS')
      families.setdefault(fam,[]).append((i,float(t['p_raw'])))
    for fam,items in families.items():
      adj=holm_adjust(items)
      for i,_ in items:tests[i]['p_holm']=adj[i]
    write(OUT/'sac_regularization_replication_final_seed_metrics.csv',seedrows)
    write(OUT/'sac_regularization_replication_final_region_metrics.csv',regional)
    write(OUT/'sac_regularization_replication_final_tests.csv',tests)
    # Paired seed differences B-A; bootstrap RNG is dedicated and deterministic.
    rng=np.random.default_rng(20261107); effects=[]
    for comparison in ('fixed30k','selected'):
      byseed={r['sac_seed']:{x['arm']:x for x in seedrows if x['comparison']==comparison and x['sac_seed']==r['sac_seed']} for r in seedrows if r['comparison']==comparison}
      for seed in SEEDS:
       a,b=byseed[seed]['A'],byseed[seed]['B']
       effects.append(dict(comparison=comparison,sac_seed=seed,
        success_rate_diff_b_minus_a_pp=100*(b['success_rate']-a['success_rate']),
        rescue_diff_b_minus_a=b['rescue_count']-a['rescue_count'],
        regression_diff_b_minus_a=b['regression_count']-a['regression_count'],
        net_improvement_diff_b_minus_a=b['net_improvement']-a['net_improvement'],
        activation_rate_diff_b_minus_a_pp=100*(b['sac_activation_rate']-a['sac_activation_rate']),
        sac_calls_saved_vs_always_a=a['episodes']-a['sac_episodes'],sac_calls_saved_vs_always_b=b['episodes']-b['sac_episodes']))
    write(OUT/'sac_regularization_replication_seed_effects.csv',effects)
    effect_stats=[]
    for comparison in ('fixed30k','selected'):
      rr=[r for r in effects if r['comparison']==comparison]
      for metric in ('success_rate_diff_b_minus_a_pp','rescue_diff_b_minus_a','regression_diff_b_minus_a','net_improvement_diff_b_minus_a','activation_rate_diff_b_minus_a_pp'):
        effect_stats.append(summarize_effects(rr,metric,rng,comparison))
    write(OUT/'sac_regularization_replication_seed_effect_summary.csv',effect_stats)
    # ES/WS Arm B minus Arm A conditional rates, summarized over independent training seeds.
    regional_effects=[]
    for comparison in ('fixed30k','selected'):
      for seed in SEEDS:
       bc_a,a=datasets[comparison,seed,'A']; bc_b,b=datasets[comparison,seed,'B']
       for region in REGIONS:
        ra,rb=region_counts(bc_a,a,region),region_counts(bc_b,b,region)
        regional_effects.append(dict(comparison=comparison,sac_seed=seed,region=region,
          regression_rate_diff_b_minus_a_pp=100*(float(rb['regression_rate'])-float(ra['regression_rate'])),
          rescue_rate_diff_b_minus_a_pp=100*(float(rb['rescue_rate'])-float(ra['rescue_rate'])),
          a_regressions=ra['regressions'],b_regressions=rb['regressions'],a_rescues=ra['rescues'],b_rescues=rb['rescues'],
          bc_success_positions=ra['bc_successes'],bc_failure_positions=ra['bc_failures']))
    write(OUT/'sac_regularization_replication_regional_seed_effects.csv',regional_effects)
    regional_stats=[]
    for comparison in ('fixed30k','selected'):
      for region in REGIONS:
       rr=[x for x in regional_effects if x['comparison']==comparison and x['region']==region]
       for metric in ('regression_rate_diff_b_minus_a_pp','rescue_rate_diff_b_minus_a_pp'):
        regional_stats.append(summarize_effects(rr,metric,rng,comparison+':'+region))
    write(OUT/'sac_regularization_replication_regional_seed_effect_summary.csv',regional_stats)
    training=training_diagnostics()
    write(OUT/'sac_regularization_replication_training_diagnostics.csv',training)
    # summaries and paired exact position tests against BC for each arm/seed and arm contrasts
    summary={'protocol_sha256':freeze['protocol_sha256'],'bc_checkpoint_sha256':freeze['bc_checkpoint_sha256'],
      'final_seed_range':[260000,261999],'training_seeds':list(SEEDS),'selection_steps':{f'{s}_{a}':v for (s,a),v in selected.items()},
      'decision_criteria':{},'comparisons':{},'tests':tests,'training_diagnostics':training,
      'dependence_note':'Inference about regularization generalization uses ten paired SAC training seeds. Shared positions make episode-level tests conditional on this fixed evaluation set, not independent training runs.'}
    for comparison in ('fixed30k','selected'):
      rr=[r for r in seedrows if r['comparison']==comparison]
      summary['comparisons'][comparison]={'per_seed':[r for r in rr], 'seed_effects':[e for e in effects if e['comparison']==comparison], 'seed_effect_summaries':[x for x in effect_stats if x['comparison']==comparison]}
      es=[e for e in effects if e['comparison']==comparison]
      reductions=sum(e['regression_diff_b_minus_a']<0 for e in es)
      mean_success=float(np.mean([e['success_rate_diff_b_minus_a_pp'] for e in es])); mean_rescue=float(np.mean([e['rescue_diff_b_minus_a'] for e in es]))
      summary['decision_criteria'][comparison]={'seeds_with_fewer_regressions':reductions,'required_at_least':7,'mean_success_difference_b_minus_a_pp':mean_success,'minimum_allowed_pp':-.5,'mean_rescue_difference_b_minus_a':mean_rescue,'minimum_allowed_rescue_diff':-10,
        'criteria_met':reductions>=7 and mean_success>=-.5 and mean_rescue>=-10}
    # visuals
    plot(seedrows,effects,regional,selected)
    (OUT/'sac_regularization_replication_final_summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print('Wrote metrics, tests, JSON summary and six figures.')
def plot(rows,effects,regional,selected):
    plt.style.use('seaborn-v0_8-whitegrid')
    colors={'A':'#2878B5','B':'#D9534F'}
    def save(fig,name):
      for ext in ('png','svg'):fig.savefig(FIG/f'{name}.{ext}',dpi=180,bbox_inches='tight')
      plt.close(fig)
    # 1 success per training seed fixed
    fig,ax=plt.subplots(figsize=(9,5)); x=np.arange(len(SEEDS)); width=.36
    for j,arm in enumerate(ARMS):
      vals=[next(r['success_rate']*100 for r in rows if r['comparison']=='fixed30k' and r['sac_seed']==s and r['arm']==arm) for s in SEEDS]
      ax.bar(x+(j-.5)*width,vals,width,label=f'Arm {arm}',color=colors[arm])
    ax.set_xticks(x,SEEDS);ax.set_ylabel('Success rate (%)');ax.legend();ax.set_title('Fixed 30k: success by independent training seed');save(fig,'fixed30k_success_by_seed')
    #2 rescue-regression paired seed contrast
    fig,ax=plt.subplots(figsize=(9,5)); x=np.arange(len(SEEDS)); w=.36
    for j,key,label,col in [(0,'rescue_diff_b_minus_a','Rescue B−A','#2A9D8F'),(1,'regression_diff_b_minus_a','Regression B−A','#E76F51')]:
      v=[next(e[key] for e in effects if e['comparison']=='fixed30k' and e['sac_seed']==s) for s in SEEDS]
      ax.bar(x+(j-.5)*w,v,w,label=label,color=col)
    ax.axhline(0,color='black',lw=.8);ax.set_xticks(x,SEEDS);ax.set_ylabel('Count difference');ax.legend();ax.set_title('Fixed 30k: rescue and regression differences (B − A)');save(fig,'fixed30k_rescue_regression_seed_effects')
    #3 success rate selected
    fig,ax=plt.subplots(figsize=(9,5));x=np.arange(len(SEEDS));w=.36
    for j,arm in enumerate(ARMS):
      vals=[next(r['success_rate']*100 for r in rows if r['comparison']=='selected' and r['sac_seed']==s and r['arm']==arm) for s in SEEDS]
      ax.bar(x+(j-.5)*w,vals,w,label=f'Arm {arm}',color=colors[arm])
    ax.set_xticks(x,SEEDS);ax.set_ylabel('Success rate (%)');ax.legend();ax.set_title('Validation-selected checkpoints: success by seed');save(fig,'selected_success_by_seed')
    #4 ES/WS regression and rescue regional pooled descriptive per seed average rate
    fig,axs=plt.subplots(1,2,figsize=(10,4.5))
    for ax,metric,title in [(axs[0],'regression_rate','Regression rate | BC successes'),(axs[1],'rescue_rate','Rescue rate | BC failures')]:
      for arm,col in zip(ARMS,[colors['A'],colors['B']]):
        q=[r for r in regional if r['comparison']=='fixed30k' and r['arm']==arm and r['region'] in ('ES','WS')]
        vals=[]
        for region in ('ES','WS'):
          z=[float(r[metric]) for r in q if r['region']==region and r[metric]!='']
          vals.append(100*np.mean(z) if z else 0)
        ax.plot(['ES','WS'],vals,marker='o',label=f'Arm {arm}',color=col)
      ax.set_title(title);ax.set_ylabel('Mean of per-seed rates (%)');ax.legend()
    fig.suptitle('Fixed 30k regional outcomes (seed-level means)');save(fig,'fixed30k_es_ws_outcomes')
    #5 activation rate
    fig,ax=plt.subplots(figsize=(9,5));x=np.arange(len(SEEDS));w=.36
    for j,arm in enumerate(ARMS):
      vals=[next(r['sac_activation_rate']*100 for r in rows if r['comparison']=='selected' and r['sac_seed']==s and r['arm']==arm) for s in SEEDS]
      ax.bar(x+(j-.5)*w,vals,w,label=f'Arm {arm}',color=colors[arm])
    ax.set_xticks(x,SEEDS);ax.set_ylabel('Episodes with nonzero SAC action (%)');ax.legend();ax.set_title('Selective residual activity by seed (selected checkpoint)');save(fig,'selected_sac_activation_by_seed')
    #6 selected checkpoint step
    fig,ax=plt.subplots(figsize=(9,5));x=np.arange(len(SEEDS));w=.36
    for j,arm in enumerate(ARMS):
      vals=[selected[s,arm]/1000 for s in SEEDS]
      ax.bar(x+(j-.5)*w,vals,w,label=f'Arm {arm}',color=colors[arm])
    ax.set_xticks(x,SEEDS);ax.set_yticks([10,20,30]);ax.set_ylabel('Selected training step (thousands)');ax.legend();ax.set_title('Validation-based checkpoint selection');save(fig,'selected_checkpoint_steps')

def training_diagnostics():
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from pathlib import Path
    tags={'actor_loss':'train/actor_loss','critic_loss':'train/critic_loss','entropy_coefficient':'train/ent_coef',
      'entropy_coefficient_loss':'train/ent_coef_loss','episode_reward':'rollout/ep_rew_mean','episode_length':'rollout/ep_len_mean',
      'rollout_success_rate':'rollout/success_rate','residual_norm_mm':'diagnostics/residual_norm_episode_mean',
      'applied_residual_norm_mm':'diagnostics/applied_residual_norm_m_episode_mean',
      'clip_difference_norm_mm':'diagnostics/residual_clip_difference_norm_m_episode_mean',
      'saturation_episode_mean':'diagnostics/action_saturated_episode_mean','task_reward':'diagnostics/task_reward_episode_mean',
      'shaping_reward':'diagnostics/shaping_reward_episode_mean','residual_penalty':'diagnostics/residual_action_penalty_episode_mean'}
    result=[]
    for seed in SEEDS:
      for arm in ARMS:
        run=Path('bc_residual_sac_runs')/f'replication_{arm}_seed{seed}'
        files=list((Path('bc_residual_sac_logs')/f'replication_{arm}_seed{seed}').rglob('events.out.tfevents*'))
        if len(files)!=1:raise RuntimeError(f'Expected one TensorBoard event file: seed={seed}, arm={arm}, found={files}')
        acc=EventAccumulator(str(files[0]));acc.Reload(); record={'sac_seed':seed,'arm':arm,'step_window':'29001-30000'}
        for name,tag in tags.items():
          if tag not in acc.Tags().get('scalars',[]):record[name]='Not Available';continue
          values=[float(e.value) for e in acc.Scalars(tag) if 29000<int(e.step)<=30000]
          record[name]=float(np.mean(values)) if values else 'Not Available'
          if name in ('residual_norm_mm','applied_residual_norm_mm','clip_difference_norm_mm') and values:
            record[name]=float(np.mean(values))*1000
        diagnostic=read(run/'diagnostics_episode_metrics.csv')
        selected=[r for r in diagnostic if 29000<int(r['training_step'])<=30000]
        record['diagnostic_episode_count']=len(selected)
        result.append(record)
    return result
if __name__=='__main__':main()

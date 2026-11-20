"""Frozen Stage1 gates, paired diagnostics and six standalone figures."""
import csv,json
from pathlib import Path
import numpy as np
from action_chunking import sha256
from run_h4_motion_estimator_stage1 import OUT,PROTOCOL,ESTIMATORS


def rows_from(path):
    sequences=[json.loads(line) for line in path.open()]
    return sequences,[w for s in sequences for w in s['windows']]


def summarize(rows):
    result={}
    for estimator in ESTIMATORS:
        result[estimator]={}
        for motion in ('stationary','moving'):
            result[estimator][motion]={}
            for sigma in (0.,.01):
                rr=[r for r in rows if r['estimator']==estimator and r['motion']==motion and not r['mixed'] and np.isclose(r['target_sigma'],sigma)]
                result[estimator][motion][str(int(sigma*1000))]={
                    'n':len(rr),'MAE_mm':float(np.mean([abs(r['error'])*1000 for r in rr])) if rr else None,
                    'bias_mm':float(np.mean([r['error']*1000 for r in rr])) if rr else None,
                    'classification_accuracy':float(np.mean([r['classification_correct'] for r in rr])) if rr else None,
                    'wrong_interval_fraction':float(np.mean([r['wrong_interval'] for r in rr])) if rr else None,
                    'wrong_direction_actual_switches_per_decision':float(np.mean([r['wrong_switch'] for r in rr])) if rr else None}
    return result


def detection(sequences):
    result={}
    for estimator in ESTIMATORS:
        result[estimator]={}
        for profile in ('clean_to_noisy','noisy_to_clean'):
            target=1 if profile=='clean_to_noisy' else 4;delays=[];misses=0;already=0
            for s in sequences:
                if s['profile']!=profile:continue
                rr=[w for w in s['windows'] if w['estimator']==estimator]
                if next(r['after'] for r in rr if r['step']==40)==target:already+=1
                switches=[r['step']-40 for r in rr if r['step']>=40 and r['switched'] and r['after']==target]
                if switches:delays.append(switches[0])
                else:misses+=1
            result[estimator][profile]={'n':len(delays)+misses,'switches_detected':len(delays),'misses':misses,
                'miss_fraction':misses/(len(delays)+misses),'already_target_at40':already,
                'median_delay_steps':float(np.median(delays)) if delays else None,
                'p90_delay_steps':float(np.quantile(delays,.9)) if delays else None,
                'delays_steps':delays}
    return result


def main():
    p=json.loads(PROTOCOL.read_text());ph=sha256(PROTOCOL)
    seq,rows=rows_from(OUT/'diagnostic_sequences.jsonl');stress_seq,stress_rows=rows_from(OUT/'acceleration_stress.jsonl')
    assert len(seq)==480 and len(stress_seq)==400
    assert all(s['protocol_sha256']==ph for s in seq+stress_seq)
    summary=summarize(rows);delay=detection(seq);checks={}
    c=p['pass_criteria'];mad=summary['motion_MAD']
    checks['coverage']=all(mad[motion][sigma]['n']>=100 for motion in ('stationary','moving') for sigma in ('0','10'))
    for sigma in ('0','10'):
        r=mad['moving'][sigma]
        checks[f'moving_MAE_{sigma}mm']=r['MAE_mm']<=c['moving_MAE_mm_max_each_noise']
        checks[f'moving_classification_{sigma}mm']=r['classification_accuracy']>=c['moving_classification_accuracy_min_each_noise']
        checks[f'moving_wrong_interval_{sigma}mm']=r['wrong_interval_fraction']<=c['moving_false_selected_interval_fraction_max_each_noise']
        checks[f'moving_wrong_switch_{sigma}mm']=r['wrong_direction_actual_switches_per_decision']<=c['moving_wrong_direction_actual_switch_rate_per_decision_max_each_noise']
        checks[f'stationary_MAE_{sigma}mm']=mad['stationary'][sigma]['MAE_mm']<=c['stationary_MAE_mm_max_each_noise']
    pos=[r for r in rows if r['estimator']=='position' and r['motion']=='moving' and not r['mixed']]
    robust=[r for r in rows if r['estimator']=='motion_MAD' and r['motion']=='moving' and not r['mixed']]
    ratio=1-np.mean([abs(r['error']) for r in robust])/np.mean([abs(r['error']) for r in pos])
    checks['relative_moving_MAE_reduction']=ratio>=c['moving_MAE_relative_reduction_vs_position_min']
    for profile,d in delay['motion_MAD'].items():
        checks[f'detection_p90_{profile}']=d['p90_delay_steps'] is not None and d['p90_delay_steps']<=c['dynamic_switch_detection_p90_steps_max_each_direction']
        checks[f'detection_misses_{profile}']=d['miss_fraction']<=c['dynamic_switch_miss_fraction_max_each_direction']
    # Resample the20 independent reset/noise seeds jointly across source, models,
    # profiles and estimators. Conditional trajectory study, not policy-population CI.
    rng=np.random.default_rng(p['bootstrap_seed']);bootstrap={}
    for sigma in ('0','10'):
        diffs=[]
        for seed in p['seeds']:
            pp=[abs(r['error'])*1000 for r in pos if r['seed']==seed and np.isclose(r['target_sigma'],int(sigma)/1000)]
            mm=[abs(r['error'])*1000 for r in robust if r['seed']==seed and np.isclose(r['target_sigma'],int(sigma)/1000)]
            diffs.append(np.mean(mm)-np.mean(pp))
        values=np.asarray(diffs);ix=rng.integers(0,20,size=(p['bootstrap_replicates'],20));ci=np.quantile(values[ix].mean(axis=1),[.025,.975])
        bootstrap[sigma]={'mean_seed_equal_MAE_difference_mm':float(values.mean()),'paired_seed_bootstrap_95pct_CI_mm':ci.tolist(),'unit':'20 paired reset/noise seeds; conditional on fixed policies and sampled trajectories'}
    checks={k:bool(v) for k,v in checks.items()}
    by_model={str(m):summarize([r for r in rows if r['model_seed']==m]) for m in p['models']}
    stress={name:summarize([r for r in stress_rows if r['synthetic_motion']==name]) for name in ('static','velocity','acceleration','oscillation','impact')}
    mixed={e:{profile:{'n':len(rr),'MAE_vs_window_RMS_mm':float(np.mean([abs(r['error'])*1000 for r in rr]))} for profile in ('clean_to_noisy','noisy_to_clean') if (rr:=[r for r in rows if r['estimator']==e and r['profile']==profile and r['mixed']])} for e in ESTIMATORS}
    result={'stage1_passed':all(checks.values()),'checks':checks,'protocol_sha256':ph,'actual_trajectories':120,'replayed_sequences':480,'windows':len(rows),
        'metrics':summary,'moving_MAE_relative_reduction':float(ratio),'detection':delay,'by_model':by_model,'synthetic_stress':stress,
        'mixed_window_errors':mixed,'paired_seed_bootstrap':bootstrap,
        'stage2_authorized_by_gate':all(checks.values()),'new_grasp_success_evaluation':False,
        'limitations':'Offline clean-policy trajectory replay; candidate estimates did not control motion. Correlated windows and shared seeds are not independent training policies. Acceleration/noise are not universally identifiable.'}
    with (OUT/'stage1_summary.json').open('x') as f:f.write(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    with (OUT/'window_metrics.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    figures(seq,rows,stress_rows,summary,delay)
    print(json.dumps({'passed':result['stage1_passed'],'checks':checks,'metrics':summary,'relative_MAE_reduction':ratio,'detection':{e:{q:{k:v for k,v in d.items() if k!='delays_steps'} for q,d in dd.items()} for e,dd in delay.items()}},indent=2))


def figures(sequences,rows,stress_rows,summary,delay):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'position':'#CC6677','motion_MAD':'#4477AA'}
    def save(fig,name):
        fig.suptitle('Stage 1: estimator validation only; no grasp-success evaluation',fontsize=10);fig.tight_layout(rect=(0,0,1,.94))
        for ext in ('png','svg'):
            path=OUT/f'{name}.{ext}';fig.savefig(path,bbox_inches='tight')
            if ext=='svg':path.write_text('\n'.join(x.rstrip() for x in path.read_text().splitlines())+'\n')
        plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    for ax,motion in zip(axs,('stationary','moving')):
        for i,e in enumerate(ESTIMATORS):ax.bar(np.arange(2)+(i-.5)*.35,[summary[e][motion][s]['MAE_mm'] for s in ('0','10')],.35,label=e,color=colors[e])
        ax.axhline(2,color='k',ls='--',label='MAE gate');ax.set_xticks([0,1],['0','10']);ax.set_title(motion);ax.set_xlabel('Injected noise (mm)');ax.set_ylabel('Mean absolute error (mm)')
    axs[-1].legend();save(fig,'01_estimator_error')
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for ax,profile in zip(axs,('clean_to_noisy','noisy_to_clean')):
        s=next(s for s in sequences if s['identity']=={'model_seed':42,'motion_source':'R4'} and s['seed']==350000 and s['profile']==profile)
        for e in ESTIMATORS:
            rr=[r for r in s['windows'] if r['estimator']==e];ax.plot([r['step'] for r in rr],[1000*r['sigma_hat'] for r in rr],label=e,color=colors[e])
        ax.plot([r['step'] for r in rr],[1000*r['target_sigma'] for r in rr],'k--',label='Window RMS sigma');ax.axvline(40,color='grey',ls=':');ax.set_xlabel('Window end boundary');ax.set_ylabel('Estimated noise (mm)');ax.set_title(profile)
    axs[-1].legend();save(fig,'02_transition_examples')
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    for ax,key in zip(axs,('classification_accuracy','wrong_interval_fraction')):
        for i,e in enumerate(ESTIMATORS):ax.bar(np.arange(2)+(i-.5)*.35,[100*summary[e]['moving'][s][key] for s in ('0','10')],.35,label=e,color=colors[e])
        ax.set_xticks([0,1],['0','10']);ax.set_title(key.replace('_',' '));ax.set_ylabel('Moving windows (%)');ax.set_xlabel('Injected noise (mm)')
    axs[-1].legend();save(fig,'03_classification_false_switch')
    fig,ax=plt.subplots(figsize=(8,4));names=[];data=[]
    for e in ESTIMATORS:
        for profile,d in delay[e].items():names.append(e+'\n'+profile);data.append(d['delays_steps'])
    ax.boxplot(data,tick_labels=names);ax.axhline(24,color='k',ls='--');ax.set_ylabel('Appropriate actual-switch delay (steps)');save(fig,'04_detection_delay')
    fig,ax=plt.subplots(figsize=(8,4))
    for e in ESTIMATORS:
        rr=[r for r in rows if r['estimator']==e and not r['mixed'] and r['target_sigma']==0]
        ax.scatter([r['true_xy_diameter']*1000 for r in rr],[r['sigma_hat']*1000 for r in rr],s=5,alpha=.2,label=e,color=colors[e])
    ax.set_xlabel('True XY window diameter (mm; diagnostic only)');ax.set_ylabel('Estimate at zero injected noise (mm)');ax.legend();save(fig,'05_motion_contamination')
    fig,ax=plt.subplots(figsize=(10,4));names=('static','velocity','acceleration','oscillation','impact')
    for i,e in enumerate(ESTIMATORS):
        vals=[np.mean([r['sigma_hat']*1000 for r in stress_rows if r['estimator']==e and r['synthetic_motion']==n and r['profile']=='clean']) for n in names]
        ax.bar(np.arange(5)+(i-.5)*.35,vals,.35,label=e,color=colors[e])
    ax.set_xticks(np.arange(5),names);ax.set_ylabel('Mean estimate with NO sensor noise (mm)');ax.legend();save(fig,'06_acceleration_ambiguity')


if __name__=='__main__':main()

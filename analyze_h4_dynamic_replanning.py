"""Pre-Final estimator diagnostics; never treats smoke outcomes as Final evidence."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from action_chunking import sha256
from evaluate_h4_dynamic_replanning import OUT, MODELS, PROFILES, protocol


def diagnose_smoke():
    p,ph=protocol();audit=json.loads((OUT/'smoke_integrity_audit.json').read_text())
    if not audit['software_integrity_passed'] or audit['protocol_sha256']!=ph:
        raise RuntimeError('Audited smoke required')
    rows=[];windows=[];examples={}
    for cell in audit['cells']:
        path=OUT/cell['file']
        if sha256(path)!=cell['sha256']:raise RuntimeError('Smoke evidence changed')
        with path.open() as f:
            for r in csv.DictReader(f):
                if r['controller']!='Online':continue
                m=int(r['model_seed']);profile=r['profile'];seed=int(r['seed'])
                events=json.loads(r['decisions_json']);ww=json.loads(r['windows_json'])
                rows.append({'model_seed':m,'profile':profile,'seed':seed,'decisions':events,
                             'calls':int(r['bc_inference_calls']),'steps':int(r['steps']),
                             'switch41':int(r['switch_R4_to_R1']),'switch14':int(r['switch_R1_to_R4'])})
                for w in ww:windows.append({'model_seed':m,'profile':profile,'seed':seed,**w})
                if seed==342000:examples[m,profile]=(events,ww)
    groups={}
    for profile in PROFILES:
        ww=[w for w in windows if w['profile']==profile]
        rr=[r for r in rows if r['profile']==profile]
        groups[profile]={'episodes':len(rr),'windows':len(ww),
            'violating_windows':sum(w['gt_xy_diameter_m']>.001 for w in ww),
            'affected_episodes':len({(w['model_seed'],w['seed']) for w in ww if w['gt_xy_diameter_m']>.001}),
            'maximum_xy_displacement_mm':1000*max(w['gt_xy_diameter_m'] for w in ww),
            'R4_to_R1_switches':sum(r['switch41'] for r in rr),'R1_to_R4_switches':sum(r['switch14'] for r in rr),
            'mean_inference_calls':float(np.mean([r['calls'] for r in rr])),
            'window_error_mm':{}}
        for label,select in [('pre_change',lambda w:w['step']<=40),('transition',lambda w:w['mixed_window']),('post_change_only',lambda w:w['post_change_only'])]:
            selected=[w for w in ww if select(w)]
            groups[profile]['window_error_mm'][label]={'n':len(selected),'mean_absolute_error':float(np.mean([abs(w['estimated_sigma_m']-w['injected_rms_sigma_m'])*1000 for w in selected])) if selected else None}
    first=[w['step'] for w in windows if w['gt_xy_diameter_m']>.001]
    summary={'status':'Stopped after complete smoke: preregistered stationarity gate failed',
        'protocol_sha256':ph,'smoke_episodes':960,'calibration_episodes':0,'final_episodes':0,
        'Online_episodes':len(rows),'Online_windows':len(windows),'earliest_violating_window_step':min(first),
        'online_by_profile':groups,'no_success_rates_or_final_statistics_computed':True,
        'interpretation':'Observed XY variance is confounded by real cube motion. Switching is mechanically verified, but cannot be interpreted as valid sensor-noise detection. Smoke diagnostics do not establish control reliability.'}
    with (OUT/'smoke_estimator_diagnostics.json').open('x') as f:f.write(json.dumps(summary,indent=2,sort_keys=True,allow_nan=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    def save(fig,name):
        fig.suptitle('Smoke diagnostics only — Final not run',fontsize=10)
        fig.tight_layout(rect=(0,0,1,.94))
        for ext in ('svg','png'):
            path=OUT/f'{name}.{ext}'
            if path.exists():raise FileExistsError(path)
            fig.savefig(path,bbox_inches='tight')
            if ext=='svg':path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
        plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(13,4),sharey=True)
    for ax,m in zip(axs,MODELS):
        for profile in PROFILES:
            ww=examples[m,profile][1];ax.plot([w['step'] for w in ww],[1000*w['gt_xy_diameter_m'] for w in ww],label=profile)
        ax.axhline(1,color='black',ls='--',label='1 mm gate');ax.set_title(f'BC seed {m}, example 342000');ax.set_xlabel('Decision boundary')
    axs[0].set_ylabel('True cube XY window diameter (mm)');axs[-1].legend(fontsize=7);save(fig,'01_stationarity_windows')
    fig,axs=plt.subplots(1,2,figsize=(11,4),sharey=True)
    for ax,profile in zip(axs,PROFILES[2:]):
        for m in MODELS:
            ww=examples[m,profile][1];ax.plot([w['step'] for w in ww],[1000*w['estimated_sigma_m'] for w in ww],label=f'BC {m}')
        ww=examples[42,profile][1];ax.plot([w['step'] for w in ww],[1000*w['injected_rms_sigma_m'] for w in ww],'k--',label='Window injected RMS σ')
        ax.axvline(40,color='grey',ls=':');ax.set_title(profile);ax.set_xlabel('Decision boundary');ax.legend(fontsize=7)
    axs[0].set_ylabel('Sigma estimate / injected RMS (mm)');save(fig,'02_noise_estimate_examples')
    fig,axs=plt.subplots(1,2,figsize=(11,4),sharey=True)
    for ax,profile in zip(axs,PROFILES[2:]):
        for m in MODELS:
            events=examples[m,profile][0];ax.step([0]+[e['step'] for e in events]+[100],[4]+[e['after'] for e in events]+[events[-1]['after']],where='post',label=f'BC {m}',alpha=.7)
        ax.axvline(40,color='grey',ls=':');ax.axvline(80,color='grey',ls='--');ax.set_yticks([1,4]);ax.set_title(profile);ax.set_xlabel('Control step');ax.legend(fontsize=7)
    axs[0].set_ylabel('Replanning interval');save(fig,'03_interval_examples')
    fig,ax=plt.subplots(figsize=(8,4));labels=list(PROFILES)
    ax.bar(range(4),[groups[q]['affected_episodes'] for q in labels],color='#CC6677');ax.set_xticks(range(4),labels,rotation=15);ax.set_ylabel('Online episodes violating 1 mm gate / 60');save(fig,'04_affected_episodes')
    fig,ax=plt.subplots(figsize=(8,4));x=np.arange(4)
    ax.bar(x-.18,[groups[q]['R4_to_R1_switches'] for q in labels],.36,label='R4 → R1')
    ax.bar(x+.18,[groups[q]['R1_to_R4_switches'] for q in labels],.36,label='R1 → R4')
    ax.set_xticks(x,labels,rotation=15);ax.set_ylabel('Switches (includes initial t20)');ax.legend();save(fig,'05_switch_counts')
    fig,ax=plt.subplots(figsize=(8,4));ax.scatter([1000*w['gt_position_variance_sigma_m'] for w in windows],[1000*w['estimated_sigma_m'] for w in windows],s=5,alpha=.15)
    ax.set_xlabel('True-motion contribution to XY variance (mm)');ax.set_ylabel('Observed sigma estimate (mm)');save(fig,'06_motion_estimate_confounding')
    print(json.dumps(summary,indent=2))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=('smoke','final'),required=True);a=parser.parse_args()
    if a.phase=='final':
        gate=OUT/'calibration_integrity_audit.json'
        if not gate.exists() or not json.loads(gate.read_text()).get('gate_passed'):
            raise RuntimeError('Final analysis prohibited: Calibration gate not passed; no Final outcomes exist')
        raise RuntimeError('Study stopped before Final; no Final analysis is implemented or claimed')
    diagnose_smoke()


if __name__=='__main__':main()

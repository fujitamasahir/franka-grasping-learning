"""Nominal-noise switch diagnostics on smoke only; no success ranking."""
import csv
import json
import statistics
from evaluate_h4_dynamic_replanning import OUT, PROFILES


def main():
    result={}
    for profile in PROFILES:
        rows=[]
        for path in OUT.glob(f'smoke_*_Online_{profile}.csv'):
            with path.open() as stream:rows.extend(csv.DictReader(stream))
        if len(rows)!=60:raise RuntimeError('Complete smoke required')
        delays=[];missing=0;unnecessary=0;no_switch=0
        for row in rows:
            switches=[e for e in json.loads(row['decisions_json']) if e['switched']]
            no_switch+=not bool(switches)
            for event in switches:
                clean=(profile=='clean' or profile=='clean_to_noisy' and event['step']<40 or profile=='noisy_to_clean' and event['step']>=40)
                unnecessary+=event['step']>20 and event['after']!=(4 if clean else 1)
            if profile in PROFILES[2:]:
                target=1 if profile=='clean_to_noisy' else 4
                found=[e['step']-40 for e in switches if e['step']>=40 and e['after']==target]
                if found:delays.append(found[0])
                else:missing+=1
        result[profile]={'episodes':len(rows),'appropriate_post_change_switch_delay_steps':{
            'n':len(delays),'median':statistics.median(delays) if delays else None,
            'min':min(delays) if delays else None,'max':max(delays) if delays else None},
            'missing_appropriate_post_change_switch':missing if profile in PROFILES[2:] else None,
            'unnecessary_switches_after_initial_decision':unnecessary,'episodes_without_any_switch':no_switch,
            'interpretation':'Smoke-only nominal-noise labeling. Stationarity gate failed; not validated detection or causal outcome evidence.'}
    with (OUT/'smoke_switch_diagnostics.json').open('x') as stream:
        stream.write(json.dumps(result,indent=2,sort_keys=True)+'\n')


if __name__=='__main__':main()

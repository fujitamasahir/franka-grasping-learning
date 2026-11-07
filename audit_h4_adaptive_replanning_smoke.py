"""Pre-calibration smoke completeness, stationarity, and pairing audit."""
import csv,json,statistics
from pathlib import Path
from action_chunking import sha256
from evaluate_h4_adaptive_replanning import OUT,FIELDS

def main():
    protocol=json.loads(Path('bc_results/h4_adaptive_replanning_protocol.json').read_text())
    expected=list(range(333000,333020)); positions={}; noise={}; applied={}; cells=[]; displacement=[]; estimates={}; rows_by={}
    for m in (42,123,456):
      for ctrl in ('R1','R4','Adaptive','WarmupR1'):
       for mm in (0,5,10):
        p=OUT/f'smoke2_H4_seed{m}_{ctrl}_noise{mm}mm.csv'
        with p.open(newline='') as f:
         rd=csv.DictReader(f)
         if rd.fieldnames!=FIELDS:raise ValueError(f'Schema mismatch: {p}')
         rows=list(rd)
        if [int(x['seed']) for x in rows]!=expected:raise ValueError(f'Seed set mismatch: {p}')
        rows_by[m,ctrl,mm]={int(x['seed']):x for x in rows}
        for x in rows:
         s=int(x['seed']);pos=(x['cube_x'],x['cube_y']); schedule=x['base_noise_sha256']
         if s in positions and positions[s]!=pos:raise ValueError(f'Position mismatch at {s}')
         positions[s]=pos
         if s in noise and noise[s]!=schedule:raise ValueError(f'Base noise mismatch at {s}')
         noise[s]=schedule
         ak=(s,mm)
         if ak in applied and applied[ak]!=x['applied_noise_sha256']:raise ValueError(f'Applied noise mismatch at {s}/{mm}')
         applied[ak]=x['applied_noise_sha256']
         d=float(x['stationarity_gt_max_displacement_m']);displacement.append(d)
         if ctrl in ('Adaptive','WarmupR1'):
          if int(x['steps'])>=20 and (int(x['decision_made'])!=1 or int(x['steps_before_decision'])!=20):raise ValueError(f'Bad switch timing {p} seed {s}')
          if ctrl=='Adaptive' and x['estimated_sigma_m']!='':estimates.setdefault(mm,[]).append(float(x['estimated_sigma_m']))
        cells.append({'model_seed':m,'controller':ctrl,'noise_mm':mm,'episodes':len(rows),'csv_sha256':sha256(p)})
    # Verify adaptive outcomes are exactly the appropriate complete branch trajectory.
    branch_fields=('success','steps','return','cube_x','cube_y','region','first_close_step','first_lift_step','closed_fraction','bc_inference_calls','terminal_chunk_index','base_noise_sha256','applied_noise_sha256')
    branch_checks=0
    for m in (42,123,456):
      for mm in (0,5,10):
       for s,a in rows_by[m,'Adaptive',mm].items():
        chosen=int(a['chosen_interval']) if a['chosen_interval'] else 4
        branch=rows_by[m,'R4' if chosen==4 else 'WarmupR1',mm][s]
        if any(a[k]!=branch[k] for k in branch_fields):raise ValueError(f'Adaptive branch mismatch: model {m}, noise {mm}, seed {s}')
        w=rows_by[m,'WarmupR1',mm][s]
        if a['estimated_sigma_m']!=w['estimated_sigma_m']:raise ValueError(f'Estimator prefix mismatch at {m}/{mm}/{s}')
        branch_checks+=1
    maxd=max(displacement)
    if maxd>protocol['estimator']['stationarity_audit_acceptance_threshold_m']:
      raise RuntimeError(f'Cube motion during first 20 steps exceeded stationarity criterion: {maxd:.6g}m')
    result={'smoke_passed':True,'cells':len(cells),'episodes':sum(x['episodes'] for x in cells),'seed_range_inclusive':expected,
      'positions_paired_across_all_conditions':len(positions)==20,'base_noise_paired_across_all_conditions':len(noise)==20,'applied_noise_paired_by_sigma':len(applied)==60,
      'stationarity_criterion_m':protocol['estimator']['stationarity_audit_acceptance_threshold_m'],
      'max_ground_truth_cube_displacement_first_20_steps_m':maxd,
      'adaptive_branch_equivalence_checks':branch_checks,'adaptive_matches_fixed_R4_or_warmup_R1_branch_exactly':True,
      'estimated_sigma_summary_m':{str(mm):{'n':len(v),'median':statistics.median(v),'min':min(v),'max':max(v)} for mm,v in estimates.items()},
      'adaptive_threshold_used_for_smoke_m':0.0075,'smoke_outcomes_not_used_for_threshold_selection':True,'configurations':cells}
    out=OUT/'smoke2_integrity_audit.json'
    if out.exists():raise FileExistsError(out)
    out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(f"Smoke audit passed: {result['episodes']} episodes; max initial cube movement {maxd:.3e}m")
if __name__=='__main__':main()

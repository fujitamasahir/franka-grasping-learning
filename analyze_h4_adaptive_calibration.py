"""Select one global adaptive threshold from complete Calibration branch rollouts."""
import csv,json
from pathlib import Path
import numpy as np
from action_chunking import sha256
from evaluate_h4_adaptive_replanning import OUT,FIELDS,validate_rows

MODELS=(42,123,456); NOISE=(0,5,10); CANDIDATES=(.005,.0075,.01); SEEDS=range(330000,330300)

def read(m,controller,mm):
 p=OUT/f'calibration_H4_seed{m}_{controller}_noise{mm}mm.csv'
 if (OUT/f'calibration_H4_seed{m}_{controller}_noise{mm}mm.partial.csv').exists():
  raise RuntimeError(f'Partial file exists for an apparently complete calibration cell: {p}')
 meta=json.loads((OUT/f'calibration_H4_seed{m}_{controller}_noise{mm}mm.json').read_text())
 proto=json.loads(Path('bc_results/h4_adaptive_replanning_protocol.json').read_text());ph=Path('bc_results/h4_adaptive_replanning_protocol.sha256').read_text().split()[0]
 ck=Path(f'bc_models/action_chunking/H4_seed{m}.pt')
 if sha256(ck)!=proto['frozen_models'][str(m)]['sha256'] or meta['checkpoint_sha256']!=sha256(ck):raise ValueError(f'Checkpoint provenance mismatch: {p}')
 if meta['protocol_sha256']!=ph or meta['phase']!='calibration' or meta['threshold_m'] is not None:raise ValueError(f'Calibration metadata mismatch: {p}')
 if meta['model_seed']!=m or meta['controller']!=controller or meta['noise_sigma_m']!=mm/1000 or meta['seed_range_inclusive']!=[330000,330299]:raise ValueError(f'Calibration cell mismatch: {p}')
 with p.open(newline='') as f:
  rd=csv.DictReader(f)
  if rd.fieldnames!=FIELDS:raise ValueError(f'Schema mismatch: {p}')
  rows=list(rd)
 if [int(x['seed']) for x in rows]!=list(SEEDS):raise ValueError(f'Incomplete or unordered calibration cell: {p}')
 validate_rows(rows,SEEDS,m,None,controller,None,mm/1000,sha256(ck),ph)
 return {int(x['seed']):x for x in rows}

def main():
 proto=json.loads(Path('bc_results/h4_adaptive_replanning_protocol.json').read_text());ph=Path('bc_results/h4_adaptive_replanning_protocol.sha256').read_text().split()[0]
 if sha256(Path('bc_results/h4_adaptive_replanning_protocol.json'))!=ph:raise RuntimeError('Protocol SHA mismatch')
 if tuple(proto['candidate_thresholds_m'])!=CANDIDATES or proto['calibration']['seed_range_inclusive']!=[330000,330299]:raise RuntimeError('Analysis constants differ from preregistration')
 data={(m,c,n):read(m,c,n) for m in MODELS for c in ('R1','R4','WarmupR1') for n in NOISE}
 base_hashes={};applied_hashes={}
 for (m,c,n),rows in data.items():
  for seed,row in rows.items():
   if seed in base_hashes and base_hashes[seed]!=row['base_noise_sha256']:raise ValueError(f'Base noise not paired for seed {seed}')
   base_hashes[seed]=row['base_noise_sha256'];key=(seed,n)
   if key in applied_hashes and applied_hashes[key]!=row['applied_noise_sha256']:raise ValueError(f'Applied noise not paired for {key}')
   applied_hashes[key]=row['applied_noise_sha256']
 if len(base_hashes)!=300 or len(applied_hashes)!=900:raise ValueError('Calibration pairing audit cardinality mismatch')
 # The variance estimate comes from the separately executed R4-prefix/WarmupR1 branch.
 # Its R4 prefix is deterministic and common to Fixed R4, as verified by smoke branch tests.
 candidates=[];candidate_cells=[];fixed_cells=[]
 for threshold in CANDIDATES:
  rates=[];calls=[];decisions={str(n):{'R1':0,'R4':0,'n':0} for n in NOISE};estimates={str(n):[] for n in NOISE}
  for m in MODELS:
   for n in NOISE:
    successes=[];cellcalls=[]
    for s in SEEDS:
     w=data[m,'WarmupR1',n][s];r4=data[m,'R4',n][s]
     if w['estimated_sigma_m']=='':
      chosen=4;branch=r4 # terminated in the shared R4 prefix before any decision
     else:
      est=float(w['estimated_sigma_m']);chosen=4 if est<=threshold else 1;branch=r4 if chosen==4 else w
     successes.append(int(branch['success']));cellcalls.append(int(branch['bc_inference_calls']))
     if w['estimated_sigma_m']!='':decisions[str(n)]['R'+str(chosen)]+=1;decisions[str(n)]['n']+=1;estimates[str(n)].append(est)
    rate=float(np.mean(successes));mean_calls=float(np.mean(cellcalls));rates.append(rate);calls.append(mean_calls)
    candidate_cells.append({'threshold_m':threshold,'model_seed':m,'noise_mm':n,'episodes':len(successes),'successes':sum(successes),'success_rate_pct':100*rate,'mean_inference_calls':mean_calls})
  candidates.append({'threshold_m':threshold,'equally_weighted_success_rate_pct':100*float(np.mean(rates)),'equally_weighted_mean_inference_calls':float(np.mean(calls)),'selection_frequency_by_noise':decisions})
 # Fixed baselines are equally weighted over the same 9 cells; ties use fewer calls then stable R1 order.
 fixed={}
 for c in ('R1','R4'):
  rates=[];calls=[]
  for m in MODELS:
   for n in NOISE:
    rows=data[m,c,n].values();rr=list(rows);rates.append(np.mean([int(x['success']) for x in rr]));calls.append(np.mean([int(x['bc_inference_calls']) for x in rr]))
    fixed_cells.append({'model_seed':m,'controller':c,'noise_mm':n,'episodes':len(rr),'successes':sum(int(x['success']) for x in rr),'success_rate_pct':100*float(np.mean([int(x['success']) for x in rr])),'mean_inference_calls':float(np.mean([int(x['bc_inference_calls']) for x in rr]))})
  fixed[c]={'equally_weighted_success_rate_pct':100*float(np.mean(rates)),'equally_weighted_mean_inference_calls':float(np.mean(calls))}
 best_fixed=max(('R1','R4'),key=lambda c:(fixed[c]['equally_weighted_success_rate_pct'],-fixed[c]['equally_weighted_mean_inference_calls'],c=='R1'))
 selected=sorted(candidates,key=lambda x:(-x['equally_weighted_success_rate_pct'],x['equally_weighted_mean_inference_calls'],x['threshold_m']))[0]
 # Diagnostic only: nearest nominal sigma bucket, never used by the controller.
 est_summary={}; confusion={}
 for n in NOISE:
  vals=[float(x['estimated_sigma_m']) for m in MODELS for x in data[m,'WarmupR1',n].values()]
  assigned=[min(NOISE,key=lambda q:abs(v-q/1000)) for v in vals]
  est_summary[str(n)]={'n':len(vals),'mean_m':float(np.mean(vals)),'median_m':float(np.median(vals)),'sd_m':float(np.std(vals,ddof=1)),'min_m':min(vals),'max_m':max(vals),'nearest_nominal_bucket_accuracy_pct':100*float(np.mean(np.array(assigned)==n))}
  confusion[str(n)]={str(q):assigned.count(q) for q in NOISE}
 frozen={'selected_threshold_m':selected['threshold_m'],'selection_rule':'highest equally weighted success; then fewer mean BC inference calls; then smaller threshold','calibration_seed_range_inclusive':[330000,330299],'protocol_sha256':ph,'selected_calibration_summary':selected,'fixed_baselines':fixed,'calibration_selected_fixed_baseline':best_fixed,'candidate_thresholds':candidates,'estimate_diagnostics_by_true_noise_mm':est_summary,'nearest_sigma_confusion_counts':confusion,'adaptive_outcomes_from_complete_branches':True,'fixed_R1_labels_not_used_for_adaptive_R1_branch':True}
 frozen['complete_cell_count']=27;frozen['executed_calibration_episodes']=8100
 for name,obj in [('calibration_summary.json',frozen),('frozen_threshold.json',frozen)]:
  p=OUT/name
  if p.exists():raise FileExistsError(p)
  p.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')
 p=OUT/'calibration_candidate_cells.csv'
 if p.exists():raise FileExistsError(p)
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(candidate_cells[0]));w.writeheader();w.writerows(candidate_cells)
 p=OUT/'calibration_fixed_cell_summary.csv'
 if p.exists():raise FileExistsError(p)
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(fixed_cells[0]));w.writeheader();w.writerows(fixed_cells)
 print(f"Frozen global threshold {selected['threshold_m']*1000:g} mm; selected Calibration success {selected['equally_weighted_success_rate_pct']:.3f}% ; best fixed {best_fixed}")
if __name__=='__main__':main()

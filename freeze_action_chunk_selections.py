"""Select SAC milestones on the preregistered validation positions and freeze hashes."""
import csv,json,hashlib
from pathlib import Path
from action_chunking import sha256,verify_protocol

ROOT=Path('bc_results/action_chunking_sac');OUT=Path('bc_results/action_chunking_checkpoint_selections.json')
def main():
    protocol,_=verify_protocol(); selections=[]
    for arch in ('H1','H4'):
      for bc in (42,123,456):
       for sac in (1103,1117,1123):
        run=ROOT/f'{arch}_bc{bc}_sac{sac}';path=run/'validation.csv'
        if not path.exists():raise FileNotFoundError(f'Missing validation milestones: {path}')
        with path.open(newline='') as f:rows=list(csv.DictReader(f))
        by={int(r['timesteps']):r for r in rows}
        if set(by)!={10000,20000,30000}:raise RuntimeError(f'Expected all 3 validation checkpoints in {run}')
        # Highest success, then fewest paired regressions, then shortest successful
        # episode, then earliest checkpoint. Sort ascending on the stated criteria.
        chosen=min(rows,key=lambda r:(-float(r['success_rate']),int(r['regression']),
            float(r['mean_success_steps']) if r['mean_success_steps'] else float('inf'),int(r['timesteps'])))
        step=int(chosen['timesteps']);model=run/f'checkpoint_{step}.zip'
        if not model.is_file():raise FileNotFoundError(model)
        checkpoint_hashes={str(n):sha256(run/f'checkpoint_{n}.zip') for n in (10000,20000,30000)}
        selections.append({'architecture':arch,'bc_seed':bc,'sac_seed':sac,'selected_step':step,
            'selection_successes':int(chosen['successes']),'selection_episodes':int(chosen['episodes']),
            'selection_success_rate':float(chosen['success_rate']),'selection_rescues':int(chosen['rescue']),
            'selection_regressions':int(chosen['regression']),'selection_mean_success_steps':chosen['mean_success_steps'],
            'model_path':str(model),'model_sha256':sha256(model),'validation_csv_sha256':sha256(run/'validation.csv'),
            'checkpoint_hashes':checkpoint_hashes,'bc_model_sha256':sha256(f'bc_models/action_chunking/{arch}_seed{bc}.pt')})
    result={'protocol_sha256':sha256('bc_results/action_chunking_protocol.json'),
      'protocol_selection_rule':protocol['evaluation']['checkpoint_rule'],
      'frozen_before_final_evaluation':True,'selections':selections}
    data=json.dumps(result,indent=2,sort_keys=True)+'\n'
    if OUT.exists():
        if OUT.read_text()!=data:raise FileExistsError(f'Frozen selection already exists and differs: {OUT}')
        return
    OUT.write_text(data,encoding='utf-8');digest=hashlib.sha256(OUT.read_bytes()).hexdigest()
    OUT.with_suffix('.sha256').write_text(f'{digest}  {OUT.name}\n',encoding='utf-8')
    print(f'Frozen 18 selections; SHA256 {digest}',flush=True)
if __name__=='__main__':main()

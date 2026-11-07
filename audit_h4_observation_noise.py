"""Strict completeness, pairing, and frozen-checkpoint audit before analysis."""
import csv
import json
from pathlib import Path

from action_chunking import sha256
from evaluate_h4_observation_noise import FIELDS,NOISE_SEED,OUT,PROTOCOL,PROTOCOL_SHA,validate_rows


def main():
    protocol=json.loads(PROTOCOL.read_text());protocol_hash=PROTOCOL_SHA.read_text().split()[0]
    if sha256(PROTOCOL)!=protocol_hash:raise RuntimeError('Protocol SHA mismatch')
    expected=list(range(321000,322000));position_by_seed={};base_hash_by_seed={};applied_hash_by_seed_sigma={}
    cells=[]
    for model_seed in (42,123,456):
        model_path=Path(f'bc_models/action_chunking/H4_seed{model_seed}.pt')
        checkpoint_hash=sha256(model_path)
        if checkpoint_hash!=protocol['frozen_models'][str(model_seed)]['sha256']:
            raise RuntimeError(f'Frozen model changed: {model_path}')
        for interval in (1,4):
            for sigma_mm in (0,5,10):
                tag=f'final_H4_seed{model_seed}_R{interval}_noise{sigma_mm}mm'
                path=OUT/f'{tag}.csv';meta_path=OUT/f'{tag}.json';partial=OUT/f'{tag}.partial.csv'
                if partial.exists():raise FileExistsError(f'Partial output remains; preserving and stopping: {partial}')
                with path.open(newline='',encoding='utf-8') as f:
                    reader=csv.DictReader(f)
                    if reader.fieldnames!=FIELDS:raise ValueError(f'Schema mismatch: {path}')
                    rows=list(reader)
                sigma=sigma_mm/1000
                validate_rows(rows,expected,model_seed,interval,sigma,checkpoint_hash,protocol_hash)
                if len(rows)!=1000:raise ValueError(f'Expected exactly 1000 rows: {path}')
                metadata=json.loads(meta_path.read_text())
                if metadata['checkpoint_sha256']!=checkpoint_hash or metadata['protocol_sha256']!=protocol_hash:
                    raise ValueError(f'Metadata provenance mismatch: {meta_path}')
                if metadata['noise_generation_seed']!=NOISE_SEED or metadata['seed_range_inclusive']!=[321000,321999]:
                    raise ValueError(f'Metadata noise/seed mismatch: {meta_path}')
                for r in rows:
                    seed=int(r['seed']);pos=(r['cube_x'],r['cube_y'])
                    if seed in position_by_seed and position_by_seed[seed]!=pos:
                        raise ValueError(f'Initial positions differ for evaluation seed {seed}')
                    position_by_seed[seed]=pos
                    if seed in base_hash_by_seed and base_hash_by_seed[seed]!=r['base_noise_sha256']:
                        raise ValueError(f'Base noise schedule differs across cells at seed {seed}')
                    base_hash_by_seed[seed]=r['base_noise_sha256']
                    k=(seed,sigma_mm)
                    if k in applied_hash_by_seed_sigma and applied_hash_by_seed_sigma[k]!=r['applied_noise_sha256']:
                        raise ValueError(f'Applied noise differs across model/interval for seed {seed}, sigma {sigma_mm}mm')
                    applied_hash_by_seed_sigma[k]=r['applied_noise_sha256']
                cells.append({'model_seed':model_seed,'interval':interval,'noise_sigma_m':sigma,'rows':len(rows),
                    'unique_seeds':len({int(r['seed']) for r in rows}),'csv_path':str(path),'csv_sha256':sha256(path),
                    'metadata_path':str(meta_path),'metadata_sha256':sha256(meta_path),'checkpoint_sha256':checkpoint_hash})
    if len(position_by_seed)!=1000 or len(base_hash_by_seed)!=1000 or len(applied_hash_by_seed_sigma)!=3000:
        raise AssertionError('Pairing audit cardinality mismatch')
    audit={'final_evaluation_complete':True,'configuration_count':len(cells),'total_final_episodes':sum(x['rows'] for x in cells),
      'seed_range_inclusive':[321000,321999],'all_cells_exact_unique_seed_set':True,'same_initial_positions_across_18_cells':True,
      'same_base_noise_schedule_across_all_cells_by_episode_seed':True,
      'same_applied_noise_between_R1_R4_and_models_by_episode_seed_sigma':True,
      'fixed_noise_generation_seed':NOISE_SEED,'protocol_sha256':protocol_hash,
      'all_frozen_model_hashes_match_protocol_and_are_unchanged':True,'configurations':cells}
    out=OUT/'final_integrity_audit.json'
    if out.exists():raise FileExistsError(f'Refusing to overwrite audit: {out}')
    out.write_text(json.dumps(audit,indent=2,sort_keys=True)+'\n')
    print(f'Integrity audit passed: {audit["total_final_episodes"]} episodes, {len(cells)} cells, paired positions/noise/model hashes OK')


if __name__=='__main__':main()

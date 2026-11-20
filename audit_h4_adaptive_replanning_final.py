"""Validate paired Final cells; incomplete mode reports counts without outcomes."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from action_chunking import sha256
from evaluate_h4_adaptive_replanning import OUT, FIELDS, NOISE_SEED, validate_rows
from franka_autonomous_env import FrankaAutonomousEnv


def initial_position_map(seeds):
    env = FrankaAutonomousEnv(position_range=.20)
    result = {}
    for seed in seeds:
        env.reset(seed=int(seed))
        xy = env.data.xpos[env.cube_id][:2].copy()
        result[int(seed)] = (float(xy[0]), float(xy[1]))
    env.close()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--allow-incomplete', action='store_true')
    args = parser.parse_args()
    protocol_path = Path('bc_results/h4_adaptive_replanning_protocol.json')
    protocol = json.loads(protocol_path.read_text())
    if protocol['noise']['generation_seed']!=NOISE_SEED:
        raise RuntimeError('Evaluator noise seed differs from preregistration')
    protocol_hash = Path('bc_results/h4_adaptive_replanning_protocol.sha256').read_text().split()[0]
    if sha256(protocol_path) != protocol_hash:
        raise RuntimeError('Protocol hash mismatch')
    expected = list(range(331000, 332000))
    initial = initial_position_map(expected)
    base, applied, cells, complete, partial_files = {}, {}, [], 0, []
    threshold_path = OUT / 'frozen_threshold.json'
    threshold = json.loads(threshold_path.read_text())['selected_threshold_m'] if threshold_path.exists() else None
    for model_seed in (42, 123, 456):
        checkpoint = Path(f'bc_models/action_chunking/H4_seed{model_seed}.pt')
        checkpoint_hash = sha256(checkpoint)
        if checkpoint_hash != protocol['frozen_models'][str(model_seed)]['sha256']:
            raise RuntimeError(f'Checkpoint mismatch: {checkpoint}')
        for controller in ('R1', 'R4', 'Adaptive'):
            for noise_mm in (0, 5, 10):
                tag = f'final_H4_seed{model_seed}_{controller}_noise{noise_mm}mm'
                final = OUT / f'{tag}.csv'
                partial = OUT / f'{tag}.partial.csv'
                metadata_path = OUT / f'{tag}.json'
                if final.exists() and partial.exists():
                    raise ValueError(f'Both final and partial exist: {tag}')
                path = final if final.exists() else partial if partial.exists() else None
                if path is None:
                    cells.append({'model_seed': model_seed, 'controller': controller, 'noise_mm': noise_mm, 'rows': 0, 'complete': False})
                    continue
                if not metadata_path.exists():
                    raise FileNotFoundError(f'Metadata missing: {metadata_path}')
                metadata = json.loads(metadata_path.read_text())
                cell_threshold = threshold if controller == 'Adaptive' else None
                if (metadata['checkpoint_sha256'] != checkpoint_hash or metadata['protocol_sha256'] != protocol_hash
                    or metadata['model_seed'] != model_seed or metadata['controller'] != controller
                    or metadata['phase']!='final' or metadata['seed_range_inclusive']!=[331000,331999]
                    or metadata['noise_generation_seed']!=NOISE_SEED or metadata['noise_sigma_m'] != noise_mm / 1000
                    or metadata['threshold_m'] != cell_threshold):
                    raise ValueError(f'Metadata mismatch: {metadata_path}')
                with path.open(newline='') as stream:
                    reader = csv.DictReader(stream)
                    if reader.fieldnames != FIELDS:
                        raise ValueError(f'Schema mismatch: {path}')
                    rows = list(reader)
                seeds = validate_rows(rows, expected, model_seed, None, controller, cell_threshold,
                                      noise_mm / 1000, checkpoint_hash, protocol_hash)
                is_complete = final.exists()
                if is_complete and (len(rows) != 1000 or seeds != set(expected)):
                    raise ValueError(f'Final seed set invalid: {final}')
                if not is_complete and len(rows) >= 1000:
                    raise ValueError(f'Full partial should be atomically finalized: {partial}')
                if is_complete:
                    complete += 1
                else:
                    partial_files.append(str(partial))
                for row in rows:
                    seed = int(row['seed'])
                    x, y = initial[seed]
                    expected_region = ('E' if x >= .55 else 'W') + ('N' if y >= 0 else 'S')
                    if row['region'] != expected_region:
                        raise ValueError(f'Initial-position region mismatch at seed {seed}: {row["region"]} != {expected_region}')
                    base_hash = row['base_noise_sha256']
                    if seed in base and base[seed] != base_hash:
                        raise ValueError(f'Base noise mismatch at seed {seed}')
                    base[seed] = base_hash
                    noise_key = (seed, noise_mm)
                    applied_hash = row['applied_noise_sha256']
                    if noise_key in applied and applied[noise_key] != applied_hash:
                        raise ValueError(f'Applied noise mismatch at {noise_key}')
                    applied[noise_key] = applied_hash
                cells.append({'model_seed': model_seed, 'controller': controller, 'noise_mm': noise_mm,
                              'rows': len(rows), 'complete': is_complete, 'csv_sha256': sha256(path)})
    passed = complete == 27 and all(cell['rows'] == 1000 for cell in cells)
    if passed and (len(base) != 1000 or len(applied) != 3000):
        raise ValueError('Final pairing audit cardinality mismatch')
    if not passed and not args.allow_incomplete:
        raise RuntimeError(f'Final incomplete: {complete}/27 cells complete; use --allow-incomplete to record progress')
    position_payload = json.dumps([[seed, *initial[seed]] for seed in expected], separators=(',', ':')).encode()
    result = {
        'final_evaluation_complete': passed, 'complete_cells': complete, 'configuration_count': 27,
        'total_rows_written': sum(cell['rows'] for cell in cells), 'expected_total_final_episodes': 27000,
        'seed_range_inclusive': [331000, 331999], 'initial_positions_reconstructed_from_seeded_reset': True,
        'initial_position_map_sha256': hashlib.sha256(position_payload).hexdigest(),
        'base_noise_shared_by_seed': True, 'applied_noise_shared_by_seed_sigma': True,
        'protocol_sha256': protocol_hash, 'incomplete_files': partial_files, 'cells': cells,
    }
    output = OUT / ('final_integrity_audit.json' if passed else 'final_progress_audit.json')
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(f"Final progress: {complete}/27 cells complete; {result['total_rows_written']} rows written; outcomes not summarized")
    if not passed:
        print(f'Progress record: {output}')


if __name__ == '__main__':
    main()

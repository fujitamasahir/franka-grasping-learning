"""Check sigma-zero behavior against the preceding H4 R1/R4 Final records."""
import csv
import json
import math
from pathlib import Path

from action_chunking import load_bc_policy,sha256
from evaluate_h4_observation_noise import rollout


def rows(path):
    with Path(path).open(newline='',encoding='utf-8') as f:return {int(r['seed']):r for r in csv.DictReader(f)}


def compare_row(actual,expected,seed,model_seed,interval):
    checks={'success':actual['success']==int(expected['success']),
      'steps':actual['steps']==int(expected['steps']),
      'cube_x':math.isclose(actual['cube_x'],float(expected['cube_x']),abs_tol=1e-12),
      'cube_y':math.isclose(actual['cube_y'],float(expected['cube_y']),abs_tol=1e-12),
      'first_close_step':str(actual['first_close_step'])==str(expected['first_close_step']),
      'first_lift_step':str(actual['first_lift_step'])==str(expected['first_lift_step']),
      'closed_fraction':math.isclose(actual['closed_fraction'],float(expected['closed_fraction']),abs_tol=1e-12),
      'inference_calls':actual['bc_inference_calls']==int(expected['bc_inference_calls']),
      'terminal_chunk_index':actual['terminal_chunk_index']==int(expected['terminal_chunk_index']),
      'return':math.isclose(actual['return'],float(expected['return']),abs_tol=1e-8)}
    if not all(checks.values()):raise AssertionError(f'Zero-noise regression mismatch model={model_seed} R{interval} seed={seed}: {checks}')
    if actual['bc_inference_calls']!=(actual['steps']+interval-1)//interval:
        raise AssertionError(f'Unexpected inference schedule for seed {seed}, R{interval}')
    return checks


def main():
    all_checks=[];noise_hashes={}
    for model_seed in (42,123,456):
        r1_path=f'bc_results/h4_replanning/final_H4_seed{model_seed}_R1.csv'
        if model_seed==456:r1_path='bc_results/h4_replanning/final_H4_seed456_R1_recovery1.csv'
        baselines={1:rows(r1_path),4:rows(f'bc_results/h4_replanning/final_H4_seed{model_seed}_R4.csv')}
        model_path=f'bc_models/action_chunking/H4_seed{model_seed}.pt'
        model,ck=load_bc_policy(model_path);digest=sha256(model_path)
        for interval in (1,4):
            for seed in range(310000,310020):
                actual,_=rollout(model,ck,model_seed,interval,0.,seed,digest,'0'*64)
                checks=compare_row(actual,baselines[interval][seed],seed,model_seed,interval)
                all_checks.append({'model_seed':model_seed,'interval':interval,'seed':seed,'checks':checks})
                key=(model_seed,seed)
                pair=(actual['base_noise_sha256'],actual['applied_noise_sha256'])
                if key in noise_hashes:assert noise_hashes[key]==pair
                noise_hashes[key]=pair
    # The preserved original H4 R4 action trace supplies a stepwise action check.
    model,ck=load_bc_policy('bc_models/action_chunking/H4_seed42.pt')
    actual,trace=rollout(model,ck,42,4,0.,290042,sha256('bc_models/action_chunking/H4_seed42.pt'),'0'*64,trace=True)
    original=json.loads(Path('bc_results/action_chunking_eval/trajectory_bc_final_H4_seed42_BC.json').read_text())['steps']
    assert len(trace)==len(original)==121
    for step,(new,old) in enumerate(zip(trace,original)):
        assert new['chunk_index']==old['chunk_index']==step%4
        assert all(math.isclose(a,b,abs_tol=1e-7) for a,b in zip(new['planned_action'][:3],old['planned_xyz']))
        assert math.isclose(new['planned_action'][3],old['gripper'],abs_tol=1e-7)
    out=Path('bc_results/h4_observation_noise/software_zero_noise_regression.json')
    data={'purpose':'software regression only; excluded from new noise-study outcomes',
      'prior_final_seed_range_inclusive':[310000,310019],'models':[42,123,456],'intervals':[1,4],
      'episodes_compared':len(all_checks),'all_outcome_timing_metrics_match':True,
      'metrics':['success','steps','initial cube position','close/lift step','closed fraction','inference calls','terminal chunk index','return'],
      'shared_schedule_hashes_match_between_R1_and_R4_per_seed':len(noise_hashes),
      'stepwise_saved_R4_trace_seed_290042':{'model_seed':42,'steps':len(trace),'xyz_gripper_chunk_timing_match_atol_1e_7':True},
      'per_episode_checks':all_checks}
    if out.exists():raise FileExistsError(out)
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(data,indent=2,sort_keys=True)+'\n')
    print(f'Zero-noise regression passed: {len(all_checks)} episodes plus saved 121-step trace')


if __name__=='__main__':main()

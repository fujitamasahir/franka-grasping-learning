"""Render the English experiment report from committed, machine-readable outputs."""
import csv,json
from pathlib import Path
import numpy as np

ROOT=Path('bc_results/action_chunking_analysis');REPORT=Path('EXPERIMENTS_ACTION_CHUNKED_BC_RESIDUAL_RL.md')
def read(p):
    with Path(p).open(newline='') as f:return list(csv.DictReader(f))
def pct(x):return f'{100*float(x):.2f}%'
def num(x):return f'{float(x):.2f}'
def table(headers,rows):
    return '| '+' | '.join(headers)+' |\n|'+'|'.join(['---']*len(headers))+'|\n'+'\n'.join('| '+' | '.join(map(str,r))+' |' for r in rows)
def main():
    bc=read(ROOT/'bc_final_metrics.csv');cells=read(ROOT/'final_cell_metrics.csv');gains=read(ROOT/'difference_in_gains.csv')
    boot=read(ROOT/'difference_in_gains_bootstrap.csv');tests=read(ROOT/'paired_mcnemar.csv');bct=read(ROOT/'h1_vs_h4_bc_mcnemar.csv')
    variability=read(ROOT/'seed_variability.csv');regions=read(ROOT/'regional_rescue_regression.csv');sel=read(ROOT/'checkpoint_selection_effect.csv')
    protocol=json.loads(Path('bc_results/action_chunking_protocol.json').read_text());freeze=json.loads(Path('bc_results/action_chunking_checkpoint_selections.json').read_text())
    correction=json.loads(Path('bc_results/action_chunking_eval/implementation_correction_log.json').read_text())
    # BC architecture-level descriptive results.
    bc_table=[]
    for h in ('H1','H4'):
        rr=[x for x in bc if x['architecture']==h];rates=[float(x['success_rate']) for x in rr]
        calls=np.mean([float(x['mean_bc_calls']) for x in rr]);lat=np.mean([float(x['mean_bc_latency_ms']) for x in rr])
        bc_table.append([h,f'{np.mean(rates)*100:.2f}% ± {np.std(rates,ddof=1)*100:.2f}%',
            f'{calls:.1f}',f'{lat:.3f}',f'{np.mean([float(x["mean_episode_seconds"]) for x in rr]):.3f}'])
    bc_cells=[[r['architecture'],r['bc_seed'],f'{100*float(r["success_rate"]):.2f}%',r['successes'],num(r['mean_steps']),num(r['mean_first_close_step'])] for r in bc]
    selected_table=[]
    for r in cells:
      if r['checkpoint']=='selected':selected_table.append([r['architecture'],r['bc_seed'],r['sac_seed'],r['selected_step'],pct(r['success_rate']),
          f'{float(r["gain_pp"]):+.2f}',r['rescue'],r['regression'],r['net_improvement'],num(r['regression_rate_given_bc_success']),num(r['rescue_rate_given_bc_failure'])])
    fixed_table=[]
    for r in cells:
      if r['checkpoint']=='fixed30k':fixed_table.append([r['architecture'],r['bc_seed'],r['sac_seed'],pct(r['success_rate']),f'{float(r["gain_pp"]):+.2f}',r['rescue'],r['regression'],r['net_improvement']])
    gain_table=[[r['checkpoint'],r['bc_seed'],r['sac_seed'],f'{float(r["gain_h1_pp"]):+.2f}',f'{float(r["gain_h4_pp"]):+.2f}',f'{float(r["difference_in_gains_pp"]):+.2f}'] for r in gains]
    h1h4=[]
    for h in ('H1','H4'):
      rr=[r for r in cells if r['architecture']==h and r['checkpoint']=='selected']
      h1h4.append([h,pct(np.mean([float(x['success_rate']) for x in rr])),f'{np.mean([float(x["gain_pp"]) for x in rr]):+.2f}',
          f'{np.mean([int(x["rescue"]) for x in rr]):.1f}',f'{np.mean([int(x["regression"]) for x in rr]):.1f}',
          f'{np.mean([int(x["mean_bc_calls"]) for x in rr]):.1f}',f'{np.mean([int(x["always_on_sac_calls"]) for x in rr]):.0f}'])
    cost=[];sac_hours=[];bc_train=[]
    for h in ('H1','H4'):
        rr=[r for r in cells if r['architecture']==h and r['checkpoint']=='selected']
        base=[r for r in bc if r['architecture']==h]
        cost.append([h,f'{np.mean([float(r["mean_bc_calls"]) for r in base]):.1f}',
            f'{np.mean([float(r["mean_bc_latency_ms"]) for r in base]):.3f}',
            f'{np.mean([float(r["mean_sac_latency_ms"]) for r in rr]):.3f}',
            f'{np.mean([float(r["mean_episode_seconds"]) for r in rr]):.3f}',
            f'{np.mean([float(r["always_on_sac_calls"]) for r in rr]):.0f}'])
        for b in (42,123,456):
            summary=json.loads(Path(f'bc_results/action_chunking_bc_runs/{h}_seed{b}/summary.json').read_text())
            bc_train.append([h,b,num(summary['training_seconds']),summary['best_epoch']])
        for b in (42,123,456):
            for s in (1103,1117,1123):
                q=Path(f'bc_results/action_chunking_sac/{h}_bc{b}_sac{s}/training_runtime.json')
                if q.exists():sac_hours.append(float(json.loads(q.read_text())['accumulated_seconds']))
    bg={r['checkpoint']:r for r in boot}
    significance=sum(float(x['p_holm'])<.05 for x in tests)
    bcsig=sum(float(x['p_holm'])<.05 for x in bct)
    selected_steps=[int(x['selected_step']) for x in freeze['selections']]
    selection_changes=[float(x['final_selection_minus_fixed_pp']) for x in sel]
    cell_seed=[]
    for h in ('H1','H4'):
      for factor in ('bc_seed','sac_seed'):
        rr=[r for r in variability if r['architecture']==h and r['checkpoint']=='selected' and r['factor']==factor]
        for x in rr:cell_seed.append([h,factor,x['seed'],f'{float(x["mean_gain_pp"]):+.2f}',num(x['sd_over_crossed_other_factor_pp'])])
    reg_summary=[]
    for h in ('H1','H4'):
      for q in ('EN','ES','WN','WS'):
        rr=[r for r in regions if r['architecture']==h and r['checkpoint']=='selected' and r['region']==q]
        reg_summary.append([h,q,sum(int(r['episodes']) for r in rr),sum(int(r['rescue']) for r in rr),sum(int(r['regression']) for r in rr),sum(int(r['net_improvement']) for r in rr)])
    report=f'''# Action-Chunked Behavior Cloning and Residual SAC

## Abstract

This preregistered study compares single-step (H1) and four-step (H4) state-conditioned MLP behavior cloning, then applies an always-on, stepwise SAC XYZ residual to each frozen BC policy. Demonstrations, valid chunk starts, robot dynamics, evaluation positions, action limits, reward terms, and checkpoint selection were fixed before training. Final conclusions are intentionally generated only after the disjoint Final set was evaluated. The study tests a small deterministic-state MLP on one simulated Franka task; it is not a reproduction of ACT or Diffusion Policy.

## Motivation and Related Work

H4 predicts four future actions from the current 12D state and executes them open loop while physics still advances at every control step. This can lower BC replanning cost but lets state error accumulate within a chunk. SAC observes the current state, the exact next planned BC action, and chunk phase, and can add a bounded XYZ correction on every step. The design tests whether that closed-loop residual changes the reliability of H4 relative to H1 under matched starts.

Ankile et al. describe ResiP as residual RL for precise assembly, using a learned imitation policy with residual refinement ([arXiv:2407.16677](https://arxiv.org/abs/2407.16677)). Their setting motivates correcting a nominal policy during execution. Ankile et al. also study residual off-policy RL for fine-tuning BC ([arXiv:2509.19301](https://arxiv.org/abs/2509.19301)). This experiment differs by freezing the BC policy, learning only a small per-step XYZ residual, and explicitly exposing the current action-chunk state. It uses a lightweight state-conditioned chunked MLP, not ACT or Diffusion Policy.

## Research Questions

1. Does H4 action chunking change BC grasp success under matched training episodes and starting observations?
2. Does stepwise residual SAC improve H4 more reliably than H1 relative to each method's own BC baseline?
3. How do rescue, regression, and seed variability change with chunking?
4. Does H4 reduce BC replanning while SAC continues acting every control step?

## Hypotheses

Chunking may improve temporal consistency or reduce closed-loop responsiveness. Stepwise residual SAC may correct errors within an open-loop chunk. H4 plus SAC may have a different rescue/regression balance than H1 plus SAC. BC and SAC training seeds may interact. No method is assumed to win.

## Dataset Audit and Data Integrity

The study uses the original unsampled `demonstrations/franka_expert_train.npz` and `demonstrations/franka_expert_wide.npz`: 200 and 100 additional successful episodes, respectively, with 35,377 observations total. Observation/action row counts aligned; all 12D observations and 4D actions were finite; gripper labels were `+1=open`, `-1=closed`; `max_move=0.15 m`, `max_time=16 s`, and `control_dt=0.1 s` matched. Within every episode, rows were contiguous and normalized observation time advanced by 0.1 s within 1e-4 s. The original wide file had 100 additional trajectories and no exact duplicates of the narrow file. Source hashes and the full audit are in `bc_results/action_chunking_data_audit.json`.

The historical `fair_wide_expert.npz` is excluded from chunk targets because random row subsampling left 73 nonconsecutive control intervals. Episode IDs were remapped for the wide source, then 30 of the 300 episodes were held out with fixed NumPy seed 20261106. H1 and H4 use the same 31,029 training windows and 3,448 validation starts; windows never cross episode boundaries or temporal gaps. H1 supervises one action per start (31,029 targets); H4 supervises four (124,116 targets). This extra future-action supervision is a method difference. The split lists, hashes, and audit are committed in the preregistration.

## Single-Step vs Action-Chunked BC

Both policies take 12D observations and use a 128-128 ReLU MLP. H1 predicts one tanh XYZ action plus one gripper logit and replans each step. H4 predicts four XYZ actions and four gripper logits, caching and executing the sequence for four consecutive physics steps. Both use Adam, learning rate 3e-4, batch 256, maximum 100 epochs, the same train-only observation normalization, SmoothL1 XYZ loss plus 0.25 BCE gripper loss, and early stopping on held-out episodes (patience 15, minimum loss improvement 1e-6). Three seeds (42, 123, 456) were trained per horizon. H1 has 18,692 parameters and H4 has 20,240.

## Chunk-Aware Residual SAC

The SAC observation has 17 values: environment state (12), the exact planned action being corrected (4), and normalized chunk phase (1; always 0 for H1 and 0, 1/3, 2/3, 1 for H4). SAC supplies only XYZ residuals, clipped to ±5 mm per axis; BC retains gripper control and its parameters remain frozen. SAC acts every control step, including within H4 chunks. No spatial gate is used. Condition C was retained: 30,000 steps, checkpoints at 10k/20k/30k, 90% hard samples outside the ±12 cm inner square, shaped reward, residual penalty coefficient 0.005, and SAC seeds 1103/1117/1123. Stable-Baselines3 SAC uses CPU, MLP [256,256], learning rate 3e-4, replay buffer 50,000, learning starts 1,000, batch 128, gamma .99, train frequency 1, gradient steps 1, and automatic entropy coefficient. The 18 runs cross 2 horizons × 3 BC seeds × 3 SAC seeds; these are not 18 independent BC training replications.

## Experimental Protocol

BC validation uses shared seeds 270000–270499. SAC checkpoint validation uses shared seeds 280000–280499, plus a separate 40-position diagnostic panel at 300000–300039 that does not select checkpoints. Selection is highest validation success, then fewest regressions against its paired BC, shortest successful episode, then earliest checkpoint. Selection hashes were frozen before Final evaluation. Final uses shared seeds 290000–291999. BC Final rollouts are reused across associated SAC policies. Final fixed-30k and selected policies are evaluated separately, except identical checkpoints are evaluated once and reused. Episode outputs are append-resumable CSVs. Source code, test outputs, data provenance, model hashes, and selection hashes are recorded with the artifacts.

## BC Results

Final BC success, mean episode BC calls, inference latency, and episode execution time are summarized below. The ± term is the sample standard deviation across the three BC training seeds; it is descriptive, not a population-level estimate.

{table(['Policy','Success mean ± SD','BC calls / episode','BC latency (ms/call)','Episode time (s)'],bc_table)}

{table(['Horizon','BC seed','Final success','Successes','Mean steps','Mean first close step'],bc_cells)}

The historical `fair_wide_expert_seed42.pt` is not used as a matched baseline: it was trained from a different, randomly subsampled dataset and does not provide temporally reliable H4 targets. Its prior results are contextual only.

During validation implementation QA, an initial partial set used the environment defaults (`max_move=0.025 m`, `max_time=12 s`) instead of checkpoint values (`0.15 m`, `16 s`). These rollouts were invalidated and excluded; all six corrected BC validations were rerun with the checkpoint settings. The correction log records hashes and episode counts. No Final seed or training/model-selection decision was exposed to these discarded rollouts.

## SAC Results

Selected-checkpoint absolute success, success gain relative to the matching BC policy, rescue, regression, and conditional rates for every crossed seed cell:

{table(['H','BC seed','SAC seed','Selected step','Success','Gain (pp)','Rescue','Regression','Net','Reg./BC success','Rescue/BC fail'],selected_table)}

Fixed 30k results are shown separately so validation selection effects remain visible:

{table(['H','BC seed','SAC seed','Success','Gain (pp)','Rescue','Regression','Net'],fixed_table)}

The primary differential benefit is computed from each architecture's own BC baseline:

{table(['Checkpoint family','BC seed','SAC seed','Gain H1 (pp)','Gain H4 (pp)','Difference-in-gains (pp)'],gain_table)}

{table(['Horizon','Mean selected success','Mean gain (pp)','Mean rescue','Mean regression','BC calls / ep','SAC calls / ep'],h1h4)}

## Rescue and Regression

For each rollout pair, rescue is a BC failure corrected to success, and regression is a BC success changed to failure. Their difference equals the paired success-count difference. Conditional rates use BC-failure and BC-success denominators respectively. H1 and H4 have different baseline failure sets, so all residual results are paired only with their own BC policy. `bc_results/action_chunking_analysis/final_cell_metrics.csv` contains exact per-cell denominators and counts.

Regional paired rescue/regression counts for selected checkpoints:

{table(['Horizon','Region','Episodes across crossed cells','Rescue','Regression','Net'],reg_summary)}

## Training-Seed Variability

The selected-policy mean gains by BC or SAC seed, averaged over the crossed other factor, are listed below. The SD column is variation over the three values of that other factor, not an independent replication estimate.

{table(['H','Factor','Seed','Mean gain (pp)','SD over crossed other factor'],cell_seed)}

Checkpoint selection chose steps {', '.join(map(str,sorted(set(selected_steps))))} across 18 policies. The mean final selected-minus-fixed30k success difference was {np.mean(selection_changes):+.2f} percentage points (range {min(selection_changes):+.2f} to {max(selection_changes):+.2f}). See `checkpoint_selection_effect.csv` for all paired rows. Validation success and Final success are separate measurements; selection was frozen using validation only.

## Computational Cost

H1 replans BC every step. H4 replans once per four steps, so the measured BC call frequency should be interpreted with per-call latency and episode time. SAC remains always-on: its inference-call rate is reported, and this experiment makes no SAC-call-saving claim. Residual norm, saturation, intervention rate, gripper-close timing, lift timing, and chunk phase at events are in the cell CSV. The selected H4 trajectory plot is a single preregistered illustrative seed and is not representative.

{table(['Horizon','BC calls / episode','BC latency (ms)','SAC latency (ms)','Residual episode time (s)','SAC calls / episode'],cost)}

BC optimization time by training seed:

{table(['Horizon','BC seed','Training seconds','Best epoch'],bc_train)}

Total accumulated SAC training wall time across completed runs: **{sum(sac_hours)/3600:.2f} hours** ({len(sac_hours)} complete runs with runtime records). Per-run durations are stored in `training_runtime.json`.

## Statistical Analysis

Exact two-sided McNemar tests compare paired final positions for each BC-vs-SAC cell and H1-vs-H4 BC seed. Holm corrections are applied within the preregistered architecture/checkpoint families. {significance} of {len(tests)} paired BC-vs-SAC tests were Holm-adjusted below .05; {bcsig} of the 3 H1-vs-H4 BC comparisons were below .05 after Holm correction. These tests quantify paired position outcomes for these trained models, not variation over model training.

The primary differential SAC effect is `Gain_H4 − Gain_H1`, each gain measured against its own matched BC rollout. A 10,000-replicate hierarchical bootstrap resamples the 3 BC seed indices and 3 SAC seed indices independently and forms the crossed 3×3 cells. It preserves the crossed training structure; shared positions remain paired within cells.

{table(['Checkpoint family','Mean difference-in-gains (pp)','Across-cell SD (pp)','Hierarchical 95% interval (pp)'],[[r['checkpoint'],num(r['mean_difference_in_gains_pp']),num(r['sd_across_9_crossed_cells_pp']),f'[{num(r["hierarchical_bootstrap_95_low_pp"])}, {num(r["hierarchical_bootstrap_95_high_pp"])}]'] for r in boot])}

## Discussion

The H1/H4 BC comparison uses identical demonstration episodes, chunk-valid start observations, and Final positions. If H4 improves BC success, the result is specific to this horizon-four deterministic state MLP and includes fourfold future-action supervision per start. If H4 underperforms, reduced closed-loop responsiveness, optimization or data limitations, or execution implementation should be considered; this does not refute action chunking generally.

Residual SAC is judged by both rescue and regression against each architecture's own BC baseline. A positive gain can still contain substantial regressions, while low regression can coexist with limited rescue. Checkpoint-selected results are selection-conditioned and should be read alongside fixed-30k outcomes and validation curves. With only three BC and three SAC seeds, the crossed cell variation is large relative to what can support broad generalization.

## Limitations

- One simulated Franka grasping task and one short horizon (H=4).
- State-conditioned MLP, not ACT or Diffusion Policy.
- Demonstrations are successful scripted-expert trajectories.
- H4 has four times as many future-action labels per valid start.
- Only three BC seeds and three SAC seeds; seed-population uncertainty is weakly identified.
- Thousands of positions are repeated across policies and do not create independent training replications.
- SAC is always active, and results are simulation-only.
- Checkpoint selection uses one shared 500-position validation set and can be noisy.

## Implications for ResiP / ResFiT

The design shares the residual-control idea with ResiP: preserve a nominal imitation policy and use feedback to refine execution. Its H4 arm makes an explicit four-action open-loop plan, with SAC correcting each next action. The study does not reproduce ResiP's assembly setup or claim comparable performance. Relative to ResFiT's residual off-policy BC fine-tuning direction, this study keeps BC frozen and isolates chunking under fixed SAC settings; it does not test residual policy parameterization or online BC fine-tuning. Any observed benefit is therefore evidence about this controlled implementation only.

## Next Research Questions

Use these results to decide whether a small ACT or diffusion-policy baseline is warranted. A follow-up should retain matched demonstration starts, include a properly capacity-matched sequence model and multiple horizons, and preserve paired residual comparisons. If H4 gains are weak or seed-sensitive, first test whether the limitation is chunk execution or model expressivity before scaling to a larger policy family. Include additional independent BC/SAC seeds before population-level claims.

## Reproduction Instructions

1. Use branch `experiment/action-chunked-bc-residual-rl` at source base `10d607fe7cacaae3004c8a291348c5022e41f4a9`.
2. Verify the protocol SHA in `bc_results/action_chunking_protocol.sha256` and the dataset audit hashes.
3. Run `.venv/bin/python -m unittest discover -p 'test_*.py' -v`.
4. Train BC: `.venv/bin/python train_action_chunk_bc.py`.
5. Generate BC validation rollouts: `.venv/bin/python evaluate_action_chunking.py --phase bc-validation`.
6. Train the 18 crossed SAC runs: `.venv/bin/python train_action_chunk_sac.py`; resume an interrupted run with `--resume --arch H1 --bc-seed 42 --sac-seed 1103` (choose the interrupted tuple).
7. Freeze validation selections: `.venv/bin/python freeze_action_chunk_selections.py`.
8. Run BC Final and then SAC Final: `.venv/bin/python evaluate_action_chunking.py --phase bc-final`; `.venv/bin/python evaluate_action_chunking.py --phase sac-final`.
9. Analyze and create report figures: `.venv/bin/python analyze_action_chunking.py`; `.venv/bin/python plot_action_chunking.py`; `.venv/bin/python write_action_chunking_report.py`.

Protocol SHA-256: `{freeze['protocol_sha256']}`. Selection manifest SHA-256 is recorded in `bc_results/action_chunking_checkpoint_selections.sha256`. Final tables and analysis live under `bc_results/action_chunking_analysis/`; SVG/PNG figures are under `bc_results/action_chunking_figures/`.
'''
    REPORT.write_text(report,encoding='utf-8');print(f'Wrote {REPORT}',flush=True)
if __name__=='__main__':main()

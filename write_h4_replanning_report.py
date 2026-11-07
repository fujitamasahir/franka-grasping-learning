"""Generate the English H4 replanning ablation report from audited outputs."""
import json
import os
from pathlib import Path

ROOT=Path(__file__).parent
DATA=ROOT/'bc_results/h4_replanning'
REPORT=ROOT/'EXPERIMENTS_H4_REPLANNING_ABLATION.md'


def pct(x): return f'{x:.2f}'
def num(x): return f'{x:.3f}'
def pval(x): return f'{x:.3g}'


def main():
    s=json.loads((DATA/'summary.json').read_text())
    integrity=json.loads((DATA/'final_integrity_audit.json').read_text())
    lines=[]
    add=lines.append
    mean_rates={i:sum(s['per_model'][str(seed)][f'R{i}']['success_rate_pct'] for seed in (42,123,456))/3 for i in (1,2,4)}
    seed_eff=s['seed_level_primary_R4_minus_R1_pp']
    r4_cmp=[s['per_model'][str(seed)]['paired_comparisons']['R4_vs_R1'] for seed in (42,123,456)]
    r4r1_mean=seed_eff['mean']
    add('# H4 Replanning-Frequency Ablation of Frozen Action-Chunked Behavior Cloning')
    add('')
    add('**Study status:** Final evaluation completed with a transparent recovery run for one corrupted output. **Protocol SHA-256:** `'+s['protocol_sha256']+'`.')
    add('')
    add('## 1. Abstract')
    add('')
    add(f'We compared runtime replanning intervals of 1, 2, and 4 control steps using the same frozen H4 behavior-cloning (BC) policy for each of three BC training seeds. Across the three fixed checkpoints, mean success was {mean_rates[1]:.2f}% (R1), {mean_rates[2]:.2f}% (R2), and {mean_rates[4]:.2f}% (R4). R4 exceeded R1 by {r4r1_mean:.2f} percentage points for all three checkpoints, with seed-specific differences of {r4_cmp[0]["difference_pp"]:.2f}, {r4_cmp[1]["difference_pp"]:.2f}, and {r4_cmp[2]["difference_pp"]:.2f} pp. The effect was heterogeneous: the position-level Holm-adjusted exact McNemar test was significant for BC seeds 42 and 456, but not seed 123. R4 used about 75% fewer BC inference calls than R1, while mean episode wall time changed by only about 1.9%. These are conditional results for three fixed models and 2,000 shared positions; they do not establish population-level superiority.')
    add('')
    add('## 2. Motivation')
    add('')
    add('The earlier H1-versus-H4 comparison changed the prediction horizon, number of supervised future actions, output-head size, and runtime replanning rate together. It therefore could not isolate which factor explained the success-rate difference. This study keeps each H4 checkpoint and its normalization fixed and changes only how many predicted actions are executed before replanning.')
    add('')
    add('## 3. Research Questions')
    add('')
    add('- **RQ:** Does executing cached action chunks open-loop improve grasping success compared with replanning more frequently using the same trained action-chunked policy?')
    add('- **Primary contrast:** paired R4 − R1 success for each frozen BC model.')
    add('- **Secondary contrast:** paired R2 − R1; R4 − R2 is descriptive.')
    add('')
    add('## 4. Preregistered Hypotheses')
    add('')
    add('- **H1:** R4 may outperform R1 through temporally consistent action execution.')
    add('- **H2:** R1 may outperform R4 through more frequent feedback and state correction.')
    add('- **H3:** R2 may trade off temporal consistency and feedback frequency.')
    add('- **H4:** R4 should reduce BC inference calls; this need not proportionally reduce total episode time.')
    add('')
    add('These hypotheses were treated as competing possibilities, not assumed conclusions.')
    add('')
    add('## 5. Frozen Model and Dataset Provenance')
    add('')
    add('The study used three existing, frozen, state-conditioned H4 MLP BC checkpoints. No new BC or SAC training, parameter updates, reward changes, or residual controller were used. Every condition retained prediction horizon 4 and output shape `(4, 4)`. The three checkpoints have identical observation-normalization arrays. Their SHA-256 hashes matched the existing training provenance summaries before and after evaluation.')
    add('')
    add('| BC training seed | Checkpoint | SHA-256 | Parameters |')
    add('|---:|---|---|---:|')
    for seed in (42,123,456):
        entry=json.loads((ROOT/f'bc_results/action_chunking_bc_runs/H4_seed{seed}/summary.json').read_text())
        ck=json.loads((DATA/'final_integrity_audit.json').read_text())
        model=next(x for x in integrity['configurations'] if x['model_seed']==seed)
        add(f'| {seed} | `bc_models/action_chunking/H4_seed{seed}.pt` | `{model["checkpoint_sha256"]}` | {entry["parameter_count"]:,} |')
    add('')
    add('Observation-mean SHA-256: `d833cc7443412972194a8dedfee2bc5b29ad3ce854e8acd51735941a68bec62b`; observation-standard-deviation SHA-256: `8902f6e7f5947d2e4f8fc2860a7799aa600dfe9bc8476c7be58b5b32a127a3c3`. The model training position range in checkpoint metadata is ±0.12 m; evaluation used the preregistered ±0.20 m position range.')
    add('')
    add('## 6. Experimental Conditions')
    add('')
    add('| Condition | Actions executed from each H4 prediction | Replan interval |')
    add('|---|---|---:|')
    add('| R1 | Index 0 only; remaining predictions discarded | 1 step |')
    add('| R2 | Indices 0 and 1; indices 2–3 discarded | 2 steps |')
    add('| R4 | Indices 0–3 | 4 steps |')
    add('')
    add('All conditions used the same checkpoint within each BC seed, checkpoint normalization, gripper convention (`>=0` open; `<0` close), FrankaAutonomousEnv dynamics, `max_move=0.15 m`, `max_time=16 s`, `control_dt=0.1 s`, and existing success rule (lift ≥0.10 m for at least five control steps). No SAC, learned gate, or reward modification was used.')
    add('')
    add('## 7. Software Equivalence and R4 Regression Test')
    add('')
    add('R4 was compared against existing H4 final rollout records on the predetermined software-test positions 290000–290019 for all three BC seeds (60 episodes total). Success/failure, episode steps, deterministic initial cube coordinates, first close/lift step, inference-call count, terminal chunk index, closed fraction, and return matched the saved records. The recorded seed-290042 trajectory also matched all 121 stepwise XYZ commands, gripper commands, and chunk indices within 1e-7. These old positions were used only for software verification and are excluded from the new hypothesis analysis.')
    add('')
    add('## 8. Evaluation Protocol')
    add('')
    add('A 20-position-per-cell smoke test used seeds 312000–312019 (180 episodes), followed by the fixed Final range 310000–311999. Each of the nine model-by-interval cells contains exactly 2,000 unique seeds. All nine cells share identical initial positions. Runs were sequential; per-episode CSV rows were flushed and resume-validated. The recovery output for H4 seed 456 / R1 is versioned as `final_H4_seed456_R1_recovery1.csv`; it follows the same preregistered conditions and seeds. The corrupt partial is excluded from analysis and retained separately.')
    add('')
    add(f'Integrity audit: {integrity["total_final_episodes"]:,} valid episodes across {integrity["configuration_count"]} cells; exact seed sets, checkpoint hashes, protocol hashes, and cross-cell initial positions passed. Model weights were unchanged. See `bc_results/h4_replanning/final_integrity_audit.json`.')
    add('')
    add('A post-run analysis QA check found that the first generated summary multiplied the mean per-call latency by 1,000 twice. That summary is preserved as `summary_initial_latency_scale_error.json`; `summary_latency_correction.json` records the correction. The reported per-call latency was recomputed from unchanged raw inference milliseconds and call counts. Success outcomes, inference-call counts, total inference time, and tests were unaffected.')
    add('')
    add('### Recovery and data-integrity disclosure')
    add('')
    add('The first H4 seed 456 / R1 partial file contained 2,062 rows for 2,000 unique seeds, with duplicated seeds 311938–311999. Duplicate records had matching non-timing fields but different inference and wall-clock measurements. The file was preserved byte-for-byte and excluded. With explicit user authorization, that entire 2,000-episode cell was rerun to a separate `recovery1` output; no rows were carried over. During diagnosis, two example duplicated records (311938 and 311939) were directly viewed and both were successes. This limited outcome exposure occurred before the recovery run; no aggregate condition comparison was computed and no condition or setting was selected or tuned. This deviation and exposure are retained in `recovery_authorization.json` and `corrupt_partial_seed456_R1_audit.json`.')
    add('')
    add('## 9. Success Rate Results')
    add('')
    add('| Frozen BC seed | R1 | R2 | R4 | R4 − R1 (pp) |')
    add('|---:|---:|---:|---:|---:|')
    for seed in (42,123,456):
        d=s['per_model'][str(seed)]
        add(f'| {seed} | {d["R1"]["successes"]}/2000 ({d["R1"]["success_rate_pct"]:.2f}%) | {d["R2"]["successes"]}/2000 ({d["R2"]["success_rate_pct"]:.2f}%) | {d["R4"]["successes"]}/2000 ({d["R4"]["success_rate_pct"]:.2f}%) | {d["paired_comparisons"]["R4_vs_R1"]["difference_pp"]:+.2f} |')
    add(f'| Mean across three fixed models | {mean_rates[1]:.2f}% | {mean_rates[2]:.2f}% | {mean_rates[4]:.2f}% | **{r4r1_mean:+.2f}** |')
    add('')
    add('The three-model mean is descriptive and weights the three existing checkpoints equally. It is not a 6,000-policy estimate.')
    add('')
    add('## 10. Paired Rescue and Regression Analysis')
    add('')
    add('“Rescue” and “regression” below mean paired success transitions between replanning conditions, not SAC interventions.')
    add('')
    add('| BC seed | R2 rescue / regression vs R1 | Net | R4 rescue / regression vs R1 | Net |')
    add('|---:|---:|---:|---:|---:|')
    for seed in (42,123,456):
        d=s['per_model'][str(seed)]['paired_comparisons'];a=d['R2_vs_R1'];b=d['R4_vs_R1']
        add(f'| {seed} | {a["rescue"]} / {a["regression"]} | {a["net_improvement"]:+d} | {b["rescue"]} / {b["regression"]} | {b["net_improvement"]:+d} |')
    add('')
    add('For R4−R1, the per-seed success differences equal rescue minus regression: +259, +4, and +81 successes. Across the three paired model-position cells, the descriptive total is 417 rescue transitions and 73 regressions (net +344); because evaluation positions are shared across models, this aggregate is not an independent-sample test. R2−R1 yielded 328 rescues and 37 regressions (net +291). R4 exceeded R2 in all three seeds by +1.05, +0.85, and +0.75 pp; its paired transitions and arithmetic are in `summary.json`.')
    add('')
    add('Regional success and transition counts are exploratory. Each fixed model has 493 EN, 512 ES, 503 WN, and 492 WS positions.')
    add('')
    add('| BC seed | Region (n) | R1 success | R2 success | R4 success | R2 rescue / regression vs R1 | R4 rescue / regression vs R1 |')
    add('|---:|:---:|---:|---:|---:|---:|---:|')
    for seed in (42,123,456):
        rg=s['regions'][str(seed)]
        for region in ('EN','ES','WN','WS'):
            n=rg['R1'][region]['n'];r2=rg['paired_transitions_vs_R1']['R2'][region];r4=rg['paired_transitions_vs_R1']['R4'][region]
            add(f'| {seed} | {region} ({n}) | {rg["R1"][region]["success_rate_pct"]:.2f}% | {rg["R2"][region]["success_rate_pct"]:.2f}% | {rg["R4"][region]["success_rate_pct"]:.2f}% | {r2["rescue"]} / {r2["regression"]} | {r4["rescue"]} / {r4["regression"]} |')
    add('')
    add('The largest R4−R1 regional changes occurred for seed 42 in WN and WS; seed 123 had small declines in EN/ES but gains in WN/WS; seed 456 gains were largest in ES/WS. These region-level patterns should not be generalized because the analysis is exploratory and each region contains about 500 positions per fixed model.')
    add('')
    add('## 11. BC Inference Frequency')
    add('')
    add('| BC seed | Interval | Calls / episode | Mean latency / call (ms) | Total inference / episode (ms) |')
    add('|---:|---:|---:|---:|---:|')
    for seed in (42,123,456):
        d=s['per_model'][str(seed)]
        for i in (1,2,4):
            c=d[f'R{i}']
            add(f'| {seed} | R{i} | {c["mean_inference_calls"]:.2f} | {c["mean_inference_latency_ms_per_call"]:.4f} | {c["mean_total_inference_ms"]:.3f} |')
    add('')
    r1calls=sum(s['per_model'][str(seed)]['R1']['mean_inference_calls'] for seed in (42,123,456))/3
    r4calls=sum(s['per_model'][str(seed)]['R4']['mean_inference_calls'] for seed in (42,123,456))/3
    r1infer=sum(s['per_model'][str(seed)]['R1']['mean_total_inference_ms'] for seed in (42,123,456))/3
    r4infer=sum(s['per_model'][str(seed)]['R4']['mean_total_inference_ms'] for seed in (42,123,456))/3
    add(f'Across the three models, mean calls fell from {r1calls:.2f} (R1) to {r4calls:.2f} (R4), a {(1-r4calls/r1calls)*100:.1f}% reduction. Mean measured inference time fell from {r1infer:.3f} to {r4infer:.3f} ms per episode ({(1-r4infer/r1infer)*100:.1f}%). Per-call latency fluctuated across sequential runs and is descriptive.')
    add('')
    add('## 12. Execution Time')
    add('')
    add('| BC seed | Interval | Mean control steps | Mean successful steps | Mean total episode wall time (s) | Mean first close step | Mean first lift step |')
    add('|---:|---:|---:|---:|---:|---:|---:|')
    for seed in (42,123,456):
        for i in (1,2,4):
            c=s['per_model'][str(seed)][f'R{i}']
            add(f'| {seed} | R{i} | {c["mean_control_steps"]:.2f} | {c["mean_successful_episode_steps"]:.2f} | {c["mean_episode_wall_seconds"]:.4f} | {c["mean_first_close_step"]:.2f} | {c["mean_first_lift_step"]:.2f} |')
    add('')
    wall1=sum(s['per_model'][str(seed)]['R1']['mean_episode_wall_seconds'] for seed in (42,123,456))/3
    wall4=sum(s['per_model'][str(seed)]['R4']['mean_episode_wall_seconds'] for seed in (42,123,456))/3
    add(f'Mean total episode wall time was {wall1:.4f} s for R1 and {wall4:.4f} s for R4, a descriptive {(1-wall4/wall1)*100:.1f}% difference. Episode timing starts immediately before the initial policy inference and ends at the terminal transition; environment reset and model loading are excluded. The runtime figure separates full episode time from millisecond-scale model inference. The modest wall-time change is not evidence of a proportional end-to-end speedup; hardware load and serial run order affect timings.')
    add('')
    add('## 13. Statistical Analysis')
    add('')
    add('For each fixed model, paired success outcomes on the same 2,000 positions were compared with a two-sided exact McNemar test. Holm correction was applied across the three model-specific tests within each comparison family. A 95% percentile paired bootstrap interval uses 10,000 resamples of positions and fixed RNG seed 20261106. These inferential quantities are conditional on the evaluated model and position distribution.')
    add('')
    add('| BC seed | R4−R1 (pp) | 95% paired bootstrap CI (pp) | Exact McNemar p | Holm p |')
    add('|---:|---:|---:|---:|---:|')
    for seed in (42,123,456):
        c=s['per_model'][str(seed)]['paired_comparisons']['R4_vs_R1'];lo,hi=c['paired_difference_95pct_bootstrap_ci_pp']
        add(f'| {seed} | {c["difference_pp"]:+.2f} | [{lo:+.2f}, {hi:+.2f}] | {pval(c["exact_two_sided_p"])} | {pval(c["holm_adjusted_p_across_3_models"])} |')
    add('')
    add('The primary position-level contrast is significant after the specified correction for seeds 42 and 456, but not for seed 123. The three seed-level effects are +12.95, +0.20, and +4.05 pp (mean +5.73, median +4.05, sample SD 6.54, range +0.20 to +12.95). All three directions are positive, but three trained seeds are too few to support strong population-level generalization. The shared 2,000 positions are not independent policy replications.')
    add('')
    add('R2−R1 exact paired tests are reported in `summary.json`; Holm-adjusted results are significant for each seed, with effects +11.90, −0.65, and +3.30 pp. These position-level tests do not establish that a replanning interval is generally superior across training runs or tasks.')
    add('')
    add('## 14. Interpretation of R1/R2/R4')
    add('')
    add(f'R4 yielded higher success than R1 for all three frozen checkpoints, and it exceeded R2 for all three by less than 1.1 pp. R1 did not outperform R4 in any of the three seed-level comparisons. The magnitude varied sharply: seed 42 accounted for the largest observed R4−R1 gain, while seed 123 was nearly unchanged. R2 was not a uniformly better compromise: it was below R1 for seed 123 and above R1 for seeds 42 and 456.')
    add('')
    add('## 15. Implications for Action Chunking and Closed-Loop Feedback')
    add('')
    add('For these H4 policies and this simulated grasping task, executing cached actions for longer blocks was associated with higher success than replanning every step. This is consistent with temporal consistency being useful here; it does not show that open-loop chunks are universally better. Frequent feedback did not yield higher mean success in this experiment. R4 reduced BC inference demand substantially, while measured end-to-end episode runtime changed little.')
    add('')
    add('The earlier H1-versus-H4 result remains confounded by architecture, supervision count, output parameterization, and runtime frequency. This controlled H4 study shows that runtime interval alone can matter for some checkpoints, especially seed 42, but it cannot attribute the entire H1/H4 gap to runtime replanning. The R4 replication closely matches earlier H4 Final rates: new minus prior differences were −0.05, −0.15, and +0.05 pp for seeds 42, 123, and 456, respectively.')
    add('')
    add('## 16. Limitations')
    add('')
    add('- Only three independently trained BC seeds and one simulated task were studied; positions are shared within each seed across conditions.')
    add('- Position-level p-values and confidence intervals are conditional on fixed checkpoints and the chosen position distribution; they do not replace more independent training seeds.')
    add('- Sequential wall-clock timing is sensitive to machine load. Call reduction, inference-time reduction, and end-to-end runtime reduction are distinct quantities.')
    add('- The recovery run followed a documented duplicate-output incident. Two duplicated success outcomes were directly observed before the full-cell recovery; no tuning or interval selection followed, but this is a limited data-exposure deviation.')
    add('- The experiment isolates runtime replanning within H4 only. It does not isolate H4 multi-step supervision from policy architecture or compare additional chunk sizes.')
    add('')
    add('## 17. Relationship to ResiP / ResFiT')
    add('')
    add('[ResiP](https://arxiv.org/abs/2407.16677) combines frozen chunked behavior cloning with a closed-loop residual policy learned using reinforcement learning. [ResFiT](https://arxiv.org/abs/2509.19301) studies off-policy residual fine-tuning around a frozen BC base. Both motivate preserving a pretrained policy while adding corrections. The present experiment is a prerequisite-style runtime ablation: it trains neither a residual policy nor any RL controller, and evaluates only how a frozen H4 policy behaves when the cached chunk is consumed at different rates.')
    add('')
    add('## 18. Next Research Direction')
    add('')
    add('Before making broad claims, replicate the same preregistered R1/R2/R4 comparison with more independently trained H4 seeds and a separately held-out position distribution. A subsequent study can test whether the observed seed- and region-specific failures are corrected by a residual controller, but that should be a distinct experiment with its own training and selection protocol. A matched timing protocol or repeated interleaved run order is needed before making strong runtime claims.')
    add('')
    add('## 19. Reproduction Instructions')
    add('')
    add('From the repository root on branch `experiment/h4-replanning-ablation`:')
    add('')
    add('```bash')
    add('./.venv/bin/python -m unittest discover -v')
    for seed in (42,123,456):
        for interval in (1,2,4):
            label=' --run-label recovery1' if (seed,interval)==(456,1) else ''
            add(f'./.venv/bin/python evaluate_h4_replanning.py --phase final --model-seed {seed} --interval {interval}{label}')
    add('MPLCONFIGDIR=/private/tmp/matplotlib-h4 ./.venv/bin/python analyze_h4_replanning.py')
    add('./.venv/bin/python write_h4_replanning_report.py')
    add('```')
    add('')
    add('Completed output CSVs validate against the expected unique seed set and provenance hashes. Do not run duplicate processes for the same configuration. The `recovery1` label is specific to the documented seed-456/R1 recovery and keeps the preserved corrupt partial separate.')
    text='\n'.join(lines)+'\n'
    temp=REPORT.with_suffix('.md.tmp')
    temp.write_text(text,encoding='utf-8')
    os.replace(temp,REPORT)
    print(f'Wrote {REPORT}')


if __name__=='__main__': main()

# Frozen Kalman Online closed-loop experiment: prepared, not executed

Preparation includes synthetic unit tests only. No MuJoCo Smoke/Final was run.
The previous failed Dynamic experiment is not resumed or modified. The accepted
independent diagnostic summary, audit and manifest are preserved byte-for-byte,
versioned, and pinned by the new protocol. Its other local data remain unchanged.

## Fixed design

Three frozen H4 models42/123/456; two profiles0→10 and10→0mm at absolute
observation index40; R1/R4/Kalman OneShot/Kalman Online. Internal CSV controller
names `OneShot` and `Online` always mean the new frozen Kalman variants here.
Smoke uses362000–362009 (240 episodes total); Final uses361000–361999 (24,000).
Noise generator seed20261110 is new, paired by absolute observation step across
all controllers/profiles. These ranges are disjoint from previous study ranges.

The unchanged `kalman_noise_estimator.py` defaults are pinned by SHA-256. At t20,
OneShot and Online both estimate from observations0–19, using7.5mm. Online then
uses windows[t−20:t] at t24,28,…,80, with strict6.5/8.5mm hysteresis; after80 its
interval freezes. Current/future observation t is excluded. Decisions occur on
completed old-chunk boundaries and start a fresh prediction at index0, including
R1→R4 transitions. No new model training, calibration, threshold search or
Final-based adjustment is allowed. Ground truth/noise sigma/transition time and
success labels are not controller/estimator inputs. True motion and injected
noise are evaluator-only diagnostics. The former stationary-object gate is
explicitly NOT used.

Protocol: `bc_results/h4_kalman_online_protocol.json` and `.sha256`, committed
before data acquisition. Model, estimator and legacy source hashes are verified
on each command. New runner hash is also fixed in cell metadata and audits,
preventing code changes between resumed cells or Smoke and Final. Legacy files
and output directories are never edited; scoped in-process function reuse directs
rollouts to the new study only. Do not execute another Python thread using the
legacy module while those scoped overrides are active.

## Terminal commands

Small synthetic tests (no MuJoCo):

```sh
.venv/bin/python -m unittest test_kalman_online_experiment test_kalman_noise_estimator
```

The following perform the actual experiment when the user runs them:

```sh
#240 Smoke episodes; only one compact completion/interruption JSON.
.venv/bin/python kalman_online_experiment.py run --phase smoke

#Software integrity/finite estimates/windows/chunks/noise/prefix audit; no rates.
.venv/bin/python kalman_online_experiment.py audit --phase smoke

#Requires complete passing Smoke audit and unchanged evidence.
.venv/bin/python kalman_online_experiment.py run --phase final

#Requires all24 complete cells and passing Smoke evidence.
.venv/bin/python kalman_online_experiment.py audit --phase final

#Requires complete passing Final audit; no analysis of partial outcomes.
.venv/bin/python kalman_online_experiment.py analyze
```

Output directory: `bc_results/h4_kalman_online_dynamic/`. Final results are
`final_summary.json`, `final_cells.csv` and `final_pairs.csv`. Audit commands
print only counts/gate status and the estimated Final runtime, not success rates.
Existing outputs are never overwritten with different content; a duplicate
identical JSON snapshot is verified instead of rewritten. CSV analysis artifacts
refuse replacement. Do not change settings after inspecting Final outcomes.

Count-only progress, if operationally needed:

```sh
.venv/bin/python kalman_online_experiment.py audit --phase final --allow-incomplete
```

This validates written data and records exact partial counts without calculating
success statistics. Such inspection must not be used for outcome-based stopping
or adjustment. Do not open raw partial success labels to make study decisions.

## Resume, time estimate and deadline

Every episode is appended and fsynced to a `.partial.csv`. Before resuming, its
schema, complete ordered seed prefix, provenance, noise hashes, finite counters,
interval exposures and decision records are validated. Completed cells are
validated/skipped, never overwritten. On interruption, run the same `run` command
again without editing outputs. Truncated/mismatched files stop execution and are
preserved for investigation. A global study lock excludes concurrent writers;
verify its owner before handling a stale lock after an unclean OS shutdown.

`runtime_journal.jsonl` records elapsed active run time, new episode count and
termination reason, without outcomes. The Smoke audit extrapolates total Final
wall hours from that measurement. It includes model loading/validation/reset and
diagnostic work inside measured runs; auditing is additional. The estimate is
descriptive under changing hardware/load, not a precise completion guarantee.
Final start/resume compares its estimated remaining work with the deadline and
preserves evidence rather than starting if the forecast exceeds remaining time.
The hard cutoff is November14,2026 in America/Denver; before every new episode
the date is checked. A running episode finishes append-safely at the boundary.

For a strictly logistical cell-boundary pause, create
`bc_results/h4_kalman_online_dynamic/.STOP_AFTER_CELL`; after exit, deliberately
remove only that marker before resuming. Never base a pause/stop on partial Final
performance. Integrity failure, interruption or the fixed deadline preserve all
valid partial evidence; no partial performance conclusion is generated.

## Analysis and interpretation

Primary: Online−OneShot mean success difference, equally weighting the six
model×profile cells.10,000 paired bootstrap resamples use the same evaluation
position indices jointly across models, profiles and controllers. The JSON
reports the primary95% CI, model-specific effects and their mean/median/sample
SD/range. Results condition on three frozen policies; evaluation positions are
not independent policy-training replications. A positive mean whose CI contains
zero is explicitly not called clear improvement.

Secondary: per-cell success and counterfactual beneficial/harmful/both outcomes
versus OneShot/R1/R4, checking net-success arithmetic. Per-cell exact McNemar
p-values use separate Holm families of six per baseline. Switch counts include
the initial warm-up change. Delay requires an actual appropriate switch at or
after40; already-target-at40 is reported separately. OneShot has no post40 updates
and thus cannot produce a new appropriate post-change switch. Timing, BC calls,
R1/R4 exposure, decisions and episode steps are reported separately. Wall time
includes diagnostic computation and estimator overhead, and is not a physical
robot speed or inference-call speedup claim. No graphs are produced.

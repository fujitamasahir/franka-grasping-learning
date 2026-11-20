# Independent Kalman estimator diagnostic Final: preparation only

No physical trajectories or Final sequences were generated in this change.
No grasp-success evaluation, training, plots or research report is included.
The previous failed studies and their artifacts remain unchanged.

## Frozen specification

The committed protocol is `bc_results/h4_kalman_independent_protocol.json` with
its `.sha256`. It reuses the Stage 1 acceptance criteria without modification.
Three fixed H4 models × fixed R1/R4 × reset seeds **352000–352019** produce
120 new clean-policy MuJoCo trajectories. Noise seed **20261109** is independent
of development seed 20261108. Four replays (0→0, 10→10, 0→10, 10→0 mm; change at
observation40) produce480 diagnostic sequences. Existing development seeds
350000–350019 and synthetic stress seeds351000–351019 are excluded.

The existing Stage 1 collector, observation/noise replay, pure/mixed-window
labeling, metric summarization and detection functions are reused by temporarily
redirecting their in-process globals. Their source files are never edited.
All output goes into the new `bc_results/h4_kalman_independent_validation/`.

Kalman source hash and defaults are frozen:20-sample window, dt0.1s,
acceleration SD0.05m/s², initial position covariance R, velocity variance0.1²,
sigma grid0–30mm in0.25mm steps, and numerical R floor1e-12m². The two existing
estimators are unchanged. No values may be tuned after opening Final results.

Estimators receive only noisy XY. Ground-truth motion and injected RMS sigma are
used after estimation for diagnostic labels/errors. Pure-window metrics and
mixed-window errors are kept separate. Counterfactual interval choices are not
executed by the policy. Unknown acceleration, contact and Q misspecification can
still confound inferred measurement noise.

## Commands for the user to run in Terminal

Related synthetic tests (no simulator rollout or Final data access):

```sh
.venv/bin/python -m unittest test_h4_kalman_independent test_kalman_noise_estimator
```

**The following commands execute the actual independent evaluation; they were
not run while preparing this branch.** Run from the repository root with the
existing `.venv`, frozen checkpoints and Menagerie assets.

1. Generate120 physical trajectories and480 replay sequences:

```sh
.venv/bin/python validate_h4_kalman_independent.py generate
```

2. Audit all new data, including full reconstruction of noise and estimates:

```sh
.venv/bin/python validate_h4_kalman_independent.py audit
```

3. Compute all estimator metrics, gates and20-seed summaries (no plots):

```sh
.venv/bin/python validate_h4_kalman_independent.py analyze
```

4. Display the compact JSON summary:

```sh
.venv/bin/python validate_h4_kalman_independent.py summary
```

Full results: `bc_results/h4_kalman_independent_validation/summary.json`.
`criteria_by_estimator` reports every criterion as PASS/FAIL; `kalman_accepted`
requires all Kalman gates, not selection of the best estimator after evaluation.
The position baseline necessarily fails its own25% improvement-over-position
criterion. Missing coverage or undefined relative improvement fails explicitly.
An accepted estimator does not automatically authorize a control experiment.

`metrics` has stationary/moving MAE, bias, classification, wrong selected interval
and wrong-direction actual switches per decision. `detection` has both directions'
delays, misses and already-correct-at40 counts. The inherited definition requires
an actual switch at/after40; already-correct states are separate, not silently
counted as detections. `mixed_windows` reports errors against injected window RMS
without forcing a binary true-noise label.

`per_seed` retains metrics/detection for each of20 reset/noise units.
`seed_equal_moving_metrics` and `paired_seed_bootstrap` use those20 units, preserving
model/source/profile/estimator pairing. Overlapping windows, trajectories sharing
seeds and three fixed models are not independent policy-training replications.

## Integrity and interruption

The protocol must already be committed, and frozen code/checkpoint hashes must
match. `dataset_manifest.json` hashes every physical trajectory and the replay
file. Audit verifies exact120/480 identities, reserved seeds, new noise seed,
source model/protocol hashes, seed-reconstructed initial physical positions,
and every observed sample/estimate/window label. Historical tracked modifications
cause an audit failure. Hashes and reset reconstruction provide provenance checks;
seed metadata alone is not proof that a trajectory was independently acquired.

One new-directory lock prevents concurrent generate/audit/analyze writers.
Completed replay, manifest, summary and audit files are never overwritten. The
collector can validate/skip completed trajectory files after interruption. A
partial replay file is retained and blocks regeneration; preserve and inspect
it before a separate recovery decision. Do not remove a lock without checking
whether its owner is still running. Existing Stage 1 data must never be copied
into this new directory as independent Final evidence.

Analysis reruns audit before computing results and verifies an existing audit
rather than replacing it. Full audit intentionally recomputes all estimator
windows and can take additional time. No such full audit was run in preparation;
only small synthetic fixtures exercised the implementation. Tests use synthetic
noise indices990000+, not the reserved Final noise indices.

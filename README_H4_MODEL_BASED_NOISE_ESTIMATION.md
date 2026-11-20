# Model-based sensor-noise estimator: implementation only

This branch adds a constant-velocity Kalman observation-noise estimator and a
bounded comparison CLI. No Stage 1 rerun, new physics rollout, grasp-success
Final, training, figures or research report was produced. Prior studies remain
unchanged. This implementation is not a validated controller or permission to
proceed to Stage 2.

## Model and assumptions

Per axis, state is `[position, velocity]`, `F=[[1,dt],[0,1]]`, `H=[1,0]`.
Independent per-step Gaussian acceleration defines `Q=a_std² G Gᵀ`, where
`G=[dt²/2,dt]`. XY share isotropic, independent measurement covariance `R=σ² I`.
Defaults fixed before data comparison: `dt=0.1 s`, acceleration SD `0.05 m/s²`,
initial velocity SD `0.1 m/s`, and a sigma grid from 0 to 30 mm in 0.25 mm steps.
These are engineering assumptions, not calibrated or Final-optimized values.

Each 20-observation window conditions on its first observation. Initial position
covariance is R; velocity starts at zero with the stated prior uncertainty.
For each candidate R, the Kalman filter propagates P and evaluates innovations
with `S=H P_pred Hᵀ+R`. Summed Gaussian innovation negative log likelihood selects
R. The state-uncertainty contribution is therefore explicitly included, rather
than equating raw residual variance with R. A Joseph covariance update preserves
numerical stability. Zero-grid R uses a 1e-12 m² internal numerical floor.

The estimator receives only noisy XY. Truth, injected sigma, noise seed and
success labels are not estimator inputs. Existing position-variance and
second-difference MAD functions are imported unchanged. The CLI reads the existing
JSONL format (`seed`, `observed_xy`, optional other Stage 1 fields); it does not
read historical `windows` estimates or success labels to fit parameters.

Fixed Q is essential: twenty noisy positions generally cannot identify unknown
process noise and measurement noise simultaneously. Accelerations/contact,
incorrect Q, correlated noise, changing sigma within a window, anisotropy and an
incorrect initial prior can invalidate sensor-noise attribution. In particular,
contact errors can increase estimated R. The returned grid boundary flag and
likelihood support are diagnostics, not a validity guarantee or calibrated 95%
confidence interval. Near-zero/boundary estimates need separate interpretation.

## Development and independent diagnostic Final

Existing Stage 1 data (350000–350019, stress 351000–351019) are development-only.
They were not reanalyzed to choose the defaults in this change. Reserve new reset
and noise indices **352000–352019** for a separately acquired independent
estimator diagnostic Final. Those data have not been generated or evaluated.
Do not relabel old observations with new seeds. Generate independent trajectories
and noise, record provenance, and freeze model/configuration and criteria before
opening their diagnostic labels. This reserved set is not a grasp-success Final.

`--role final` rejects seeds outside the reserved range; this is a metadata guard,
not proof of independent acquisition. A future experiment must verify provenance
and completeness separately. Do not tune parameters using that set. No claim of
estimator validity or pass/fail gate is made by this implementation-only CLI.

## Terminal commands

Run related small tests only:

```sh
.venv/bin/python -m unittest test_kalman_noise_estimator test_motion_aware_noise_estimator
```

Run a bounded development comparison of the first four existing sequences:

```sh
.venv/bin/python compare_h4_noise_estimators.py \
  --input bc_results/h4_motion_estimator/diagnostic_sequences.jsonl \
  --limit 4 --role development \
  --output /private/tmp/h4_kalman_development.json
```

The output includes three estimates per window, fixed configuration, sample
counts and Kalman diagnostics. The means summarize estimates, not estimation
accuracy; this CLI does not use truth labels or perform significance tests.
It prints a compact summary and saves detailed windows in JSON. Increase `--limit`
only deliberately; the default is four, not all 480 sequences. Existing output
files are never overwritten.

Show the saved JSON summary without dumping window logs:

```sh
.venv/bin/python -c 'import json; p="/private/tmp/h4_kalman_development.json"; d=json.load(open(p)); print(json.dumps({k:v for k,v in d.items() if k!="window_results"}, indent=2))'
```

Only after independently acquiring the reserved diagnostic set, a corresponding
command can use its new JSONL path with `--role final`; do not run it against the
old development file or use its outcomes to retune the model.

# Python source layout

The executable implementations are grouped under `scripts/`. Run them from the repository root through `scripts/run.py`; for example:

```sh
.venv/bin/python scripts/run.py action_chunking/train_action_chunk_bc.py
```

| Folder | Purpose |
|---|---|
| [`scripts/core/`](scripts/core/) | Shared environments, controllers, estimators, and common code. |
| [`scripts/behavior_cloning/`](scripts/behavior_cloning/) | Demonstration collection, BC/DAgger training and evaluation. |
| [`scripts/residual_sac/`](scripts/residual_sac/) | Residual SAC training, gates, evaluation and analysis. |
| [`scripts/h4_replanning/`](scripts/h4_replanning/) | H4 fixed, adaptive and dynamic replanning. |
| [`scripts/h4_noise_estimation/`](scripts/h4_noise_estimation/) | Motion-aware and Kalman noise-estimation studies. |
| [`scripts/action_chunking/`](scripts/action_chunking/) | Action-chunked BC/SAC training, evaluation and analysis. |
| [`scripts/other_tools/`](scripts/other_tools/) | Older Franka demos, diagnostics and visualization utilities. |

Ten source files remain at the repository root because committed Kalman protocols pin their names and SHA-256 hashes. Three small compatibility shims remain for modules imported by those frozen files. See [`scripts/README.md`](scripts/README.md).

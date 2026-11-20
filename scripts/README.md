# Python source

Executable implementations are grouped in this folder. Run commands from the repository root using `.venv/bin/python scripts/run.py <group>/<file.py> [arguments]`. The runner adds the project and script folders to the import path.

| Folder | Contents |
|---|---|
| `core/` | Shared environments, controllers, and estimators. |
| `behavior_cloning/` | Demonstration collection, BC/DAgger training, and evaluation. |
| `residual_sac/` | Residual SAC training, gates, evaluation, and analysis. |
| `h4_replanning/` | Fixed, adaptive, and dynamic H4 replanning. |
| `h4_noise_estimation/` | Motion-aware and Kalman studies. |
| `action_chunking/` | H1/H4 action-chunked BC and residual SAC. |
| `other_tools/` | Older Franka demos, diagnostics, and visualization. |

Ten root Python files are source files frozen by study protocol hashes. Three root files are small compatibility shims imported by those frozen sources. This exception keeps archived audit evidence verifiable.

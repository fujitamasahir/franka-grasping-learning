# Franka Grasping Project

Franka grasping研究リポジトリの入口です。実験の問い・結果・Python実装・保存データを対応づけています。

## 最初に見る場所

- [実験一覧](EXPERIMENTS.md)：研究の流れと各実験の短い説明。
- [Pythonソース一覧](PYTHON.md)：ソース配置、起動方法、固定名の互換ファイル。
- [`experiments/`](experiments/)：各実験の手法、結果解釈、再現手順。
- [`bc_results/README.md`](bc_results/README.md)：結果データの分類とリンク。
- [`scripts/README.md`](scripts/README.md)：Python実装ディレクトリの案内。

## 実験・コード・レポート・データの対応

コマンドを実行するときはリポジトリルートから `.venv/bin/python scripts/run.py <分類>/<ファイル.py> [引数]` を使います。例: `.venv/bin/python scripts/run.py action_chunking/train_action_chunk_bc.py`. 表のPython名は [`scripts/`](scripts/) 内の実装ファイル名です。「root固定元」と記したファイルはプロトコルが名前を固定しているためルートにあります。

| 段階 | 実験 | Python実装 | 結果レポート | 保存先 |
|---|---|---|---|---|
| BC | BC / DAgger ベースライン: デモ収集、BC学習、Fair dataset比較、一般化評価。 | `collect_demonstrations.py`, `collect_dagger.py`, `prepare_fair_datasets.py`, `train_behavior_cloning.py`, `evaluate_behavior_cloning.py`, `evaluate_generalization.py`, `analyze_fair_experiment.py` | [BC overview](experiments/behavior-cloning/README.md) | [baselines](bc_results/baselines/) |
| Residual SAC | Controlled residual SAC: 固定BC方策に対するサンプリング方式・報酬設計の比較。 | `train_bc_residual_sac.py`, `evaluate_bc_residual_sac.py` | [EXPERIMENTS_SAC.md](experiments/residual-sac/EXPERIMENTS_SAC.md) | [residual_sac](bc_results/residual_sac/) |
| Residual SAC | Selective residual gate: BCが失敗しそうな場合だけSACを適用するgateの評価。 | `analyze_selective_residual_gate.py`, `evaluate_selective_residual_sac.py` | [EXPERIMENTS_SELECTIVE_GATE.md](experiments/residual-sac/EXPERIMENTS_SELECTIVE_GATE.md) | [selective_gate](bc_results/selective_gate/) |
| Residual SAC | Selective gate multi-seed: 複数のBC/SAC seedでselective gateの結果を再評価。 | `analyze_selective_gate_multiseed.py` | [EXPERIMENTS_SELECTIVE_MULTISEED.md](experiments/residual-sac/EXPERIMENTS_SELECTIVE_MULTISEED.md) | [selective_gate](bc_results/selective_gate/) |
| Residual SAC | Policy-specific gate: 個別方策ごとにgateを較正し、独立seedで検証。 | `analyze_policy_specific_gate.py`, `calibrate_policy_specific_gate.py`, `verify_policy_specific_gate_live.py` | [EXPERIMENTS_POLICY_SPECIFIC_GATE.md](experiments/residual-sac/EXPERIMENTS_POLICY_SPECIFIC_GATE.md) | [selective_gate](bc_results/selective_gate/) |
| Residual SAC | SAC training stability: checkpointと学習seedによる性能のばらつきを測定。 | `run_sac_stability_training.py`, `run_sac_stability_evaluations.py`, `freeze_sac_checkpoint_selection.py`, `analyze_sac_stability.py`, `plot_sac_stability.py` | [EXPERIMENTS_SAC_TRAINING_STABILITY.md](experiments/residual-sac/EXPERIMENTS_SAC_TRAINING_STABILITY.md) | [sac_training_stability](bc_results/sac_training_stability/) |
| Residual SAC | SAC regression control: SACによるBC成功の退行を診断し、正則化を比較。 | `run_sac_regression_training.py`, `run_sac_regression_evaluations.py`, `freeze_sac_regression_selection.py`, `analyze_sac_regression_diagnostics.py`, `analyze_sac_regression_results.py`, `plot_sac_regression_results.py` | [EXPERIMENTS_SAC_REGRESSION_CONTROL.md](experiments/residual-sac/EXPERIMENTS_SAC_REGRESSION_CONTROL.md) | [sac_regression_control](bc_results/sac_regression_control/) |
| Residual SAC | SAC regularization replication: 独立学習seedで正則化比較を再現。 | `run_sac_regularization_replication_training.py`, `run_sac_regularization_replication_evaluations.py`, `freeze_sac_regularization_replication_selection.py`, `analyze_sac_regularization_replication.py` | [EXPERIMENTS_SAC_REGULARIZATION_REPLICATION.md](experiments/residual-sac/EXPERIMENTS_SAC_REGULARIZATION_REPLICATION.md) | [sac_regularization_replication](bc_results/sac_regularization_replication/) |
| Residual SAC | Action-chunked BC + SAC: H1/H4 BCとchunk-aware residual SACを比較。 | `train_action_chunk_bc.py`, `train_action_chunk_sac.py`, `evaluate_action_chunking.py`, `freeze_action_chunk_selections.py`, `analyze_action_chunking.py`, `plot_action_chunking.py`, `write_action_chunking_report.py` | [EXPERIMENTS_ACTION_CHUNKED_BC_RESIDUAL_RL.md](experiments/residual-sac/EXPERIMENTS_ACTION_CHUNKED_BC_RESIDUAL_RL.md) | [action_chunking](bc_results/action_chunking/) |
| H4 replanning | Fixed H4 replanning ablation: 固定replanning間隔による性能・推論コストを比較。 | rootの固定元 `evaluate_h4_replanning.py`; `analyze_h4_replanning.py`, `write_h4_replanning_report.py` | [EXPERIMENTS_H4_REPLANNING_ABLATION.md](experiments/h4-replanning/EXPERIMENTS_H4_REPLANNING_ABLATION.md) | [h4_replanning](bc_results/h4_replanning/) |
| H4 replanning | Observation-noise robustness: 観測ノイズ量を変えて固定replanning方策を評価。 | `evaluate_h4_observation_noise.py`, `audit_h4_observation_noise.py`, `analyze_h4_observation_noise.py`, `verify_h4_observation_noise_zero.py` | [EXPERIMENTS_H4_OBSERVATION_NOISE.md](experiments/h4-replanning/EXPERIMENTS_H4_OBSERVATION_NOISE.md) | [observation_noise](bc_results/h4_replanning/observation_noise/) |
| H4 replanning | Uncertainty-adaptive replanning: 観測から推定した不確かさでR1/R4を選択。 | `evaluate_h4_adaptive_replanning.py`, `analyze_h4_adaptive_calibration.py`, `analyze_h4_adaptive_replanning.py`, `audit_h4_adaptive_replanning_smoke.py`, `audit_h4_adaptive_replanning_final.py` | [EXPERIMENTS_H4_UNCERTAINTY_ADAPTIVE_REPLANNING.md](experiments/h4-replanning/EXPERIMENTS_H4_UNCERTAINTY_ADAPTIVE_REPLANNING.md) | [adaptive](bc_results/h4_replanning/adaptive/) |
| H4 replanning | Dynamic-noise replanning: 変化するノイズ下でのonline switchingを評価。報告書記載のstationarity gateで研究を停止。 | rootの固定元 `evaluate_h4_dynamic_replanning.py`; `analyze_h4_dynamic_replanning.py`, `audit_h4_dynamic_replanning.py`, `diagnose_h4_dynamic_switches.py`, `validate_h4_dynamic_legacy.py` | [EXPERIMENTS_H4_DYNAMIC_UNCERTAINTY_REPLANNING.md](experiments/h4-replanning/EXPERIMENTS_H4_DYNAMIC_UNCERTAINTY_REPLANNING.md) | [dynamic](bc_results/h4_replanning/dynamic/) |
| H4 replanning | Motion-aware noise estimator: 動作由来の変動と観測ノイズの分離を検証。preregistered gateを通過せず後段へ進まず。 | rootの固定元 `run_h4_motion_estimator_stage1.py`, `analyze_h4_motion_estimator_stage1.py`; `audit_h4_motion_estimator_stage1.py` | [EXPERIMENTS_H4_MOTION_AWARE_NOISE_ESTIMATION.md](experiments/h4-noise-estimation/EXPERIMENTS_H4_MOTION_AWARE_NOISE_ESTIMATION.md) | [motion_aware](bc_results/h4_noise_estimation/motion_aware/) |
| H4 replanning | Model-based Kalman diagnostic: CV Kalman noise estimatorの実装・比較準備。説明書上、この作業自体は診断Final未実施。 | `compare_h4_noise_estimators.py`;共通推定器はrootの `kalman_noise_estimator.py` | [README_H4_MODEL_BASED_NOISE_ESTIMATION.md](experiments/h4-noise-estimation/README_H4_MODEL_BASED_NOISE_ESTIMATION.md) | 専用の完了済みFinal結果なし。診断用データはmotion-awareフォルダを参照。 |
| H4 replanning | Independent Kalman validation: 独立データ上でestimator診断を生成・監査・解析。 | `validate_h4_kalman_independent.py`;関連監査・解析は`run_h4_motion_estimator_stage1.py`系 | [README_H4_KALMAN_INDEPENDENT_VALIDATION.md](experiments/h4-noise-estimation/README_H4_KALMAN_INDEPENDENT_VALIDATION.md) | [independent_validation](bc_results/h4_noise_estimation/independent_validation/) |
| H4 replanning | Kalman online replanning: Kalman推定を使うonline replanning閉ループ研究。 | `kalman_online_experiment.py`;共通rule/controllerはrootの固定元ファイル | [README_H4_KALMAN_ONLINE_DYNAMIC.md](experiments/h4-noise-estimation/README_H4_KALMAN_ONLINE_DYNAMIC.md) | [kalman_online](bc_results/h4_noise_estimation/kalman_online/) |

## フォルダ案内

### 研究資料・コード・データ

| パス | 内容 |
|---|---|
| [`experiments/behavior-cloning/`](experiments/behavior-cloning/) | BC/DAgger baselineの案内。専用の長文結果レポートはなく、表の保存CSVが結果記録。 |
| [`experiments/residual-sac/`](experiments/residual-sac/) | residual SAC、gate、安定性、回帰、正則化、action chunkingの各レポート。 |
| [`experiments/h4-replanning/`](experiments/h4-replanning/) | 固定、観測ノイズ、adaptive、dynamic replanningのレポート。 |
| [`experiments/h4-noise-estimation/`](experiments/h4-noise-estimation/) | motion-aware、model-based、独立Kalman診断、online Kalmanのレポート。 |
| [`scripts/core/`](scripts/core/) | 共通環境、controller、estimator、再利用コード。 |
| [`scripts/behavior_cloning/`](scripts/behavior_cloning/) | demonstration、BC/DAggerの収集・学習・評価・分析。 |
| [`scripts/residual_sac/`](scripts/residual_sac/) | residual SAC、gate、評価、選択、統計、図表。 |
| [`scripts/action_chunking/`](scripts/action_chunking/) | H1/H4 chunked BC/SACの学習・評価・分析。 |
| [`scripts/h4_replanning/`](scripts/h4_replanning/) | H4 replanningの評価、監査、分析、レポート生成。 |
| [`scripts/h4_noise_estimation/`](scripts/h4_noise_estimation/) | ノイズ推定、Kalman実験、検証。 |
| [`scripts/other_tools/`](scripts/other_tools/) | 初期Frankaデモ、診断、可視化、動画変換など。 |
| [`scripts/tests/`](scripts/tests/) | 過去の共通テスト置き場。現在のテストは各実装フォルダ内の `test_*.py`。 |
| [`bc_results/baselines/`](bc_results/baselines/) | BC/DAgger、fair dataset、generalizationのCSVと要約。 |
| [`bc_results/residual_sac/`](bc_results/residual_sac/) | 初期residual SAC pilotとresidual scale比較。 |
| [`bc_results/selective_gate/`](bc_results/selective_gate/) | selective/policy-specific gateの較正・Final・要約。 |
| [`bc_results/sac_training_stability/`](bc_results/sac_training_stability/) | SAC seed/checkpoint安定性の評価・監査・図表。 |
| [`bc_results/sac_regression_control/`](bc_results/sac_regression_control/) | regression-controlのCSV、要約、図。 |
| [`bc_results/sac_regularization_replication/`](bc_results/sac_regularization_replication/) | 正則化replicationのprotocol、rollout、要約、図。 |
| [`bc_results/action_chunking/`](bc_results/action_chunking/) | chunking研究のprotocol、学習runs、rollout、analysis、figures。 |
| [`bc_results/h4_replanning/`](bc_results/h4_replanning/) | 固定間隔H4 replanningのrollout、監査、分析。 |
| [`bc_results/h4_replanning/observation_noise/`](bc_results/h4_replanning/observation_noise/) | 観測ノイズrobustnessのrollout・summary・figures。 |
| [`bc_results/h4_replanning/adaptive/`](bc_results/h4_replanning/adaptive/) | adaptive replanningのprotocolと成果物。 |
| [`bc_results/h4_replanning/dynamic/`](bc_results/h4_replanning/dynamic/) | dynamic replanningのsmoke・gate・分析資料。 |
| [`bc_results/h4_noise_estimation/`](bc_results/h4_noise_estimation/) | motion-aware/Kalman系protocolと研究グループ。 |
| [`bc_results/h4_noise_estimation/motion_aware/`](bc_results/h4_noise_estimation/motion_aware/) | motion-aware Stage 1データ・audit・summary・figures。 |
| [`bc_results/h4_noise_estimation/independent_validation/`](bc_results/h4_noise_estimation/independent_validation/) | 独立Kalman診断の出力。 |
| [`bc_results/h4_noise_estimation/kalman_online/`](bc_results/h4_noise_estimation/kalman_online/) | online Kalman studyのprotocol、監査、rollout、analysis。 |

### ルート直下のその他フォルダ

| パス | 内容 |
|---|---|
| `assets/` | プロジェクト用の画像・静的アセット。 |
| `demonstrations/` | 学習用expert/DAgger demonstration。 |
| `bc_models/` | BC checkpointとモデル設定。 |
| `best_franka_model/`, `autonomous_best/`, `autonomous_v2_best/`, `bc_residual_sac_best/` | 学習済みbest modelのローカル保存場所。 |
| `autonomous_checkpoints/`, `bc_residual_sac_runs/` | 学習run/checkpointのローカル出力。 |
| `autonomous_eval/`, `autonomous_v2_eval/`, `bc_residual_sac_eval/`, `eval_logs/` | 評価結果・実行ログ。 |
| `autonomous_logs/`, `autonomous_v2_logs/`, `bc_residual_sac_logs/`, `tensorboard_logs/` | 学習/TensorBoardログ。 |
| `mujoco_menagerie/` | MuJoCo Menagerie Franka asset checkout（環境依存）。 |

### ルート直下の個別ファイル

| ファイル | 役割 |
|---|---|
| [`action_chunking.py`](action_chunking.py) | Action chunkingの共通処理。 |
| [`franka_ai_env.py`](franka_ai_env.py), [`franka_autonomous_env.py`](franka_autonomous_env.py) | 初期Franka AI / autonomous学習用環境。 |
| [`analyze_h4_motion_estimator_stage1.py`](analyze_h4_motion_estimator_stage1.py), [`run_h4_motion_estimator_stage1.py`](run_h4_motion_estimator_stage1.py) | Motion-aware estimator Stage 1のprotocol固定ソース。 |
| [`evaluate_h4_replanning.py`](evaluate_h4_replanning.py), [`evaluate_h4_adaptive_replanning.py`](evaluate_h4_adaptive_replanning.py), [`evaluate_h4_observation_noise.py`](evaluate_h4_observation_noise.py), [`evaluate_h4_dynamic_replanning.py`](evaluate_h4_dynamic_replanning.py) | H4評価の互換entry point。内容により一部はprotocol固定ソース。 |
| [`h4_dynamic_controller.py`](h4_dynamic_controller.py), [`h4_kalman_online_rule.py`](h4_kalman_online_rule.py), [`kalman_noise_estimator.py`](kalman_noise_estimator.py), [`motion_aware_noise_estimator.py`](motion_aware_noise_estimator.py) | H4/Kalman研究protocolでファイル名・SHA-256を固定した共通ソース。 |
| [`bc_residual_sac_final.zip`](bc_residual_sac_final.zip), [`franka_autonomous_sac.zip`](franka_autonomous_sac.zip), [`franka_autonomous_v2_sac.zip`](franka_autonomous_v2_sac.zip), [`franka_sac.zip`](franka_sac.zip) | 旧学習済みモデルの圧縮ファイル。 |
| [`franka_ai_comparison.csv`](franka_ai_comparison.csv), [`grasp_results.csv`](grasp_results.csv) | 初期Franka実験の比較・把持結果CSV。 |

## ルート直下の固定ファイル

| ファイル | 役割 |
|---|---|
| [`EXPERIMENTS.md`](EXPERIMENTS.md) | 短い実験一覧と結果への入口。 |
| [`PYTHON.md`](PYTHON.md) | Python配置・起動方法。 |
| [`scripts/run.py`](scripts/run.py) | roleフォルダ内のPythonを実行する共通runner。 |
| [`action_chunking.py`](action_chunking.py), [`franka_autonomous_env.py`](franka_autonomous_env.py), [`franka_ai_env.py`](franka_ai_env.py), [`franka_bc_residual_env.py`](scripts/core/franka_bc_residual_env.py), [`franka_chunk_aware_residual_env.py`](scripts/core/franka_chunk_aware_residual_env.py) | モデル環境・共通コード。プロトコル固定元の一部はルート配置。 |
| [`evaluate_h4_replanning.py`](evaluate_h4_replanning.py), [`evaluate_h4_adaptive_replanning.py`](evaluate_h4_adaptive_replanning.py), [`evaluate_h4_observation_noise.py`](evaluate_h4_observation_noise.py), [`evaluate_h4_dynamic_replanning.py`](evaluate_h4_dynamic_replanning.py) | H4の互換entry point/プロトコル固定元。実装の一部は `scripts/h4_replanning/`。 |
| [`analyze_h4_motion_estimator_stage1.py`](analyze_h4_motion_estimator_stage1.py), [`run_h4_motion_estimator_stage1.py`](run_h4_motion_estimator_stage1.py), [`h4_dynamic_controller.py`](h4_dynamic_controller.py), [`h4_kalman_online_rule.py`](h4_kalman_online_rule.py), [`kalman_noise_estimator.py`](kalman_noise_estimator.py), [`motion_aware_noise_estimator.py`](motion_aware_noise_estimator.py) | Kalman protocolがファイル名・SHAを固定する研究ソース。 |
| [`h4_kalman_online_rule.py`](h4_kalman_online_rule.py) | Kalman online controller rule。 |
| `*.zip` model files and `franka_ai_comparison.csv`, `grasp_results.csv` | ルートに残る旧実験のモデル/結果ファイル。ファイル名を保持。 |

`.venv/`, `.git/`, `__pycache__/` は環境・Git・Python cacheです。学習ログ、checkpoint、動画、MuJoCo assetの一部はgitignore対象またはローカル依存です。各 `README.md` は該当フォルダのファイル命名と役割を説明します。

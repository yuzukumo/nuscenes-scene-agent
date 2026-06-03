# Benchmark Snapshot Notes

Large generated outputs are kept local by default. Local runs nonetheless provide a compact summary of system behavior.

## Benchmark Scale

The small case counts in the committed benchmark files are sampling choices, not dataset limits. The pipeline first mines and validates candidate risk cases, then exports benchmark slices with explicit caps so that each case can be inspected and reused across scenario mining, perception, BEV occupancy, and world-model evaluation.

| Source | Candidate or Anchor Count | Exported Benchmark Count | Notes |
| --- | ---: | ---: | --- |
| `nuScenes` trainval case library | `33` unique cases, `29` unique passed cases | configurable | produced by natural-language mining and deterministic validation |
| `nuScenes` scenario-mining benchmark | `24` anchors | `48` queries | two reference-aware query variants per anchor |
| `nuScenes` perception, BEV occupancy, and world-model slices | `24` anchors | `24` cases per layer | derived from the scenario-mining anchors |
| `nuPlan` mini replay scan | `243` candidate anchors | `16` replay cases | limited by `max_cases` for balanced inspection |
| `nuPlan` cross-split replay sweep | `1556` candidate anchors | `112` replay cases | controlled by per-study caps in the sweep config |

The `nuScenes` labels in these benchmark files are automatically mined from validated cases. They should be described as weak-supervised or auto-mined reference anchors, not as independent human annotations. Increasing `max_cases` expands the benchmark up to the number of validated anchors available in the source case library; increasing coverage further requires mining additional queries or adding new failure-derived scenarios.

Structured entry point for the default `nuScenes` benchmark suite:

```bash
python -m nusc_scene_agent run-experiment-config \
  --config configs/risk_benchmark_suite.yaml
```

This command generates:

- `benchmarks/trainval_scenario_mining_v1.yaml`
- `benchmarks/trainval_perception_slices_v1.json`
- `benchmarks/trainval_world_model_slices_v1.json`
- `benchmarks/trainval_bev_occupancy_slices_v1.json`
- `outputs/trainval_perception_proxy_study_v1`
- `outputs/trainval_world_model_proxy_study_v1`
- `outputs/trainval_bev_occupancy_proxy_study_v1`

## Metric and Evidence Boundaries

Scenario-mining metrics evaluate retrieval and grounding: `Pass@K`, scene match, actor match, reference-case match, event-window IoU, and peak-sample error.

Perception-slice metrics evaluate short-window actor coverage: anchor recall, full-track success, event recall, contiguous temporal coverage, center error, and first-match lag.

Sparse BEV occupancy metrics evaluate actor-center occupancy rather than dense semantic occupancy: occupancy IoU, primary actor recall, context recall, anchor-frame IoU, and a composite risk-fidelity score.

World-model metrics evaluate future trajectory and sparse future occupancy: ADE, FDE, `MinADE@K`, `MinFDE@K`, `MissRate@K`, occupancy IoU, closest-approach distance and time errors, and risk fidelity.

`nuPlan` replay-regression metrics compare predicted ego rollouts with logged replay windows: ego ADE/FDE, minimum-distance error, minimum-TTC error, red-light context recall, comfort-target errors, collision-proxy mismatch, and risk fidelity.

Proxy profiles are controlled perturbations for sensitivity analysis. They validate that the benchmark detects delayed initialization, sparse tracking, missing context occupancy, and simple rollout errors. They are not competitive model baselines. Real or external baselines are handled separately through official `nuScenes` adapters, physics forecast baselines, `ContextVAE`, and user-provided prediction files.

## Core Trainval Suite

Dataset:

- `nuScenes v1.0-trainval`
- `map expansion`

Snapshot:

| Metric | Value |
| --- | --- |
| Queries | 16 |
| Pass@1 | 16/16 |
| Pass@K | 16/16 |
| Selected cases | 36 |
| Unique cases | 33 |
| Unique passed cases | 29 |
| Mean best validation score | 89.26 |

Behavior coverage:

- `crossing`
- `stopped_lead`
- `oncoming`
- `cut_in`
- close-proximity queries without explicit behavior labels

## Language Robustness Comparison

The `trainval_language_stress_v1` suite measures phrasing robustness rather than replaying structured benchmark labels.

Snapshot:

| Profile | Pass@1 | Mean Best Score |
| --- | --- | --- |
| `rule_only` | 16/16 | 90.64 |
| `llm_planner` | 15/16 | 86.29 |
| `hybrid_agent` | 16/16 | 91.34 |

Additional observations:

- queries: `16`
- planner disagreement count: `12`

## Scenario-Mining Benchmark

The default scenario-mining benchmark is `benchmarks/trainval_scenario_mining_v1.yaml`. It is generated from the validated trainval case library and stores reference supervision for scene identity, actor identity, event-window range, and event peak sample.

| Quantity | Value |
| --- | --- |
| Anchor cases | 24 |
| Query variants | 48 |
| Variants per anchor | canonical query and paraphrase query |
| Source case library | `outputs/trainval_suite_llm_hybrid_en_v1/case_library_enriched.json` |

Behavior distribution:

| Behavior | Anchors | Queries |
| --- | ---: | ---: |
| `stopped_lead` | 7 | 14 |
| `crossing` | 5 | 10 |
| `proximity` | 5 | 10 |
| `oncoming` | 4 | 8 |
| `cut_in` | 3 | 6 |

The benchmark is intended for reference-aware retrieval evaluation. A valid result should retrieve the expected scene, ground the expected actor, and localize the event window, not only return a plausible risky scene.

## Scenario-Conditioned Perception Slice Evaluation

The repository also supports a scenario-conditioned perception evaluation layer derived from the validated scenario-mining anchors. This layer exports short event-window actor tracks in ego coordinates and evaluates external perception outputs on these mined slices.

The same layer also exposes:

- benchmark-side `risk_facets` for distance band, TTC band, visibility band, map relation, and occlusion proxy
- an adapter from official `nuScenes` detection or tracking JSON into the local evaluation schema
- per-profile `CSV` exports for case-level perception metrics

Local snapshot on `benchmarks/trainval_perception_slices_v1.json`:

| Profile | Cases | Anchor Recall | Full Track | Mean Event Recall | Mean Contiguous Coverage | Mean Center Error |
| --- | --- | --- | --- | --- | --- | --- |
| `oracle_tracking` | 24 | 24/24 | 24/24 | 1.000 | 1.000 | 0.000 |
| `delayed_track` | 24 | 23/24 | 0/24 | 0.719 | 0.719 | 0.403 |
| `crossing_sparse_track` | 24 | 22/24 | 19/24 | 0.901 | 0.823 | 0.126 |

Behavior-sensitive findings:

- the slice set contains `24` anchors across `stopped_lead`, `crossing`, `oncoming`, `cut_in`, and close-proximity interactions
- `delayed_track` preserves most anchor matches but fails full-window tracking on all cases, separating initialization delay from localization accuracy
- `crossing_sparse_track` keeps high mean event recall while reducing full-track success, which isolates temporal sparsity effects

Risk-conditioned findings:

- all slices fall into `urgent_ttc`, so the split is primarily discriminated by geometry and map relation rather than TTC strata
- `crossing_emergence` is the most sensitive occlusion group for sparse tracking, with mean event recall `0.536`
- `near_range` cases regress under sparse tracking, while `critical_range` cases remain stable in this proxy setting
- `shared_lane_supported` cases are robust to sparse tracking but remain sensitive to delayed initialization

The detailed local exports are:

- `outputs/trainval_perception_proxy_study_v1/perception_comparison_summary.md`
- `outputs/trainval_perception_proxy_study_v1/perception_comparison_summary.html`
- `outputs/trainval_perception_proxy_study_v1/perception_leaderboard.csv`
- `outputs/trainval_perception_proxy_study_v1/oracle_tracking/perception_metrics_summary.md`
- `outputs/trainval_perception_proxy_study_v1/crossing_sparse_track/perception_metrics_summary.md`
- `outputs/trainval_perception_proxy_study_v1/delayed_track/perception_metrics_summary.md`
- `outputs/trainval_perception_proxy_study_v1/oracle_tracking/perception_case_metrics.csv`

## Risk-Conditioned BEV Occupancy Slice Evaluation

The repository also derives a sparse BEV occupancy benchmark from the perception-slice cases and the indexed `nuScenes` actor table. Each frame stores primary risk actor cells, context actor cells, and union occupancy cells on a fixed ego-frame grid.

Local snapshot on `benchmarks/trainval_bev_occupancy_slices_v1.json`:

| Profile | Cases | Mean Occupancy IoU | Primary Recall | Context Recall | Anchor IoU | Risk Fidelity |
| --- | --- | --- | --- | --- | --- | --- |
| `oracle_occupancy` | 24 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| `context_drop_occupancy` | 24 | 0.553 | 1.000 | 0.502 | 0.554 | 0.699 |
| `risk_actor_only` | 24 | 0.105 | 1.000 | 0.003 | 0.104 | 0.398 |

Observed behavior:

- `risk_actor_only` preserves primary actor recall but fails to cover surrounding BEV context
- `context_drop_occupancy` separates partial context coverage from complete occupancy coverage
- this layer evaluates sparse actor-center occupancy, not dense semantic occupancy labels

The detailed local exports are:

- `benchmarks/trainval_bev_occupancy_slices_v1.json`
- `outputs/trainval_bev_occupancy_proxy_study_v1/bev_occupancy_comparison_summary.md`
- `outputs/trainval_bev_occupancy_proxy_study_v1/bev_occupancy_leaderboard.csv`
- `outputs/trainval_bev_occupancy_proxy_study_v1/oracle_occupancy/bev_occupancy_metrics_summary.md`
- `outputs/trainval_bev_occupancy_proxy_study_v1/risk_actor_only/bev_occupancy_case_metrics.csv`

## Scenario-Conditioned World-Model Evaluation

The repository supports a scenario-conditioned world-model benchmark derived from the perception-slice layer. Each case exposes an observed history, a short future horizon for the primary risk actor, sparse future occupancy supervision, and the inherited risk facets from the validated scenario anchor.

The benchmark metadata carries named challenge tracks such as `challenge/crossing_emergence`, `challenge/large_lead_occluder`, `challenge/lateral_merge`, and `challenge/opposite_direction_conflict`.

For multi-modal forecasts, the evaluation export also reports benchmark-aligned forecast metrics:

- `MinADE@1`, `MinADE@5`, `MinADE@10`
- `MinFDE@1`, `MinFDE@5`, `MinFDE@10`
- `MissRate@1`, `MissRate@5`, `MissRate@10`

Local snapshot on `benchmarks/trainval_world_model_slices_v1.json`:

| Profile | Cases | Full Horizon | Mean Horizon Recall | Mean ADE | Mean FDE | Mean Occupancy IoU | Mean Risk Fidelity |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `oracle_rollout` | 24 | 24/24 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 |
| `risk_underreach_rollout` | 24 | 24/24 | 1.000 | 0.725 | 0.921 | 0.975 | 0.892 |
| `kinematic_rollout` | 24 | 24/24 | 1.000 | 1.010 | 1.347 | 0.958 | 0.869 |

Observed behavior:

- all `24` cases retain full future-horizon coverage in the proxy study
- `challenge/lateral_merge` is the lowest-performing challenge track for both non-oracle profiles
- `risk_underreach_rollout` and `kinematic_rollout` separate trajectory error from occupancy alignment error
- evaluation exports include challenge-track summaries in addition to behavior and risk-facet summaries

External-adaptation support:

- compact external rollouts can be adapted from benchmark-keyed `xy_ego` arrays with optional occupancy cells
- the adapter can rasterize actor occupancy from trajectory-only predictions when occupancy grids are not provided
- `nuScenes prediction challenge` style multi-modal forecasts can be adapted from `(instance, sample, prediction, probabilities)` records
- the forecast adapter supports `top_probability`, `oracle_ade`, and `oracle_fde` mode selection policies

Replay export snapshot:

- `outputs/trainval_world_model_replay_v1/replay_manifest.json`
- replay files in newline-delimited `JSONL`
- metadata, history-track, future-track, and future-occupancy topics

The detailed local exports are:

- `outputs/trainval_world_model_proxy_study_v1/world_model_comparison_summary.md`
- `outputs/trainval_world_model_proxy_study_v1/world_model_comparison_summary.html`
- `outputs/trainval_world_model_proxy_study_v1/world_model_leaderboard.csv`
- `outputs/trainval_world_model_proxy_study_v1/oracle_rollout/world_model_metrics_summary.md`
- `outputs/trainval_world_model_proxy_study_v1/kinematic_rollout/world_model_metrics_summary.md`
- `outputs/trainval_world_model_proxy_study_v1/risk_underreach_rollout/world_model_metrics_summary.md`

## nuPlan Replay-Regression Evaluation

The repository includes a compact `nuPlan` replay-regression layer that reads SQLite logs directly. It uses agent-specific scenario tags as anchors, exports ego replay windows, and evaluates predicted ego rollouts against logged risk interactions.

Local snapshot on `outputs/nuplan_mini_replay_study_v2`:

| Profile | Cases | Full Horizon | Mean Ego ADE | Mean Ego FDE | Mean Min-Distance Error | Mean Min-TTC Error | Mean Risk Fidelity |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `logged_ego` | 16 | 16/16 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |
| `constant_velocity` | 16 | 16/16 | 14.077 | 28.039 | 5.289 | 0.602 | 0.772 |
| `stopped` | 16 | 16/16 | 9.854 | 19.651 | 5.197 | 3.121 | 0.743 |

The local run scans all `64` mini SQLite logs, collects `243` candidate anchors, and exports `16` balanced replay cases. The selected cases cover `vru_interaction`, `high_speed_interaction`, `large_vehicle_interaction`, and `static_obstacle_context` families across `las_vegas`, `sg-one-north`, `us-ma-boston`, and `us-pa-pittsburgh-hazelwood`.

Case distribution:

| Dimension | Distribution |
| --- | --- |
| Scenario family | `vru_interaction: 5`, `high_speed_interaction: 2`, `large_vehicle_interaction: 7`, `static_obstacle_context: 2` |
| Difficulty | `easy: 1`, `medium: 3`, `hard: 12` |
| Scenario tag | `near_pedestrian_on_crosswalk: 4`, `near_pedestrian_at_pickup_dropoff: 1`, `near_high_speed_vehicle: 2`, `near_long_vehicle: 7`, `near_trafficcone_on_driveable: 2` |

The detailed local exports are:

- `outputs/nuplan_mini_replay_study_v2/nuplan_replay_benchmark.json`
- `outputs/nuplan_mini_replay_study_v2/unified_risk_cases.json`
- `outputs/nuplan_mini_replay_study_v2/artifact_manifest.json`
- `outputs/nuplan_mini_replay_study_v2/experiment_result.json`
- `outputs/nuplan_mini_replay_study_v2/nuplan_replay_study_summary.md`
- `outputs/nuplan_mini_replay_study_v2/comparison/nuplan_replay_comparison_summary.md`
- `outputs/nuplan_mini_replay_study_v2/comparison/nuplan_replay_leaderboard.csv`
- `outputs/nuplan_mini_replay_study_v2/case_studies/nuplan_replay_case_studies.png`
- `outputs/nuplan_mini_replay_study_v2/case_studies/nuplan_replay_case_studies.md`
- `outputs/nuplan_mini_replay_study_v2/case_studies/nuplan_replay_case_studies.html`
- `outputs/nuplan_mini_replay_study_v2/case_studies/nuplan_replay_case_studies.json`
- `outputs/nuplan_mini_replay_study_v2/logged_ego_evaluation/nuplan_replay_metrics_summary.md`
- `outputs/nuplan_mini_replay_study_v2/constant_velocity_evaluation/nuplan_replay_metrics_summary.md`
- `outputs/nuplan_mini_replay_study_v2/stopped_evaluation/nuplan_replay_metrics_summary.md`

The structured experiment entry point is `configs/nuplan_replay_mini.yaml`. The local artifact manifest records `28` generated artifacts, all present in the latest run.

### nuPlan Cross-Split Replay Sweep

The structured sweep entry point is `configs/nuplan_replay_sweep_medium.yaml`. The latest local run covers `5` studies, scans `576` SQLite logs, collects `1556` candidate anchors, and exports `112` replay cases.

Study coverage:

| Study | DBs | Candidates | Cases |
| --- | --- | --- | --- |
| `mini` | 64 | 243 | 16 |
| `val_sample` | 128 | 429 | 24 |
| `train_boston_sample` | 128 | 340 | 24 |
| `train_pittsburgh_sample` | 128 | 294 | 24 |
| `train_singapore_sample` | 128 | 250 | 24 |

Overall profile leaderboard:

| Profile | Cases | Full Horizon | Mean Ego ADE | Mean Ego FDE | Mean Min-Distance Error | Mean Min-TTC Error | Mean Risk Fidelity |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `logged_ego` | 112 | 112/112 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |
| `constant_velocity` | 112 | 112/112 | 12.774 | 25.640 | 4.269 | 0.304 | 0.808 |
| `stopped` | 112 | 112/112 | 10.666 | 21.372 | 3.772 | 2.045 | 0.796 |

Most frequent failure tags:

| Failure Tag | Count |
| --- | --- |
| `risk_distance_error` | 115 |
| `ttc_error` | 17 |
| `collision_proxy_mismatch` | 10 |

The detailed sweep exports are:

- `outputs/nuplan_replay_sweep_v1/nuplan_replay_sweep_summary.md`
- `outputs/nuplan_replay_sweep_v1/nuplan_replay_sweep_leaderboard.csv`
- `outputs/nuplan_replay_sweep_v1/nuplan_replay_sweep_family_matrix.csv`
- `outputs/nuplan_replay_sweep_v1/nuplan_replay_sweep_failure_taxonomy.csv`
- `outputs/nuplan_replay_sweep_v1/artifact_manifest.json`

### Official Physics Baselines On Benchmark Anchors

The repository can also run the official `nuScenes` physics forecast baselines directly on the world-model benchmark anchors instead of relying on the prediction-challenge split files.

Local snapshot on `outputs/nuscenes_forecast_baselines_eval`:

| Profile | Cases | Full Horizon | Mean ADE | Mean FDE | Mean MinADE@1 | Mean MissRate@5 | Mean Occupancy IoU | Mean Risk Fidelity |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `physics_oracle` | 8 | 8/8 | 0.212 | 0.214 | 0.212 | 0.000 | 0.175 | 0.840 |
| `cv_heading` | 8 | 8/8 | 0.288 | 0.394 | 0.288 | 0.125 | 0.175 | 0.823 |

Observed comparison:

- `physics_oracle` improves trajectory accuracy and closest-approach timing relative to `cv_heading`
- `physics_oracle` also improves the newly exported forecast metrics, driving `MissRate@5` to `0.000`
- both baselines retain low occupancy IoU because they only predict the primary actor trajectory while the occupancy target also includes surrounding context actors
- `challenge/shared_lane_lead` is the clearest separation point, with risk fidelity `0.834` for `physics_oracle` and `0.799` for `cv_heading`

The detailed local exports are:

- `outputs/nuscenes_forecast_baselines_eval/predictions/baseline_manifest.json`
- `outputs/nuscenes_forecast_baselines_eval/cv_heading/world_model_metrics_summary.md`
- `outputs/nuscenes_forecast_baselines_eval/physics_oracle/world_model_metrics_summary.md`
- `outputs/nuscenes_forecast_baselines_eval/comparison/world_model_comparison_summary.md`
- `assets/world_model_case_studies/world_model_case_studies.png`

### Real Multi-Modal Paper Baseline Example

The repository also supports an external multi-modal forecasting baseline through `ContextVAE` (IEEE RA-L 2023). Because this family of models expects a standard `12`-step forecast horizon, the integration first derives a forecast-compatible subset from the scenario-conditioned world-model benchmark, then evaluates the external model and the local physics baselines on the same subset.

Local snapshot on `outputs/contextvae_world_model_study_v1`:

| Profile | Cases | Full Horizon | Mean ADE | Mean MinADE@5 | Mean Occupancy IoU | Mean Risk Fidelity |
| --- | --- | --- | --- | --- | --- | --- |
| `physics_oracle` | 7 | 7/7 | 0.135 | 0.135 | 0.197 | 0.860 |
| `cv_heading` | 7 | 7/7 | 0.139 | 0.139 | 0.197 | 0.859 |
| `contextvae` | 7 | 7/7 | 0.280 | 0.207 | 0.181 | 0.841 |

Observed comparison:

- `7/8` world-model slices remain forecast-compatible under the standard `12`-step prediction requirement
- `ContextVAE` preserves full-horizon coverage on the compatible subset and improves from `0.280` `MinADE@1` to `0.207` `MinADE@5`
- the largest residual gap remains occupancy agreement, which is expected because the benchmark occupancy target includes surrounding context actors while the external baseline predicts only the primary actor trajectory

### External Official Baseline Example

The repository supports direct evaluation of official `nuScenes` prediction JSON. Because official files such as `results_val_megvii.json` only cover the validation split, the recommended path is to filter the perception benchmark to the covered subset before scoring.

Local example on:

- prediction file: `external_predictions/nuscenes_official/megvii_tracking/unpacked/results_val_megvii.json`
- aligned output: `outputs/external_megvii_tracking_eval_covered`
- coverage mode: `full_window`

Coverage summary:

| Quantity | Value |
| --- | --- |
| Original slices | 8 |
| Anchor-covered slices | 2 |
| Full-window-covered slices | 2 |

Aligned result snapshot:

| Profile | Cases | Anchor Recall | Full Track | Mean Event Recall | Mean Contiguous Coverage | Mean Center Error | Mean First Match Lag |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `megvii_tracking_val_covered` | 2 | 1/2 | 1/2 | 0.500 | 0.500 | 0.271 | 2.50 |

Observed behavior on the aligned subset:

- the covered subset contains one `stopped_lead` slice and one `oncoming` slice
- the `stopped_lead` slice is tracked successfully with a center error of `0.271 m`
- the `oncoming` slice is missed entirely, so this external example primarily illustrates adapter and coverage-alignment behavior

### External Official Tracking Comparison

The repository also supports direct comparison across multiple external evaluation outputs.

Local comparison on the shared covered subset:

| Profile | Cases | Anchor Recall | Full Track | Mean Event Recall | Mean Contiguous Coverage | Mean Center Error |
| --- | --- | --- | --- | --- | --- | --- |
| `pointpillars_tracking_val_covered` | 2 | 1/2 | 1/2 | 0.600 | 0.600 | 0.323 |
| `megvii_tracking_val_covered` | 2 | 1/2 | 1/2 | 0.500 | 0.500 | 0.271 |

Observed comparison:

- both methods solve the `stopped_lead` slice on the aligned subset
- `PointPillars` achieves partial event recall on the `oncoming` slice (`0.200`), whereas `Megvii` misses it completely
- `Megvii` remains slightly tighter on center localization for the one successful slice

The detailed local comparison export is:

- `outputs/external_official_tracking_comparison/perception_comparison_summary.md`

## Agentic Mining Extensions

### Structural Multimodal Retrieval

Local snapshot on `outputs/multimodal_retrieval_demo_v1`:

| Quantity | Value |
| --- | --- |
| Query | `pedestrian crossing close in front of ego with dense surrounding traffic context` |
| Candidate count | 10 |
| Top candidate | `scene-0947`, pedestrian, `5.566 m`, `TTC 0.003 s` |
| Top fused score | `0.874` |

The reranked query pipeline also writes validated case reports and evidence figures under:

- `outputs/multimodal_query_demo_v1/pedestrian_crossing_close_in_front_of_ego_with_dense_surrounding_traffic_context`

### Learned Scene Reranking

Local snapshot on `outputs/learned_retriever_trainval_large_v2`:

| Quantity | Value |
| --- | --- |
| Training source | weak labels mined from `artifacts/index/v1.0-trainval.sqlite` |
| Scenario families | `vru_crossing_front`, `stopped_lead_vehicle`, `lateral_cut_in`, `oncoming_vehicle` |
| Total training groups | 4,000 |
| Train groups | 2,779 |
| Scene-held-out validation groups | 1,221 |
| Train Recall@1 | 1.000 |
| Scene-held-out Recall@1 | 1.000 |
| Scene-held-out mean rank | 1.000 |
| Model | `query_scene_pairwise_mlp_v1` |

Weak-supervised family pools:

| Family | Positives | Negatives | Groups |
| --- | ---: | ---: | ---: |
| `vru_crossing_front` | 1,000 | 1,502 | 1,000 |
| `stopped_lead_vehicle` | 1,000 | 4,523 | 1,000 |
| `lateral_cut_in` | 1,000 | 3,441 | 1,000 |
| `oncoming_vehicle` | 1,000 | 24,000 | 1,000 |

The learned scorer is a post-retrieval reranker trained from trainval weak labels and near-miss negatives. The weak-supervised validation split is scene-level, so validation scenes are excluded from training.

The trained checkpoint can be evaluated on the current reference-aware scenario-mining benchmark with:

```bash
python -m nusc_scene_agent benchmark \
  --config benchmarks/trainval_scenario_mining_v1.yaml \
  --db artifacts/index/v1.0-trainval.sqlite \
  --output outputs/trainval_scenario_mining_v1_learned \
  --query-mode rule \
  --rerank-mode learned \
  --learned-reranker-checkpoint outputs/learned_retriever_trainval_large_v2/learned_retriever.pt
```

Demo outputs:

- `outputs/learned_retriever_trainval_large_v2/training_report.md`
- `outputs/learned_retrieval_trainval_large_demo_v1/learned_retrieval_report.md`
- `outputs/learned_query_trainval_large_demo_v1/pedestrian_crossing_close_in_front_of_ego_with_dense_surrounding_traffic_context`

### Model-in-the-loop Failure Mining

Local snapshot on `outputs/model_in_the_loop_failure_mining_v1`:

| Quantity | Value |
| --- | --- |
| Source files | 25 |
| Failure records | 305 |
| Failure clusters | 56 |
| Benchmark update queries | 24 |

Top failure clusters:

| Source | Failure | Context | Actor/Scenario | Count |
| --- | --- | --- | --- | ---: |
| `nuplan_replay` | `risk_distance_error` | `large_vehicle_interaction` | `near_long_vehicle` | 42 |
| `nuplan_replay` | `risk_distance_error` | `vru_interaction` | `near_pedestrian_on_crosswalk` | 30 |
| `nuplan_replay` | `risk_distance_error` | `static_obstacle_context` | `near_trafficcone_on_driveable` | 25 |

The generated update queries are written to:

- `outputs/model_in_the_loop_failure_mining_v1/failure_update_queries.yaml`
- `outputs/model_in_the_loop_failure_mining_v1/failure_mining_report.md`
- `outputs/model_in_the_loop_failure_mining_v1/failure_records.csv`

## Interpretation

Interpretation:

- case-level pass metrics can hide reference-scene and actor-grounding errors
- planner improvements should be measured against scene-level and event-level supervision, not only retrieval score
- deterministic parsing remains a strong baseline on tightly specified scenario-mining queries
- local-model and `hybrid` profiles remain informative because they expose planner disagreement and paraphrase sensitivity that the benchmark can quantify
- mined scenario anchors can also support perception-side evaluation without converting the repository into a full detector-training stack

Taken together, the repository supports both retrieval experiments and reference-aware benchmark construction for planner robustness studies.

## Counterfactual Benchmark Track

The repository includes a counterfactual benchmark generation path.

Given a validated case library, it can generate:

- positive canonical queries
- positive paraphrase queries
- actor-swap negatives
- behavior-swap negatives

These generated benchmarks carry explicit `reference_case_keys`, and when reference event windows are available they can also support localization-aware metrics such as event-window overlap.

## Scenario Mining Track

The repository supports a reference-aware scenario mining benchmark derived from validated case libraries.

This track keeps explicit reference supervision for:

- scene identity
- actor identity
- event start and end samples
- event peak sample

As a result, benchmark metrics can evaluate:

- scene retrieval correctness
- actor grounding correctness
- event-window overlap
- peak localization error

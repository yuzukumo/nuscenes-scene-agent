# Benchmark Snapshot Notes

Large generated outputs are kept local by default. Local runs nonetheless provide a compact summary of system behavior.

## Core Trainval Suite

Dataset:

- `nuScenes v1.0-trainval`
- `map expansion`

Current snapshot:

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

Current snapshot:

| Profile | Pass@1 | Mean Best Score |
| --- | --- | --- |
| `rule_only` | 16/16 | 90.64 |
| `llm_planner` | 15/16 | 86.29 |
| `hybrid_agent` | 16/16 | 91.34 |

Additional observations:

- queries: `16`
- planner disagreement count: `12`

## Scenario Mining Comparison

The reference-aware scenario mining suite evaluates whether a profile retrieves not only a risky case, but also the correct scene anchor, actor identity, and event window.

Local snapshot on `benchmarks/trainval_scenario_mining_v1.yaml`:

| Profile | Pass@1 | Mean Best Score | Scene@1 | Actor@1 | Reference@1 | Scenario Group Success@1 | Mean Event IoU | Mean Peak Error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `rule_only` | 16/16 | 92.42 | 15/16 | 15/16 | 15/16 | 7/8 | 1.000 | 0.00 |
| `llm_planner` | 16/16 | 90.67 | 12/16 | 12/16 | 12/16 | 6/8 | 0.963 | 0.00 |
| `hybrid_agent` | 16/16 | 92.67 | 13/16 | 13/16 | 13/16 | 6/8 | 1.000 | 0.00 |

Group-level scenario consistency:

| Profile | Groups | Scene@1 | Actor@1 | Reference@1 | Mean Event IoU | Mean Peak Error |
| --- | --- | --- | --- | --- | --- | --- |
| `rule_only` | 8 | 7/8 | 7/8 | 7/8 | 0.938 | 1.12 |
| `llm_planner` | 8 | 6/8 | 6/8 | 6/8 | 0.751 | 1.62 |
| `hybrid_agent` | 8 | 6/8 | 6/8 | 6/8 | 0.812 | 2.25 |

The detailed local comparison is exported to:

- `outputs/trainval_scenario_profile_comparison_v1/benchmark_profile_comparison_summary.md`
- `outputs/trainval_scenario_profile_comparison_v1/benchmark_leaderboard.html`
- `outputs/trainval_scenario_profile_comparison_v1/benchmark_leaderboard.csv`
- `outputs/trainval_scenario_profile_comparison_v1/behavior_error_analysis.html`
- `outputs/trainval_scenario_profile_comparison_v1/behavior_error_analysis.csv`
- `outputs/trainval_scenario_profile_comparison_v1/comparison_browser.html`
- `outputs/trainval_scenario_mining_v1_rule/query_gallery.html`
- `outputs/trainval_scenario_mining_v1_llm/query_gallery.html`
- `outputs/trainval_scenario_mining_v1_hybrid/query_gallery.html`

Observed failure patterns:

- all three profiles achieve `16/16` case-level `Pass@1`, so plain risky-scene retrieval alone is not discriminative enough
- `rule_only` remains the strongest profile once evaluation requires the correct reference scene and actor
- `llm_planner` underperforms most clearly on crossing anchors and loses localization quality on paraphrased groups
- `hybrid_agent` recovers some query-level grounding relative to `llm_planner`, but still trails the deterministic baseline on group-level consistency
- planner disagreement appears on `11/16` queries in the scenario-mining comparison, indicating that the benchmark captures planner robustness rather than retrieval recall alone

## Hybrid Ablation

The repository includes a module-level ablation track over the hybrid scenario-mining pipeline.

Local snapshot on `outputs/trainval_hybrid_ablation_v1`:

| Variant | Pass@1 | Mean Best Score | Scene@1 | Actor@1 | Reference@1 | Scenario Group Success@1 | Mean Event IoU | Mean Peak Error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `full_system` | 16/16 | 92.67 | 13/16 | 13/16 | 13/16 | 6/8 | 1.000 | 0.00 |
| `no_rerank` | 16/16 | 92.67 | 13/16 | 13/16 | 13/16 | 6/8 | 1.000 | 0.00 |
| `no_map_context` | 12/16 | 79.97 | 7/16 | 7/16 | 7/16 | 3/8 | 0.636 | 7.64 |
| `no_event_localization` | 16/16 | 92.67 | 13/16 | 13/16 | 13/16 | 6/8 | 0.000 | 0.00 |

The detailed local ablation exports are:

- `outputs/trainval_hybrid_ablation_v1/benchmark_profile_comparison_summary.md`
- `outputs/trainval_hybrid_ablation_v1/benchmark_leaderboard.html`
- `outputs/trainval_hybrid_ablation_v1/behavior_error_analysis.html`
- `outputs/trainval_hybrid_ablation_v1/comparison_browser.html`
- `outputs/trainval_hybrid_ablation_v1/ablation_manifest.json`

Observed ablation conclusions:

- removing `LLM` reranking does not change the current scenario-mining ranking on this benchmark
- removing map context causes the largest regression, dropping `Pass@1` from `16/16` to `12/16`
- the map ablation is especially damaging on `crossing` and `stopped_lead`, where lane and crosswalk support stabilize the final match
- disabling event localization preserves retrieval outcomes but collapses localization-aware metrics to zero, which is expected and validates that the benchmark can isolate localization contributions

## Scenario-Conditioned Perception Slice Evaluation

The repository also supports a scenario-conditioned perception evaluation layer derived from the validated scenario-mining anchors. This layer exports short event-window actor tracks in ego coordinates and evaluates external perception outputs on these mined slices.

The same layer also exposes:

- benchmark-side `risk_facets` for distance band, TTC band, visibility band, map relation, and occlusion proxy
- an adapter from official `nuScenes` detection or tracking JSON into the local evaluation schema
- per-profile `CSV` exports for case-level perception metrics

Local snapshot on `benchmarks/trainval_perception_slices_v1.json`:

| Profile | Cases | Anchor Recall | Full Track | Mean Event Recall | Mean Contiguous Coverage | Mean Center Error | Mean First Match Lag |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `oracle_tracking` | 8 | 8/8 | 8/8 | 1.000 | 1.000 | 0.000 | 0.00 |
| `crossing_sparse_track` | 8 | 8/8 | 6/8 | 0.884 | 0.789 | 0.151 | 0.00 |
| `delayed_track` | 8 | 8/8 | 0/8 | 0.691 | 0.691 | 0.403 | 1.62 |

Behavior-sensitive findings:

- the current slice set contains `8` anchors spanning `stopped_lead`, `crossing`, `oncoming`, `cut_in`, and close-proximity interactions
- `crossing_sparse_track` preserves anchor recall but collapses full-track success on the two `crossing` slices, where mean event recall drops to `0.536`
- `delayed_track` preserves anchor recall at the current anchor placement but reduces full-track success to `0/8` and introduces a mean first-match lag of `1.62` frames
- `stopped_lead` slices remain robust to temporal sparsity but not to delayed initialization, which helps separate fragmentation from track warm-up failure

Risk-conditioned findings:

- all current slices fall into `urgent_ttc`, so the current split is primarily discriminated by geometry and map relation rather than TTC strata
- `crossing_emergence` cases account for the full regression of `crossing_sparse_track`, while `large_lead_occluder` remains stable in that proxy setting
- `near_range` slices regress under sparse tracking, whereas `critical_range` slices remain stable for the current proxy profiles
- `shared_lane_supported` cases remain robust to sparse tracking but not to delayed initialization, which indicates that the benchmark can separate warm-up failure from sampling sparsity

The detailed local exports are:

- `outputs/trainval_perception_proxy_study_v1/perception_comparison_summary.md`
- `outputs/trainval_perception_proxy_study_v1/perception_comparison_summary.html`
- `outputs/trainval_perception_proxy_study_v1/perception_leaderboard.csv`
- `outputs/trainval_perception_proxy_study_v1/oracle_tracking/perception_metrics_summary.md`
- `outputs/trainval_perception_proxy_study_v1/crossing_sparse_track/perception_metrics_summary.md`
- `outputs/trainval_perception_proxy_study_v1/delayed_track/perception_metrics_summary.md`
- `outputs/trainval_perception_proxy_study_v1/oracle_tracking/perception_case_metrics.csv`

## Scenario-Conditioned World-Model Evaluation

The repository now also supports a scenario-conditioned world-model benchmark derived from the perception-slice layer. Each case exposes an observed history, a short future horizon for the primary risk actor, sparse future occupancy supervision, and the inherited risk facets from the validated scenario anchor.

The benchmark metadata now also carries named challenge tracks such as `challenge/crossing_emergence`, `challenge/large_lead_occluder`, `challenge/lateral_merge`, and `challenge/opposite_direction_conflict`.

For multi-modal forecasts, the evaluation export also reports benchmark-aligned forecast metrics:

- `MinADE@1`, `MinADE@5`, `MinADE@10`
- `MinFDE@1`, `MinFDE@5`, `MinFDE@10`
- `MissRate@1`, `MissRate@5`, `MissRate@10`

Local snapshot on `benchmarks/trainval_world_model_slices_v1.json`:

| Profile | Cases | Full Horizon | Mean Horizon Recall | Mean ADE | Mean FDE | Mean Occupancy IoU | Mean Risk Fidelity |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `oracle_rollout` | 8 | 8/8 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 |
| `risk_underreach_rollout` | 8 | 8/8 | 1.000 | 0.782 | 0.905 | 0.889 | 0.853 |
| `kinematic_rollout` | 8 | 8/8 | 1.000 | 0.838 | 0.959 | 0.883 | 0.853 |

Observed behavior:

- the current world-model slice set retains all `8` mined anchors by backing the rollout anchor off by one frame when the original event anchor falls at the end of the localized window
- `crossing` remains relatively stable for both non-oracle profiles, with mean risk fidelity above `0.93`
- `cut_in` is the lowest-performing family in the current proxy study, where `kinematic_rollout` drops to `0.678` mean risk fidelity and `0.500` occupancy IoU
- `stopped_lead` and generic `proximity` cases expose risk-alignment errors even when horizon coverage remains complete
- evaluation exports now include challenge-track summaries in addition to behavior and risk-facet summaries

External-adaptation support:

- compact external rollouts can now be adapted from benchmark-keyed `xy_ego` arrays with optional occupancy cells
- the adapter can rasterize actor occupancy from trajectory-only predictions when occupancy grids are not provided
- `nuScenes prediction challenge` style multi-modal forecasts can be adapted from `(instance, sample, prediction, probabilities)` records
- the forecast adapter supports `top_probability`, `oracle_ade`, and `oracle_fde` mode selection policies

Replay export snapshot:

- `outputs/trainval_world_model_replay_v1/replay_manifest.json`
- `8` replay files in newline-delimited `JSONL`
- `66` total replay messages across metadata, history tracks, future tracks, and future occupancy topics

The detailed local exports are:

- `outputs/trainval_world_model_proxy_study_v1/world_model_comparison_summary.md`
- `outputs/trainval_world_model_proxy_study_v1/world_model_comparison_summary.html`
- `outputs/trainval_world_model_proxy_study_v1/world_model_leaderboard.csv`
- `outputs/trainval_world_model_proxy_study_v1/oracle_rollout/world_model_metrics_summary.md`
- `outputs/trainval_world_model_proxy_study_v1/kinematic_rollout/world_model_metrics_summary.md`
- `outputs/trainval_world_model_proxy_study_v1/risk_underreach_rollout/world_model_metrics_summary.md`

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
- both baselines retain low occupancy IoU because they only predict the primary actor trajectory while the current occupancy target also includes surrounding context actors
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

- `7/8` current world-model slices remain forecast-compatible under the standard `12`-step prediction requirement
- `ContextVAE` preserves full-horizon coverage on the compatible subset and improves from `0.280` `MinADE@1` to `0.207` `MinADE@5`
- the largest residual gap remains occupancy agreement, which is expected because the benchmark occupancy target includes surrounding context actors while the external baseline predicts only the primary actor trajectory

### External Official Baseline Example

The repository now also supports direct evaluation of official `nuScenes` prediction JSON. Because official files such as `results_val_megvii.json` only cover the validation split, the recommended path is to filter the perception benchmark to the covered subset before scoring.

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

- the covered subset currently contains one `stopped_lead` slice and one `oncoming` slice
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
- `PointPillars` achieves partial event recall on the `oncoming` slice (`0.200`), whereas the current `Megvii` result misses it completely
- `Megvii` remains slightly tighter on center localization for the one successful slice

The detailed local comparison export is:

- `outputs/external_official_tracking_comparison/perception_comparison_summary.md`

## Interpretation

The current results indicate:

- case-level pass metrics can hide reference-scene and actor-grounding errors
- planner improvements should be measured against scene-level and event-level supervision, not only retrieval score
- deterministic parsing remains a strong baseline on tightly specified scenario-mining queries
- `LLM` and `hybrid` profiles remain informative because they expose planner disagreement and paraphrase sensitivity that the benchmark can quantify
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

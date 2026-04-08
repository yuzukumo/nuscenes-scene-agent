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

## Language-Stress Comparison

The language-stress suite is designed to test phrasing robustness rather than simply replay structured benchmark labels.

Current snapshot:

| Profile | Pass@1 | Mean Best Score |
| --- | --- | --- |
| `rule_only` | 16/16 | 90.64 |
| `llm_planner` | 15/16 | 86.29 |
| `hybrid_agent` | 16/16 | 91.34 |

Additional signal:

- queries: `16`
- planner signal divergence: `12`

## Scenario Mining Comparison

The planning-centric scenario mining suite evaluates whether a profile retrieves not only a risky case, but also the correct scene anchor, actor identity, and event window.

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
- planner signal divergence appears on `11/16` queries in the scenario-mining comparison, indicating that the benchmark captures planner robustness rather than retrieval recall alone

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
- `stopped_lead` slices remain robust to temporal sparsity but not to delayed initialization, which makes the layer useful for distinguishing fragmentation from track warm-up failure

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
- the `oncoming` slice is missed entirely, so this external example primarily demonstrates the adapter and coverage-alignment path

### External Official Tracking Comparison

The repository also supports direct comparison across multiple external evaluation outputs.

Local comparison on the shared covered subset:

| Profile | Cases | Anchor Recall | Full Track | Mean Event Recall | Mean Contiguous Coverage | Mean Center Error |
| --- | --- | --- | --- | --- | --- | --- |
| `pointpillars_tracking_val_covered` | 2 | 1/2 | 1/2 | 0.600 | 0.600 | 0.323 |
| `megvii_tracking_val_covered` | 2 | 1/2 | 1/2 | 0.500 | 0.500 | 0.271 |

Observed comparison signal:

- both methods solve the `stopped_lead` slice on the aligned subset
- `PointPillars` achieves partial event recall on the `oncoming` slice (`0.200`), whereas the current `Megvii` result misses it completely
- `Megvii` remains slightly tighter on center localization for the one successful slice

The detailed local comparison export is:

- `outputs/external_official_tracking_comparison/perception_comparison_summary.md`

## Why These Results Matter

These results indicate:

- case-level pass metrics can hide reference-scene and actor-grounding errors
- planner improvements should be measured against scene-level and event-level supervision, not only retrieval score
- deterministic parsing remains a strong baseline on tightly specified scenario-mining queries
- `LLM` and `hybrid` profiles remain informative because they expose signal divergence and paraphrase sensitivity that the benchmark can quantify
- mined scenario anchors can also support perception-side evaluation without converting the repository into a full detector-training stack

The repository therefore serves both as a retrieval toolkit and as a benchmark design environment for reference-aware studies of planner robustness.

## Counterfactual Benchmark Track

The repository includes a counterfactual benchmark generation path.

Given a validated case library, it can generate:

- positive canonical queries
- positive paraphrase queries
- actor-swap negatives
- behavior-swap negatives

These generated benchmarks carry explicit `reference_case_keys`, and when reference event windows are available they can also support localization-aware metrics such as event-window overlap.

## Scenario Mining Track

The repository supports a planning-centric scenario mining benchmark derived from validated case libraries.

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

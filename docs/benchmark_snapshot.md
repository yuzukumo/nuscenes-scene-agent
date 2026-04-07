# Benchmark Snapshot Notes

Large generated outputs are kept local by default. The current local runs nonetheless provide a compact summary of system behavior.

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

Current local snapshot on `benchmarks/trainval_scenario_mining_v1.yaml`:

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

Current local snapshot on `outputs/trainval_hybrid_ablation_v1`:

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

## Why These Results Matter

These results indicate:

- case-level pass metrics can hide reference-scene and actor-grounding errors
- planner improvements should be measured against scene-level and event-level supervision, not only retrieval score
- deterministic parsing remains a strong baseline on tightly specified scenario-mining queries
- `LLM` and `hybrid` profiles remain informative because they expose signal divergence and paraphrase sensitivity that the benchmark can quantify

Accordingly, the repository functions both as a retrieval toolkit and as a benchmark design environment for reference-aware studies of planner robustness.

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

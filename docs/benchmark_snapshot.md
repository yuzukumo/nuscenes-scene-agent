# Benchmark Snapshot Notes

Benchmark specifications under `benchmarks/` are versioned; generated outputs under `outputs/` are excluded from version control by default. This document records benchmark scale, metric boundaries, and representative local results.

The benchmark stack is scenario-centric. `nuScenes`, `nuPlan`, `Bench2Drive`, and `CARLA` are connected through the shared taxonomy in [configs/scenario_taxonomy.yaml](../configs/scenario_taxonomy.yaml). `nuScenes` anchors define and validate real-world risk semantics; `nuPlan` evaluates logged replay behavior; `Bench2Drive` trains and diagnoses a vision trajectory planner; `CARLA` provides semantically matched closed-loop visual evidence.

## Benchmark Scale

| Source | Candidate or Anchor Count | Exported Benchmark Count | Notes |
| --- | ---: | ---: | --- |
| `nuScenes` trainval case library | `33` unique cases, `30` passed cases | configurable | mined with natural-language queries and deterministic validation |
| `nuScenes` scenario-mining benchmark | `24` anchors | `48` queries | canonical and paraphrase query variants |
| `nuScenes` perception, BEV occupancy, and world-model slices | `24` anchors | `24` cases per layer | derived from scenario-mining anchors |
| `nuPlan` cross-split replay sweep | `1556` candidate anchors | `112` replay cases | controlled by per-study caps |
| `nuPlan` cross-split closed-loop replay sweep | `1556` candidate anchors | `112` replay-simulation cases | uses the same scenario sampling as replay regression |
| `Bench2Drive` vision planner | `44,940` cached samples | `4,334` held-out test samples | multi-camera vision imitation training |
| `Bench2Drive` vision closed-loop | `44,940` cached samples | `64` held-out test cases | model-in-the-loop rollout with vehicle dynamics |
| `CARLA` semantic demo mining | `1` configured target class | audit-gated | model-waypoint ego control with Traffic Manager ambient vehicles and semantic evidence checks |

The versioned benchmark specifications use sampling caps over validated mined cases. `nuScenes` reference labels are weakly supervised anchors derived from the validated case library, including scene identity, actor identity, event-window range, and peak sample. These anchors are deterministic benchmark targets, not independent human annotations; metrics that use them should be interpreted as anchor consistency rather than true semantic recall.

Main entry point:

```bash
python -m nusc_scene_agent run-full-benchmark-suite
```

The suite writes a top-level summary under `outputs/full_benchmark_suite_v1` and a compact registry under `outputs/full_benchmark_suite_v1/result_registry`.

## Metric Boundaries

Scenario-mining metrics evaluate retrieval and grounding against weakly supervised anchors: validation acceptance@K, scene match, actor match, reference-case consistency, event-window IoU, and peak-sample error.

The reported best validation quality follows the selection policy: accepted cases take precedence, then quality is maximized. Artifacts also retain the ungated maximum quality as a separate diagnostic field.

Perception-slice metrics evaluate short-window actor coverage: anchor recall, full-track success, event recall, contiguous temporal coverage, center error, and first-match lag.

Sparse BEV occupancy metrics evaluate actor-center occupancy: occupancy IoU, primary actor recall, context recall, anchor-frame IoU, and risk-fidelity score.

World-model metrics evaluate future trajectory and sparse future occupancy: ADE, FDE, `MinADE@K`, `MinFDE@K`, `MissRate@K`, occupancy IoU, closest-approach distance and time errors, and risk fidelity.

`nuPlan` replay-regression metrics compare predicted ego rollouts with logged replay windows: ego ADE/FDE, minimum-distance error, minimum-TTC error, red-light context recall, comfort-target errors, collision-proxy mismatch, and risk fidelity.

`nuPlan` closed-loop replay metrics roll the ego state forward with planner profiles while replaying logged actors and traffic-light context. Reported metrics include ego ADE/FDE, minimum-distance error, minimum-TTC error, a bounded progress ratio, the raw progress ratio used by the score, collision-proxy mismatch, comfort violations, closed-loop drift, and closed-loop score. The bounded ratio is for reporting; the raw ratio preserves over-progress penalties in the score.

Bench2Drive vision-planner metrics evaluate supervised ego-trajectory prediction from six camera views and route features. Reported metrics include waypoint ADE/FDE, control loss, brake accuracy, and training throughput. The closed-loop layer rolls the trained model forward in a waypoint-control bicycle-model simulation; reported metrics include closed-loop ADE/FDE, route completion, lateral error, and closed-loop score. This layer is diagnostic and should not be interpreted as a full simulator benchmark.

CARLA semantic demo mining evaluates the trained multi-camera planner in synchronous urban rollouts selected from the shared scenario taxonomy. Ego control follows model-predicted waypoints through a low-level vehicle controller, with optional safety-brake override. CARLA Traffic Manager controls ambient vehicles. A rollout is retained as a semantic demo only if video, control-attribution, traffic-context, and scenario-specific evidence checks pass. The retained rollout is qualitative audit evidence, not a statistically powered closed-loop benchmark.

Proxy profiles are controlled perturbations for sensitivity analysis. External baselines are evaluated through adapter interfaces for official `nuScenes` prediction files, forecast outputs, and `ContextVAE`. Baseline rows with different case counts should be compared as subset diagnostics rather than as significance-tested model rankings. Validation quality is a continuous diagnostic in `[0, 100]`; validation acceptance is a separate deterministic gate. Reported best quality uses accepted cases first, while ungated maxima are retained as diagnostics.

Retrieval weights, validation-quality weights, pass gates, and behavior thresholds are exported in query and case artifacts. `benchmark-score-sweep` compares retrieval profiles, `benchmark-validation-score-sweep` compares validation-quality profiles, and `benchmark-threshold-sweep` scales metric-valued behavior thresholds. Each command varies one component while holding the other components fixed.

## Core `nuScenes` Suite

Dataset:

- `nuScenes v1.0-trainval`
- map expansion

Case-library snapshot:

| Metric | Value |
| --- | --- |
| Queries | 16 |
| Validation acceptance@1 | 16/16 |
| Validation acceptance@K | 16/16 |
| Selected cases | 36 |
| Unique cases | 33 |
| Unique passed cases | 30 |
| Mean best validation quality | 89.97 |

Scenario-mining benchmark:

| Quantity | Value |
| --- | --- |
| Anchor cases | 24 |
| Query variants | 48 |
| Variants per anchor | canonical query and paraphrase query |
| Source case library | `outputs/trainval_case_library_v1/case_library_enriched.json` |

Behavior distribution:

| Behavior | Anchors | Queries |
| --- | ---: | ---: |
| `stopped_lead` | 7 | 14 |
| `crossing` | 5 | 10 |
| `proximity` | 5 | 10 |
| `oncoming` | 4 | 8 |
| `cut_in` | 3 | 6 |

## Sensitivity Analysis

The following deterministic sweeps keep benchmark queries and acceptance gates fixed. They quantify sensitivity to retrieval weights, validation-quality weights, and metric-valued behavior thresholds; they do not provide independent semantic ground truth.

| Study | Profile | Validation acceptance@1 | Mean best quality | Unique passed cases |
| --- | --- | ---: | ---: | ---: |
| Retrieval weights | `default` | `16/16` | `89.97` | `30` |
| Retrieval weights | `equal` | `16/16` | `89.03` | `29` |
| Validation-quality weights | `default` | `16/16` | `89.97` | `30` |
| Validation-quality weights | `equal` | `16/16` | `89.31` | `30` |
| Behavior thresholds | `0.85x` | `16/16` | `89.49` | `28` |
| Behavior thresholds | `1.00x` | `16/16` | `89.97` | `30` |
| Behavior thresholds | `1.15x` | `16/16` | `89.97` | `31` |

Validation acceptance@1 is unchanged across all profiles. Equal validation-quality weights reduce mean best quality by `0.66` without changing the accepted case count, so the named default profile remains the primary configuration. Retrieval and threshold variants alter case-library composition modestly; these changes are reported as sensitivity diagnostics rather than evidence of semantic superiority.

## Perception And BEV Occupancy

Perception-slice snapshot:

| Profile | Cases | Anchor Recall | Full Track | Mean Event Recall | Mean Contiguous Coverage | Mean Center Error |
| --- | ---: | --- | --- | ---: | ---: | ---: |
| `oracle_tracking` | 24 | 24/24 | 24/24 | 1.000 | 1.000 | 0.000 |
| `delayed_track` | 24 | 23/24 | 0/24 | 0.719 | 0.719 | 0.403 |
| `crossing_sparse_track` | 24 | 22/24 | 19/24 | 0.901 | 0.823 | 0.126 |

BEV occupancy snapshot:

| Profile | Cases | Mean Occupancy IoU | Primary Recall | Context Recall | Anchor IoU | Risk Fidelity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `oracle_occupancy` | 24 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| `context_drop_occupancy` | 24 | 0.553 | 1.000 | 0.502 | 0.554 | 0.699 |
| `risk_actor_only` | 24 | 0.105 | 1.000 | 0.003 | 0.104 | 0.398 |

The sparse BEV layer stores primary risk actor cells, surrounding context actor cells, and union occupancy cells on a fixed ego-frame grid. It evaluates whether predictions cover the risk actor and nearby dynamic context during the mined event window.

## World-Model Evaluation

Scenario-conditioned world-model snapshot:

| Profile | Cases | Full Horizon | Mean Horizon Recall | Mean ADE | Mean FDE | Mean Occupancy IoU | Mean Risk Fidelity |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `oracle_rollout` | 24 | 24/24 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 |
| `risk_underreach_rollout` | 24 | 24/24 | 1.000 | 0.725 | 0.921 | 0.975 | 0.892 |
| `kinematic_rollout` | 24 | 24/24 | 1.000 | 1.010 | 1.347 | 0.958 | 0.869 |

Official physics baselines on the common benchmark anchors:

| Profile | Cases | Full Horizon | Mean ADE | Mean FDE | Mean MinADE@1 | Mean MissRate@5 | Mean Occupancy IoU | Mean Risk Fidelity |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `physics_oracle` | 8 | 8/8 | 0.212 | 0.214 | 0.212 | 0.000 | 0.175 | 0.840 |
| `cv_heading` | 8 | 8/8 | 0.288 | 0.394 | 0.288 | 0.125 | 0.175 | 0.823 |

`ContextVAE` baseline on the forecast-compatible subset:

| Profile | Cases | Full Horizon | Mean ADE | Mean MinADE@5 | Mean Occupancy IoU | Mean Risk Fidelity |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| `physics_oracle` | 7 | 7/7 | 0.135 | 0.135 | 0.197 | 0.860 |
| `cv_heading` | 7 | 7/7 | 0.139 | 0.139 | 0.197 | 0.859 |
| `contextvae` | 7 | 7/7 | 0.280 | 0.207 | 0.181 | 0.841 |

The `ContextVAE` comparison evaluates the three forecast profiles on the same seven forecast-compatible cases; the preceding eight-case rows are common-anchor diagnostics. The comparison artifact includes clip-level bootstrap intervals; it does not support a statistically powered ranking at this sample size.

## `nuPlan` Replay Evaluation

Cross-split replay-regression snapshot:

| Profile | Cases | Full Horizon | Mean Ego ADE | Mean Ego FDE | Mean Min-Distance Error | Mean Min-TTC Error | Mean Risk Fidelity |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `logged_ego` | 112 | 112/112 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |
| `history_kinematic` | 112 | 112/112 | 0.916 | 2.656 | 0.337 | 0.285 | 0.965 |
| `constant_velocity` | 112 | 112/112 | 12.774 | 25.640 | 4.269 | 0.304 | 0.808 |
| `stopped` | 112 | 112/112 | 10.666 | 21.372 | 3.772 | 2.045 | 0.796 |

Closed-loop replay snapshot:

| Profile | Cases | Full Horizon | Mean Ego ADE | Mean Ego FDE | Mean Min-Distance Error | Mean Min-TTC Error | Mean Progress Ratio | Closed-Loop Score |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `logged_ego_oracle` | 112 | 112/112 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 |
| `history_kinematic` | 112 | 112/112 | 1.027 | 2.858 | 0.353 | 0.286 | 0.921 | 1.022 | 0.950 |
| `idm_like_following` | 112 | 112/112 | 1.500 | 4.152 | 0.427 | 0.343 | 0.826 | 0.905 | 0.935 |

Study coverage:

| Study | DBs | Candidates | Cases |
| --- | ---: | ---: | ---: |
| `mini` | 64 | 243 | 16 |
| `val_sample` | 128 | 429 | 24 |
| `train_boston_sample` | 128 | 340 | 24 |
| `train_pittsburgh_sample` | 128 | 294 | 24 |
| `train_singapore_sample` | 128 | 250 | 24 |

## Bench2Drive Vision Planner

Dataset and cache:

| Quantity | Value |
| --- | ---: |
| Source archives | 1,000 |
| Manifest rows after finite-value filtering | 44,940 |
| Train samples | 35,629 |
| Validation samples | 4,977 |
| Held-out test samples | 4,334 |
| Cameras | 6 |
| Tensor cache image size | 160 x 160 |
| Tensor cache size | 20 GB |

Training configuration:

| Quantity | Value |
| --- | --- |
| Model | `research` trajectory transformer |
| Input | six camera views and route features |
| Target | future ego waypoints, control values, brake state |
| Training | `8`-GPU distributed data parallel |
| Per-GPU batch size | 64 |
| Epochs | 24 |
| Precision | `fp16`; TF32 and cuDNN benchmark enabled |
| Runtime | 289.538 seconds |
| Loss additions | displacement `0.1`; endpoint `0.025`; path length `0.025` |
| Trajectory modes | 4 |
| Trajectory selection | expected mixture over mode probabilities |
| Trajectory temperature | 0.5 |

Held-out test evaluation:

| Metric | Value |
| --- | ---: |
| ADE | 1.653 |
| FDE | 2.697 |
| Lateral MAE | 0.552 m |
| Turn lateral MAE | 0.815 m |
| Brake accuracy | 0.847 |
| Brake F1 | 0.828 |
| Oracle ADE over trajectory modes | 0.992 |
| Oracle FDE over trajectory modes | 1.313 |

Validation trajectory-selection search:

| Model | ADE | FDE | Brake F1 |
| --- | ---: | ---: | ---: |
| `trajectory_transformer_argmax` | 1.665 | 2.676 | 0.827 |
| `trajectory_transformer_topk_expected` | 1.636 | 2.639 | 0.827 |
| `trajectory_transformer_expected_t0.5` | 1.610 | 2.624 | 0.827 |

Planner diagnostic summary:

| Metric | Value |
| --- | ---: |
| Samples | 4,334 |
| Underreach rate | 0.2439 |
| Severe underreach rate | 0.0708 |
| Near-stop prediction rate | 0.0000 |
| Mean lateral error | 0.552 m |
| Predicted-to-target speed ratio | 0.959 |
| Brake F1 | 0.828 |
| Diagnostic status | `ready_for_closed_loop_diagnostics` |

Planner diagnostics are used as readiness checks for subsequent closed-loop analysis. The CARLA stage uses semantic audit gates and reports safety-override attribution instead of treating supervised ADE/FDE as sufficient closed-loop evidence.

Model-in-the-loop closed-loop validation:

| Metric | Value |
| --- | ---: |
| Cases | 64 held-out test cases |
| Mean closed-loop ADE | 9.721 m |
| Mean closed-loop FDE | 19.471 m |
| Mean lateral error | 1.568 m |
| Mean route completion | 0.799 |
| Mean closed-loop score | 0.157 |

Paired comparison with the original trajectory-only objective on the same 64 cases:

| Metric | Baseline | Candidate | Oriented improvement | 95% CI | Candidate win rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Closed-loop ADE | 11.260 m | 9.721 m | `+1.539 m` | `[+0.739, +2.382]` | 0.625 |
| Closed-loop FDE | 23.690 m | 19.471 m | `+4.218 m` | `[+2.051, +6.541]` | 0.641 |
| Mean lateral error | 1.436 m | 1.568 m | `-0.132 m` | `[-0.406, +0.140]` | 0.484 |
| Route completion | 0.746 | 0.799 | `+0.053` | `[+0.009, +0.105]` | 0.484 |
| Closed-loop score | 0.087 | 0.157 | `+0.070` | `[+0.023, +0.120]` | 0.609 |

Positive values are oriented so that higher is better. The intervals are case-level percentile-bootstrap intervals with seed `7` and `10,000` replicates. The candidate improves the closed-loop-oriented metrics while having no statistically conclusive change in lateral error.

Paired held-out open-loop comparison on `4,334` samples from `97` clips:

| Metric | Baseline | Candidate | Oriented improvement | 95% CI |
| --- | ---: | ---: | ---: | ---: |
| ADE | 1.622 m | 1.653 m | `-0.031 m` | `[-0.069, +0.010]` |
| FDE | 2.693 m | 2.697 m | `-0.003 m` | `[-0.089, +0.084]` |
| Lateral MAE | 0.525 m | 0.552 m | `-0.028 m` | `[-0.045, -0.013]` |
| Path-length error | 0.465 m | 0.396 m | `+0.069 m` | `[+0.036, +0.101]` |
| Brake F1 | 0.827 | 0.828 | `+0.002` | `[-0.028, +0.034]` |

Open-loop intervals use clip-level percentile bootstrap with seed `7`, `10,000` replicates, and `97` clip clusters. The regularized objective improves path-length error but slightly worsens lateral imitation error; ADE, FDE, and brake F1 changes are inconclusive. This trade-off is the reason the closed-loop comparison is reported as the primary model result rather than a claim of uniform improvement.

The Bench2Drive model-in-the-loop layer is a proxy diagnostic using simplified vehicle dynamics. The CARLA section reports the retained visual rollout evidence.

## CARLA Semantic Demo Mining

The CARLA stage searches route and traffic configurations for an audit-gated right-turn pedestrian-yield target:

| Target | Evidence Gate |
| --- | --- |
| `pedestrian_yield` | crosswalk pedestrian actor, ego yield/brake response, no collision |

Only passing attempts are promoted from trial runs to `outputs/carla_semantic_demo_final`. The retained report records MP4 paths, states CSV, route traces, control attribution, and semantic audit status. The README references a curated copy of the retained demo under `assets/`.

Semantic-gated CARLA run:

| Quantity | Value |
| --- | ---: |
| Target classes | 1 |
| Passed target classes | 1 |
| Attempts | 1 |
| Retained rollouts | 1 |
| Total video frames | 272 |
| Duration | 27.1 s |
| Traffic Manager vehicles | 13 |
| Scripted vehicles | 0 |
| Crosswalk pedestrians | 9 |
| Collisions | 0 |
| Semantic audit failures | 0 |
| Semantic audit warnings | 0 |
| Model waypoint-controller ratio | 1.000 |
| Safety override ratio | 0.096 |
| Mean lateral error | 0.728 m |
| Route completion | 0.982 |
| Video | `1920x1080` HEVC MP4 |

Retained rollout media:

| Scenario | Type | Frames | Traffic Manager Vehicles | Scripted Vehicles | Collisions | Video |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `right_turn_pedestrian_yield` | `pedestrian_crossing` | 272 | 13 | 0 | 0 | `1920x1080` HEVC MP4 |

## Learned Retrieval And Failure Mining

Weakly supervised learned retrieval:

| Quantity | Value |
| --- | --- |
| Training source | `artifacts/index/v1.0-trainval.sqlite` |
| Scenario families | `vru_crossing_front`, `stopped_lead_vehicle`, `lateral_cut_in`, `oncoming_vehicle` |
| Total training groups | 4,000 |
| Train groups | 2,779 |
| Scene-held-out validation groups | 1,221 |
| Train weak-anchor consistency@1 | 1.000 |
| Scene-held-out weak-anchor consistency@1 | 1.000 |
| Model | `query_scene_pairwise_mlp_v1` |

Failure-aware candidate-generation diagnostic:

| Metric | Rule Ranking | Learned Candidate Generation |
| --- | ---: | ---: |
| Queries | 24 | 24 |
| Validation-gated acceptance@1 | 20/24 | 20/24 |
| Validation-gated acceptance@K | 20/24 | 24/24 |
| Mean top-1 score delta | - | -4.1621 |
| Mean acceptance-gated best quality delta | - | +2.095 |
| Mean ungated maximum quality delta | - | +2.095 |

The learned model expands candidate coverage on failure-mined queries. Deterministic validation performs final case selection. The generated artifact reports acceptance-gated best quality separately from the ungated maximum; both are diagnostic and neither is independent semantic recall.

Failure-mining snapshot:

| Quantity | Value |
| --- | ---: |
| Source files | 48 |
| Failure records | 401 |
| Failure clusters | 83 |
| Benchmark update queries | 24 |

Top failure clusters:

| Source | Failure | Context | Actor/Scenario | Count |
| --- | --- | --- | --- | ---: |
| `nuplan_replay` | `risk_distance_error` | `large_vehicle_interaction` | `near_long_vehicle` | 44 |
| `nuplan_replay` | `risk_distance_error` | `vru_interaction` | `near_pedestrian_on_crosswalk` | 30 |
| `nuplan_replay` | `risk_distance_error` | `static_obstacle_context` | `near_trafficcone_on_driveable` | 25 |
| `nuplan_replay_sweep` | `risk_distance_error` | `large_vehicle_interaction` | `near_long_vehicle` | 17 |
| `nuplan_closed_loop` | `closed_loop_drift` | `large_vehicle_interaction` | `near_long_vehicle` | 7 |

## Result Registry

| Layer | Key Result |
| --- | --- |
| `risk_benchmark_suite` | `24` scenario anchors, `48` queries, and `24` cases for perception, BEV occupancy, and world-model layers |
| `nuplan_replay_regression` | `112` replay cases; best non-oracle risk fidelity `0.965` |
| `nuplan_closed_loop_replay` | `112` replay-simulation cases; best non-oracle closed-loop score `0.950` |
| `bench2drive_vision_planner` | `44,940` cached samples; held-out test trajectory-transformer ADE `1.653`; FDE `2.697`; brake F1 `0.828` |
| `bench2drive_vision_closed_loop` | `64` held-out test model-in-the-loop cases; route completion `0.799`; mean lateral error `1.568 m`; closed-loop score `0.157` |
| `carla_semantic_demo` | `1/1` semantic target passed; `272` frames; `13` Traffic Manager vehicles; `9` crosswalk pedestrians; `0` scripted vehicles; `0` collisions |
| `failure_mining` | `401` failure records, `83` clusters, and `24` update queries |

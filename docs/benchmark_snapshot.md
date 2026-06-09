# Benchmark Snapshot Notes

Generated outputs are excluded from version control by default. This document records benchmark scale, metric boundaries, and representative local results.

The benchmark stack is scenario-centric. `nuScenes`, `nuPlan`, `Bench2Drive`, and `CARLA` are connected through the shared taxonomy in [configs/scenario_taxonomy.yaml](../configs/scenario_taxonomy.yaml). `nuScenes` anchors define and validate real-world risk semantics; `nuPlan` evaluates logged replay behavior; `Bench2Drive` trains and diagnoses a vision trajectory planner; `CARLA` provides semantically matched closed-loop visual evidence.

## Benchmark Scale

| Source | Candidate or Anchor Count | Exported Benchmark Count | Notes |
| --- | ---: | ---: | --- |
| `nuScenes` trainval case library | `33` unique cases, `29` passed cases | configurable | mined with natural-language queries and deterministic validation |
| `nuScenes` scenario-mining benchmark | `24` anchors | `48` queries | canonical and paraphrase query variants |
| `nuScenes` perception, BEV occupancy, and world-model slices | `24` anchors | `24` cases per layer | derived from scenario-mining anchors |
| `nuPlan` cross-split replay sweep | `1556` candidate anchors | `112` replay cases | controlled by per-study caps |
| `nuPlan` cross-split closed-loop replay sweep | `1556` candidate anchors | `112` replay-simulation cases | uses the same scenario sampling as replay regression |
| `Bench2Drive` vision planner | `44,940` cached samples | `4,717` validation samples | multi-camera vision imitation training |
| `Bench2Drive` vision closed-loop | `44,940` cached samples | `64` closed-loop cases | model-in-the-loop rollout with vehicle dynamics |
| `CARLA` semantic demo mining | `1` configured target class | audit-gated | model-waypoint ego control with Traffic Manager ambient vehicles and semantic evidence checks |

The committed benchmark slices use sampling caps over validated mined cases. `nuScenes` reference labels are weak-supervised anchors derived from the validated case library, including scene identity, actor identity, event-window range, and peak sample.

Main entry point:

```bash
python -m nusc_scene_agent run-full-benchmark-suite
```

The suite writes a top-level summary under `outputs/full_benchmark_suite_v1` and a compact registry under `outputs/full_benchmark_suite_v1/result_registry`.

## Metric Boundaries

Scenario-mining metrics evaluate retrieval and grounding: `Pass@K`, scene match, actor match, reference-case match, event-window IoU, and peak-sample error.

Perception-slice metrics evaluate short-window actor coverage: anchor recall, full-track success, event recall, contiguous temporal coverage, center error, and first-match lag.

Sparse BEV occupancy metrics evaluate actor-center occupancy: occupancy IoU, primary actor recall, context recall, anchor-frame IoU, and risk-fidelity score.

World-model metrics evaluate future trajectory and sparse future occupancy: ADE, FDE, `MinADE@K`, `MinFDE@K`, `MissRate@K`, occupancy IoU, closest-approach distance and time errors, and risk fidelity.

`nuPlan` replay-regression metrics compare predicted ego rollouts with logged replay windows: ego ADE/FDE, minimum-distance error, minimum-TTC error, red-light context recall, comfort-target errors, collision-proxy mismatch, and risk fidelity.

`nuPlan` closed-loop replay metrics roll the ego state forward with planner profiles while replaying logged actors and traffic-light context. Reported metrics include ego ADE/FDE, minimum-distance error, minimum-TTC error, progress ratio, collision-proxy mismatch, comfort violations, closed-loop drift, and closed-loop score.

Bench2Drive vision-planner metrics evaluate supervised ego-trajectory prediction from six camera views and route features. Reported metrics include waypoint ADE/FDE, control loss, brake accuracy, and training throughput. The closed-loop layer rolls the trained model forward in a waypoint-control bicycle-model simulation; reported metrics include closed-loop ADE/FDE, route completion, lateral error, and closed-loop score.

CARLA semantic demo mining evaluates the trained multi-camera planner in synchronous urban rollouts selected from the shared scenario taxonomy. Ego control follows model-predicted waypoints through a low-level vehicle controller, with optional safety-brake override. CARLA Traffic Manager controls ambient vehicles. A rollout is retained as a semantic demo only if video, control-attribution, traffic-context, and scenario-specific evidence checks pass.

Proxy profiles are controlled perturbations for sensitivity analysis. External baselines are evaluated through adapter interfaces for official `nuScenes` prediction files, forecast outputs, and `ContextVAE`.

## Core `nuScenes` Suite

Dataset:

- `nuScenes v1.0-trainval`
- map expansion

Case-library snapshot:

| Metric | Value |
| --- | --- |
| Queries | 16 |
| Pass@1 | 16/16 |
| Pass@K | 16/16 |
| Selected cases | 36 |
| Unique cases | 33 |
| Unique passed cases | 29 |
| Mean best validation score | 89.26 |

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

Official physics baselines on benchmark anchors:

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
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `logged_ego_oracle` | 112 | 112/112 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 |
| `history_kinematic` | 112 | 112/112 | 1.027 | 2.858 | 0.353 | 0.286 | 1.022 | 0.950 |
| `idm_like_following` | 112 | 112/112 | 1.500 | 4.152 | 0.427 | 0.343 | 0.905 | 0.935 |

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
| Train samples | 40,223 |
| Validation samples | 4,717 |
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
| Runtime | 315.983 seconds |
| Trajectory modes | 4 |
| Trajectory selection | expected mixture over mode probabilities |
| Trajectory temperature | 0.5 |

Independent validation:

| Metric | Value |
| --- | ---: |
| ADE | 1.599 |
| FDE | 2.625 |
| Lateral MAE | 0.561 m |
| Turn lateral MAE | 0.830 m |
| Brake accuracy | 0.903 |
| Brake F1 | 0.884 |
| Oracle ADE over trajectory modes | 0.937 |
| Oracle FDE over trajectory modes | 1.239 |

Trajectory-selection ablation:

| Model | ADE | FDE | Brake F1 |
| --- | ---: | ---: | ---: |
| `trajectory_transformer_argmax` | 1.709 | 2.790 | 0.886 |
| `trajectory_transformer_topk_expected` | 1.631 | 2.684 | 0.886 |
| `trajectory_transformer_expected_t0.5` | 1.599 | 2.625 | 0.884 |

Planner diagnostic summary:

| Metric | Value |
| --- | ---: |
| Samples | 4,717 |
| Underreach rate | 0.389 |
| Severe underreach rate | 0.329 |
| Near-stop prediction rate | 0.205 |
| Mean lateral error | 0.561 m |
| Predicted-to-target speed ratio | 0.887 |
| Brake F1 | 0.884 |
| Diagnostic status | `requires_planner_improvement_before_carla_rollout` |

The diagnostic flag is caused by conservative horizon length on part of the validation set. The CARLA stage therefore uses semantic audit gates and reports safety-override attribution instead of treating supervised ADE/FDE as sufficient closed-loop evidence.

Model-in-the-loop closed-loop validation:

| Metric | Value |
| --- | ---: |
| Cases | 64 |
| Mean closed-loop ADE | 12.438 |
| Mean closed-loop FDE | 24.902 |
| Mean lateral error | 1.332 |
| Mean route completion | 0.754 |
| Mean closed-loop score | 0.105 |

The Bench2Drive model-in-the-loop layer is a proxy diagnostic using simplified vehicle dynamics. The CARLA section reports the retained visual rollout evidence.

## CARLA Semantic Demo Mining

The CARLA stage searches route and traffic configurations for an audit-gated right-turn pedestrian-yield target:

| Target | Evidence Gate |
| --- | --- |
| `pedestrian_yield` | crosswalk pedestrian actor, ego yield/brake response, no collision |

Only passing attempts are promoted from trial runs to `outputs/carla_semantic_demo_trajectory_transformer_final`. The retained report records MP4 paths, states CSV, route traces, control attribution, and semantic audit status. The README references a curated copy of the retained demo under `assets/`.

Semantic-gated CARLA run:

| Quantity | Value |
| --- | ---: |
| Target classes | 1 |
| Passed target classes | 1 |
| Attempts | 2 |
| Retained rollouts | 1 |
| Total video frames | 267 |
| Duration | 26.6 s |
| Traffic Manager vehicles | 13 |
| Scripted vehicles | 0 |
| Crosswalk pedestrians | 9 |
| Collisions | 0 |
| Semantic audit failures | 0 |
| Semantic audit warnings | 0 |
| Direct model-control ratio | 1.000 |
| Safety override ratio | 0.105 |
| Mean lateral error | 0.642 m |
| Route completion | 0.983 |
| Video | `1920x1080` HEVC MP4 |

Retained rollout media:

| Scenario | Type | Frames | Traffic Manager Vehicles | Scripted Vehicles | Collisions | Video |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `right_turn_pedestrian_yield` | `pedestrian_crossing` | 267 | 13 | 0 | 0 | `1920x1080` HEVC MP4 |

## Learned Retrieval And Failure Mining

Weakly supervised learned retrieval:

| Quantity | Value |
| --- | --- |
| Training source | `artifacts/index/v1.0-trainval.sqlite` |
| Scenario families | `vru_crossing_front`, `stopped_lead_vehicle`, `lateral_cut_in`, `oncoming_vehicle` |
| Total training groups | 4,000 |
| Train groups | 2,779 |
| Scene-held-out validation groups | 1,221 |
| Train Recall@1 | 1.000 |
| Scene-held-out Recall@1 | 1.000 |
| Model | `query_scene_pairwise_mlp_v1` |

Failure-aware candidate-generation diagnostic:

| Metric | Rule Ranking | Learned Candidate Generation |
| --- | ---: | ---: |
| Queries | 24 | 24 |
| Pass@1 | 20/24 | 20/24 |
| Pass@K | 20/24 | 24/24 |
| Mean top-1 score delta | - | -4.429 |
| Mean best score delta | - | +2.165 |

The learned model expands candidate coverage on failure-mined queries. Deterministic validation performs final case selection.

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
| `bench2drive_vision_planner` | `44,940` cached samples; temperature-calibrated trajectory-transformer ADE `1.599`; FDE `2.625`; brake F1 `0.884` |
| `bench2drive_vision_closed_loop` | `64` model-in-the-loop cases; route completion `0.754`; mean lateral error `1.332 m`; closed-loop score `0.105` |
| `carla_semantic_demo` | `1/1` semantic target passed; `267` frames; `13` Traffic Manager vehicles; `9` crosswalk pedestrians; `0` scripted vehicles; `0` collisions |
| `failure_mining` | `401` failure records, `83` clusters, and `24` update queries |

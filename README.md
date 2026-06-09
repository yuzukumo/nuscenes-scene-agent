<div align="center">

# Autonomous Driving Risk Scenario Benchmark Agent

Scenario-centric risk mining, benchmark generation, replay evaluation, and vision E2E planner validation across `nuScenes`, `nuPlan`, `Bench2Drive`, and `CARLA`.

`Python 3.10+` `Conda` `nuScenes` `nuPlan` `CARLA` `Bench2Drive` `Ollama` `Benchmarking`

[English](README.md) | [简体中文](README.zh-CN.md)

</div>

<p align="center">
  <video src="./assets/carla_trajectory_transformer_demo.mp4" controls muted playsinline width="100%"></video>
  <br>
  <sub>CARLA closed-loop demo, 1080p MP4.</sub>
</p>

`Autonomous Driving Risk Scenario Benchmark Agent` is organized around a shared risk-scenario taxonomy. `nuScenes` mines and validates real-world scenario anchors, `nuPlan` evaluates logged replay and replay-based closed-loop behavior, `Bench2Drive` trains and diagnoses a vision E2E trajectory planner, and `CARLA` provides audit-gated closed-loop visual evidence under semantically matched scenarios. Each backend has a distinct role and is linked by the same scenario definitions.

## Scenario-Centric Design

The taxonomy in [configs/scenario_taxonomy.yaml](configs/scenario_taxonomy.yaml) defines the bridge between data mining and model evaluation.

| Backend | Role |
| --- | --- |
| `nuScenes` | Mine risk anchors from real-world logs and export retrieval, perception, BEV occupancy, and world-model slices. |
| `nuPlan` | Replay logged ego behavior and evaluate accumulated closed-loop error under the same scenario families. |
| `Bench2Drive` | Train and evaluate a multi-camera vision E2E trajectory planner on simulator driving data. |
| `CARLA` | Run audit-gated visual closed-loop rollouts for selected scenario targets. |

<p align="center">
  <img src="./assets/pipeline_overview.png" alt="Pipeline overview" width="100%">
</p>

## Vision E2E Planner Training

The Bench2Drive component trains a vision E2E trajectory planner from six RGB cameras and route features. The model predicts multimodal future ego waypoints together with control and brake heads, using transformer pooling over camera, route, and trajectory-mode tokens. The trained checkpoint is evaluated by supervised validation, a simplified model-in-the-loop rollout, and selected CARLA semantic rollouts.

| Item | Value |
| --- | --- |
| Input | six RGB camera views and route features |
| Model | `research` trajectory transformer with `4` trajectory modes |
| Training set | `40,223` train samples and `4,717` validation samples |
| Training runtime | `8`-GPU DDP, `24` epochs, `315.983s` |
| Supervised validation | ADE `1.599`, FDE `2.625`, brake F1 `0.884` |
| Closed-loop diagnostic | `64` proxy rollout cases; route completion `0.754`; closed-loop score `0.105` |
| CARLA evidence | one retained closed-loop demo with `0` collisions and safety override ratio `0.105` |

## Results

The trainval suite exports `24` scenario anchors, `48` paired scenario-mining queries, and aligned perception, BEV occupancy, and world-model slices. The exported counts are sampling caps over validated mined cases.

| Layer | Snapshot |
| --- | --- |
| Scenario mining | `24` anchors and `48` reference-aware queries |
| Learned reranking | `4,000` weakly labeled trainval groups; scene-held-out Recall@1 `1.000` |
| Perception slices | `24` mined risk slices with event-window actor supervision |
| BEV occupancy slices | `oracle_occupancy` IoU `1.000`; `context_drop_occupancy` IoU `0.553`; `risk_actor_only` IoU `0.105` |
| World-model benchmark | `24` scenario-conditioned slices; `kinematic_rollout` risk fidelity `0.869` |
| `ContextVAE` baseline | `7` forecast-compatible slices; `ADE 0.280`; `MinADE@5 0.207`; risk fidelity `0.841` |
| `nuPlan` replay regression | `576` SQLite logs scanned; `1556` candidates; `112` replay cases; `history_kinematic` ADE `0.916` |
| `nuPlan` closed-loop replay | `112` replay-simulation cases; `history_kinematic` ADE `1.027`; closed-loop score `0.950` |
| Bench2Drive vision E2E trajectory transformer | `44,940` cached multi-camera samples; `8`-GPU DDP runtime `316.0s`; temperature-calibrated ADE `1.599`; FDE `2.625`; brake F1 `0.884` |
| Bench2Drive model-in-the-loop proxy | `64` cases; route completion `0.754`; mean lateral error `1.332 m`; closed-loop score `0.105` |
| CARLA semantic demo mining | `1/1` audit-gated demo target; `267` frames; `13` Traffic Manager vehicles; `9` crosswalk pedestrians; direct model-control ratio `1.000`; safety override ratio `0.105`; `0` scripted vehicles; `0` collisions |
| Failure mining | `401` failure records, `83` clusters, and `24` benchmark update queries |
| Failure-aware ML retrieval | Pass@K improves from `20/24` to `24/24` with validation-gated candidate generation |

<p align="center">
  <img src="./assets/readme_overview.png" alt="Representative scene-mining outputs" width="100%">
</p>

<p align="center">
  <img src="./assets/world_model_results_overview.png" alt="World-model evaluation overview" width="100%">
</p>

<p align="center">
  <img src="./assets/nuplan_replay_case_studies.png" alt="nuPlan replay-regression case studies" width="100%">
</p>

<p align="center">
  <img src="./assets/nuplan_closed_loop_case_studies.png" alt="nuPlan closed-loop replay case studies" width="100%">
</p>

Detailed benchmark tables are in [docs/benchmark_snapshot.md](docs/benchmark_snapshot.md).

## Capabilities

- Local-Ollama natural-language query planning with deterministic retrieval and validation.
- Actor grounding, event localization, TTC, lane relation, crosswalk context, and BEV evidence rendering.
- Reference-aware scenario-mining benchmarks with scene, actor, and event-window supervision.
- Scenario-conditioned perception, sparse BEV occupancy, and world-model benchmark slices.
- Weakly supervised query-scene reranking and failure-aware candidate generation.
- Model-in-the-loop failure mining across perception, occupancy, world-model, replay-regression, and closed-loop metrics.
- Scenario-taxonomy alignment across `nuScenes` mining, `nuPlan` replay, Bench2Drive vision E2E planner training, and CARLA semantic demo mining.
- Result registry, artifact manifests, and dataset-backend inspection.

## Quickstart

```bash
conda env create -f environment.yml
conda activate nuscenes
```

Dataset links and archive layout are listed in [docs/dataset_downloads.md](docs/dataset_downloads.md).

Prepare data and build the `nuScenes` trainval index:

```bash
python -m nusc_scene_agent inspect-archives --workspace .
python -m nusc_scene_agent prepare-data --workspace . --profile trainval-full

python -m nusc_scene_agent build-index \
  --version v1.0-trainval \
  --dataroot data/sets/nuscenes \
  --db artifacts/index/v1.0-trainval.sqlite
```

Start the local model endpoint. Run `ollama serve` in a separate shell if the service is not already active:

```bash
ollama pull gemma4:latest
ollama serve
```

Run the full benchmark suite:

```bash
python -m nusc_scene_agent run-full-benchmark-suite
```

The suite is configured in [configs/full_benchmark_suite.yaml](configs/full_benchmark_suite.yaml). Stage-level commands are documented in [docs/usage.md](docs/usage.md).

## Data Policy

Dataset archives, extracted datasets, map files, SQLite indices, generated outputs, external repositories, and external prediction files are excluded from version control. The relevant directories include `archives/`, `data/`, `artifacts/`, `outputs/`, `external/`, and `external_predictions/`.

## Repository Layout

```text
src/nusc_scene_agent/    core library and CLI
benchmarks/              benchmark configs and exported benchmark JSON
configs/                 structured experiment configs
assets/                  static figures referenced by README and docs
docs/                    architecture, benchmark notes, usage, and dataset links
tests/                   unit tests for retrieval, validation, reporting, and benchmarks
environment.yml          conda-first environment
```

## Documentation

- [Usage](docs/usage.md)
- [Architecture Notes](docs/architecture.md)
- [Benchmark Snapshot Notes](docs/benchmark_snapshot.md)
- [Dataset Downloads](docs/dataset_downloads.md)

## License

Released under the [MIT License](LICENSE).

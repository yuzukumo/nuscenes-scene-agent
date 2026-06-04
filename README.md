<div align="center">

# nuScenes Scene Mining Agent

Natural-language risky-scene mining, validation, benchmark generation, and replay-based simulation evaluation on `nuScenes` and `nuPlan`.

`Python 3.10+` `Conda` `nuScenes` `nuPlan` `Ollama` `Scene Mining` `Benchmarking`

</div>

<p align="center">
  <img src="./assets/pipeline_overview.png" alt="Pipeline overview" width="100%">
</p>

`nuScenes Scene Mining Agent` builds a structured workflow from open-ended risk descriptions to validated driving cases and benchmark artifacts. The pipeline indexes datasets, plans structured scene queries with a local Ollama model, retrieves candidate scenes, validates them with geometry and map context, and exports benchmark layers for retrieval, perception, BEV occupancy, world-model evaluation, replay regression, and closed-loop replay.

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
- `nuPlan` replay-regression and closed-loop replay evaluation from SQLite scenario tags.
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

Run the end-to-end benchmark suite:

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

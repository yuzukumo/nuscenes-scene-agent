<div align="center">

# nuScenes Scene Mining Agent

Natural-language risky-scene retrieval, validation, and benchmark generation on `nuScenes`, with optional `nuPlan` replay-regression support.

`Python 3.10+` `Conda` `nuScenes` `nuPlan` `Ollama` `Scene Mining` `Benchmarking`

</div>

<p align="center">
  <img src="./assets/pipeline_overview.png" alt="Pipeline overview" width="100%">
</p>

`nuScenes Scene Mining Agent` mines risky driving scenes from `nuScenes`, validates them with geometric and map-based evidence, and converts the results into benchmark artifacts for scenario retrieval, perception, BEV occupancy, world-model evaluation, and replay regression.

The project focuses on dataset indexing, scene mining, validation, and benchmark construction. Learned modules are used as compact retrieval or forecasting components and are evaluated against explicit benchmark slices.

## Results

The default trainval suite exports `24` scenario anchors, `48` paired scenario-mining queries, and `24` aligned perception, BEV occupancy, and world-model cases. Counts are sampling caps over validated mined cases, not dataset limits.

| Layer | Snapshot |
| --- | --- |
| Scenario mining | `24` anchors and `48` reference-aware queries |
| Learned reranking | PyTorch query-scene scorer trained on `4,000` weakly labeled trainval groups; held-out Recall@1 `1.000` |
| Perception slices | `24` mined risk slices; proxy profiles separate delayed initialization and sparse crossing tracks |
| BEV occupancy slices | `24` sparse occupancy slices; `oracle_occupancy` IoU `1.000`, `context_drop_occupancy` `0.553`, `risk_actor_only` `0.105` |
| World-model benchmark | `24` scenario-conditioned slices with challenge tracks; `kinematic_rollout` risk fidelity `0.869` |
| `ContextVAE` baseline | forecast-compatible subset: `7` slices; `ADE 0.280`, `MinADE@5 0.207`, risk fidelity `0.841` |
| `nuPlan` replay regression | `576` SQLite logs scanned, `1556` candidate anchors collected, `112` balanced replay cases exported |
| Failure mining | `305` failure records, `56` clusters, and `24` benchmark update queries |

<p align="center">
  <img src="./assets/readme_overview.png" alt="Representative scene-mining outputs" width="100%">
</p>

<p align="center">
  <img src="./assets/world_model_results_overview.png" alt="World-model evaluation overview" width="100%">
</p>

<p align="center">
  <img src="./assets/nuplan_replay_case_studies.png" alt="nuPlan replay-regression case studies" width="100%">
</p>

Detailed benchmark tables are in [docs/benchmark_snapshot.md](docs/benchmark_snapshot.md).

## Capabilities

- Natural-language scene retrieval with `rule`, `llm`, and `hybrid` query modes.
- Deterministic validation using actor grounding, event localization, TTC, lane relation, and crosswalk context.
- Reference-aware scenario-mining benchmarks with scene, actor, and event-window supervision.
- Scenario-conditioned perception and sparse BEV occupancy slices for external tracking or detection outputs.
- Scenario-conditioned world-model slices for future rollouts and multi-modal forecasts.
- Structural multimodal reranking over language intent, BEV geometry, motion, and sensor visibility.
- Learned query-scene reranking from weakly supervised trainval scenario families.
- Model-in-the-loop failure mining from perception, occupancy, world-model, and replay-regression metrics.
- Optional `nuPlan` replay-regression benchmark generation from SQLite scenario tags.
- LangGraph research agent that reviews local artifacts through an Ollama model.

## Quickstart

```bash
conda env create -f environment.yml
conda activate nuscenes
```

Dataset links and archive layout are listed in [docs/dataset_downloads.md](docs/dataset_downloads.md).

Prepare `v1.0-mini`, build an index, and run one query:

```bash
python -m nusc_scene_agent prepare-data --workspace . --profile mini

python -m nusc_scene_agent build-index \
  --version v1.0-mini \
  --dataroot data/sets/nuscenes \
  --db artifacts/index/v1.0-mini.sqlite

python -m nusc_scene_agent query \
  "pedestrian crossing at close range ahead of ego lane" \
  --db artifacts/index/v1.0-mini.sqlite \
  --output outputs/mini_query \
  --query-mode rule
```

Generate the default trainval benchmark suite:

```bash
python -m nusc_scene_agent run-experiment-config \
  --config configs/risk_benchmark_suite.yaml
```

Additional workflows for Ollama, learned reranking, `ContextVAE`, `nuPlan`, failure mining, and registry export are in [docs/usage.md](docs/usage.md).

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

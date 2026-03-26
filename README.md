<div align="center">

# nuScenes Scene Mining Agent

Turn natural-language risk descriptions into validated `nuScenes` cases, evidence figures, and benchmark-ready case libraries.

`Python 3.10+` `Conda` `nuScenes` `Responses API` `Scene Mining & Benchmarking`

</div>

<p align="center">
  <img src="./assets/pipeline_overview.png" alt="Pipeline overview" width="100%">
</p>

`nuScenes Scene Mining Agent` is an agentic toolkit for the part of autonomous driving work that usually stays manual and messy: mining risky scenes, validating them with explicit evidence, and turning them into reusable evaluation assets.

It is intentionally not a driving policy, not an end-to-end training stack, and not a CARLA-heavy simulator project. The focus is the data-centric loop:

1. describe a risky scenario in natural language
2. translate it into structured retrieval hypotheses
3. search a `SQLite` scene index built from `nuScenes`
4. validate candidates with geometry, motion, TTC, and map context
5. export evidence figures, reports, case libraries, and benchmark summaries

## Why This Repo Is Useful

- Mine corner cases from `nuScenes` without manually scanning scenes one by one.
- Turn open-ended safety language into a reproducible retrieval workflow.
- Keep the system interpretable with deterministic validation instead of black-box matching alone.
- Build benchmark suites and failure-analysis artifacts that are useful for research demos, interviews, and portfolio projects.
- Compare `rule_only`, `llm_planner`, and `hybrid_agent` on the same risky-scene benchmark.

## Showcase

<p align="center">
  <img src="./assets/readme_showcase.png" alt="Risky scene mining showcase" width="100%">
</p>

Representative local outputs already produced by the pipeline include:

| Scenario family | What the export shows |
| --- | --- |
| Pedestrian crossing | crosswalk-aware front crossing with risky proximity |
| Stopped lead vehicle | same-lane blocking behavior ahead of ego |
| Oncoming vehicle | opposite-direction conflict with map-aware validation |
| Right-side cut-in | lateral merge into ego path with temporal evidence |

The default output surface is `BEV evidence PNG + Markdown/HTML reports`. No ready-made `mp4` video files are currently committed.

## Why It Counts As An Agent

The "agent" here is an orchestration agent, not a driving agent.

It combines:

- an `LLM planner` that converts free-form scene descriptions into structured retrieval hypotheses
- a retrieval engine over a prebuilt `SQLite` scene index
- an optional `LLM reranker` for semantic fit
- deterministic validators for geometry, motion, TTC, lane relation, and crosswalk context
- a reporting layer that exports evidence images, case reports, case libraries, and benchmark summaries

The core design choice is hybridization: `LLM for intent understanding and ranking`, `deterministic code for evidence, filtering, and reproducibility`.

## Data Policy

This repository does **not** include:

- `nuScenes` archives
- extracted dataset files
- map files
- local `SQLite` artifacts
- generated benchmark outputs

Those directories are intentionally excluded from version control through `.gitignore`, so the repo stays lightweight and GitHub-friendly.

Download links are listed here:

- [Dataset Downloads](docs/dataset_downloads.md)

If you want the shortest path:

- mini: `v1.0-mini.tgz`
- maps: `nuScenes-map-expansion-v1.3.zip`
- full trainval: `v1.0-trainval_meta.tgz` plus `v1.0-trainval01_blobs.tgz` to `v1.0-trainval10_blobs.tgz`

## Quickstart

### 1. Create the environment

```bash
conda env create -f environment.yml
conda activate nuscenes
```

If the environment already exists:

```bash
conda env update -f environment.yml --prune
conda activate nuscenes
```

### 2. Download data and place it locally

See:

- [Dataset Downloads](docs/dataset_downloads.md)

Expected local layout:

```text
archives/
  mini/
    v1.0-mini.tgz
  maps/
    nuScenes-map-expansion-v1.3.zip
  trainval/
    v1.0-trainval_meta.tgz
    v1.0-trainval01_blobs.tgz
    ...
    v1.0-trainval10_blobs.tgz
```

Check readiness:

```bash
python -m nusc_scene_agent inspect-archives --workspace .
```

### 3. Run the smallest end-to-end demo

Prepare `v1.0-mini`:

```bash
python -m nusc_scene_agent prepare-data \
  --workspace . \
  --profile mini
```

Build the mini index:

```bash
python -m nusc_scene_agent build-index \
  --version v1.0-mini \
  --dataroot data/sets/nuscenes \
  --db artifacts/index/v1.0-mini.sqlite
```

Run one query:

```bash
python -m nusc_scene_agent query \
  "pedestrian crossing very close in front of ego lane and risky" \
  --db artifacts/index/v1.0-mini.sqlite \
  --output outputs/demo_query \
  --query-mode rule
```

### 4. Scale to trainval

Prepare `v1.0-trainval + map expansion`:

```bash
python -m nusc_scene_agent prepare-data \
  --workspace . \
  --profile trainval-full
```

Build the full index:

```bash
python -m nusc_scene_agent build-index \
  --version v1.0-trainval \
  --dataroot data/sets/nuscenes \
  --db artifacts/index/v1.0-trainval.sqlite
```

Run the full benchmark:

```bash
python -m nusc_scene_agent benchmark \
  --config benchmarks/trainval_suite_v1.yaml \
  --db artifacts/index/v1.0-trainval.sqlite \
  --output outputs/trainval_suite_llm_hybrid_en_v1 \
  --query-mode hybrid \
  --rerank-mode llm
```

Run the language-stress comparison:

```bash
python -m nusc_scene_agent benchmark-compare \
  --config benchmarks/trainval_language_stress_v1.yaml \
  --db artifacts/index/v1.0-trainval.sqlite \
  --output outputs/trainval_language_stress_comparison_v2
```

## LLM Configuration

`LLM` support is optional. Rule-only retrieval works without any API configuration.

To enable `--query-mode llm|hybrid` or `--rerank-mode llm`, set:

```bash
export NUSC_SCENE_AGENT_LLM_BASE_URL="https://your-openai-compatible-endpoint"
export NUSC_SCENE_AGENT_LLM_API_KEY="your-key"
export NUSC_SCENE_AGENT_LLM_MODEL="your-model"
```

The runtime uses an OpenAI-compatible `Responses API` flow and keeps retrieval and validation runnable even if planning or reranking fails.

## Benchmark Snapshot

Current local snapshot on `v1.0-trainval + map expansion`:

### Core Suite

| Metric | Value |
| --- | --- |
| Queries | `16` |
| Pass@1 | `16/16` |
| Pass@K | `16/16` |
| Selected cases | `36` |
| Unique cases | `33` |
| Unique passed cases | `29` |
| Mean best validation score | `89.26` |

### Language-Stress Comparison

| Profile | Pass@1 | Mean best score |
| --- | --- | --- |
| `rule_only` | `16/16` | `90.64` |
| `llm_planner` | `15/16` | `86.29` |
| `hybrid_agent` | `16/16` | `91.34` |

Additional signal:

- planner signal divergence: `12 / 16`
- hybrid arbitration improves robustness over naive planner-only behavior
- hard-case taxonomy exposes overlap, borderline, and behavior-gap failure modes

More context:

- [Architecture Notes](docs/architecture.md)
- [Benchmark Snapshot Notes](docs/benchmark_snapshot.md)

## Repository Layout

```text
src/nusc_scene_agent/    core library and CLI
benchmarks/              benchmark configs
assets/                  repo-safe showcase figures
docs/                    architecture and dataset notes
tests/                   unit tests for retrieval, validation, reporting, and benchmarks
environment.yml          conda-first environment
```

Large local-only directories are intentionally not tracked:

- `archives/`
- `data/`
- `artifacts/`
- `outputs/`

## Command Surface

The CLI currently supports:

- `inspect-archives`
- `prepare-data`
- `build-index`
- `query`
- `benchmark`
- `benchmark-compare`
- `demo`

This gives the repo a practical workflow from archive inspection to dataset preparation, indexing, scene retrieval, validation, and benchmark export.

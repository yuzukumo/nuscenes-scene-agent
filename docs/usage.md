# Usage

This page lists commands for reproducing the local benchmark pipeline. Generated files are written under `outputs/`, `artifacts/`, or `external/`; these directories are excluded from version control.

## Environment

```bash
conda env create -f environment.yml
conda activate nuscenes
```

To update an existing environment:

```bash
conda env update -f environment.yml --prune
conda activate nuscenes
```

## Data Preparation

Check local archives:

```bash
python -m nusc_scene_agent inspect-archives --workspace .
```

Prepare `nuScenes v1.0-trainval` with map expansion:

```bash
python -m nusc_scene_agent prepare-data --workspace . --profile trainval-full
```

Build the trainval index:

```bash
python -m nusc_scene_agent build-index \
  --version v1.0-trainval \
  --dataroot data/sets/nuscenes \
  --db artifacts/index/v1.0-trainval.sqlite
```

## Local LLM

The benchmark suite uses an Ollama server for structured query planning and artifact review. Run `ollama serve` in a separate shell if the service is not already active.

```bash
ollama pull gemma4:latest
ollama serve
```

The default endpoint and model are also configurable:

```bash
export NUSC_SCENE_AGENT_OLLAMA_BASE_URL="http://127.0.0.1:11434"
export NUSC_SCENE_AGENT_OLLAMA_MODEL="gemma4:latest"
```

## End-to-End Suite

```bash
python -m nusc_scene_agent run-full-benchmark-suite
```

This executes the configured pipeline in [configs/full_benchmark_suite.yaml](../configs/full_benchmark_suite.yaml):

- case-library generation from `benchmarks/trainval_suite_v1.yaml`
- `nuScenes` scenario-mining, perception, BEV occupancy, and world-model benchmark generation
- `nuPlan` replay-regression sweep
- `nuPlan` closed-loop replay sweep
- model-in-the-loop failure mining
- result-registry export

## Stage-Level Commands

Run the `nuScenes` benchmark layers from an existing case library:

```bash
python -m nusc_scene_agent run-experiment-config \
  --config configs/risk_benchmark_suite.yaml
```

Train the weakly supervised query-scene reranker:

```bash
pip install -e ".[learned]"

python -m nusc_scene_agent train-large-learned-retriever \
  --db artifacts/index/v1.0-trainval.sqlite \
  --output outputs/learned_retriever_trainval_large_v2 \
  --max-groups-per-family 1000 \
  --epochs 20 \
  --negatives-per-query 12
```

Evaluate failure-aware candidate generation:

```bash
python -m nusc_scene_agent run-experiment-config \
  --config configs/failure_aware_reranking.yaml
```

Run structural multimodal retrieval for a single query:

```bash
python -m nusc_scene_agent run-multimodal-retrieval \
  "pedestrian crossing close in front of ego with dense surrounding traffic context" \
  --db artifacts/index/v1.0-trainval.sqlite \
  --output outputs/multimodal_retrieval_v1 \
  --top-k 10 \
  --candidate-pool 64
```

Run the `ContextVAE` world-model baseline:

```bash
pip install -e ".[forecast]"
git clone https://github.com/xupei0610/ContextVAE.git external/ContextVAE

python -m nusc_scene_agent run-contextvae-world-model-study \
  --benchmark benchmarks/trainval_world_model_slices_v1.json \
  --dataroot data/sets/nuscenes \
  --version v1.0-trainval \
  --output outputs/contextvae_world_model_study_v1 \
  --repo external/ContextVAE \
  --checkpoint external/ContextVAE/models/nuscenes_res18 \
  --device cuda \
  --batch-size 4 \
  --mode-count 5 \
  --clustering-samples 2000
```

Run `nuPlan` replay and closed-loop sweeps:

```bash
python -m nusc_scene_agent inspect-nuplan --dataset-root data/nuplan/dataset

python -m nusc_scene_agent run-experiment-config \
  --config configs/nuplan_replay_sweep_medium.yaml

python -m nusc_scene_agent run-experiment-config \
  --config configs/nuplan_closed_loop_sweep_medium.yaml
```

Run failure mining from generated metric artifacts:

```bash
python -m nusc_scene_agent run-failure-mining \
  --input outputs/trainval_bev_occupancy_proxy_study_v1 \
  --input outputs/trainval_world_model_proxy_study_v1 \
  --input outputs/contextvae_world_model_study_v1 \
  --input outputs/nuplan_replay_sweep_v1 \
  --input outputs/nuplan_closed_loop_sweep_v1 \
  --output outputs/model_in_the_loop_failure_mining_v1 \
  --max-queries 24
```

Export the result registry:

```bash
python -m nusc_scene_agent run-experiment-config \
  --config configs/result_registry.yaml
```

Run the LangGraph artifact review workflow:

```bash
pip install -e ".[agent]"

python -m nusc_scene_agent run-research-agent \
  --output outputs/research_agent_review_v1
```

Inspect local benchmark and dataset metadata:

```bash
python -m nusc_scene_agent inspect-dataset-backends \
  --output outputs/dataset_backends_inventory.json

python -m nusc_scene_agent export-benchmark-registry \
  --output outputs/benchmark_registry.json
```

# Usage

This page contains the main commands for reproducing local experiments. Generated outputs are written under `outputs/` and are excluded from version control.

## Environment

```bash
conda env create -f environment.yml
conda activate nuscenes
```

If the environment already exists:

```bash
conda env update -f environment.yml --prune
conda activate nuscenes
```

## Data Preparation

Check archive readiness:

```bash
python -m nusc_scene_agent inspect-archives --workspace .
```

Prepare `v1.0-mini`:

```bash
python -m nusc_scene_agent prepare-data --workspace . --profile mini
```

Prepare `v1.0-trainval`:

```bash
python -m nusc_scene_agent prepare-data --workspace . --profile trainval-full
```

Build indices:

```bash
python -m nusc_scene_agent build-index \
  --version v1.0-mini \
  --dataroot data/sets/nuscenes \
  --db artifacts/index/v1.0-mini.sqlite

python -m nusc_scene_agent build-index \
  --version v1.0-trainval \
  --dataroot data/sets/nuscenes \
  --db artifacts/index/v1.0-trainval.sqlite
```

## Ollama Query Modes

Rule-only retrieval runs without model configuration. The `llm` query mode, `hybrid` query mode, `llm` reranker, and research-agent commands use a local Ollama server.

```bash
export NUSC_SCENE_AGENT_OLLAMA_BASE_URL="http://127.0.0.1:11434"
export NUSC_SCENE_AGENT_OLLAMA_MODEL="gemma4:latest"
```

Run a hybrid benchmark:

```bash
python -m nusc_scene_agent benchmark \
  --config benchmarks/trainval_suite_v1.yaml \
  --db artifacts/index/v1.0-trainval.sqlite \
  --output outputs/trainval_suite_llm_hybrid_en_v1 \
  --query-mode hybrid \
  --rerank-mode llm
```

## Benchmark Suite

```bash
python -m nusc_scene_agent run-experiment-config \
  --config configs/risk_benchmark_suite.yaml
```

This generates scenario-mining, perception, BEV occupancy, and world-model benchmark artifacts, then runs proxy studies for the derived benchmark layers.

## Learned Scene Reranking

Install the optional dependency:

```bash
pip install -e ".[learned]"
```

Train the reranker:

```bash
python -m nusc_scene_agent train-large-learned-retriever \
  --db artifacts/index/v1.0-trainval.sqlite \
  --output outputs/learned_retriever_trainval_large_v2 \
  --max-groups-per-family 1000 \
  --epochs 20 \
  --negatives-per-query 12
```

Use the checkpoint:

```bash
python -m nusc_scene_agent query \
  "pedestrian crossing close in front of ego with dense surrounding traffic context" \
  --db artifacts/index/v1.0-trainval.sqlite \
  --output outputs/learned_query_demo_v1 \
  --query-mode rule \
  --rerank-mode learned \
  --learned-reranker-checkpoint outputs/learned_retriever_trainval_large_v2/learned_retriever.pt
```

## Structural Multimodal Reranking

```bash
python -m nusc_scene_agent run-multimodal-retrieval \
  "pedestrian crossing close in front of ego with dense surrounding traffic context" \
  --db artifacts/index/v1.0-trainval.sqlite \
  --output outputs/multimodal_retrieval_demo_v1 \
  --top-k 10 \
  --candidate-pool 64

python -m nusc_scene_agent query \
  "pedestrian crossing close in front of ego with dense surrounding traffic context" \
  --db artifacts/index/v1.0-trainval.sqlite \
  --output outputs/multimodal_query_demo_v1 \
  --query-mode rule \
  --rerank-mode multimodal
```

## ContextVAE World-Model Baseline

Install optional forecast dependencies:

```bash
pip install -e ".[forecast]"
```

Clone the external repository:

```bash
git clone https://github.com/xupei0610/ContextVAE.git external/ContextVAE
```

Run the integration:

```bash
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

If the checkpoint is missing, the integration downloads the public `nuscenes_res18` release automatically.

## nuPlan Replay Regression

After extracting `nuPlan` under `data/nuplan/dataset`:

```bash
python -m nusc_scene_agent inspect-nuplan --dataset-root data/nuplan/dataset

python -m nusc_scene_agent run-nuplan-replay-study \
  --split-dir data/nuplan/dataset/nuplan-v1.1/splits/mini \
  --output outputs/nuplan_mini_replay_study_v2 \
  --max-dbs 64 \
  --max-cases 16 \
  --max-cases-per-db 4
```

Equivalent YAML entry point:

```bash
python -m nusc_scene_agent run-experiment-config \
  --config configs/nuplan_replay_mini.yaml
```

Cross-split sweep:

```bash
python -m nusc_scene_agent run-experiment-config \
  --config configs/nuplan_replay_sweep_medium.yaml
```

## Failure Mining

```bash
python -m nusc_scene_agent run-failure-mining \
  --input outputs/trainval_bev_occupancy_proxy_study_v1 \
  --input outputs/trainval_world_model_proxy_study_v1 \
  --input outputs/contextvae_world_model_study_v1 \
  --input outputs/nuplan_replay_sweep_v1 \
  --output outputs/model_in_the_loop_failure_mining_v1 \
  --max-queries 16
```

## Research Agent

Install optional LangGraph support:

```bash
pip install -e ".[agent]"
```

Run the Ollama-backed artifact review:

```bash
python -m nusc_scene_agent run-research-agent \
  --output outputs/research_agent_review_v1
```

## Registry and Backend Inspection

```bash
python -m nusc_scene_agent inspect-dataset-backends \
  --output outputs/dataset_backends_inventory.json

python -m nusc_scene_agent export-benchmark-registry \
  --output outputs/benchmark_registry.json
```

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

from nusc_scene_agent.benchmark_comparison import (
    build_benchmark_comparison,
    default_benchmark_profiles,
    write_benchmark_comparison,
)
from nusc_scene_agent.ablation_study import default_ablation_profiles, write_ablation_manifest
from nusc_scene_agent.case_library_enrichment import enrich_case_library
from nusc_scene_agent.counterfactual_benchmark import generate_counterfactual_benchmark_from_case_library
from nusc_scene_agent.benchmark_exports import write_benchmark_exports
from nusc_scene_agent.benchmark_metrics import build_benchmark_metrics, write_benchmark_metrics
from nusc_scene_agent.bev_occupancy_benchmark import (
    BEV_OCCUPANCY_PROXY_PROFILES,
    DEFAULT_BEV_OCCUPANCY_BENCHMARK,
    adapt_perception_predictions_to_bev_occupancy,
    compare_bev_occupancy_evaluations,
    evaluate_bev_occupancy_predictions,
    generate_bev_occupancy_benchmark_from_perception_benchmark,
    generate_proxy_bev_occupancy_predictions,
    run_proxy_bev_occupancy_study,
)
from nusc_scene_agent.case_library import build_case_library, write_case_library
from nusc_scene_agent.benchmark_schema import load_benchmark_config
from nusc_scene_agent.benchmark_registry import build_default_benchmark_registry, write_benchmark_registry
from nusc_scene_agent.contextvae_integration import (
    DEFAULT_CONTEXTVAE_CHECKPOINT,
    DEFAULT_CONTEXTVAE_OUTPUT,
    DEFAULT_CONTEXTVAE_REPO,
    run_contextvae_world_model_study,
)
from nusc_scene_agent.data_utils import DEFAULT_DATAROOT, PREPARE_PROFILES, discover_archive_inventory, prepare_data
from nusc_scene_agent.dataset_backends import inspect_dataset_backends, write_dataset_backend_inventory
from nusc_scene_agent.experiment_config import run_experiment_config
from nusc_scene_agent.failure_mining import mine_model_failures
from nusc_scene_agent.failure_aware_reranking import (
    DEFAULT_FAILURE_AWARE_RERANKING_OUTPUT,
    DEFAULT_FAILURE_UPDATE_QUERIES,
    run_failure_aware_reranking_eval,
)
from nusc_scene_agent.gallery import build_benchmark_gallery, build_comparison_browser
from nusc_scene_agent.indexing import build_index
from nusc_scene_agent.langgraph_agent import run_langgraph_query_pipeline
from nusc_scene_agent.learned_retrieval import (
    DEFAULT_LARGE_LEARNED_RETRIEVER_OUTPUT,
    DEFAULT_LEARNED_RETRIEVER_CHECKPOINT,
    DEFAULT_LEARNED_RETRIEVER_OUTPUT,
    LearnedRetrieverConfig,
    run_learned_retrieval_report,
    train_learned_scene_retriever,
    train_weakly_supervised_scene_retriever,
)
from nusc_scene_agent.llm_client import DEFAULT_TIMEOUT_S, LLMConfig
from nusc_scene_agent.multimodal_retrieval import run_multimodal_retrieval_report
from nusc_scene_agent.nuplan_closed_loop import (
    DEFAULT_NUPLAN_CLOSED_LOOP_OUTPUT,
    DEFAULT_NUPLAN_CLOSED_LOOP_PROFILES,
    run_nuplan_closed_loop_study,
)
from nusc_scene_agent.nuplan_closed_loop_sweep import DEFAULT_NUPLAN_CLOSED_LOOP_SWEEP_OUTPUT
from nusc_scene_agent.nuplan_replay import (
    DEFAULT_NUPLAN_REPLAY_BENCHMARK,
    DEFAULT_NUPLAN_SPLIT,
    NUPLAN_REPLAY_PROFILES,
    compare_nuplan_replay_evaluations,
    evaluate_nuplan_rollouts,
    generate_nuplan_proxy_rollouts,
    generate_nuplan_replay_benchmark,
    inspect_nuplan_dataset,
    render_nuplan_replay_case_studies,
    run_nuplan_replay_study,
)
from nusc_scene_agent.nuplan_sweep import DEFAULT_NUPLAN_REPLAY_SWEEP_OUTPUT
from nusc_scene_agent.perception_benchmark import (
    PROXY_PERCEPTION_PROFILES,
    PREDICTION_COVERAGE_MODES,
    adapt_filter_and_evaluate_nuscenes_predictions,
    adapt_and_evaluate_nuscenes_predictions,
    adapt_nuscenes_predictions,
    compare_perception_evaluations,
    evaluate_perception_predictions,
    filter_perception_benchmark_by_predictions,
    generate_perception_benchmark_from_scenario_config,
    generate_proxy_perception_predictions,
    run_proxy_perception_study,
)
from nusc_scene_agent.pipeline import run_query_pipeline
from nusc_scene_agent.research_agent import (
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_RESEARCH_AGENT_OUTPUT,
    run_research_agent,
)
from nusc_scene_agent.scenario_mining_benchmark import generate_scenario_mining_benchmark_from_case_library
from nusc_scene_agent.unified_schema import load_unified_case_source, write_unified_case_collection
from nusc_scene_agent.world_model_benchmark import (
    NUSCENES_FORECAST_MODE_SELECTIONS,
    NUSCENES_FORECAST_BASELINE_PROFILES,
    WORLD_MODEL_PROXY_PROFILES,
    adapt_and_evaluate_nuscenes_forecast_predictions,
    adapt_nuscenes_forecast_predictions,
    adapt_world_model_predictions,
    compare_world_model_evaluations,
    evaluate_world_model_predictions,
    export_world_model_replay,
    generate_proxy_world_model_predictions,
    generate_nuscenes_forecast_baselines,
    generate_world_model_benchmark_from_perception_benchmark,
    run_nuscenes_forecast_baselines,
    run_proxy_world_model_study,
)
from nusc_scene_agent.validation import ValidationConfig
from nusc_scene_agent.world_model_case_studies import render_world_model_case_studies


DEFAULT_DB = Path("artifacts/index/v1.0-trainval.sqlite")
DEFAULT_BENCHMARK = Path("benchmarks/smoke_queries.yaml")
DEFAULT_TRAINVAL_BENCHMARK = Path("benchmarks/trainval_suite_v1.yaml")
DEFAULT_COMPARE_BENCHMARK = Path("benchmarks/trainval_language_stress_v1.yaml")
DEFAULT_SCENARIO_MINING_BENCHMARK = Path("benchmarks/trainval_scenario_mining_v1.yaml")
DEFAULT_PERCEPTION_BENCHMARK = Path("benchmarks/trainval_perception_slices_v1.json")
DEFAULT_WORLD_MODEL_BENCHMARK = Path("benchmarks/trainval_world_model_slices_v1.json")


def _add_llm_connection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ollama-base-url", default="")
    parser.add_argument("--ollama-model", default="")
    parser.add_argument("--ollama-timeout", type=float, default=DEFAULT_TIMEOUT_S)


def _add_llm_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--query-mode", choices=["rule", "llm", "hybrid"], default="rule")
    parser.add_argument("--rerank-mode", choices=["none", "llm", "multimodal", "learned"], default="none")
    parser.add_argument("--learned-reranker-checkpoint", default="")
    _add_llm_connection_args(parser)


def _resolve_llm_config(
    args: argparse.Namespace,
    require: bool = False,
    default_base_url: str = "",
    default_model: str = "",
) -> Optional[LLMConfig]:
    env_config = LLMConfig.from_env()
    base_url = str(
        getattr(args, "ollama_base_url", "")
        or (env_config.base_url if env_config else "")
        or default_base_url
    ).strip()
    model = str(getattr(args, "ollama_model", "") or (env_config.model if env_config else "") or default_model).strip()
    timeout_s = float(getattr(args, "ollama_timeout", DEFAULT_TIMEOUT_S))
    if not (base_url and model):
        if require:
            raise ValueError("Ollama configuration requires --ollama-base-url and --ollama-model.")
        return None
    return LLMConfig(base_url=base_url, model=model, timeout_s=timeout_s)


def _optional_path(value: str) -> Optional[Path]:
    value = str(value or "").strip()
    return Path(value) if value else None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="nuScenes scene mining and benchmark generation toolkit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect-archives", help="Inspect local nuScenes archives and readiness.")
    inspect_parser.add_argument("--workspace", default=".", help="Workspace root that contains the archives.")

    prepare_parser = subparsers.add_parser("prepare-data", help="Unpack nuScenes archives into a dataroot.")
    prepare_parser.add_argument("--archive", action="append", default=[], help="Archive path, repeatable.")
    prepare_parser.add_argument("--workspace", default=".", help="Workspace root that contains the archives.")
    prepare_parser.add_argument("--dataroot", default=str(DEFAULT_DATAROOT), help="Extraction target.")
    prepare_parser.add_argument("--profile", choices=PREPARE_PROFILES, default="trainval-full")

    build_parser = subparsers.add_parser("build-index", help="Build a SQLite index from nuScenes annotations.")
    build_parser.add_argument("--version", default="v1.0-trainval")
    build_parser.add_argument("--dataroot", default=str(DEFAULT_DATAROOT))
    build_parser.add_argument("--db", default=str(DEFAULT_DB))
    build_parser.add_argument("--scene-limit", type=int, default=0)

    query_parser = subparsers.add_parser("query", help="Run one natural-language risk query.")
    query_parser.add_argument("text", help="Natural-language risk description.")
    query_parser.add_argument("--db", default=str(DEFAULT_DB))
    query_parser.add_argument("--output", default="outputs/query")
    query_parser.add_argument("--top-k", type=int, default=5)
    query_parser.add_argument("--candidate-pool", type=int, default=12)
    _add_llm_args(query_parser)

    multimodal_retrieval_parser = subparsers.add_parser(
        "run-multimodal-retrieval",
        help="Score one risk query with the structural multimodal retrieval model.",
    )
    multimodal_retrieval_parser.add_argument("text", help="Natural-language risk description.")
    multimodal_retrieval_parser.add_argument("--db", default=str(DEFAULT_DB))
    multimodal_retrieval_parser.add_argument("--output", default="outputs/multimodal_retrieval")
    multimodal_retrieval_parser.add_argument("--top-k", type=int, default=20)
    multimodal_retrieval_parser.add_argument("--candidate-pool", type=int, default=64)
    multimodal_retrieval_parser.add_argument("--query-mode", choices=["rule", "llm"], default="rule")
    _add_llm_connection_args(multimodal_retrieval_parser)

    train_learned_parser = subparsers.add_parser(
        "train-learned-retriever",
        help="Train a compact query-scene pairwise reranker from reference-aware scenario-mining anchors.",
    )
    train_learned_parser.add_argument("--benchmark", default=str(DEFAULT_SCENARIO_MINING_BENCHMARK))
    train_learned_parser.add_argument("--db", default="artifacts/index/v1.0-trainval.sqlite")
    train_learned_parser.add_argument("--output", default=str(DEFAULT_LEARNED_RETRIEVER_OUTPUT))
    train_learned_parser.add_argument("--epochs", type=int, default=80)
    train_learned_parser.add_argument("--negatives-per-query", type=int, default=12)
    train_learned_parser.add_argument("--candidate-pool", type=int, default=64)
    train_learned_parser.add_argument("--text-hash-dim", type=int, default=256)
    train_learned_parser.add_argument("--hidden-dim", type=int, default=128)
    train_learned_parser.add_argument("--embedding-dim", type=int, default=64)
    train_learned_parser.add_argument("--learning-rate", type=float, default=1e-3)
    train_learned_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_learned_parser.add_argument("--validation-fraction", type=float, default=0.25)
    train_learned_parser.add_argument("--seed", type=int, default=7)
    train_learned_parser.add_argument("--device", default="")

    train_large_learned_parser = subparsers.add_parser(
        "train-large-learned-retriever",
        help="Train a learned reranker from weakly labeled trainval scenario families.",
    )
    train_large_learned_parser.add_argument("--db", default="artifacts/index/v1.0-trainval.sqlite")
    train_large_learned_parser.add_argument("--output", default=str(DEFAULT_LARGE_LEARNED_RETRIEVER_OUTPUT))
    train_large_learned_parser.add_argument("--max-groups-per-family", type=int, default=1000)
    train_large_learned_parser.add_argument("--epochs", type=int, default=20)
    train_large_learned_parser.add_argument("--negatives-per-query", type=int, default=12)
    train_large_learned_parser.add_argument("--text-hash-dim", type=int, default=256)
    train_large_learned_parser.add_argument("--hidden-dim", type=int, default=128)
    train_large_learned_parser.add_argument("--embedding-dim", type=int, default=64)
    train_large_learned_parser.add_argument("--learning-rate", type=float, default=1e-3)
    train_large_learned_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_large_learned_parser.add_argument("--validation-fraction", type=float, default=0.2)
    train_large_learned_parser.add_argument("--seed", type=int, default=7)
    train_large_learned_parser.add_argument("--device", default="")

    learned_retrieval_parser = subparsers.add_parser(
        "run-learned-retrieval",
        help="Score one risk query with a trained query-scene reranker checkpoint.",
    )
    learned_retrieval_parser.add_argument("text", help="Natural-language risk description.")
    learned_retrieval_parser.add_argument("--db", default="artifacts/index/v1.0-trainval.sqlite")
    learned_retrieval_parser.add_argument("--checkpoint", default=str(DEFAULT_LEARNED_RETRIEVER_CHECKPOINT))
    learned_retrieval_parser.add_argument("--output", default="outputs/learned_retrieval_v1")
    learned_retrieval_parser.add_argument("--top-k", type=int, default=20)
    learned_retrieval_parser.add_argument("--candidate-pool", type=int, default=64)

    failure_aware_reranking_parser = subparsers.add_parser(
        "run-failure-aware-reranking",
        help="Evaluate learned reranking on failure-mined scene queries.",
    )
    failure_aware_reranking_parser.add_argument("--query-config", default=str(DEFAULT_FAILURE_UPDATE_QUERIES))
    failure_aware_reranking_parser.add_argument("--db", default="artifacts/index/v1.0-trainval.sqlite")
    failure_aware_reranking_parser.add_argument("--checkpoint", default=str(DEFAULT_LEARNED_RETRIEVER_CHECKPOINT))
    failure_aware_reranking_parser.add_argument("--output", default=str(DEFAULT_FAILURE_AWARE_RERANKING_OUTPUT))
    failure_aware_reranking_parser.add_argument("--candidate-pool", type=int, default=48)
    failure_aware_reranking_parser.add_argument("--top-k", type=int, default=3)
    failure_aware_reranking_parser.add_argument("--max-queries", type=int, default=24)

    langgraph_query_parser = subparsers.add_parser(
        "langgraph-query",
        help="Run one natural-language risk query through a LangGraph orchestration layer.",
    )
    langgraph_query_parser.add_argument("text", help="Natural-language risk description.")
    langgraph_query_parser.add_argument("--db", default=str(DEFAULT_DB))
    langgraph_query_parser.add_argument("--output", default="outputs/langgraph_query")
    langgraph_query_parser.add_argument("--top-k", type=int, default=5)
    langgraph_query_parser.add_argument("--candidate-pool", type=int, default=12)
    _add_llm_args(langgraph_query_parser)

    research_agent_parser = subparsers.add_parser(
        "run-research-agent",
        help="Run an Ollama-backed research agent over local benchmark artifacts.",
    )
    research_agent_parser.add_argument("--output", default=str(DEFAULT_RESEARCH_AGENT_OUTPUT))
    research_agent_parser.add_argument("--artifact", action="append", default=[], help="Benchmark artifact path, repeatable.")
    research_agent_parser.add_argument(
        "--focus",
        default="Assess benchmark gaps and propose evidence-based next actions.",
        help="Focused instruction for the research agent.",
    )
    research_agent_parser.add_argument("--ollama-base-url", default=DEFAULT_OLLAMA_BASE_URL)
    research_agent_parser.add_argument("--ollama-model", default=DEFAULT_OLLAMA_MODEL)
    research_agent_parser.add_argument("--ollama-timeout", type=float, default=DEFAULT_TIMEOUT_S)

    failure_mining_parser = subparsers.add_parser(
        "run-failure-mining",
        help="Mine model-evaluation failures and generate benchmark update queries.",
    )
    failure_mining_parser.add_argument(
        "--input",
        action="append",
        default=[],
        required=True,
        help="Evaluation file or directory. Repeatable.",
    )
    failure_mining_parser.add_argument("--output", default="outputs/model_in_the_loop_failure_mining_v1")
    failure_mining_parser.add_argument("--max-queries", type=int, default=24)
    failure_mining_parser.add_argument("--min-count", type=int, default=1)

    benchmark_parser = subparsers.add_parser("benchmark", help="Run a YAML benchmark query suite.")
    benchmark_parser.add_argument("--config", default=str(DEFAULT_BENCHMARK))
    benchmark_parser.add_argument("--db", default=str(DEFAULT_DB))
    benchmark_parser.add_argument("--output", default="outputs/benchmark")
    benchmark_parser.add_argument("--candidate-pool", type=int, default=12)
    _add_llm_args(benchmark_parser)

    generate_counterfactual_parser = subparsers.add_parser(
        "generate-counterfactual-benchmark",
        help="Generate a contrastive counterfactual benchmark from a case library.",
    )
    generate_counterfactual_parser.add_argument(
        "--case-library",
        default="outputs/trainval_case_library_v1/case_library.json",
        help="Path to case_library.json used as benchmark anchors.",
    )
    generate_counterfactual_parser.add_argument(
        "--output",
        default="benchmarks/trainval_counterfactual_reference_v1.yaml",
        help="Output YAML path for the generated benchmark.",
    )
    generate_counterfactual_parser.add_argument("--max-cases", type=int, default=6)

    generate_scenario_parser = subparsers.add_parser(
        "generate-scenario-mining-benchmark",
        help="Generate a reference-aware scenario mining benchmark from a case library.",
    )
    generate_scenario_parser.add_argument(
        "--case-library",
        default="outputs/trainval_case_library_v1/case_library.json",
        help="Path to case_library.json used as scenario anchors.",
    )
    generate_scenario_parser.add_argument(
        "--output",
        default="benchmarks/trainval_scenario_mining_v1.yaml",
        help="Output YAML path for the generated benchmark.",
    )
    generate_scenario_parser.add_argument("--max-cases", type=int, default=8)

    generate_perception_parser = subparsers.add_parser(
        "generate-perception-benchmark",
        help="Generate a scenario-conditioned perception benchmark from a scenario mining benchmark.",
    )
    generate_perception_parser.add_argument("--config", default=str(DEFAULT_SCENARIO_MINING_BENCHMARK))
    generate_perception_parser.add_argument("--db", default="artifacts/index/v1.0-trainval.sqlite")
    generate_perception_parser.add_argument("--output", default=str(DEFAULT_PERCEPTION_BENCHMARK))

    generate_world_model_parser = subparsers.add_parser(
        "generate-world-model-benchmark",
        help="Generate a scenario-conditioned world-model benchmark from a perception benchmark.",
    )
    generate_world_model_parser.add_argument("--benchmark", default=str(DEFAULT_PERCEPTION_BENCHMARK))
    generate_world_model_parser.add_argument("--db", default="artifacts/index/v1.0-trainval.sqlite")
    generate_world_model_parser.add_argument("--output", default=str(DEFAULT_WORLD_MODEL_BENCHMARK))

    proxy_predictions_parser = subparsers.add_parser(
        "generate-proxy-perception-predictions",
        help="Generate proxy prediction tracks for a perception benchmark.",
    )
    proxy_predictions_parser.add_argument("--benchmark", default=str(DEFAULT_PERCEPTION_BENCHMARK))
    proxy_predictions_parser.add_argument("--output", default="outputs/proxy_perception_predictions.json")
    proxy_predictions_parser.add_argument("--profile", choices=PROXY_PERCEPTION_PROFILES, default="oracle_tracking")

    adapt_predictions_parser = subparsers.add_parser(
        "adapt-nuscenes-predictions",
        help="Adapt official nuScenes detection or tracking outputs to the local perception-evaluation schema.",
    )
    adapt_predictions_parser.add_argument("--benchmark", default=str(DEFAULT_PERCEPTION_BENCHMARK))
    adapt_predictions_parser.add_argument("--db", default="artifacts/index/v1.0-trainval.sqlite")
    adapt_predictions_parser.add_argument("--input", required=True, help="Path to the official nuScenes prediction JSON.")
    adapt_predictions_parser.add_argument("--output", default="outputs/adapted_nuscenes_predictions.json")
    adapt_predictions_parser.add_argument("--task", choices=["tracking", "detection"], default="tracking")

    filter_perception_parser = subparsers.add_parser(
        "filter-perception-benchmark",
        help="Filter a perception benchmark to cases covered by a prediction file.",
    )
    filter_perception_parser.add_argument("--benchmark", default=str(DEFAULT_PERCEPTION_BENCHMARK))
    filter_perception_parser.add_argument("--predictions", required=True)
    filter_perception_parser.add_argument("--output", default="outputs/filtered_perception_benchmark.json")
    filter_perception_parser.add_argument("--coverage-mode", choices=PREDICTION_COVERAGE_MODES, default="full_window")

    evaluate_perception_parser = subparsers.add_parser(
        "evaluate-perception-predictions",
        help="Evaluate prediction tracks on the scenario-conditioned perception benchmark.",
    )
    evaluate_perception_parser.add_argument("--benchmark", default=str(DEFAULT_PERCEPTION_BENCHMARK))
    evaluate_perception_parser.add_argument(
        "--predictions",
        required=True,
        help="Prediction JSON with per-sample actor tracks.",
    )
    evaluate_perception_parser.add_argument("--output", default="outputs/perception_evaluation")
    evaluate_perception_parser.add_argument("--profile-name", default="")
    evaluate_perception_parser.add_argument("--match-distance-m", type=float, default=2.0)

    proxy_study_parser = subparsers.add_parser(
        "run-proxy-perception-study",
        help="Run a proxy perception comparison on the scenario-conditioned benchmark.",
    )
    proxy_study_parser.add_argument("--benchmark", default=str(DEFAULT_PERCEPTION_BENCHMARK))
    proxy_study_parser.add_argument("--output", default="outputs/trainval_perception_proxy_study_v1")

    proxy_world_model_predictions_parser = subparsers.add_parser(
        "generate-proxy-world-model-predictions",
        help="Generate proxy world-model rollouts for a world-model benchmark.",
    )
    proxy_world_model_predictions_parser.add_argument("--benchmark", default=str(DEFAULT_WORLD_MODEL_BENCHMARK))
    proxy_world_model_predictions_parser.add_argument("--output", default="outputs/proxy_world_model_predictions.json")
    proxy_world_model_predictions_parser.add_argument("--profile", choices=WORLD_MODEL_PROXY_PROFILES, default="oracle_rollout")

    adapt_world_model_predictions_parser = subparsers.add_parser(
        "adapt-world-model-predictions",
        help="Adapt compact external world-model rollouts to the local evaluation schema.",
    )
    adapt_world_model_predictions_parser.add_argument("--benchmark", default=str(DEFAULT_WORLD_MODEL_BENCHMARK))
    adapt_world_model_predictions_parser.add_argument("--input", required=True)
    adapt_world_model_predictions_parser.add_argument("--output", default="outputs/adapted_world_model_predictions.json")
    adapt_world_model_predictions_parser.add_argument("--no-rasterize-trajectory", action="store_true")

    adapt_nuscenes_forecast_parser = subparsers.add_parser(
        "adapt-nuscenes-forecast-predictions",
        help="Adapt nuScenes prediction-challenge style trajectory forecasts to the local world-model schema.",
    )
    adapt_nuscenes_forecast_parser.add_argument("--benchmark", default=str(DEFAULT_WORLD_MODEL_BENCHMARK))
    adapt_nuscenes_forecast_parser.add_argument("--input", required=True)
    adapt_nuscenes_forecast_parser.add_argument("--output", default="outputs/adapted_nuscenes_forecasts.json")
    adapt_nuscenes_forecast_parser.add_argument("--mode-selection", choices=NUSCENES_FORECAST_MODE_SELECTIONS, default="top_probability")
    adapt_nuscenes_forecast_parser.add_argument("--no-rasterize-trajectory", action="store_true")

    evaluate_nuscenes_forecast_parser = subparsers.add_parser(
        "evaluate-nuscenes-forecast-predictions",
        help="Adapt nuScenes prediction-challenge forecasts and evaluate them on the world-model benchmark.",
    )
    evaluate_nuscenes_forecast_parser.add_argument("--benchmark", default=str(DEFAULT_WORLD_MODEL_BENCHMARK))
    evaluate_nuscenes_forecast_parser.add_argument("--input", required=True)
    evaluate_nuscenes_forecast_parser.add_argument("--output", default="outputs/nuscenes_forecast_eval")
    evaluate_nuscenes_forecast_parser.add_argument("--mode-selection", choices=NUSCENES_FORECAST_MODE_SELECTIONS, default="top_probability")
    evaluate_nuscenes_forecast_parser.add_argument("--profile-name", default="")
    evaluate_nuscenes_forecast_parser.add_argument("--no-rasterize-trajectory", action="store_true")

    generate_nuscenes_forecast_baselines_parser = subparsers.add_parser(
        "generate-nuscenes-forecast-baselines",
        help="Generate official nuScenes physics forecast baselines directly on the world-model benchmark anchors.",
    )
    generate_nuscenes_forecast_baselines_parser.add_argument("--benchmark", default=str(DEFAULT_WORLD_MODEL_BENCHMARK))
    generate_nuscenes_forecast_baselines_parser.add_argument("--dataroot", default="data/sets/nuscenes")
    generate_nuscenes_forecast_baselines_parser.add_argument("--version", default="v1.0-trainval")
    generate_nuscenes_forecast_baselines_parser.add_argument("--output", default="outputs/nuscenes_forecast_baselines")

    run_nuscenes_forecast_baselines_parser = subparsers.add_parser(
        "run-nuscenes-forecast-baselines",
        help="Generate, evaluate, and compare official nuScenes physics forecast baselines on the world-model benchmark.",
    )
    run_nuscenes_forecast_baselines_parser.add_argument("--benchmark", default=str(DEFAULT_WORLD_MODEL_BENCHMARK))
    run_nuscenes_forecast_baselines_parser.add_argument("--dataroot", default="data/sets/nuscenes")
    run_nuscenes_forecast_baselines_parser.add_argument("--version", default="v1.0-trainval")
    run_nuscenes_forecast_baselines_parser.add_argument("--output", default="outputs/nuscenes_forecast_baselines_eval")
    run_nuscenes_forecast_baselines_parser.add_argument("--mode-selection", choices=NUSCENES_FORECAST_MODE_SELECTIONS, default="top_probability")

    run_contextvae_world_model_parser = subparsers.add_parser(
        "run-contextvae-world-model-study",
        help="Run the ContextVAE multimodal forecasting baseline on the forecast-compatible world-model benchmark subset.",
    )
    run_contextvae_world_model_parser.add_argument("--benchmark", default=str(DEFAULT_WORLD_MODEL_BENCHMARK))
    run_contextvae_world_model_parser.add_argument("--dataroot", default=str(DEFAULT_DATAROOT))
    run_contextvae_world_model_parser.add_argument("--version", default="v1.0-trainval")
    run_contextvae_world_model_parser.add_argument("--output", default=str(DEFAULT_CONTEXTVAE_OUTPUT))
    run_contextvae_world_model_parser.add_argument("--repo", default=str(DEFAULT_CONTEXTVAE_REPO))
    run_contextvae_world_model_parser.add_argument("--checkpoint", default=str(DEFAULT_CONTEXTVAE_CHECKPOINT))
    run_contextvae_world_model_parser.add_argument("--device", default="")
    run_contextvae_world_model_parser.add_argument("--batch-size", type=int, default=8)
    run_contextvae_world_model_parser.add_argument("--mode-count", type=int, default=5)
    run_contextvae_world_model_parser.add_argument("--clustering-samples", type=int, default=2000)
    run_contextvae_world_model_parser.add_argument("--map-scale", type=int, default=1)
    run_contextvae_world_model_parser.add_argument("--seed", type=int, default=1)

    evaluate_world_model_parser = subparsers.add_parser(
        "evaluate-world-model-predictions",
        help="Evaluate future trajectory and occupancy predictions on the scenario-conditioned world-model benchmark.",
    )
    evaluate_world_model_parser.add_argument("--benchmark", default=str(DEFAULT_WORLD_MODEL_BENCHMARK))
    evaluate_world_model_parser.add_argument("--predictions", required=True)
    evaluate_world_model_parser.add_argument("--output", default="outputs/world_model_evaluation")
    evaluate_world_model_parser.add_argument("--profile-name", default="")

    proxy_world_model_study_parser = subparsers.add_parser(
        "run-proxy-world-model-study",
        help="Run a proxy comparison on the scenario-conditioned world-model benchmark.",
    )
    proxy_world_model_study_parser.add_argument("--benchmark", default=str(DEFAULT_WORLD_MODEL_BENCHMARK))
    proxy_world_model_study_parser.add_argument("--output", default="outputs/trainval_world_model_proxy_study_v1")

    compare_world_model_parser = subparsers.add_parser(
        "compare-world-model-evaluations",
        help="Compare multiple world-model evaluation output directories.",
    )
    compare_world_model_parser.add_argument(
        "--eval-dir",
        action="append",
        default=[],
        required=True,
        help="Evaluation directory that contains world_model_metrics.json. Repeatable.",
    )
    compare_world_model_parser.add_argument("--output", default="outputs/world_model_comparison")

    export_world_model_parser = subparsers.add_parser(
        "export-world-model-replay",
        help="Export world-model benchmark cases as replay-ready JSONL or MCAP artifacts.",
    )
    export_world_model_parser.add_argument("--benchmark", default=str(DEFAULT_WORLD_MODEL_BENCHMARK))
    export_world_model_parser.add_argument("--output", default="outputs/world_model_replay")
    export_world_model_parser.add_argument("--format", choices=["jsonl", "mcap"], default="jsonl")

    render_world_model_case_studies_parser = subparsers.add_parser(
        "render-world-model-case-studies",
        help="Render qualitative case studies from world-model evaluation outputs.",
    )
    render_world_model_case_studies_parser.add_argument("--benchmark", default=str(DEFAULT_WORLD_MODEL_BENCHMARK))
    render_world_model_case_studies_parser.add_argument(
        "--eval-dir",
        action="append",
        default=[],
        required=True,
        help="Evaluation directory that contains world_model_metrics.json and adapted_predictions.json. Repeatable.",
    )
    render_world_model_case_studies_parser.add_argument("--output", default="outputs/world_model_case_studies")
    render_world_model_case_studies_parser.add_argument("--max-cases", type=int, default=4)

    inspect_nuplan_parser = subparsers.add_parser(
        "inspect-nuplan",
        help="Inspect local nuPlan dataset layout and split readiness.",
    )
    inspect_nuplan_parser.add_argument("--dataset-root", default="data/nuplan/dataset")

    generate_nuplan_replay_parser = subparsers.add_parser(
        "generate-nuplan-replay-benchmark",
        help="Generate a compact nuPlan replay-regression benchmark from SQLite scenario tags.",
    )
    generate_nuplan_replay_parser.add_argument("--split-dir", default=str(DEFAULT_NUPLAN_SPLIT))
    generate_nuplan_replay_parser.add_argument("--output", default=str(DEFAULT_NUPLAN_REPLAY_BENCHMARK))
    generate_nuplan_replay_parser.add_argument("--max-dbs", type=int, default=4)
    generate_nuplan_replay_parser.add_argument("--max-cases", type=int, default=16)
    generate_nuplan_replay_parser.add_argument("--max-cases-per-db", type=int, default=4)
    generate_nuplan_replay_parser.add_argument("--history-s", type=float, default=2.0)
    generate_nuplan_replay_parser.add_argument("--future-s", type=float, default=4.0)
    generate_nuplan_replay_parser.add_argument("--frame-hz", type=float, default=2.0)
    generate_nuplan_replay_parser.add_argument("--scenario-tag", action="append", default=[])

    generate_nuplan_rollout_parser = subparsers.add_parser(
        "generate-nuplan-proxy-rollouts",
        help="Generate simple ego rollout baselines for a nuPlan replay benchmark.",
    )
    generate_nuplan_rollout_parser.add_argument("--benchmark", default=str(DEFAULT_NUPLAN_REPLAY_BENCHMARK))
    generate_nuplan_rollout_parser.add_argument("--output", default="outputs/nuplan_proxy_rollouts.json")
    generate_nuplan_rollout_parser.add_argument(
        "--profile",
        choices=NUPLAN_REPLAY_PROFILES,
        default="constant_velocity",
    )

    evaluate_nuplan_rollout_parser = subparsers.add_parser(
        "evaluate-nuplan-rollouts",
        help="Evaluate ego rollout predictions on a nuPlan replay benchmark.",
    )
    evaluate_nuplan_rollout_parser.add_argument("--benchmark", default=str(DEFAULT_NUPLAN_REPLAY_BENCHMARK))
    evaluate_nuplan_rollout_parser.add_argument("--predictions", required=True)
    evaluate_nuplan_rollout_parser.add_argument("--output", default="outputs/nuplan_replay_evaluation")
    evaluate_nuplan_rollout_parser.add_argument("--profile-name", default="")
    evaluate_nuplan_rollout_parser.add_argument("--collision-distance-m", type=float, default=2.0)

    run_nuplan_replay_parser = subparsers.add_parser(
        "run-nuplan-replay-study",
        help="Generate a nuPlan replay benchmark, proxy rollouts, and evaluation summaries.",
    )
    run_nuplan_replay_parser.add_argument("--split-dir", default=str(DEFAULT_NUPLAN_SPLIT))
    run_nuplan_replay_parser.add_argument("--output", default="outputs/nuplan_replay_study_v1")
    run_nuplan_replay_parser.add_argument("--max-dbs", type=int, default=4)
    run_nuplan_replay_parser.add_argument("--max-cases", type=int, default=16)
    run_nuplan_replay_parser.add_argument("--max-cases-per-db", type=int, default=4)
    run_nuplan_replay_parser.add_argument("--history-s", type=float, default=2.0)
    run_nuplan_replay_parser.add_argument("--future-s", type=float, default=4.0)
    run_nuplan_replay_parser.add_argument("--frame-hz", type=float, default=2.0)
    run_nuplan_replay_parser.add_argument("--min-anchor-gap-s", type=float, default=4.0)
    run_nuplan_replay_parser.add_argument("--scenario-tag", action="append", default=[])
    run_nuplan_replay_parser.add_argument(
        "--profile",
        action="append",
        default=[],
        choices=NUPLAN_REPLAY_PROFILES,
        help="Proxy rollout profile. Repeatable. Defaults to all profiles.",
    )

    run_nuplan_sweep_parser = subparsers.add_parser(
        "run-nuplan-replay-sweep",
        help="Run a cross-split nuPlan replay-regression sweep.",
    )
    run_nuplan_sweep_parser.add_argument("--config", default="configs/nuplan_replay_sweep_medium.yaml")

    run_nuplan_closed_loop_sweep_parser = subparsers.add_parser(
        "run-nuplan-closed-loop-sweep",
        help="Run a cross-split nuPlan closed-loop replay sweep.",
    )
    run_nuplan_closed_loop_sweep_parser.add_argument("--config", default="configs/nuplan_closed_loop_sweep_medium.yaml")

    run_nuplan_closed_loop_parser = subparsers.add_parser(
        "run-nuplan-closed-loop-study",
        help="Run closed-loop replay simulation on compact nuPlan cases.",
    )
    run_nuplan_closed_loop_parser.add_argument("--split-dir", default=str(DEFAULT_NUPLAN_SPLIT))
    run_nuplan_closed_loop_parser.add_argument("--output", default=str(DEFAULT_NUPLAN_CLOSED_LOOP_OUTPUT))
    run_nuplan_closed_loop_parser.add_argument("--max-dbs", type=int, default=64)
    run_nuplan_closed_loop_parser.add_argument("--max-cases", type=int, default=16)
    run_nuplan_closed_loop_parser.add_argument("--max-cases-per-db", type=int, default=4)
    run_nuplan_closed_loop_parser.add_argument("--history-s", type=float, default=2.0)
    run_nuplan_closed_loop_parser.add_argument("--future-s", type=float, default=4.0)
    run_nuplan_closed_loop_parser.add_argument("--frame-hz", type=float, default=2.0)
    run_nuplan_closed_loop_parser.add_argument("--min-anchor-gap-s", type=float, default=4.0)
    run_nuplan_closed_loop_parser.add_argument("--scenario-tag", action="append", default=[])
    run_nuplan_closed_loop_parser.add_argument(
        "--profile",
        action="append",
        default=[],
        choices=DEFAULT_NUPLAN_CLOSED_LOOP_PROFILES,
        help="Closed-loop planner profile. Repeatable. Defaults to all profiles.",
    )

    compare_nuplan_replay_parser = subparsers.add_parser(
        "compare-nuplan-replay-evaluations",
        help="Compare multiple nuPlan replay evaluation output directories.",
    )
    compare_nuplan_replay_parser.add_argument(
        "--eval-dir",
        action="append",
        default=[],
        required=True,
        help="Evaluation directory that contains nuplan_replay_metrics.json. Repeatable.",
    )
    compare_nuplan_replay_parser.add_argument("--output", default="outputs/nuplan_replay_comparison")

    render_nuplan_case_studies_parser = subparsers.add_parser(
        "render-nuplan-replay-case-studies",
        help="Render qualitative case studies from nuPlan replay evaluation outputs.",
    )
    render_nuplan_case_studies_parser.add_argument("--benchmark", default=str(DEFAULT_NUPLAN_REPLAY_BENCHMARK))
    render_nuplan_case_studies_parser.add_argument(
        "--eval-dir",
        action="append",
        default=[],
        required=True,
        help="Evaluation directory that contains nuplan_replay_metrics.json. Repeatable.",
    )
    render_nuplan_case_studies_parser.add_argument("--output", default="outputs/nuplan_replay_case_studies")
    render_nuplan_case_studies_parser.add_argument("--max-cases", type=int, default=4)

    inspect_backends_parser = subparsers.add_parser(
        "inspect-dataset-backends",
        help="Inspect local nuScenes and nuPlan backend readiness.",
    )
    inspect_backends_parser.add_argument("--nuscenes-root", default="data/sets/nuscenes")
    inspect_backends_parser.add_argument("--nuplan-root", default="data/nuplan/dataset")
    inspect_backends_parser.add_argument("--index-root", default="artifacts/index")
    inspect_backends_parser.add_argument("--output", default="")

    registry_parser = subparsers.add_parser(
        "export-benchmark-registry",
        help="Export the benchmark-layer registry.",
    )
    registry_parser.add_argument("--output", default="outputs/benchmark_registry.json")

    full_suite_parser = subparsers.add_parser(
        "run-full-benchmark-suite",
        help="Run the end-to-end benchmark suite config.",
    )
    full_suite_parser.add_argument("--config", default="configs/full_benchmark_suite.yaml")

    unified_cases_parser = subparsers.add_parser(
        "export-unified-cases",
        help="Convert a supported case source to the unified risk-case schema.",
    )
    unified_cases_parser.add_argument("--source", required=True)
    unified_cases_parser.add_argument(
        "--source-type",
        choices=["nuplan_replay_benchmark", "nuscenes_case_library"],
        required=True,
    )
    unified_cases_parser.add_argument("--output", default="outputs/unified_risk_cases.json")

    experiment_config_parser = subparsers.add_parser(
        "run-experiment-config",
        help="Run a structured experiment YAML config.",
    )
    experiment_config_parser.add_argument("--config", required=True)

    generate_bev_occupancy_parser = subparsers.add_parser(
        "generate-bev-occupancy-benchmark",
        help="Generate a sparse BEV occupancy benchmark from perception slices.",
    )
    generate_bev_occupancy_parser.add_argument("--perception-benchmark", default=str(DEFAULT_PERCEPTION_BENCHMARK))
    generate_bev_occupancy_parser.add_argument("--db", default="artifacts/index/v1.0-trainval.sqlite")
    generate_bev_occupancy_parser.add_argument("--output", default=str(DEFAULT_BEV_OCCUPANCY_BENCHMARK))

    generate_bev_occupancy_predictions_parser = subparsers.add_parser(
        "generate-proxy-bev-occupancy-predictions",
        help="Generate proxy BEV occupancy predictions.",
    )
    generate_bev_occupancy_predictions_parser.add_argument("--benchmark", default=str(DEFAULT_BEV_OCCUPANCY_BENCHMARK))
    generate_bev_occupancy_predictions_parser.add_argument("--output", default="outputs/bev_occupancy_predictions.json")
    generate_bev_occupancy_predictions_parser.add_argument(
        "--profile",
        choices=BEV_OCCUPANCY_PROXY_PROFILES,
        default="risk_actor_only",
    )

    adapt_bev_occupancy_parser = subparsers.add_parser(
        "adapt-perception-to-bev-occupancy",
        help="Rasterize adapted perception predictions into sparse BEV occupancy predictions.",
    )
    adapt_bev_occupancy_parser.add_argument("--benchmark", default=str(DEFAULT_BEV_OCCUPANCY_BENCHMARK))
    adapt_bev_occupancy_parser.add_argument("--predictions", required=True)
    adapt_bev_occupancy_parser.add_argument("--output", default="outputs/adapted_bev_occupancy_predictions.json")
    adapt_bev_occupancy_parser.add_argument("--profile-name", default="")

    evaluate_bev_occupancy_parser = subparsers.add_parser(
        "evaluate-bev-occupancy",
        help="Evaluate sparse BEV occupancy predictions on risk slices.",
    )
    evaluate_bev_occupancy_parser.add_argument("--benchmark", default=str(DEFAULT_BEV_OCCUPANCY_BENCHMARK))
    evaluate_bev_occupancy_parser.add_argument("--predictions", required=True)
    evaluate_bev_occupancy_parser.add_argument("--output", default="outputs/bev_occupancy_evaluation")
    evaluate_bev_occupancy_parser.add_argument("--profile-name", default="")

    run_bev_occupancy_parser = subparsers.add_parser(
        "run-proxy-bev-occupancy-study",
        help="Run proxy BEV occupancy profiles on a sparse occupancy benchmark.",
    )
    run_bev_occupancy_parser.add_argument("--benchmark", default=str(DEFAULT_BEV_OCCUPANCY_BENCHMARK))
    run_bev_occupancy_parser.add_argument("--output", default="outputs/trainval_bev_occupancy_proxy_study_v1")

    compare_bev_occupancy_parser = subparsers.add_parser(
        "compare-bev-occupancy-evaluations",
        help="Compare multiple BEV occupancy evaluation directories.",
    )
    compare_bev_occupancy_parser.add_argument("--eval-dir", action="append", default=[], required=True)
    compare_bev_occupancy_parser.add_argument("--output", default="outputs/bev_occupancy_comparison")

    compare_perception_parser = subparsers.add_parser(
        "compare-perception-evaluations",
        help="Compare multiple perception evaluation output directories.",
    )
    compare_perception_parser.add_argument(
        "--eval-dir",
        action="append",
        default=[],
        required=True,
        help="Evaluation directory that contains perception_metrics.json. Repeatable.",
    )
    compare_perception_parser.add_argument("--output", default="outputs/perception_comparison")

    evaluate_nuscenes_parser = subparsers.add_parser(
        "evaluate-nuscenes-predictions",
        help="Adapt official nuScenes detection or tracking outputs and evaluate them on the perception benchmark.",
    )
    evaluate_nuscenes_parser.add_argument("--benchmark", default=str(DEFAULT_PERCEPTION_BENCHMARK))
    evaluate_nuscenes_parser.add_argument("--db", default="artifacts/index/v1.0-trainval.sqlite")
    evaluate_nuscenes_parser.add_argument("--input", required=True, help="Path to the official nuScenes prediction JSON.")
    evaluate_nuscenes_parser.add_argument("--output", default="outputs/nuscenes_perception_eval")
    evaluate_nuscenes_parser.add_argument("--task", choices=["tracking", "detection"], default="tracking")
    evaluate_nuscenes_parser.add_argument("--profile-name", default="")
    evaluate_nuscenes_parser.add_argument("--match-distance-m", type=float, default=2.0)

    evaluate_nuscenes_filtered_parser = subparsers.add_parser(
        "evaluate-nuscenes-predictions-covered",
        help="Adapt official nuScenes outputs, filter the benchmark to covered cases, and evaluate the aligned subset.",
    )
    evaluate_nuscenes_filtered_parser.add_argument("--benchmark", default=str(DEFAULT_PERCEPTION_BENCHMARK))
    evaluate_nuscenes_filtered_parser.add_argument("--db", default="artifacts/index/v1.0-trainval.sqlite")
    evaluate_nuscenes_filtered_parser.add_argument("--input", required=True, help="Path to the official nuScenes prediction JSON.")
    evaluate_nuscenes_filtered_parser.add_argument("--output", default="outputs/nuscenes_perception_eval_covered")
    evaluate_nuscenes_filtered_parser.add_argument("--task", choices=["tracking", "detection"], default="tracking")
    evaluate_nuscenes_filtered_parser.add_argument("--profile-name", default="")
    evaluate_nuscenes_filtered_parser.add_argument("--match-distance-m", type=float, default=2.0)
    evaluate_nuscenes_filtered_parser.add_argument("--coverage-mode", choices=PREDICTION_COVERAGE_MODES, default="full_window")

    enrich_case_library_parser = subparsers.add_parser(
        "enrich-case-library",
        help="Re-validate a case library to populate actor grounding and event localization fields.",
    )
    enrich_case_library_parser.add_argument(
        "--case-library",
        default="outputs/trainval_case_library_v1/case_library.json",
        help="Input case library JSON path.",
    )
    enrich_case_library_parser.add_argument(
        "--db",
        default="artifacts/index/v1.0-trainval.sqlite",
        help="SQLite index used for re-validation.",
    )
    enrich_case_library_parser.add_argument(
        "--output",
        default="outputs/trainval_case_library_v1/case_library_enriched.json",
        help="Output JSON path for the enriched case library.",
    )

    compare_parser = subparsers.add_parser(
        "benchmark-compare",
        help="Run the benchmark across rule, llm, and hybrid profiles and export a comparison summary.",
    )
    compare_parser.add_argument("--config", default=str(DEFAULT_COMPARE_BENCHMARK))
    compare_parser.add_argument("--db", default="artifacts/index/v1.0-trainval.sqlite")
    compare_parser.add_argument("--output", default="outputs/trainval_profile_comparison_v1")
    compare_parser.add_argument("--candidate-pool", type=int, default=12)
    _add_llm_connection_args(compare_parser)

    ablate_parser = subparsers.add_parser(
        "benchmark-ablate",
        help="Run an ablation study over the benchmark and export comparison artifacts.",
    )
    ablate_parser.add_argument("--config", default="benchmarks/trainval_scenario_mining_v1.yaml")
    ablate_parser.add_argument("--db", default="artifacts/index/v1.0-trainval.sqlite")
    ablate_parser.add_argument("--output", default="outputs/trainval_hybrid_ablation_v1")
    ablate_parser.add_argument("--candidate-pool", type=int, default=12)
    ablate_parser.add_argument("--base-query-mode", choices=["rule", "llm", "hybrid"], default="hybrid")
    ablate_parser.add_argument("--base-rerank-mode", choices=["none", "llm"], default="llm")
    ablate_parser.add_argument("--reuse-existing", action="store_true")
    _add_llm_connection_args(ablate_parser)

    gallery_parser = subparsers.add_parser(
        "build-gallery",
        help="Build a static query gallery for one benchmark output or a side-by-side browser for a comparison output.",
    )
    gallery_group = gallery_parser.add_mutually_exclusive_group(required=True)
    gallery_group.add_argument(
        "--benchmark-output",
        default="",
        help="Benchmark output directory that contains benchmark_summary.json.",
    )
    gallery_group.add_argument(
        "--comparison-output",
        default="",
        help="Comparison output directory that contains benchmark_profile_comparison.json.",
    )
    gallery_parser.add_argument("--title", default="")

    return parser


def _run_benchmark(
    config_path: Path,
    db_path: Path,
    output_dir: Path,
    candidate_pool: int,
    query_mode: str = "rule",
    rerank_mode: str = "none",
    llm_config: Optional[LLMConfig] = None,
    learned_reranker_checkpoint: Optional[Path] = None,
    validation_config: Optional[ValidationConfig] = None,
) -> List[dict]:
    queries = load_benchmark_config(config_path)
    summaries: List[dict] = []
    benchmark_results: List[dict] = []
    for spec in queries:
        result = run_query_pipeline(
            db_path=db_path,
            query_text=spec.natural_language,
            output_root=output_dir,
            top_k=spec.top_k,
            candidate_pool=spec.candidate_pool or candidate_pool,
            benchmark_spec=spec,
            query_mode=query_mode,
            rerank_mode=rerank_mode,
            llm_config=llm_config,
            learned_reranker_checkpoint=learned_reranker_checkpoint,
            validation_config=validation_config,
        )
        result["id"] = spec.id
        benchmark_results.append(result)
        summaries.append(
            {
                "id": spec.id,
                "description": spec.description,
                "query_dir": str(result["query_dir"]),
                "candidate_count": result["candidate_count"],
                "selected_count": result["selected_count"],
                "tags": spec.tags,
                "behaviors": spec.behaviors,
                "actors": spec.actors,
            }
        )

    (output_dir / "benchmark_summary.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    case_library_entries = build_case_library(benchmark_results)
    write_case_library(case_library_entries, output_dir)
    write_benchmark_metrics(build_benchmark_metrics(benchmark_results, case_library_entries), output_dir)
    write_benchmark_exports(benchmark_results, case_library_entries, output_dir)
    return summaries


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "inspect-archives":
        inventory = discover_archive_inventory(Path(args.workspace))
        print(json.dumps(inventory.to_dict(), indent=2, ensure_ascii=False))
        return

    if args.command == "prepare-data":
        extracted = prepare_data(
            workspace=Path(args.workspace),
            dataroot=Path(args.dataroot),
            archives=args.archive or None,
            profile=args.profile,
        )
        print("Prepared dataroot:", Path(args.dataroot).resolve())
        for item in extracted:
            print("  extracted:", item)
        return

    if args.command == "build-index":
        stats = build_index(
            version=args.version,
            dataroot=Path(args.dataroot),
            db_path=Path(args.db),
            scene_limit=args.scene_limit,
            verbose=True,
        )
        print("Built index:", Path(args.db).resolve())
        print(json.dumps(stats, indent=2))
        return

    if args.command == "query":
        llm_config = _resolve_llm_config(args)
        result = run_query_pipeline(
            db_path=Path(args.db),
            query_text=args.text,
            output_root=Path(args.output),
            top_k=args.top_k,
            candidate_pool=args.candidate_pool,
            query_mode=args.query_mode,
            rerank_mode=args.rerank_mode,
            llm_config=llm_config,
            learned_reranker_checkpoint=_optional_path(args.learned_reranker_checkpoint),
        )
        print("Query output:", result["query_dir"])
        print("Candidates:", result["candidate_count"], "Selected:", result["selected_count"])
        return

    if args.command == "run-multimodal-retrieval":
        llm_config = _resolve_llm_config(args) if args.query_mode == "llm" else None
        result = run_multimodal_retrieval_report(
            db_path=Path(args.db),
            query_text=args.text,
            output_dir=Path(args.output),
            top_k=args.top_k,
            candidate_pool=args.candidate_pool,
            query_mode=args.query_mode,
            llm_config=llm_config,
        )
        print("Multimodal retrieval report:", Path(result["markdown"]).resolve())
        print("Candidates:", result["candidate_count"])
        return

    if args.command == "train-learned-retriever":
        report = train_learned_scene_retriever(
            benchmark_path=Path(args.benchmark),
            db_path=Path(args.db),
            output_dir=Path(args.output),
            config=LearnedRetrieverConfig(
                text_hash_dim=args.text_hash_dim,
                hidden_dim=args.hidden_dim,
                embedding_dim=args.embedding_dim,
                negatives_per_query=args.negatives_per_query,
                candidate_pool=args.candidate_pool,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                validation_fraction=args.validation_fraction,
                seed=args.seed,
                device=args.device,
            ),
        )
        print("Learned retriever checkpoint:", Path(report["checkpoint_path"]).resolve())
        print(
            "Training groups:",
            report["train_group_count"],
            "Validation groups:",
            report["validation_group_count"],
        )
        print("Validation Recall@1:", dict(report.get("validation_metrics") or {}).get("recall_at_1"))
        return

    if args.command == "train-large-learned-retriever":
        report = train_weakly_supervised_scene_retriever(
            db_path=Path(args.db),
            output_dir=Path(args.output),
            max_groups_per_family=args.max_groups_per_family,
            config=LearnedRetrieverConfig(
                text_hash_dim=args.text_hash_dim,
                hidden_dim=args.hidden_dim,
                embedding_dim=args.embedding_dim,
                negatives_per_query=args.negatives_per_query,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                validation_fraction=args.validation_fraction,
                seed=args.seed,
                device=args.device,
            ),
        )
        print("Large learned retriever checkpoint:", Path(report["checkpoint_path"]).resolve())
        print("Training groups:", report["train_group_count"], "Validation groups:", report["validation_group_count"])
        print("Validation Recall@1:", dict(report.get("validation_metrics") or {}).get("recall_at_1"))
        return

    if args.command == "run-learned-retrieval":
        result = run_learned_retrieval_report(
            db_path=Path(args.db),
            query_text=args.text,
            checkpoint_path=Path(args.checkpoint),
            output_dir=Path(args.output),
            top_k=args.top_k,
            candidate_pool=args.candidate_pool,
        )
        print("Learned retrieval report:", Path(result["markdown"]).resolve())
        print("Candidates:", result["candidate_count"])
        return

    if args.command == "run-failure-aware-reranking":
        result = run_failure_aware_reranking_eval(
            query_config=Path(args.query_config),
            db_path=Path(args.db),
            output_dir=Path(args.output),
            learned_checkpoint=Path(args.checkpoint),
            candidate_pool=args.candidate_pool,
            top_k=args.top_k,
            max_queries=args.max_queries,
        )
        print("Failure-aware reranking:", Path(args.output).resolve())
        print(json.dumps(result["overview"], indent=2, ensure_ascii=False))
        return

    if args.command == "langgraph-query":
        llm_config = _resolve_llm_config(args)
        result = run_langgraph_query_pipeline(
            db_path=Path(args.db),
            query_text=args.text,
            output_root=Path(args.output),
            top_k=args.top_k,
            candidate_pool=args.candidate_pool,
            query_mode=args.query_mode,
            rerank_mode=args.rerank_mode,
            llm_config=llm_config,
            learned_reranker_checkpoint=_optional_path(args.learned_reranker_checkpoint),
        )
        print("LangGraph query output:", result["query_dir"])
        print("Candidates:", result["candidate_count"], "Selected:", result["selected_count"])
        return

    if args.command == "run-research-agent":
        llm_config = _resolve_llm_config(
            args,
            require=True,
            default_base_url=DEFAULT_OLLAMA_BASE_URL,
            default_model=DEFAULT_OLLAMA_MODEL,
        )
        result = run_research_agent(
            output_dir=Path(args.output),
            llm_config=llm_config,
            artifact_paths=[Path(path) for path in args.artifact] if args.artifact else None,
            focus=args.focus,
        )
        print("Research agent report:", result["report_path"])
        print("Output:", result["output_dir"])
        return

    if args.command == "run-failure-mining":
        result = mine_model_failures(
            inputs=[Path(path) for path in args.input],
            output_dir=Path(args.output),
            max_queries=args.max_queries,
            min_count=args.min_count,
        )
        print("Failure mining report:", result["report_md"])
        print("Benchmark update YAML:", result["benchmark_yaml"])
        print("Failures:", result["failure_record_count"], "Update queries:", result["update_query_count"])
        return

    if args.command == "benchmark":
        output_dir = Path(args.output).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        llm_config = _resolve_llm_config(args)
        summaries = _run_benchmark(
            config_path=Path(args.config),
            db_path=Path(args.db),
            output_dir=output_dir,
            candidate_pool=args.candidate_pool,
            query_mode=args.query_mode,
            rerank_mode=args.rerank_mode,
            llm_config=llm_config,
            learned_reranker_checkpoint=_optional_path(args.learned_reranker_checkpoint),
        )
        print("Benchmark output:", output_dir)
        print("Queries:", len(summaries))
        return

    if args.command == "generate-counterfactual-benchmark":
        metadata = generate_counterfactual_benchmark_from_case_library(
            case_library_path=Path(args.case_library),
            output_path=Path(args.output),
            max_cases=args.max_cases,
        )
        print("Counterfactual benchmark:", Path(args.output).resolve())
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        return

    if args.command == "generate-scenario-mining-benchmark":
        metadata = generate_scenario_mining_benchmark_from_case_library(
            case_library_path=Path(args.case_library),
            output_path=Path(args.output),
            max_cases=args.max_cases,
        )
        print("Scenario mining benchmark:", Path(args.output).resolve())
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        return

    if args.command == "generate-perception-benchmark":
        metadata = generate_perception_benchmark_from_scenario_config(
            config_path=Path(args.config),
            db_path=Path(args.db),
            output_path=Path(args.output),
        )
        print("Perception benchmark:", Path(args.output).resolve())
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        return

    if args.command == "generate-world-model-benchmark":
        metadata = generate_world_model_benchmark_from_perception_benchmark(
            perception_benchmark_path=Path(args.benchmark),
            db_path=Path(args.db),
            output_path=Path(args.output),
        )
        print("World-model benchmark:", Path(args.output).resolve())
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        return

    if args.command == "generate-proxy-perception-predictions":
        metadata = generate_proxy_perception_predictions(
            benchmark_path=Path(args.benchmark),
            output_path=Path(args.output),
            profile_name=args.profile,
        )
        print("Proxy predictions:", Path(args.output).resolve())
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        return

    if args.command == "adapt-nuscenes-predictions":
        metadata = adapt_nuscenes_predictions(
            benchmark_path=Path(args.benchmark),
            db_path=Path(args.db),
            input_path=Path(args.input),
            output_path=Path(args.output),
            task_type=args.task,
        )
        print("Adapted predictions:", Path(args.output).resolve())
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        return

    if args.command == "filter-perception-benchmark":
        metadata = filter_perception_benchmark_by_predictions(
            benchmark_path=Path(args.benchmark),
            predictions_path=Path(args.predictions),
            output_path=Path(args.output),
            coverage_mode=args.coverage_mode,
        )
        print("Filtered perception benchmark:", Path(args.output).resolve())
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        return

    if args.command == "evaluate-perception-predictions":
        summary = evaluate_perception_predictions(
            benchmark_path=Path(args.benchmark),
            predictions_path=Path(args.predictions),
            output_dir=Path(args.output),
            profile_name=args.profile_name,
            match_distance_m=args.match_distance_m,
        )
        print("Perception evaluation:", Path(args.output).resolve())
        print(json.dumps(summary["overview"], indent=2, ensure_ascii=False))
        return

    if args.command == "evaluate-nuscenes-predictions":
        metadata = adapt_and_evaluate_nuscenes_predictions(
            benchmark_path=Path(args.benchmark),
            db_path=Path(args.db),
            input_path=Path(args.input),
            output_dir=Path(args.output),
            task_type=args.task,
            profile_name=args.profile_name,
            match_distance_m=args.match_distance_m,
        )
        print("nuScenes perception evaluation:", Path(args.output).resolve())
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        return

    if args.command == "evaluate-nuscenes-predictions-covered":
        metadata = adapt_filter_and_evaluate_nuscenes_predictions(
            benchmark_path=Path(args.benchmark),
            db_path=Path(args.db),
            input_path=Path(args.input),
            output_dir=Path(args.output),
            task_type=args.task,
            profile_name=args.profile_name,
            match_distance_m=args.match_distance_m,
            coverage_mode=args.coverage_mode,
        )
        print("Covered nuScenes perception evaluation:", Path(args.output).resolve())
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        return

    if args.command == "run-proxy-perception-study":
        metadata = run_proxy_perception_study(
            benchmark_path=Path(args.benchmark),
            output_dir=Path(args.output),
        )
        print("Proxy perception study:", Path(args.output).resolve())
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        return

    if args.command == "generate-proxy-world-model-predictions":
        metadata = generate_proxy_world_model_predictions(
            benchmark_path=Path(args.benchmark),
            output_path=Path(args.output),
            profile_name=args.profile,
        )
        print("Proxy world-model predictions:", Path(args.output).resolve())
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        return

    if args.command == "adapt-world-model-predictions":
        metadata = adapt_world_model_predictions(
            benchmark_path=Path(args.benchmark),
            input_path=Path(args.input),
            output_path=Path(args.output),
            rasterize_trajectory=not bool(args.no_rasterize_trajectory),
        )
        print("Adapted world-model predictions:", Path(args.output).resolve())
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        return

    if args.command == "adapt-nuscenes-forecast-predictions":
        metadata = adapt_nuscenes_forecast_predictions(
            benchmark_path=Path(args.benchmark),
            input_path=Path(args.input),
            output_path=Path(args.output),
            mode_selection=args.mode_selection,
            rasterize_trajectory=not bool(args.no_rasterize_trajectory),
        )
        print("Adapted nuScenes forecasts:", Path(args.output).resolve())
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        return

    if args.command == "evaluate-world-model-predictions":
        summary = evaluate_world_model_predictions(
            benchmark_path=Path(args.benchmark),
            predictions_path=Path(args.predictions),
            output_dir=Path(args.output),
            profile_name=args.profile_name,
        )
        print("World-model evaluation:", Path(args.output).resolve())
        print(json.dumps(summary["overview"], indent=2, ensure_ascii=False))
        return

    if args.command == "evaluate-nuscenes-forecast-predictions":
        metadata = adapt_and_evaluate_nuscenes_forecast_predictions(
            benchmark_path=Path(args.benchmark),
            input_path=Path(args.input),
            output_dir=Path(args.output),
            mode_selection=args.mode_selection,
            profile_name=args.profile_name,
            rasterize_trajectory=not bool(args.no_rasterize_trajectory),
        )
        print("nuScenes forecast evaluation:", Path(args.output).resolve())
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        return

    if args.command == "generate-nuscenes-forecast-baselines":
        metadata = generate_nuscenes_forecast_baselines(
            benchmark_path=Path(args.benchmark),
            dataroot=Path(args.dataroot),
            version=args.version,
            output_dir=Path(args.output),
        )
        print("nuScenes forecast baselines:", Path(args.output).resolve())
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        return

    if args.command == "run-nuscenes-forecast-baselines":
        metadata = run_nuscenes_forecast_baselines(
            benchmark_path=Path(args.benchmark),
            dataroot=Path(args.dataroot),
            version=args.version,
            output_dir=Path(args.output),
            mode_selection=args.mode_selection,
        )
        print("nuScenes forecast baseline evaluation:", Path(args.output).resolve())
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        return

    if args.command == "run-contextvae-world-model-study":
        metadata = run_contextvae_world_model_study(
            benchmark_path=Path(args.benchmark),
            dataroot=Path(args.dataroot),
            version=args.version,
            output_dir=Path(args.output),
            repo_dir=Path(args.repo),
            checkpoint_path=Path(args.checkpoint),
            device=args.device,
            batch_size=args.batch_size,
            mode_count=args.mode_count,
            clustering_samples=args.clustering_samples,
            map_scale=args.map_scale,
            seed=args.seed,
        )
        print("ContextVAE world-model study:", Path(args.output).resolve())
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        return

    if args.command == "run-proxy-world-model-study":
        metadata = run_proxy_world_model_study(
            benchmark_path=Path(args.benchmark),
            output_dir=Path(args.output),
        )
        print("Proxy world-model study:", Path(args.output).resolve())
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        return

    if args.command == "compare-world-model-evaluations":
        metadata = compare_world_model_evaluations(
            evaluation_dirs=[Path(path) for path in args.eval_dir],
            output_dir=Path(args.output),
        )
        print("World-model comparison:", Path(args.output).resolve())
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        return

    if args.command == "export-world-model-replay":
        metadata = export_world_model_replay(
            benchmark_path=Path(args.benchmark),
            output_dir=Path(args.output),
            export_format=args.format,
        )
        print("World-model replay export:", Path(args.output).resolve())
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        return

    if args.command == "render-world-model-case-studies":
        metadata = render_world_model_case_studies(
            benchmark_path=Path(args.benchmark),
            evaluation_dirs=[Path(path) for path in args.eval_dir],
            output_dir=Path(args.output),
            max_cases=args.max_cases,
        )
        print("World-model case studies:", Path(args.output).resolve())
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        return

    if args.command == "inspect-nuplan":
        inventory = inspect_nuplan_dataset(dataset_root=Path(args.dataset_root))
        print(json.dumps(inventory, indent=2, ensure_ascii=False))
        return

    if args.command == "generate-nuplan-replay-benchmark":
        metadata = generate_nuplan_replay_benchmark(
            split_dir=Path(args.split_dir),
            output_path=Path(args.output),
            max_dbs=args.max_dbs,
            max_cases=args.max_cases,
            max_cases_per_db=args.max_cases_per_db,
            history_s=args.history_s,
            future_s=args.future_s,
            frame_hz=args.frame_hz,
            scenario_tags=args.scenario_tag or None,
        )
        print("nuPlan replay benchmark:", Path(args.output).resolve())
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        return

    if args.command == "generate-nuplan-proxy-rollouts":
        metadata = generate_nuplan_proxy_rollouts(
            benchmark_path=Path(args.benchmark),
            output_path=Path(args.output),
            profile_name=args.profile,
        )
        print("nuPlan proxy rollouts:", Path(args.output).resolve())
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        return

    if args.command == "evaluate-nuplan-rollouts":
        summary = evaluate_nuplan_rollouts(
            benchmark_path=Path(args.benchmark),
            predictions_path=Path(args.predictions),
            output_dir=Path(args.output),
            profile_name=args.profile_name,
            collision_distance_m=args.collision_distance_m,
        )
        print("nuPlan replay evaluation:", Path(args.output).resolve())
        print(json.dumps(summary["overview"], indent=2, ensure_ascii=False))
        return

    if args.command == "run-nuplan-replay-study":
        manifest = run_nuplan_replay_study(
            split_dir=Path(args.split_dir),
            output_dir=Path(args.output),
            max_dbs=args.max_dbs,
            max_cases=args.max_cases,
            max_cases_per_db=args.max_cases_per_db,
            history_s=args.history_s,
            future_s=args.future_s,
            frame_hz=args.frame_hz,
            min_anchor_gap_s=args.min_anchor_gap_s,
            scenario_tags=args.scenario_tag or None,
            profiles=args.profile or None,
        )
        print("nuPlan replay study:", Path(args.output).resolve())
        print(json.dumps(manifest["benchmark"]["metadata"], indent=2, ensure_ascii=False))
        return

    if args.command == "run-nuplan-replay-sweep":
        result = run_experiment_config(Path(args.config))
        print("nuPlan replay sweep:", Path(result["result"].get("output_dir", DEFAULT_NUPLAN_REPLAY_SWEEP_OUTPUT)).resolve())
        print(json.dumps(result["result"].get("overview", {}), indent=2, ensure_ascii=False))
        return

    if args.command == "run-nuplan-closed-loop-sweep":
        result = run_experiment_config(Path(args.config))
        print(
            "nuPlan closed-loop sweep:",
            Path(result["result"].get("output_dir", DEFAULT_NUPLAN_CLOSED_LOOP_SWEEP_OUTPUT)).resolve(),
        )
        print(json.dumps(result["result"].get("overview", {}), indent=2, ensure_ascii=False))
        return

    if args.command == "run-nuplan-closed-loop-study":
        manifest = run_nuplan_closed_loop_study(
            split_dir=Path(args.split_dir),
            output_dir=Path(args.output),
            max_dbs=args.max_dbs,
            max_cases=args.max_cases,
            max_cases_per_db=args.max_cases_per_db,
            history_s=args.history_s,
            future_s=args.future_s,
            frame_hz=args.frame_hz,
            min_anchor_gap_s=args.min_anchor_gap_s,
            scenario_tags=args.scenario_tag or None,
            profiles=args.profile or None,
        )
        print("nuPlan closed-loop study:", Path(args.output).resolve())
        print(json.dumps(manifest["comparison"]["overview"], indent=2, ensure_ascii=False))
        return

    if args.command == "compare-nuplan-replay-evaluations":
        comparison = compare_nuplan_replay_evaluations(
            evaluation_dirs=[Path(path) for path in args.eval_dir],
            output_dir=Path(args.output),
        )
        print("nuPlan replay comparison:", Path(args.output).resolve())
        print(json.dumps(comparison["overview"], indent=2, ensure_ascii=False))
        return

    if args.command == "render-nuplan-replay-case-studies":
        metadata = render_nuplan_replay_case_studies(
            benchmark_path=Path(args.benchmark),
            evaluation_dirs=[Path(path) for path in args.eval_dir],
            output_dir=Path(args.output),
            max_cases=args.max_cases,
        )
        print("nuPlan replay case studies:", Path(args.output).resolve())
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        return

    if args.command == "inspect-dataset-backends":
        inventory = inspect_dataset_backends(
            nuscenes_root=Path(args.nuscenes_root),
            nuplan_root=Path(args.nuplan_root),
            index_root=Path(args.index_root),
        )
        if args.output:
            write_dataset_backend_inventory(inventory, Path(args.output))
            print("Dataset backend inventory:", Path(args.output).resolve())
        print(json.dumps(inventory, indent=2, ensure_ascii=False))
        return

    if args.command == "export-benchmark-registry":
        registry = write_benchmark_registry(Path(args.output), build_default_benchmark_registry())
        print("Benchmark registry:", Path(args.output).resolve())
        print(json.dumps({"schema": registry["schema"], "layer_count": len(registry["layers"])}, indent=2))
        return

    if args.command == "export-unified-cases":
        cases = load_unified_case_source(Path(args.source), args.source_type)
        payload = write_unified_case_collection(
            cases,
            Path(args.output),
            metadata={"source": str(args.source), "source_type": args.source_type},
        )
        print("Unified risk cases:", Path(args.output).resolve())
        print(json.dumps({"schema": payload["schema"], "case_count": len(payload["cases"])}, indent=2))
        return

    if args.command == "run-experiment-config":
        result = run_experiment_config(Path(args.config))
        print("Experiment result:", result["experiment_id"])
        print(json.dumps(result["result"], indent=2, ensure_ascii=False))
        return

    if args.command == "run-full-benchmark-suite":
        result = run_experiment_config(Path(args.config))
        print("Full benchmark suite:", Path(result["result"].get("output_dir", "outputs/full_benchmark_suite_v1")).resolve())
        print(json.dumps({"stages": list(result["result"].get("stages", {}).keys())}, indent=2))
        return

    if args.command == "generate-bev-occupancy-benchmark":
        metadata = generate_bev_occupancy_benchmark_from_perception_benchmark(
            perception_benchmark_path=Path(args.perception_benchmark),
            db_path=Path(args.db),
            output_path=Path(args.output),
        )
        print("BEV occupancy benchmark:", Path(args.output).resolve())
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        return

    if args.command == "generate-proxy-bev-occupancy-predictions":
        metadata = generate_proxy_bev_occupancy_predictions(
            benchmark_path=Path(args.benchmark),
            output_path=Path(args.output),
            profile_name=args.profile,
        )
        print("BEV occupancy predictions:", Path(args.output).resolve())
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        return

    if args.command == "adapt-perception-to-bev-occupancy":
        metadata = adapt_perception_predictions_to_bev_occupancy(
            benchmark_path=Path(args.benchmark),
            perception_predictions_path=Path(args.predictions),
            output_path=Path(args.output),
            profile_name=args.profile_name,
        )
        print("Adapted BEV occupancy predictions:", Path(args.output).resolve())
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        return

    if args.command == "evaluate-bev-occupancy":
        summary = evaluate_bev_occupancy_predictions(
            benchmark_path=Path(args.benchmark),
            predictions_path=Path(args.predictions),
            output_dir=Path(args.output),
            profile_name=args.profile_name,
        )
        print("BEV occupancy evaluation:", Path(args.output).resolve())
        print(json.dumps(summary["overview"], indent=2, ensure_ascii=False))
        return

    if args.command == "run-proxy-bev-occupancy-study":
        metadata = run_proxy_bev_occupancy_study(
            benchmark_path=Path(args.benchmark),
            output_dir=Path(args.output),
        )
        print("BEV occupancy study:", Path(args.output).resolve())
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        return

    if args.command == "compare-bev-occupancy-evaluations":
        metadata = compare_bev_occupancy_evaluations(
            evaluation_dirs=[Path(path) for path in args.eval_dir],
            output_dir=Path(args.output),
        )
        print("BEV occupancy comparison:", Path(args.output).resolve())
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        return

    if args.command == "compare-perception-evaluations":
        metadata = compare_perception_evaluations(
            evaluation_dirs=[Path(path) for path in args.eval_dir],
            output_dir=Path(args.output),
        )
        print("Perception comparison:", Path(args.output).resolve())
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        return

    if args.command == "enrich-case-library":
        metadata = enrich_case_library(
            case_library_path=Path(args.case_library),
            db_path=Path(args.db),
            output_path=Path(args.output),
        )
        print("Enriched case library:", Path(args.output).resolve())
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        return

    if args.command == "benchmark-compare":
        output_dir = Path(args.output).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        llm_config = _resolve_llm_config(args)
        profile_runs = default_benchmark_profiles(include_llm=llm_config is not None)

        for profile in profile_runs:
            profile_output = output_dir / profile["name"]
            profile_output.mkdir(parents=True, exist_ok=True)
            _run_benchmark(
                config_path=Path(args.config),
                db_path=Path(args.db),
                output_dir=profile_output,
                candidate_pool=args.candidate_pool,
                query_mode=str(profile["query_mode"]),
                rerank_mode=str(profile["rerank_mode"]),
                llm_config=llm_config,
            )
            profile["output_dir"] = str(profile_output)

        comparison = build_benchmark_comparison(profile_runs)
        write_benchmark_comparison(comparison, output_dir)
        print("Benchmark comparison output:", output_dir)
        print("Profiles:", ", ".join(str(profile["name"]) for profile in profile_runs))
        return

    if args.command == "benchmark-ablate":
        output_dir = Path(args.output).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        llm_config = _resolve_llm_config(args)
        ablation_runs = default_ablation_profiles(
            base_query_mode=args.base_query_mode,
            base_rerank_mode=args.base_rerank_mode,
        )
        requires_llm = any(
            str(profile["query_mode"]) != "rule" or str(profile["rerank_mode"]) == "llm"
            for profile in ablation_runs
        )
        if requires_llm and llm_config is None:
            raise ValueError("The selected ablation profiles require Ollama configuration.")

        for profile in ablation_runs:
            profile_output = output_dir / str(profile["name"])
            profile_output.mkdir(parents=True, exist_ok=True)
            benchmark_metrics_path = profile_output / "benchmark_metrics.json"
            if not (args.reuse_existing and benchmark_metrics_path.exists()):
                _run_benchmark(
                    config_path=Path(args.config),
                    db_path=Path(args.db),
                    output_dir=profile_output,
                    candidate_pool=args.candidate_pool,
                    query_mode=str(profile["query_mode"]),
                    rerank_mode=str(profile["rerank_mode"]),
                    llm_config=llm_config,
                    validation_config=profile.get("validation_config"),
                )
            profile["output_dir"] = str(profile_output)

        comparison = build_benchmark_comparison(ablation_runs)
        write_benchmark_comparison(comparison, output_dir)
        write_ablation_manifest(ablation_runs, output_dir)
        build_comparison_browser(
            comparison_output_dir=output_dir,
            title="nuScenes Ablation Browser",
        )
        print("Benchmark ablation output:", output_dir)
        print("Profiles:", ", ".join(str(profile["name"]) for profile in ablation_runs))
        return

    if args.command == "build-gallery":
        if args.benchmark_output:
            metadata = build_benchmark_gallery(
                benchmark_output_dir=Path(args.benchmark_output),
                title=args.title or "nuScenes Benchmark Query Gallery",
            )
        else:
            metadata = build_comparison_browser(
                comparison_output_dir=Path(args.comparison_output),
                title=args.title or "nuScenes Benchmark Comparison Browser",
            )
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        return


if __name__ == "__main__":
    main()

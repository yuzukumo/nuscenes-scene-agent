from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import yaml

from nusc_scene_agent.benchmark_exports import write_benchmark_exports
from nusc_scene_agent.benchmark_metrics import build_benchmark_metrics, write_benchmark_metrics
from nusc_scene_agent.bev_occupancy_benchmark import (
    generate_bev_occupancy_benchmark_from_perception_benchmark,
    run_proxy_bev_occupancy_study,
)
from nusc_scene_agent.benchmark_registry import build_default_benchmark_registry, write_benchmark_registry
from nusc_scene_agent.benchmark_schema import load_benchmark_config
from nusc_scene_agent.case_library import build_case_library, write_case_library
from nusc_scene_agent.case_library_enrichment import enrich_case_library
from nusc_scene_agent.dataset_backends import inspect_dataset_backends, write_dataset_backend_inventory
from nusc_scene_agent.failure_mining import mine_model_failures
from nusc_scene_agent.failure_aware_reranking import run_failure_aware_reranking_eval
from nusc_scene_agent.llm_client import DEFAULT_TIMEOUT_S, LLMConfig
from nusc_scene_agent.nuplan_closed_loop import run_nuplan_closed_loop_study
from nusc_scene_agent.nuplan_closed_loop_sweep import run_nuplan_closed_loop_sweep
from nusc_scene_agent.nuplan_replay import run_nuplan_replay_study
from nusc_scene_agent.nuplan_sweep import run_nuplan_replay_sweep
from nusc_scene_agent.perception_benchmark import (
    generate_perception_benchmark_from_scenario_config,
    run_proxy_perception_study,
)
from nusc_scene_agent.pipeline import run_query_pipeline
from nusc_scene_agent.scenario_mining_benchmark import generate_scenario_mining_benchmark_from_case_library
from nusc_scene_agent.result_registry import write_result_registry
from nusc_scene_agent.validation import ValidationConfig
from nusc_scene_agent.world_model_benchmark import (
    generate_world_model_benchmark_from_perception_benchmark,
    run_proxy_world_model_study,
)


EXPERIMENT_RESULT_SCHEMA = "experiment_result_v1"
DEFAULT_TRAINVAL_CASE_LIBRARY_OUTPUT = Path("outputs/trainval_case_library_v1")
DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "gemma4:latest"


def run_experiment_config(config_path: Path) -> Dict[str, Any]:
    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    experiment = dict(config.get("experiment") or {})
    experiment_id = str(experiment.get("id") or config_path.stem)
    experiment_type = str(experiment.get("type") or "")

    if experiment_type == "nuplan_replay_study":
        result = _run_nuplan_replay_experiment(config)
    elif experiment_type == "nuplan_closed_loop_study":
        result = _run_nuplan_closed_loop_experiment(config)
    elif experiment_type == "nuplan_replay_sweep":
        result = _run_nuplan_replay_sweep_experiment(config)
    elif experiment_type == "nuplan_closed_loop_sweep":
        result = _run_nuplan_closed_loop_sweep_experiment(config)
    elif experiment_type == "full_benchmark_suite":
        result = _run_full_benchmark_suite_experiment(config, config_path)
    elif experiment_type == "case_library_generation":
        result = _run_case_library_generation_experiment(config)
    elif experiment_type == "risk_benchmark_suite":
        result = _run_risk_benchmark_suite_experiment(config)
    elif experiment_type == "bev_occupancy_study":
        result = _run_bev_occupancy_experiment(config)
    elif experiment_type == "failure_mining":
        result = _run_failure_mining_experiment(config)
    elif experiment_type == "failure_aware_reranking":
        result = _run_failure_aware_reranking_experiment(config)
    elif experiment_type == "registry_export":
        result = _run_registry_export_experiment(config)
    elif experiment_type == "result_registry":
        result = _run_result_registry_experiment(config)
    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")

    payload = {
        "schema": EXPERIMENT_RESULT_SCHEMA,
        "experiment_id": experiment_id,
        "experiment_type": experiment_type,
        "config_path": str(config_path),
        "result": result,
    }
    output_path = _experiment_result_path(config, config_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _run_nuplan_replay_experiment(config: Mapping[str, Any]) -> Dict[str, Any]:
    replay = dict(config.get("nuplan_replay") or {})
    manifest = run_nuplan_replay_study(
        split_dir=Path(str(replay.get("split_dir") or "data/nuplan/dataset/nuplan-v1.1/splits/mini")),
        output_dir=Path(str(replay.get("output") or "outputs/nuplan_replay_study_v1")),
        max_dbs=int(replay.get("max_dbs") or 4),
        max_cases=int(replay.get("max_cases") or 16),
        max_cases_per_db=int(replay.get("max_cases_per_db") or 4),
        history_s=float(replay.get("history_s") if replay.get("history_s") is not None else 2.0),
        future_s=float(replay.get("future_s") if replay.get("future_s") is not None else 4.0),
        frame_hz=float(replay.get("frame_hz") if replay.get("frame_hz") is not None else 2.0),
        min_anchor_gap_s=float(
            replay.get("min_anchor_gap_s") if replay.get("min_anchor_gap_s") is not None else 4.0
        ),
        scenario_tags=replay.get("scenario_tags") or None,
        profiles=replay.get("profiles") or None,
    )
    return {
        "output_dir": str(replay.get("output") or "outputs/nuplan_replay_study_v1"),
        "benchmark": manifest.get("benchmark", {}),
        "comparison": manifest.get("comparison", {}),
        "case_studies": manifest.get("case_studies", {}),
        "unified_cases": manifest.get("unified_cases", {}),
        "artifact_manifest": manifest.get("artifact_manifest", {}),
    }


def _run_nuplan_replay_sweep_experiment(config: Mapping[str, Any]) -> Dict[str, Any]:
    sweep = dict(config.get("nuplan_replay_sweep") or {})
    output_dir = Path(str(sweep.get("output") or dict(config.get("experiment") or {}).get("output") or "outputs/nuplan_replay_sweep_v1"))
    result = run_nuplan_replay_sweep(
        studies=list(sweep.get("studies") or []),
        output_dir=output_dir,
        defaults=dict(sweep.get("defaults") or {}),
        profiles=sweep.get("profiles") or None,
    )
    return {
        "output_dir": str(output_dir),
        "overview": result.get("overview", {}),
        "studies": result.get("studies", []),
        "artifact_manifest": result.get("artifact_manifest", {}),
    }


def _run_nuplan_closed_loop_sweep_experiment(config: Mapping[str, Any]) -> Dict[str, Any]:
    sweep = dict(config.get("nuplan_closed_loop_sweep") or {})
    output_dir = Path(
        str(sweep.get("output") or dict(config.get("experiment") or {}).get("output") or "outputs/nuplan_closed_loop_sweep_v1")
    )
    result = run_nuplan_closed_loop_sweep(
        studies=list(sweep.get("studies") or []),
        output_dir=output_dir,
        defaults=dict(sweep.get("defaults") or {}),
        profiles=sweep.get("profiles") or None,
    )
    return {
        "output_dir": str(output_dir),
        "overview": result.get("overview", {}),
        "studies": result.get("studies", []),
        "artifact_manifest": result.get("artifact_manifest", {}),
    }


def _run_nuplan_closed_loop_experiment(config: Mapping[str, Any]) -> Dict[str, Any]:
    closed_loop = dict(config.get("nuplan_closed_loop") or {})
    manifest = run_nuplan_closed_loop_study(
        split_dir=Path(str(closed_loop.get("split_dir") or "data/nuplan/dataset/nuplan-v1.1/splits/mini")),
        output_dir=Path(str(closed_loop.get("output") or "outputs/nuplan_closed_loop_study_v1")),
        max_dbs=int(closed_loop.get("max_dbs") or 64),
        max_cases=int(closed_loop.get("max_cases") or 16),
        max_cases_per_db=int(closed_loop.get("max_cases_per_db") or 4),
        history_s=float(closed_loop.get("history_s") if closed_loop.get("history_s") is not None else 2.0),
        future_s=float(closed_loop.get("future_s") if closed_loop.get("future_s") is not None else 4.0),
        frame_hz=float(closed_loop.get("frame_hz") if closed_loop.get("frame_hz") is not None else 2.0),
        min_anchor_gap_s=float(
            closed_loop.get("min_anchor_gap_s") if closed_loop.get("min_anchor_gap_s") is not None else 4.0
        ),
        scenario_tags=closed_loop.get("scenario_tags") or None,
        profiles=closed_loop.get("profiles") or None,
    )
    return {
        "output_dir": str(closed_loop.get("output") or "outputs/nuplan_closed_loop_study_v1"),
        "benchmark": manifest.get("benchmark", {}),
        "comparison": manifest.get("comparison", {}),
        "case_studies": manifest.get("case_studies", {}),
        "artifact_manifest": manifest.get("artifact_manifest", {}),
    }


def _run_case_library_generation_experiment(config: Mapping[str, Any]) -> Dict[str, Any]:
    stage = dict(config.get("case_library_generation") or {})
    experiment = dict(config.get("experiment") or {})
    output_dir = Path(str(stage.get("output") or experiment.get("output") or DEFAULT_TRAINVAL_CASE_LIBRARY_OUTPUT))
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmark_path = Path(str(stage.get("benchmark") or "benchmarks/trainval_suite_v1.yaml"))
    db_path = Path(str(stage.get("db") or "artifacts/index/v1.0-trainval.sqlite"))
    candidate_pool = int(stage.get("candidate_pool") or 12)
    query_mode = str(stage.get("query_mode") or "hybrid")
    rerank_mode = str(stage.get("rerank_mode") or "llm")
    llm_config = _stage_llm_config(stage, query_mode=query_mode, rerank_mode=rerank_mode)

    summaries: List[Dict[str, Any]] = []
    benchmark_results: List[Dict[str, Any]] = []
    validation_config = ValidationConfig()
    for spec in load_benchmark_config(benchmark_path):
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

    (output_dir / "benchmark_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    case_library_entries = build_case_library(benchmark_results)
    write_case_library(case_library_entries, output_dir)
    write_benchmark_metrics(build_benchmark_metrics(benchmark_results, case_library_entries), output_dir)
    write_benchmark_exports(benchmark_results, case_library_entries, output_dir)

    case_library_path = output_dir / "case_library.json"
    enriched_path = output_dir / "case_library_enriched.json"
    enrichment: Optional[Dict[str, Any]] = None
    if bool(stage.get("enrich", True)):
        enrichment = enrich_case_library(
            case_library_path=case_library_path,
            db_path=db_path,
            output_path=Path(str(stage.get("enriched_output") or enriched_path)),
        )
        enriched_path = Path(str(enrichment["output_path"]))

    return {
        "output_dir": str(output_dir),
        "benchmark": str(benchmark_path),
        "db": str(db_path),
        "query_mode": query_mode,
        "rerank_mode": rerank_mode,
        "query_count": len(summaries),
        "case_count": len(case_library_entries),
        "case_library": str(case_library_path),
        "enriched_case_library": str(enriched_path) if enriched_path.exists() else "",
        "enrichment": enrichment or {},
    }


def _run_full_benchmark_suite_experiment(config: Mapping[str, Any], config_path: Path) -> Dict[str, Any]:
    suite = dict(config.get("full_benchmark_suite") or {})
    experiment = dict(config.get("experiment") or {})
    output_dir = Path(str(suite.get("output") or experiment.get("output") or "outputs/full_benchmark_suite_v1"))
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_results: Dict[str, Any] = {}
    stage_result_paths: Dict[str, str] = {}

    if _stage_enabled(suite, "case_library_generation", default=bool(config.get("case_library_generation"))):
        stage_config = _merged_stage_config(config, suite, "case_library_generation")
        stage_output = Path(str(stage_config.get("output") or DEFAULT_TRAINVAL_CASE_LIBRARY_OUTPUT))
        stage_experiment = {
            "id": "full_case_library_generation",
            "type": "case_library_generation",
            "output": str(stage_output),
            "result_path": str(stage_output / "experiment_result.json"),
        }
        result = _run_case_library_generation_experiment(
            {"experiment": stage_experiment, "case_library_generation": stage_config}
        )
        stage_results["case_library_generation"] = result
        stage_result_paths["case_library_generation"] = stage_experiment["result_path"]
        _write_stage_result(stage_experiment, f"{config_path}#case_library_generation", result)

    if _stage_enabled(suite, "risk_benchmark_suite", default=True):
        stage_config = _merged_stage_config(config, suite, "risk_benchmark_suite")
        generated_library = dict(stage_results.get("case_library_generation") or {})
        if generated_library and not stage_config.get("case_library"):
            stage_config["case_library"] = (
                generated_library.get("enriched_case_library")
                or generated_library.get("case_library")
                or str(DEFAULT_TRAINVAL_CASE_LIBRARY_OUTPUT / "case_library_enriched.json")
            )
        stage_experiment = {
            "id": "full_risk_benchmark_suite",
            "type": "risk_benchmark_suite",
            "output": str(stage_config.get("output") or "outputs/risk_benchmark_suite_v1"),
            "result_path": str(Path(str(stage_config.get("result_path") or "outputs/risk_benchmark_suite_v1/experiment_result.json"))),
        }
        result = _run_risk_benchmark_suite_experiment(
            {"experiment": stage_experiment, "risk_benchmark_suite": stage_config}
        )
        stage_results["risk_benchmark_suite"] = result
        stage_result_paths["risk_benchmark_suite"] = stage_experiment["result_path"]
        _write_stage_result(stage_experiment, f"{config_path}#risk_benchmark_suite", result)

    if _stage_enabled(suite, "nuplan_replay_sweep", default=True):
        stage_config = _merged_stage_config(config, suite, "nuplan_replay_sweep")
        stage_output = Path(str(stage_config.get("output") or "outputs/nuplan_replay_sweep_v1"))
        stage_experiment = {
            "id": "full_nuplan_replay_sweep",
            "type": "nuplan_replay_sweep",
            "output": str(stage_output),
            "result_path": str(stage_output / "experiment_result.json"),
        }
        result = _run_nuplan_replay_sweep_experiment(
            {"experiment": stage_experiment, "nuplan_replay_sweep": stage_config}
        )
        stage_results["nuplan_replay_sweep"] = result
        stage_result_paths["nuplan_replay_sweep"] = stage_experiment["result_path"]
        _write_stage_result(stage_experiment, f"{config_path}#nuplan_replay_sweep", result)

    if _stage_enabled(suite, "nuplan_closed_loop_sweep", default=True):
        stage_config = _merged_stage_config(config, suite, "nuplan_closed_loop_sweep")
        stage_output = Path(str(stage_config.get("output") or "outputs/nuplan_closed_loop_sweep_v1"))
        stage_experiment = {
            "id": "full_nuplan_closed_loop_sweep",
            "type": "nuplan_closed_loop_sweep",
            "output": str(stage_output),
            "result_path": str(stage_output / "experiment_result.json"),
        }
        result = _run_nuplan_closed_loop_sweep_experiment(
            {"experiment": stage_experiment, "nuplan_closed_loop_sweep": stage_config}
        )
        stage_results["nuplan_closed_loop_sweep"] = result
        stage_result_paths["nuplan_closed_loop_sweep"] = stage_experiment["result_path"]
        _write_stage_result(stage_experiment, f"{config_path}#nuplan_closed_loop_sweep", result)

    if _stage_enabled(suite, "failure_mining", default=True):
        stage_config = _merged_stage_config(config, suite, "failure_mining")
        if not stage_config.get("inputs"):
            stage_config["inputs"] = _full_suite_failure_inputs(stage_results)
        stage_output = Path(str(stage_config.get("output") or "outputs/model_in_the_loop_failure_mining_v1"))
        stage_experiment = {
            "id": "full_failure_mining",
            "type": "failure_mining",
            "output": str(stage_output),
            "result_path": str(stage_output / "experiment_result.json"),
        }
        result = _run_failure_mining_experiment({"experiment": stage_experiment, "failure_mining": stage_config})
        stage_results["failure_mining"] = result
        stage_result_paths["failure_mining"] = stage_experiment["result_path"]
        _write_stage_result(stage_experiment, f"{config_path}#failure_mining", result)

    if _stage_enabled(suite, "result_registry", default=True):
        registry_config = _merged_stage_config(config, suite, "result_registry")
        registry_output = Path(str(registry_config.get("output") or output_dir / "result_registry"))
        sources = registry_config.get("sources") or _full_suite_result_sources(stage_results, stage_result_paths)
        registry = write_result_registry(
            output_dir=registry_output,
            sources=[Path(str(path)) for path in sources],
            metadata={"suite_output": str(output_dir), "config_path": str(config_path)},
        )
        stage_results["result_registry"] = {
            "output_dir": str(registry_output),
            "overview": registry.get("overview", {}),
            "artifact_manifest": registry.get("artifact_manifest", {}),
        }

    summary = {
        "output_dir": str(output_dir),
        "stages": stage_results,
        "stage_result_paths": stage_result_paths,
    }
    (output_dir / "full_benchmark_suite_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "full_benchmark_suite_summary.md").write_text(_render_full_suite_markdown(summary), encoding="utf-8")
    return summary


def _run_bev_occupancy_experiment(config: Mapping[str, Any]) -> Dict[str, Any]:
    bev = dict(config.get("bev_occupancy") or {})
    benchmark_output = Path(str(bev.get("benchmark_output") or "benchmarks/trainval_bev_occupancy_slices_v1.json"))
    output_dir = Path(str(bev.get("output") or dict(config.get("experiment") or {}).get("output") or "outputs/trainval_bev_occupancy_proxy_study_v1"))
    benchmark = generate_bev_occupancy_benchmark_from_perception_benchmark(
        perception_benchmark_path=Path(str(bev.get("perception_benchmark") or "benchmarks/trainval_perception_slices_v1.json")),
        db_path=Path(str(bev.get("db") or "artifacts/index/v1.0-trainval.sqlite")),
        output_path=benchmark_output,
        grid_spec=bev.get("grid_spec") or None,
    )
    study = run_proxy_bev_occupancy_study(benchmark_output, output_dir)
    return {
        "benchmark": benchmark,
        "study": study,
    }


def _run_risk_benchmark_suite_experiment(config: Mapping[str, Any]) -> Dict[str, Any]:
    suite = dict(config.get("risk_benchmark_suite") or {})
    experiment = dict(config.get("experiment") or {})
    output_root = Path(str(suite.get("output") or experiment.get("output") or "outputs/risk_benchmark_suite_v1"))
    db_path = Path(str(suite.get("db") or "artifacts/index/v1.0-trainval.sqlite"))
    case_library_path = Path(
        str(suite.get("case_library") or DEFAULT_TRAINVAL_CASE_LIBRARY_OUTPUT / "case_library_enriched.json")
    )
    max_cases = int(suite.get("max_cases") or 24)
    grid_spec = suite.get("grid_spec") or None

    scenario_output = Path(str(suite.get("scenario_output") or "benchmarks/trainval_scenario_mining_v1.yaml"))
    perception_output = Path(str(suite.get("perception_output") or "benchmarks/trainval_perception_slices_v1.json"))
    world_model_output = Path(str(suite.get("world_model_output") or "benchmarks/trainval_world_model_slices_v1.json"))
    bev_output = Path(str(suite.get("bev_occupancy_output") or "benchmarks/trainval_bev_occupancy_slices_v1.json"))

    scenario = generate_scenario_mining_benchmark_from_case_library(
        case_library_path=case_library_path,
        output_path=scenario_output,
        max_cases=max_cases,
    )
    perception = generate_perception_benchmark_from_scenario_config(
        config_path=scenario_output,
        db_path=db_path,
        output_path=perception_output,
    )
    world_model = generate_world_model_benchmark_from_perception_benchmark(
        perception_benchmark_path=perception_output,
        db_path=db_path,
        output_path=world_model_output,
        grid_spec=grid_spec,
    )
    bev_occupancy = generate_bev_occupancy_benchmark_from_perception_benchmark(
        perception_benchmark_path=perception_output,
        db_path=db_path,
        output_path=bev_output,
        grid_spec=grid_spec,
    )

    proxy_studies: Dict[str, Any] = {}
    if bool(suite.get("run_proxy_studies", True)):
        proxy_outputs = dict(suite.get("proxy_outputs") or {})
        proxy_studies["perception"] = run_proxy_perception_study(
            benchmark_path=perception_output,
            output_dir=Path(str(proxy_outputs.get("perception") or output_root / "perception_proxy_study")),
        )
        proxy_studies["world_model"] = run_proxy_world_model_study(
            benchmark_path=world_model_output,
            output_dir=Path(str(proxy_outputs.get("world_model") or output_root / "world_model_proxy_study")),
        )
        proxy_studies["bev_occupancy"] = run_proxy_bev_occupancy_study(
            benchmark_path=bev_output,
            output_dir=Path(str(proxy_outputs.get("bev_occupancy") or output_root / "bev_occupancy_proxy_study")),
        )

    return {
        "output_root": str(output_root),
        "db": str(db_path),
        "case_library": str(case_library_path),
        "max_cases": max_cases,
        "outputs": {
            "scenario_mining": str(scenario_output),
            "perception": str(perception_output),
            "world_model": str(world_model_output),
            "bev_occupancy": str(bev_output),
        },
        "scenario_mining": scenario,
        "perception": perception,
        "world_model": world_model,
        "bev_occupancy": bev_occupancy,
        "proxy_studies": proxy_studies,
    }


def _run_registry_export_experiment(config: Mapping[str, Any]) -> Dict[str, Any]:
    output_dir = Path(str(dict(config.get("experiment") or {}).get("output") or "outputs/registry_export"))
    output_dir.mkdir(parents=True, exist_ok=True)
    registry = write_benchmark_registry(output_dir / "benchmark_registry.json", build_default_benchmark_registry())
    inventory = write_dataset_backend_inventory(inspect_dataset_backends(), output_dir / "dataset_backends.json")
    return {
        "output_dir": str(output_dir),
        "benchmark_registry": {
            "path": str(output_dir / "benchmark_registry.json"),
            "layer_count": len(registry.get("layers", {})),
        },
        "dataset_backends": {
            "path": str(output_dir / "dataset_backends.json"),
            "backend_count": len(inventory.get("backends", {})),
        },
    }


def _run_result_registry_experiment(config: Mapping[str, Any]) -> Dict[str, Any]:
    result_registry = dict(config.get("result_registry") or {})
    output_dir = Path(
        str(
            result_registry.get("output")
            or dict(config.get("experiment") or {}).get("output")
            or "outputs/full_benchmark_suite_v1/result_registry"
        )
    )
    sources = [Path(str(item)) for item in list(result_registry.get("sources") or [])]
    payload = write_result_registry(
        output_dir=output_dir,
        sources=sources or None,
        metadata={"experiment": dict(config.get("experiment") or {})},
    )
    return {
        "output_dir": str(output_dir),
        "overview": payload.get("overview", {}),
        "artifact_manifest": payload.get("artifact_manifest", {}),
    }


def _run_failure_mining_experiment(config: Mapping[str, Any]) -> Dict[str, Any]:
    failure_mining = dict(config.get("failure_mining") or {})
    output_dir = Path(
        str(
            failure_mining.get("output")
            or dict(config.get("experiment") or {}).get("output")
            or "outputs/model_in_the_loop_failure_mining_v1"
        )
    )
    inputs = [Path(str(item)) for item in list(failure_mining.get("inputs") or [])]
    if not inputs:
        inputs = [
            Path("outputs/trainval_bev_occupancy_proxy_study_v1"),
            Path("outputs/trainval_world_model_proxy_study_v1"),
            Path("outputs/contextvae_world_model_study_v1"),
            Path("outputs/nuplan_replay_sweep_v1"),
            Path("outputs/nuplan_closed_loop_sweep_v1"),
        ]
    return mine_model_failures(
        inputs=inputs,
        output_dir=output_dir,
        max_queries=int(failure_mining.get("max_queries") or 24),
        min_count=int(failure_mining.get("min_count") or 1),
    )


def _run_failure_aware_reranking_experiment(config: Mapping[str, Any]) -> Dict[str, Any]:
    reranking = dict(config.get("failure_aware_reranking") or {})
    output_dir = Path(
        str(
            reranking.get("output")
            or dict(config.get("experiment") or {}).get("output")
            or "outputs/failure_aware_reranking_eval_v1"
        )
    )
    payload = run_failure_aware_reranking_eval(
        query_config=Path(str(reranking.get("query_config") or "outputs/model_in_the_loop_failure_mining_v1/failure_update_queries.yaml")),
        db_path=Path(str(reranking.get("db") or "artifacts/index/v1.0-trainval.sqlite")),
        output_dir=output_dir,
        learned_checkpoint=Path(str(reranking.get("learned_checkpoint") or "outputs/learned_retriever_trainval_large_v2/learned_retriever.pt")),
        candidate_pool=int(reranking.get("candidate_pool") or 48),
        top_k=int(reranking.get("top_k") or 3),
        max_queries=int(reranking.get("max_queries") or 24),
    )
    return {
        "output_dir": str(output_dir),
        "overview": payload.get("overview", {}),
        "artifact_manifest": payload.get("artifact_manifest", {}),
    }


def _experiment_result_path(config: Mapping[str, Any], config_path: Path) -> Path:
    experiment = dict(config.get("experiment") or {})
    output = experiment.get("result_path")
    if output:
        return Path(str(output))
    output_dir = Path(str(experiment.get("output") or "outputs/experiments"))
    return output_dir / f"{config_path.stem}_result.json"


def _stage_llm_config(stage: Mapping[str, Any], *, query_mode: str, rerank_mode: str) -> Optional[LLMConfig]:
    requires_llm = query_mode in {"llm", "hybrid"} or rerank_mode == "llm"
    if not requires_llm:
        return None

    llm = dict(stage.get("llm") or {})
    base_url = str(
        llm.get("base_url")
        or os.getenv("NUSC_SCENE_AGENT_OLLAMA_BASE_URL")
        or DEFAULT_OLLAMA_BASE_URL
    ).strip()
    model = str(
        llm.get("model")
        or os.getenv("NUSC_SCENE_AGENT_OLLAMA_MODEL")
        or DEFAULT_OLLAMA_MODEL
    ).strip()
    timeout_s = float(llm.get("timeout_s") or os.getenv("NUSC_SCENE_AGENT_OLLAMA_TIMEOUT_S") or DEFAULT_TIMEOUT_S)
    if not base_url or not model:
        raise ValueError("Ollama base URL and model are required for LLM-backed experiment stages.")
    return LLMConfig(base_url=base_url, model=model, timeout_s=timeout_s)


def _stage_enabled(suite: Mapping[str, Any], name: str, *, default: bool) -> bool:
    stages = dict(suite.get("stages") or {})
    if name not in stages:
        return default
    return bool(stages.get(name))


def _merged_stage_config(config: Mapping[str, Any], suite: Mapping[str, Any], name: str) -> Dict[str, Any]:
    base = dict(config.get(name) or {})
    override = dict(suite.get(name) or {})
    return {**base, **override}


def _write_stage_result(experiment: Mapping[str, Any], config_path: str, result: Mapping[str, Any]) -> None:
    output_path = Path(str(experiment["result_path"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": EXPERIMENT_RESULT_SCHEMA,
        "experiment_id": str(experiment.get("id") or ""),
        "experiment_type": str(experiment.get("type") or ""),
        "config_path": config_path,
        "result": dict(result),
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _full_suite_failure_inputs(stage_results: Mapping[str, Any]) -> list[str]:
    inputs = [
        "outputs/trainval_bev_occupancy_proxy_study_v1",
        "outputs/trainval_world_model_proxy_study_v1",
        "outputs/contextvae_world_model_study_v1",
        "outputs/nuscenes_forecast_baselines_eval",
    ]
    replay = dict(stage_results.get("nuplan_replay_sweep") or {})
    closed_loop = dict(stage_results.get("nuplan_closed_loop_sweep") or {})
    if replay.get("output_dir"):
        inputs.append(str(replay["output_dir"]))
    else:
        inputs.append("outputs/nuplan_replay_sweep_v1")
    if closed_loop.get("output_dir"):
        inputs.append(str(closed_loop["output_dir"]))
    else:
        inputs.append("outputs/nuplan_closed_loop_sweep_v1")
    return inputs


def _full_suite_result_sources(stage_results: Mapping[str, Any], stage_result_paths: Mapping[str, str]) -> list[str]:
    sources = []
    if stage_result_paths.get("risk_benchmark_suite"):
        sources.append(stage_result_paths["risk_benchmark_suite"])
    replay = dict(stage_results.get("nuplan_replay_sweep") or {})
    if replay.get("output_dir"):
        sources.append(str(Path(str(replay["output_dir"])) / "nuplan_replay_sweep_summary.json"))
    closed_loop = dict(stage_results.get("nuplan_closed_loop_sweep") or {})
    if closed_loop.get("output_dir"):
        sources.append(str(Path(str(closed_loop["output_dir"])) / "nuplan_closed_loop_sweep_summary.json"))
    failure = dict(stage_results.get("failure_mining") or {})
    if failure.get("report_json"):
        sources.append(str(failure["report_json"]))
    return sources


def _render_full_suite_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Full Benchmark Suite",
        "",
        f"- Output: `{summary.get('output_dir', '')}`",
        "",
        "| Stage | Output |",
        "| --- | --- |",
    ]
    for name, result in dict(summary.get("stages") or {}).items():
        if isinstance(result, Mapping):
            lines.append(f"| `{name}` | `{result.get('output_dir', '')}` |")
    lines.append("")
    return "\n".join(lines)

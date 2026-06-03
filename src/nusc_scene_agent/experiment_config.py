from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml

from nusc_scene_agent.bev_occupancy_benchmark import (
    generate_bev_occupancy_benchmark_from_perception_benchmark,
    run_proxy_bev_occupancy_study,
)
from nusc_scene_agent.benchmark_registry import build_default_benchmark_registry, write_benchmark_registry
from nusc_scene_agent.dataset_backends import inspect_dataset_backends, write_dataset_backend_inventory
from nusc_scene_agent.failure_mining import mine_model_failures
from nusc_scene_agent.nuplan_replay import run_nuplan_replay_study
from nusc_scene_agent.nuplan_sweep import run_nuplan_replay_sweep
from nusc_scene_agent.perception_benchmark import (
    generate_perception_benchmark_from_scenario_config,
    run_proxy_perception_study,
)
from nusc_scene_agent.scenario_mining_benchmark import generate_scenario_mining_benchmark_from_case_library
from nusc_scene_agent.world_model_benchmark import (
    generate_world_model_benchmark_from_perception_benchmark,
    run_proxy_world_model_study,
)


EXPERIMENT_RESULT_SCHEMA = "experiment_result_v1"


def run_experiment_config(config_path: Path) -> Dict[str, Any]:
    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    experiment = dict(config.get("experiment") or {})
    experiment_id = str(experiment.get("id") or config_path.stem)
    experiment_type = str(experiment.get("type") or "")

    if experiment_type == "nuplan_replay_study":
        result = _run_nuplan_replay_experiment(config)
    elif experiment_type == "nuplan_replay_sweep":
        result = _run_nuplan_replay_sweep_experiment(config)
    elif experiment_type == "risk_benchmark_suite":
        result = _run_risk_benchmark_suite_experiment(config)
    elif experiment_type == "bev_occupancy_study":
        result = _run_bev_occupancy_experiment(config)
    elif experiment_type == "failure_mining":
        result = _run_failure_mining_experiment(config)
    elif experiment_type == "registry_export":
        result = _run_registry_export_experiment(config)
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
        output_dir=Path(str(replay.get("output") or "outputs/nuplan_mini_replay_study_v2")),
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
        "output_dir": str(replay.get("output") or "outputs/nuplan_mini_replay_study_v2"),
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
        str(suite.get("case_library") or "outputs/trainval_suite_llm_hybrid_en_v1/case_library_enriched.json")
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
        ]
    return mine_model_failures(
        inputs=inputs,
        output_dir=output_dir,
        max_queries=int(failure_mining.get("max_queries") or 24),
        min_count=int(failure_mining.get("min_count") or 1),
    )


def _experiment_result_path(config: Mapping[str, Any], config_path: Path) -> Path:
    experiment = dict(config.get("experiment") or {})
    output = experiment.get("result_path")
    if output:
        return Path(str(output))
    output_dir = Path(str(experiment.get("output") or "outputs/experiments"))
    return output_dir / f"{config_path.stem}_result.json"

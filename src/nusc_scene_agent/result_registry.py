from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from nusc_scene_agent.artifact_manifest import build_artifact_entry, write_artifact_manifest
from nusc_scene_agent.benchmark_registry import build_default_benchmark_registry


RESULT_REGISTRY_SCHEMA = "benchmark_result_registry_v1"
DEFAULT_RESULT_REGISTRY_OUTPUT = Path("outputs/full_benchmark_suite_v1/result_registry")


DEFAULT_RESULT_SOURCES = [
    Path("outputs/risk_benchmark_suite_v1/experiment_result.json"),
    Path("outputs/nuplan_replay_sweep_v1/nuplan_replay_sweep_summary.json"),
    Path("outputs/nuplan_closed_loop_sweep_v1/nuplan_closed_loop_sweep_summary.json"),
    Path("outputs/model_in_the_loop_failure_mining_v1/failure_mining_report.json"),
]


def write_result_registry(
    output_dir: Path = DEFAULT_RESULT_REGISTRY_OUTPUT,
    *,
    sources: Optional[Sequence[Path]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_paths = [Path(path) for path in (sources or DEFAULT_RESULT_SOURCES)]
    entries = [_build_result_entry(path) for path in source_paths]
    entries = [entry for entry in entries if entry is not None]
    layer_registry = build_default_benchmark_registry()
    payload = {
        "schema": RESULT_REGISTRY_SCHEMA,
        "metadata": dict(metadata or {}),
        "overview": {
            "result_count": len(entries),
            "layer_count": len({entry["layer_id"] for entry in entries}),
            "source_count": len(source_paths),
            "existing_source_count": sum(1 for path in source_paths if path.exists()),
        },
        "layers": layer_registry.get("layers", {}),
        "results": entries,
    }
    (output_dir / "result_registry.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_registry_csv(entries, output_dir / "result_registry.csv")
    (output_dir / "result_registry.md").write_text(_render_registry_markdown(payload), encoding="utf-8")
    artifact_manifest = write_artifact_manifest(
        output_dir=output_dir,
        artifacts=[
            build_artifact_entry(output_dir / "result_registry.json", "summary", "result_registry", output_dir),
            build_artifact_entry(output_dir / "result_registry.csv", "summary", "result_registry_table", output_dir),
            build_artifact_entry(output_dir / "result_registry.md", "summary", "result_registry_report", output_dir),
            *[
                build_artifact_entry(path, "source", _infer_source_kind(path), output_dir)
                for path in source_paths
            ],
        ],
        metadata={"schema": RESULT_REGISTRY_SCHEMA},
    )
    payload["artifact_manifest"] = {
        "path": str(output_dir / "artifact_manifest.json"),
        "overview": artifact_manifest.get("overview", {}),
    }
    (output_dir / "result_registry.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _build_result_entry(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    payload = _read_json(path)
    schema = str(payload.get("schema") or "")
    if schema == "experiment_result_v1" and payload.get("experiment_type") == "risk_benchmark_suite":
        return _risk_suite_entry(path, payload)
    if schema == "nuplan_replay_sweep_v1":
        return _nuplan_replay_sweep_entry(path, payload)
    if schema == "nuplan_closed_loop_sweep_v1":
        return _nuplan_closed_loop_sweep_entry(path, payload)
    if schema == "model_in_the_loop_failure_mining_report_v1":
        return _failure_mining_entry(path, payload)
    return {
        "layer_id": "unknown",
        "source_path": str(path),
        "schema": schema,
        "summary": {},
        "metrics": {},
    }


def _risk_suite_entry(path: Path, payload: Mapping[str, Any]) -> Dict[str, Any]:
    result = dict(payload.get("result") or {})
    outputs = dict(result.get("outputs") or {})
    proxy = dict(result.get("proxy_studies") or {})
    return {
        "layer_id": "risk_benchmark_suite",
        "source_path": str(path),
        "schema": str(payload.get("schema") or ""),
        "summary": {
            "experiment_id": payload.get("experiment_id"),
            "scenario_mining_output": outputs.get("scenario_mining"),
            "perception_output": outputs.get("perception"),
            "world_model_output": outputs.get("world_model"),
            "bev_occupancy_output": outputs.get("bev_occupancy"),
        },
        "metrics": {
            "scenario_anchors": _nested_int(result, ["scenario_mining", "anchor_case_count"]),
            "scenario_queries": _nested_int(result, ["scenario_mining", "query_count"]),
            "perception_cases": _nested_int(result, ["perception", "case_count"]),
            "world_model_cases": _nested_int(result, ["world_model", "case_count"]),
            "bev_occupancy_cases": _nested_int(result, ["bev_occupancy", "case_count"]),
            "perception_proxy_cases": _nested_int(proxy, ["perception", "case_count"]),
            "world_model_proxy_cases": _nested_int(proxy, ["world_model", "case_count"]),
            "bev_occupancy_proxy_cases": _nested_int(proxy, ["bev_occupancy", "case_count"]),
        },
    }


def _nuplan_replay_sweep_entry(path: Path, payload: Mapping[str, Any]) -> Dict[str, Any]:
    overview = dict(payload.get("overview") or {})
    overall = _overall_rows(payload)
    best_non_oracle = _best_profile(
        overall,
        metric="mean_risk_fidelity_score",
        exclude={"logged_ego"},
    )
    return {
        "layer_id": "nuplan_replay_regression",
        "source_path": str(path),
        "schema": str(payload.get("schema") or ""),
        "summary": {
            "studies": overview.get("study_count"),
            "profiles": overview.get("profiles", []),
            "best_non_oracle_profile": best_non_oracle.get("profile_name"),
        },
        "metrics": {
            "db_count_scanned": overview.get("db_count_scanned"),
            "candidate_case_count": overview.get("candidate_case_count"),
            "case_count": overview.get("case_count"),
            "best_non_oracle_risk_fidelity": best_non_oracle.get("mean_risk_fidelity_score"),
            "best_non_oracle_ego_ade_m": best_non_oracle.get("mean_ego_ade_m"),
        },
    }


def _nuplan_closed_loop_sweep_entry(path: Path, payload: Mapping[str, Any]) -> Dict[str, Any]:
    overview = dict(payload.get("overview") or {})
    overall = _overall_rows(payload)
    best_non_oracle = _best_profile(
        overall,
        metric="mean_closed_loop_score",
        exclude={"logged_ego_oracle"},
    )
    return {
        "layer_id": "nuplan_closed_loop_replay",
        "source_path": str(path),
        "schema": str(payload.get("schema") or ""),
        "summary": {
            "studies": overview.get("study_count"),
            "profiles": overview.get("profiles", []),
            "best_non_oracle_profile": best_non_oracle.get("profile_name"),
        },
        "metrics": {
            "db_count_scanned": overview.get("db_count_scanned"),
            "candidate_case_count": overview.get("candidate_case_count"),
            "case_count": overview.get("case_count"),
            "best_non_oracle_closed_loop_score": best_non_oracle.get("mean_closed_loop_score"),
            "best_non_oracle_ego_ade_m": best_non_oracle.get("mean_ego_ade_m"),
            "best_non_oracle_progress_ratio": best_non_oracle.get("mean_progress_ratio"),
        },
    }


def _failure_mining_entry(path: Path, payload: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "layer_id": "failure_mining",
        "source_path": str(path),
        "schema": str(payload.get("schema") or ""),
        "summary": {
            "report_md": str(path.with_suffix(".md")),
        },
        "metrics": {
            "source_count": payload.get("source_count") or payload.get("sources"),
            "failure_record_count": payload.get("failure_record_count")
            or payload.get("failure_records")
            or payload.get("record_count"),
            "cluster_count": payload.get("cluster_count") or payload.get("clusters"),
            "update_query_count": payload.get("update_query_count") or len(list(payload.get("update_queries") or [])),
        },
    }


def _overall_rows(payload: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    return [
        dict(row)
        for row in payload.get("profile_leaderboard", [])
        if str(row.get("study_name") or "") == "__overall__"
    ]


def _best_profile(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    exclude: set[str],
) -> Dict[str, Any]:
    candidates = [dict(row) for row in rows if str(row.get("profile_name") or "") not in exclude]
    candidates = [row for row in candidates if row.get(metric) is not None]
    if not candidates:
        return {}
    return max(candidates, key=lambda row: float(row.get(metric) or 0.0))


def _nested_int(payload: Mapping[str, Any], path: Sequence[str]) -> Optional[int]:
    value: Any = payload
    for key in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    if value is None:
        return None
    return int(value)


def _write_registry_csv(entries: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    fieldnames = ["layer_id", "source_path", "schema", "summary", "metrics"]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for entry in entries:
            writer.writerow(
                {
                    "layer_id": entry.get("layer_id", ""),
                    "source_path": entry.get("source_path", ""),
                    "schema": entry.get("schema", ""),
                    "summary": json.dumps(entry.get("summary", {}), sort_keys=True),
                    "metrics": json.dumps(entry.get("metrics", {}), sort_keys=True),
                }
            )


def _render_registry_markdown(payload: Mapping[str, Any]) -> str:
    overview = dict(payload.get("overview") or {})
    lines = [
        "# Benchmark Result Registry",
        "",
        f"- Results: `{overview.get('result_count', 0)}`",
        f"- Layers: `{overview.get('layer_count', 0)}`",
        f"- Existing sources: `{overview.get('existing_source_count', 0)}/{overview.get('source_count', 0)}`",
        "",
        "| Layer | Source | Key Metrics |",
        "| --- | --- | --- |",
    ]
    for entry in payload.get("results", []):
        metrics = ", ".join(f"{key}={_format_metric(value)}" for key, value in dict(entry.get("metrics") or {}).items())
        lines.append(
            f"| `{entry.get('layer_id', '')}` | `{entry.get('source_path', '')}` | {metrics} |"
        )
    lines.append("")
    return "\n".join(lines)


def _format_metric(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _infer_source_kind(path: Path) -> str:
    name = str(path).lower()
    if "replay_sweep" in name:
        return "nuplan_replay_sweep"
    if "closed_loop_sweep" in name:
        return "nuplan_closed_loop_sweep"
    if "failure_mining" in name:
        return "failure_mining_report"
    if "risk_benchmark_suite" in name:
        return "risk_benchmark_suite_result"
    return "result_source"


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


UNIFIED_CASE_SCHEMA = "unified_risk_case_v1"
UNIFIED_COLLECTION_SCHEMA = "unified_risk_case_collection_v1"


@dataclass
class UnifiedRiskCase:
    case_id: str
    dataset: str
    benchmark_layer: str
    source: Dict[str, Any]
    scenario_family: str
    scenario_tags: List[str] = field(default_factory=list)
    difficulty_label: str = ""
    location: str = ""
    timestamp_us: int | None = None
    sample_idx: int | None = None
    actors: List[Dict[str, Any]] = field(default_factory=list)
    ego_history: List[Dict[str, Any]] = field(default_factory=list)
    ego_future: List[Dict[str, Any]] = field(default_factory=list)
    risk_targets: Dict[str, Any] = field(default_factory=dict)
    comfort_targets: Dict[str, Any] = field(default_factory=dict)
    evidence: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    schema: str = UNIFIED_CASE_SCHEMA

    def to_dict(self) -> Dict[str, Any]:
        return _json_safe(asdict(self))


@dataclass
class UnifiedCaseCollection:
    cases: List[UnifiedRiskCase]
    metadata: Dict[str, Any] = field(default_factory=dict)
    schema: str = UNIFIED_COLLECTION_SCHEMA

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "metadata": dict(self.metadata),
            "cases": [case.to_dict() for case in self.cases],
        }

    def write(self, output_path: Path) -> Dict[str, Any]:
        payload = self.to_dict()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
        return payload


def write_unified_case_collection(
    cases: Sequence[UnifiedRiskCase],
    output_path: Path,
    metadata: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    return UnifiedCaseCollection(list(cases), dict(metadata or {})).write(output_path)


def unified_case_from_nuplan_replay(case: Mapping[str, Any]) -> UnifiedRiskCase:
    scenario_tag = str(case.get("scenario_tag") or "")
    actors = []
    anchor_actor = dict(case.get("anchor_frame", {}).get("primary_actor") or {})
    if anchor_actor:
        actors.append(
            {
                "role": "primary_risk_actor",
                "category_name": anchor_actor.get("category_name", case.get("category_name", "")),
                "track_token": case.get("anchor_track_token", ""),
                "state": _compact_actor_state(anchor_actor),
            }
        )

    frames = list(case.get("frames") or [])
    ego_history = [
        _compact_ego_frame(frame)
        for frame in frames
        if float(frame.get("dt_from_anchor_s", 0.0)) < 0.0
    ]
    ego_future = [_compact_ego_frame(frame) for frame in case.get("future_frames", [])]

    return UnifiedRiskCase(
        case_id=str(case.get("case_id") or ""),
        dataset="nuplan",
        benchmark_layer="replay_regression",
        source={
            "source_db": case.get("source_db", ""),
            "log_name": case.get("log_name", ""),
            "scene_name": case.get("scene_name", ""),
            "anchor_lidar_pc_token": case.get("anchor_lidar_pc_token", ""),
        },
        scenario_family=str(case.get("scenario_family") or "unknown"),
        scenario_tags=[scenario_tag] if scenario_tag else [],
        difficulty_label=str(case.get("difficulty_label") or ""),
        location=str(case.get("location") or ""),
        timestamp_us=int(case["anchor_timestamp_us"]) if case.get("anchor_timestamp_us") is not None else None,
        actors=actors,
        ego_history=ego_history,
        ego_future=ego_future,
        risk_targets=dict(case.get("risk_targets") or {}),
        comfort_targets=dict(case.get("comfort_targets") or {}),
        evidence={
            "risk_facets": dict(case.get("risk_facets") or {}),
            "map_version": case.get("map_version", ""),
            "history_frame_count": case.get("history_frame_count", 0),
            "future_frame_count": case.get("future_frame_count", 0),
        },
        metadata={
            "vehicle_name": case.get("vehicle_name", ""),
            "category_name": case.get("category_name", ""),
            "scenario_description": case.get("scenario_description", ""),
        },
    )


def unified_cases_from_nuplan_benchmark(benchmark: Mapping[str, Any]) -> List[UnifiedRiskCase]:
    return [unified_case_from_nuplan_replay(case) for case in benchmark.get("cases", [])]


def unified_case_from_nuscenes_library_entry(entry: Mapping[str, Any]) -> UnifiedRiskCase:
    case_key = str(entry.get("case_key") or "")
    tags = [str(item) for item in entry.get("source_query_tags", []) if item]
    behaviors = [str(item) for item in entry.get("all_behaviors", []) if item]
    family = _nuscenes_family_from_entry(entry, tags, behaviors)
    return UnifiedRiskCase(
        case_id=case_key,
        dataset="nuscenes",
        benchmark_layer="scenario_mining",
        source={
            "scene_name": entry.get("scene_name", ""),
            "scene_token": entry.get("scene_token", ""),
            "sample_token": entry.get("sample_token", ""),
            "instance_token": entry.get("instance_token", ""),
        },
        scenario_family=family,
        scenario_tags=tags,
        difficulty_label=_nuscenes_difficulty_label(entry),
        location=str(entry.get("location") or ""),
        sample_idx=int(entry["sample_idx"]) if entry.get("sample_idx") is not None else None,
        actors=[
            {
                "role": "primary_risk_actor",
                "category_name": entry.get("category_name", ""),
                "category_group": entry.get("category_group", ""),
                "instance_token": entry.get("instance_token", ""),
            }
        ],
        risk_targets={
            "min_distance_m": entry.get("min_distance_m"),
            "min_ttc_s": entry.get("min_ttc_s"),
        },
        evidence={
            "passed": bool(entry.get("passed")),
            "validation_score": entry.get("validation_score"),
            "retrieval_score": entry.get("retrieval_score"),
            "matched_behaviors": list(entry.get("matched_behaviors", [])),
            "all_behaviors": behaviors,
            "map_available": bool(entry.get("map_available")),
            "map_crosswalk": bool(entry.get("map_crosswalk")),
            "map_walkway": bool(entry.get("map_walkway")),
            "map_shared_lane": bool(entry.get("map_shared_lane")),
            "map_actor_uses_ego_lane": bool(entry.get("map_actor_uses_ego_lane")),
        },
        artifacts={
            "figure": str(entry.get("figure_path") or ""),
            "report_dir": str(entry.get("report_dir") or ""),
        },
        metadata={
            "source_query_ids": list(entry.get("source_query_ids", [])),
            "source_queries": list(entry.get("source_queries", [])),
        },
    )


def unified_cases_from_nuscenes_case_library(entries: Sequence[Mapping[str, Any]]) -> List[UnifiedRiskCase]:
    return [unified_case_from_nuscenes_library_entry(entry) for entry in entries]


def load_unified_case_source(source_path: Path, source_type: str) -> List[UnifiedRiskCase]:
    payload = json.loads(Path(source_path).read_text(encoding="utf-8"))
    if source_type == "nuplan_replay_benchmark":
        return unified_cases_from_nuplan_benchmark(payload)
    if source_type == "nuscenes_case_library":
        if not isinstance(payload, list):
            raise ValueError("nuScenes case-library source must be a JSON list.")
        return unified_cases_from_nuscenes_case_library(payload)
    raise ValueError(f"Unknown unified case source type: {source_type}")


def _compact_ego_frame(frame: Mapping[str, Any]) -> Dict[str, Any]:
    ego = dict(frame.get("ego") or {})
    return {
        "timestamp_us": frame.get("timestamp_us"),
        "dt_from_anchor_s": frame.get("dt_from_anchor_s"),
        "x": ego.get("x"),
        "y": ego.get("y"),
        "yaw": ego.get("yaw"),
        "vx": ego.get("vx"),
        "vy": ego.get("vy"),
        "speed_mps": ego.get("speed_mps"),
    }


def _compact_actor_state(actor: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "x": actor.get("x"),
        "y": actor.get("y"),
        "yaw": actor.get("yaw"),
        "vx": actor.get("vx"),
        "vy": actor.get("vy"),
        "distance_m": actor.get("distance_m"),
        "ttc_s": actor.get("ttc_s"),
    }


def _nuscenes_family_from_entry(
    entry: Mapping[str, Any],
    tags: Sequence[str],
    behaviors: Sequence[str],
) -> str:
    actor = str(entry.get("category_group") or entry.get("category_name") or "")
    tag_set = set(tags)
    behavior_set = set(behaviors)
    if "pedestrian" in actor or "pedestrian" in tag_set:
        return "vru_interaction"
    if behavior_set.intersection({"cut_in", "oncoming"}):
        return "dynamic_vehicle_interaction"
    if "stopped_lead" in behavior_set or "lane_blocking" in tag_set:
        return "blocked_path_interaction"
    if "proximity" in tag_set:
        return "close_proximity_interaction"
    return "general_interaction"


def _nuscenes_difficulty_label(entry: Mapping[str, Any]) -> str:
    if not bool(entry.get("passed")):
        return "failed"
    score = float(entry.get("validation_score") or 0.0)
    if score >= 90.0:
        return "high_confidence"
    if score >= 75.0:
        return "medium_confidence"
    return "low_confidence"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value

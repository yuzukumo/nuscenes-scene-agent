from __future__ import annotations

import csv
import hashlib
import html
import json
import math
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from nusc_scene_agent.artifact_manifest import build_artifact_entry, write_artifact_manifest
from nusc_scene_agent.unified_schema import unified_cases_from_nuplan_benchmark, write_unified_case_collection


DEFAULT_NUPLAN_DATASET_ROOT = Path("data/nuplan/dataset")
DEFAULT_NUPLAN_SPLIT = DEFAULT_NUPLAN_DATASET_ROOT / "nuplan-v1.1/splits/mini"
DEFAULT_NUPLAN_REPLAY_BENCHMARK = Path("outputs/nuplan_replay_benchmark.json")

NUPLAN_REPLAY_PROFILES = ["logged_ego", "history_kinematic", "constant_velocity", "stopped"]

DEFAULT_SCENARIO_TAG_PRIORITY = [
    "near_pedestrian_on_crosswalk",
    "near_pedestrian_at_pickup_dropoff",
    "near_high_speed_vehicle",
    "near_long_vehicle",
    "near_construction_zone_sign",
    "near_trafficcone_on_driveable",
]

SCENARIO_FAMILY_DESCRIPTIONS = {
    "vru_interaction": "Primary interaction with a pedestrian or other vulnerable road user.",
    "high_speed_interaction": "Primary interaction with a high-speed dynamic actor.",
    "large_vehicle_interaction": "Primary interaction with a long or large vehicle.",
    "static_obstacle_context": "Replay context around construction, cones, or other static obstacles.",
    "intersection_context": "Replay context around an intersection or traffic-light region.",
    "general_interaction": "Risk interaction that does not match a more specific family.",
}

COMFORT_THRESHOLDS = {
    "max_acceleration_mps2": 4.0,
    "max_jerk_mps3": 5.0,
    "max_yaw_rate_rps": 1.0,
}

NUPLAN_PLOT_BACKGROUND = "#f7f4ee"
NUPLAN_GRID_COLOR = "#d9d2c7"
NUPLAN_HISTORY_COLOR = "#303030"
NUPLAN_LOGGED_EGO_COLOR = "#1f6f99"
NUPLAN_ACTOR_COLOR = "#c44e38"
NUPLAN_PROFILE_COLORS = ["#2a9d8f", "#c46a00", "#6f4e7c", "#6b8e23", "#7a1f1f"]

KINEMATIC_ACCELERATION_LIMIT_MPS2 = 4.0
KINEMATIC_YAW_RATE_LIMIT_RPS = 0.8


@dataclass(frozen=True)
class NuPlanAnchor:
    db_path: Path
    log_name: str
    vehicle_name: str
    location: str
    map_version: str
    scene_name: str
    scenario_tag: str
    category_name: str
    timestamp_us: int
    lidar_pc_token: bytes
    scene_token: bytes
    track_token: bytes
    ego_x: float
    ego_y: float
    ego_yaw: float
    ego_vx: float
    ego_vy: float
    actor_x: float
    actor_y: float
    actor_vx: float
    actor_vy: float


def inspect_nuplan_dataset(dataset_root: Path = DEFAULT_NUPLAN_DATASET_ROOT) -> Dict[str, Any]:
    dataset_root = Path(dataset_root)
    cache_root = dataset_root / "data/cache"
    split_root = dataset_root / "nuplan-v1.1/splits"
    split_counts: Dict[str, int] = {}
    cache_counts: Dict[str, int] = {}

    if cache_root.exists():
        for child in sorted(cache_root.iterdir()):
            if child.is_dir():
                cache_counts[child.name] = _count_db_files(child)

    if split_root.exists():
        for child in sorted(split_root.iterdir()):
            if child.is_dir() or child.is_symlink():
                split_counts[child.name] = _count_db_files(child)

    map_root = dataset_root / "maps"
    return {
        "dataset_root": str(dataset_root),
        "exists": dataset_root.exists(),
        "cache_counts": cache_counts,
        "split_counts": split_counts,
        "map_json_count": len(list(map_root.rglob("*.json"))) if map_root.exists() else 0,
        "map_directories": sorted(str(path.relative_to(map_root)) for path in map_root.iterdir() if path.is_dir())
        if map_root.exists()
        else [],
    }


def generate_nuplan_replay_benchmark(
    split_dir: Path = DEFAULT_NUPLAN_SPLIT,
    output_path: Path = DEFAULT_NUPLAN_REPLAY_BENCHMARK,
    *,
    max_dbs: int = 4,
    max_cases: int = 16,
    max_cases_per_db: int = 4,
    history_s: float = 2.0,
    future_s: float = 4.0,
    frame_hz: float = 2.0,
    min_anchor_gap_s: float = 4.0,
    scenario_tags: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    split_dir = Path(split_dir)
    output_path = Path(output_path)
    tags = list(scenario_tags or DEFAULT_SCENARIO_TAG_PRIORITY)
    db_paths = _discover_db_paths(split_dir, max_dbs=max_dbs)

    candidate_cases: List[Dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    for db_path in db_paths:
        anchors = _select_anchors(
            db_path=db_path,
            scenario_tags=tags,
            max_cases=max_cases_per_db,
            min_anchor_gap_s=min_anchor_gap_s,
        )
        if not anchors:
            skipped["db_without_matching_anchor"] += 1
            continue
        for anchor in anchors:
            case = _build_replay_case(anchor, history_s=history_s, future_s=future_s, frame_hz=frame_hz)
            if not case:
                skipped["anchor_without_sufficient_window"] += 1
                continue
            candidate_cases.append(case)

    cases = _select_diverse_cases(candidate_cases, max_cases=max_cases)
    sampling_summary = _build_sampling_summary(cases)

    payload = {
        "metadata": {
            "schema": "nuplan_replay_benchmark_v2",
            "split_dir": str(split_dir),
            "db_count_scanned": len(db_paths),
            "candidate_case_count": len(candidate_cases),
            "case_count": len(cases),
            "history_s": history_s,
            "future_s": future_s,
            "frame_hz": frame_hz,
            "scenario_tags": tags,
            "sampling_policy": {
                "name": "family_tag_location_balanced",
                "max_dbs": max_dbs,
                "max_cases": max_cases,
                "max_cases_per_db": max_cases_per_db,
                "min_anchor_gap_s": min_anchor_gap_s,
            },
            "taxonomy": _scenario_taxonomy_payload(tags),
            "case_distribution": sampling_summary,
            "skipped": dict(skipped),
        },
        "cases": cases,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload["metadata"]


def generate_nuplan_proxy_rollouts(
    benchmark_path: Path,
    output_path: Path,
    profile_name: str = "constant_velocity",
) -> Dict[str, Any]:
    if profile_name not in NUPLAN_REPLAY_PROFILES:
        raise ValueError(f"Unknown nuPlan replay profile: {profile_name}")

    benchmark = _read_json(Path(benchmark_path))
    predictions: List[Dict[str, Any]] = []
    for case in benchmark.get("cases", []):
        anchor = case["anchor_frame"]
        future_frames = case["future_frames"]
        kinematic_context = _history_kinematic_context(case) if profile_name == "history_kinematic" else {}
        anchor_ts = int(anchor["timestamp_us"])
        anchor_x = float(anchor["ego"]["x"])
        anchor_y = float(anchor["ego"]["y"])
        anchor_yaw = float(anchor["ego"]["yaw"])
        anchor_vx = float(anchor["ego"]["vx"])
        anchor_vy = float(anchor["ego"]["vy"])
        future_ego_states: List[Dict[str, float]] = []
        for frame in future_frames:
            timestamp_us = int(frame["timestamp_us"])
            dt = (timestamp_us - anchor_ts) / 1_000_000.0
            if profile_name == "logged_ego":
                ego = frame["ego"]
                x = float(ego["x"])
                y = float(ego["y"])
                yaw = float(ego["yaw"])
                vx = float(ego["vx"])
                vy = float(ego["vy"])
                acceleration_x = float(ego.get("acceleration_x", 0.0))
                acceleration_y = float(ego.get("acceleration_y", 0.0))
                angular_rate_z = float(ego.get("angular_rate_z", 0.0))
            elif profile_name == "history_kinematic":
                state = _history_kinematic_state_at(kinematic_context, timestamp_us)
                x = state["x"]
                y = state["y"]
                yaw = state["yaw"]
                vx = state["vx"]
                vy = state["vy"]
                acceleration_x = state["acceleration_x"]
                acceleration_y = state["acceleration_y"]
                angular_rate_z = state["angular_rate_z"]
            elif profile_name == "constant_velocity":
                x = anchor_x + anchor_vx * dt
                y = anchor_y + anchor_vy * dt
                yaw = anchor_yaw
                vx = anchor_vx
                vy = anchor_vy
                acceleration_x = 0.0
                acceleration_y = 0.0
                angular_rate_z = 0.0
            else:
                x = anchor_x
                y = anchor_y
                yaw = anchor_yaw
                vx = 0.0
                vy = 0.0
                acceleration_x = 0.0
                acceleration_y = 0.0
                angular_rate_z = 0.0
            future_ego_states.append(
                {
                    "timestamp_us": timestamp_us,
                    "x": x,
                    "y": y,
                    "yaw": yaw,
                    "vx": vx,
                    "vy": vy,
                    "speed_mps": math.hypot(vx, vy),
                    "acceleration_x": acceleration_x,
                    "acceleration_y": acceleration_y,
                    "angular_rate_z": angular_rate_z,
                }
            )
        predictions.append({"case_id": case["case_id"], "future_ego_states": future_ego_states})

    payload = {
        "metadata": {
            "schema": "nuplan_ego_rollout_predictions_v1",
            "profile_name": profile_name,
            "profile_description": _nuplan_replay_profile_description(profile_name),
            "benchmark_path": str(benchmark_path),
            "prediction_count": len(predictions),
        },
        "predictions": predictions,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload["metadata"]


def evaluate_nuplan_rollouts(
    benchmark_path: Path,
    predictions_path: Path,
    output_dir: Path,
    profile_name: str = "",
    collision_distance_m: float = 2.0,
) -> Dict[str, Any]:
    benchmark = _read_json(Path(benchmark_path))
    predictions_payload = _read_json(Path(predictions_path))
    profile = profile_name or str(predictions_payload.get("metadata", {}).get("profile_name") or "unnamed")
    predictions_by_case = {
        str(prediction.get("case_id")): prediction for prediction in predictions_payload.get("predictions", [])
    }

    case_metrics: List[Dict[str, Any]] = []
    for case in benchmark.get("cases", []):
        prediction = predictions_by_case.get(str(case["case_id"]))
        case_metrics.append(
            _evaluate_replay_case(
                case=case,
                prediction=prediction,
                collision_distance_m=collision_distance_m,
            )
        )

    summary = _build_nuplan_evaluation_summary(
        profile_name=profile,
        benchmark_path=Path(benchmark_path),
        predictions_path=Path(predictions_path),
        case_metrics=case_metrics,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_nuplan_evaluation_outputs(summary, output_dir)
    return summary


def compare_nuplan_replay_evaluations(
    evaluation_dirs: Sequence[Path],
    output_dir: Path,
) -> Dict[str, Any]:
    summaries: List[Dict[str, Any]] = []
    for eval_dir in evaluation_dirs:
        metrics_path = Path(eval_dir) / "nuplan_replay_metrics.json"
        if not metrics_path.exists():
            raise FileNotFoundError(f"Missing nuPlan replay metrics: {metrics_path}")
        summary = _read_json(metrics_path)
        summary["metadata"]["evaluation_dir"] = str(eval_dir)
        summaries.append(summary)

    profiles = [_profile_row_from_summary(summary) for summary in summaries]
    case_ids = sorted(
        {
            str(row["case_id"])
            for summary in summaries
            for row in summary.get("case_metrics", [])
        }
    )
    comparison = {
        "metadata": {
            "schema": "nuplan_replay_comparison_v1",
            "evaluation_dirs": [str(path) for path in evaluation_dirs],
        },
        "overview": {
            "profile_count": len(profiles),
            "case_count": len(case_ids),
        },
        "profiles": profiles,
        "scenario_family_matrix": _comparison_matrix(summaries, "scenario_family"),
        "scenario_tag_matrix": _comparison_matrix(summaries, "scenario_tag"),
        "difficulty_matrix": _comparison_matrix(summaries, "difficulty_label"),
        "case_matrix": _case_comparison_matrix(summaries),
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_nuplan_comparison_outputs(comparison, output_dir)
    return comparison


def render_nuplan_replay_case_studies(
    benchmark_path: Path,
    evaluation_dirs: Sequence[Path],
    output_dir: Path,
    max_cases: int = 4,
) -> Dict[str, Any]:
    benchmark = _read_json(Path(benchmark_path))
    evaluation_payloads = _load_nuplan_case_study_evaluations(evaluation_dirs)
    case_rows = _build_nuplan_case_study_rows(benchmark, evaluation_payloads)
    selected = _select_nuplan_case_studies(case_rows, max_cases=max_cases)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / "nuplan_replay_case_studies.png"
    summary_rows = _render_nuplan_case_study_figure(selected, figure_path)

    metadata = {
        "schema": "nuplan_replay_case_studies_v1",
        "benchmark_path": str(benchmark_path),
        "evaluation_dirs": [str(Path(path)) for path in evaluation_dirs],
        "case_count": len(summary_rows),
        "figure_path": str(figure_path),
        "cases": summary_rows,
    }
    (output_dir / "nuplan_replay_case_studies.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    markdown = _render_nuplan_case_studies_markdown(metadata)
    (output_dir / "nuplan_replay_case_studies.md").write_text(markdown, encoding="utf-8")
    (output_dir / "nuplan_replay_case_studies.html").write_text(_markdown_to_basic_html(markdown), encoding="utf-8")
    return {
        "output_dir": str(output_dir),
        "case_count": len(summary_rows),
        "figure_path": str(figure_path),
    }


def run_nuplan_replay_study(
    split_dir: Path = DEFAULT_NUPLAN_SPLIT,
    output_dir: Path = Path("outputs/nuplan_replay_study_v1"),
    *,
    max_dbs: int = 4,
    max_cases: int = 16,
    max_cases_per_db: int = 4,
    history_s: float = 2.0,
    future_s: float = 4.0,
    frame_hz: float = 2.0,
    min_anchor_gap_s: float = 4.0,
    scenario_tags: Optional[Sequence[str]] = None,
    profiles: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    benchmark_path = output_dir / "nuplan_replay_benchmark.json"
    benchmark_metadata = generate_nuplan_replay_benchmark(
        split_dir=Path(split_dir),
        output_path=benchmark_path,
        max_dbs=max_dbs,
        max_cases=max_cases,
        max_cases_per_db=max_cases_per_db,
        history_s=history_s,
        future_s=future_s,
        frame_hz=frame_hz,
        min_anchor_gap_s=min_anchor_gap_s,
        scenario_tags=scenario_tags,
    )

    profile_names = list(profiles or NUPLAN_REPLAY_PROFILES)
    evaluations: List[Dict[str, Any]] = []
    for profile_name in profile_names:
        predictions_path = output_dir / f"{profile_name}_rollouts.json"
        eval_dir = output_dir / f"{profile_name}_evaluation"
        generate_nuplan_proxy_rollouts(benchmark_path, predictions_path, profile_name)
        summary = evaluate_nuplan_rollouts(
            benchmark_path=benchmark_path,
            predictions_path=predictions_path,
            output_dir=eval_dir,
            profile_name=profile_name,
        )
        evaluations.append(
            {
                "profile_name": profile_name,
                "predictions_path": str(predictions_path),
                "evaluation_dir": str(eval_dir),
                "overview": summary["overview"],
            }
        )

    comparison = compare_nuplan_replay_evaluations(
        evaluation_dirs=[Path(item["evaluation_dir"]) for item in evaluations],
        output_dir=output_dir / "comparison",
    )
    case_studies = render_nuplan_replay_case_studies(
        benchmark_path=benchmark_path,
        evaluation_dirs=[Path(item["evaluation_dir"]) for item in evaluations],
        output_dir=output_dir / "case_studies",
        max_cases=min(4, max_cases),
    )
    benchmark_payload = _read_json(benchmark_path)
    unified_cases_path = output_dir / "unified_risk_cases.json"
    unified_cases_payload = write_unified_case_collection(
        unified_cases_from_nuplan_benchmark(benchmark_payload),
        unified_cases_path,
        metadata={
            "dataset": "nuplan",
            "benchmark_layer": "replay_regression",
            "benchmark_path": str(benchmark_path),
        },
    )

    manifest = {
        "benchmark": {
            "path": str(benchmark_path),
            "metadata": benchmark_metadata,
        },
        "evaluations": evaluations,
        "comparison": {
            "output_dir": str(output_dir / "comparison"),
            "overview": comparison["overview"],
        },
        "case_studies": case_studies,
        "unified_cases": {
            "path": str(unified_cases_path),
            "case_count": len(unified_cases_payload.get("cases", [])),
            "schema": unified_cases_payload.get("schema", ""),
        },
    }
    (output_dir / "nuplan_replay_study_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    _write_nuplan_study_summary(manifest, output_dir)
    artifact_manifest = _write_nuplan_study_artifact_manifest(output_dir, evaluations)
    manifest["artifact_manifest"] = {
        "path": str(output_dir / "artifact_manifest.json"),
        "overview": artifact_manifest.get("overview", {}),
    }
    (output_dir / "nuplan_replay_study_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return manifest


def _discover_db_paths(split_dir: Path, max_dbs: int = 0) -> List[Path]:
    db_paths = sorted({path.resolve() for path in Path(split_dir).rglob("*.db")})
    if max_dbs and max_dbs > 0:
        db_paths = db_paths[:max_dbs]
    return db_paths


def _count_db_files(path: Path) -> int:
    return len(list(Path(path).rglob("*.db")))


def _select_diverse_cases(cases: Sequence[Dict[str, Any]], max_cases: int) -> List[Dict[str, Any]]:
    if max_cases <= 0:
        return []
    ordered = sorted(
        cases,
        key=lambda case: (
            str(case.get("scenario_family", "")),
            str(case.get("scenario_tag", "")),
            str(case.get("location", "")),
            -float(case.get("risk_targets", {}).get("risk_severity_score") or 0.0),
            str(case.get("case_id", "")),
        ),
    )
    buckets: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for case in ordered:
        buckets[
            (
                str(case.get("scenario_family", "unknown")),
                str(case.get("scenario_tag", "unknown")),
                str(case.get("location", "unknown")),
            )
        ].append(case)

    selected: List[Dict[str, Any]] = []
    bucket_keys = sorted(
        buckets,
        key=lambda key: (
            key[0],
            key[1],
            key[2],
        ),
    )
    while len(selected) < max_cases:
        added = False
        for key in bucket_keys:
            if buckets[key]:
                selected.append(buckets[key].pop(0))
                added = True
                if len(selected) >= max_cases:
                    break
        if not added:
            break
    return selected


def _build_sampling_summary(cases: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by_family = Counter(str(case.get("scenario_family", "unknown")) for case in cases)
    by_tag = Counter(str(case.get("scenario_tag", "unknown")) for case in cases)
    by_location = Counter(str(case.get("location", "unknown")) for case in cases)
    by_difficulty = Counter(str(case.get("difficulty_label", "unknown")) for case in cases)
    return {
        "by_family": dict(sorted(by_family.items())),
        "by_tag": dict(sorted(by_tag.items())),
        "by_location": dict(sorted(by_location.items())),
        "by_difficulty": dict(sorted(by_difficulty.items())),
    }


def _scenario_taxonomy_payload(tags: Sequence[str]) -> Dict[str, Any]:
    family_to_tags: Dict[str, List[str]] = defaultdict(list)
    for tag in tags:
        family_to_tags[_scenario_family(tag, "")].append(tag)
    return {
        family: {
            "description": SCENARIO_FAMILY_DESCRIPTIONS.get(family, ""),
            "scenario_tags": sorted(tag_values),
        }
        for family, tag_values in sorted(family_to_tags.items())
    }


def _select_anchors(
    db_path: Path,
    scenario_tags: Sequence[str],
    max_cases: int,
    min_anchor_gap_s: float,
) -> List[NuPlanAnchor]:
    if max_cases <= 0:
        return []
    tag_order = {tag: idx for idx, tag in enumerate(scenario_tags)}
    placeholders = ",".join("?" for _ in scenario_tags)
    query = f"""
        SELECT
            st.type,
            lp.timestamp,
            lp.token,
            lp.scene_token,
            COALESCE(s.name, ''),
            st.agent_track_token,
            c.name,
            ep.x,
            ep.y,
            ep.qw,
            ep.qx,
            ep.qy,
            ep.qz,
            ep.vx,
            ep.vy,
            lb.x,
            lb.y,
            lb.vx,
            lb.vy
        FROM scenario_tag st
        JOIN lidar_pc lp ON lp.token = st.lidar_pc_token
        JOIN ego_pose ep ON ep.token = lp.ego_pose_token
        JOIN lidar_box lb ON lb.lidar_pc_token = st.lidar_pc_token
            AND lb.track_token = st.agent_track_token
        JOIN track tr ON tr.token = lb.track_token
        JOIN category c ON c.token = tr.category_token
        LEFT JOIN scene s ON s.token = lp.scene_token
        WHERE st.agent_track_token IS NOT NULL
            AND st.type IN ({placeholders})
        ORDER BY st.type, lp.timestamp
    """
    with sqlite3.connect(str(db_path)) as conn:
        log_record = _read_log_record(conn)
        rows = conn.execute(query, list(scenario_tags)).fetchall()

    rows = sorted(rows, key=lambda row: (tag_order.get(str(row[0]), len(tag_order)), int(row[1])))
    anchors: List[NuPlanAnchor] = []
    last_selected: Dict[Tuple[str, bytes], int] = {}
    min_gap_us = int(min_anchor_gap_s * 1_000_000)
    for row in rows:
        scenario_tag = str(row[0])
        timestamp_us = int(row[1])
        track_token = bytes(row[5])
        key = (scenario_tag, track_token)
        if key in last_selected and timestamp_us - last_selected[key] < min_gap_us:
            continue
        last_selected[key] = timestamp_us
        anchors.append(
            NuPlanAnchor(
                db_path=db_path,
                log_name=log_record["log_name"],
                vehicle_name=log_record["vehicle_name"],
                location=log_record["location"],
                map_version=log_record["map_version"],
                scene_name=str(row[4]) or "unknown_scene",
                scenario_tag=scenario_tag,
                category_name=str(row[6]),
                timestamp_us=timestamp_us,
                lidar_pc_token=bytes(row[2]),
                scene_token=bytes(row[3]),
                track_token=track_token,
                ego_x=float(row[7]),
                ego_y=float(row[8]),
                ego_yaw=_yaw_from_quaternion(float(row[9]), float(row[10]), float(row[11]), float(row[12])),
                ego_vx=float(row[13] or 0.0),
                ego_vy=float(row[14] or 0.0),
                actor_x=float(row[15]),
                actor_y=float(row[16]),
                actor_vx=float(row[17] or 0.0),
                actor_vy=float(row[18] or 0.0),
            )
        )
        if len(anchors) >= max_cases:
            break
    return anchors


def _read_log_record(conn: sqlite3.Connection) -> Dict[str, str]:
    row = conn.execute(
        "SELECT vehicle_name, logfile, location, map_version FROM log LIMIT 1"
    ).fetchone()
    if not row:
        return {"vehicle_name": "", "log_name": "", "location": "", "map_version": ""}
    return {
        "vehicle_name": str(row[0] or ""),
        "log_name": str(row[1] or ""),
        "location": str(row[2] or ""),
        "map_version": str(row[3] or ""),
    }


def _build_replay_case(anchor: NuPlanAnchor, history_s: float, future_s: float, frame_hz: float) -> Optional[Dict[str, Any]]:
    start_us = anchor.timestamp_us - int(history_s * 1_000_000)
    end_us = anchor.timestamp_us + int(future_s * 1_000_000)
    min_step_us = int(1_000_000 / frame_hz) if frame_hz > 0 else 0
    with sqlite3.connect(str(anchor.db_path)) as conn:
        frame_rows = conn.execute(
            """
            SELECT
                lp.token,
                lp.timestamp,
                ep.x,
                ep.y,
                ep.z,
                ep.qw,
                ep.qx,
                ep.qy,
                ep.qz,
                ep.vx,
                ep.vy,
                ep.acceleration_x,
                ep.acceleration_y,
                ep.angular_rate_z
            FROM lidar_pc lp
            JOIN ego_pose ep ON ep.token = lp.ego_pose_token
            WHERE lp.scene_token = ?
                AND lp.timestamp BETWEEN ? AND ?
            ORDER BY lp.timestamp
            """,
            (anchor.scene_token, start_us, end_us),
        ).fetchall()
        selected_frame_rows = _downsample_frame_rows(frame_rows, anchor.timestamp_us, min_step_us)
        if not any(int(row[1]) >= anchor.timestamp_us for row in selected_frame_rows):
            return None
        lidar_tokens = [bytes(row[0]) for row in selected_frame_rows]
        actor_by_lidar = _read_primary_actor_states(conn, lidar_tokens, anchor.track_token)
        tags_by_lidar = _read_tags_by_lidar(conn, lidar_tokens)
        traffic_by_lidar = _read_traffic_lights_by_lidar(conn, lidar_tokens)

    frames: List[Dict[str, Any]] = []
    anchor_frame: Optional[Dict[str, Any]] = None
    for row in selected_frame_rows:
        lidar_token = bytes(row[0])
        timestamp_us = int(row[1])
        ego_yaw = _yaw_from_quaternion(float(row[5]), float(row[6]), float(row[7]), float(row[8]))
        ego = {
            "x": float(row[2]),
            "y": float(row[3]),
            "z": float(row[4] or 0.0),
            "yaw": ego_yaw,
            "vx": float(row[9] or 0.0),
            "vy": float(row[10] or 0.0),
            "speed_mps": math.hypot(float(row[9] or 0.0), float(row[10] or 0.0)),
            "acceleration_x": float(row[11] or 0.0),
            "acceleration_y": float(row[12] or 0.0),
            "angular_rate_z": float(row[13] or 0.0),
        }
        actor = actor_by_lidar.get(lidar_token)
        primary_actor = _format_actor_state(actor, ego) if actor else None
        frame = {
            "timestamp_us": timestamp_us,
            "dt_from_anchor_s": (timestamp_us - anchor.timestamp_us) / 1_000_000.0,
            "lidar_pc_token": _hex(lidar_token),
            "ego": ego,
            "primary_actor": primary_actor,
            "scenario_tags": tags_by_lidar.get(lidar_token, []),
            "traffic_light_status": traffic_by_lidar.get(lidar_token, {}),
        }
        frames.append(frame)
        if lidar_token == anchor.lidar_pc_token or (
            anchor_frame is None
            and timestamp_us >= anchor.timestamp_us
        ):
            anchor_frame = frame

    if anchor_frame is None:
        return None
    future_frames = [frame for frame in frames if int(frame["timestamp_us"]) >= anchor.timestamp_us]
    if not future_frames:
        return None

    risk_targets = _compute_risk_targets(future_frames)
    comfort_targets = _compute_comfort_targets(future_frames)
    scenario_family = _scenario_family(anchor.scenario_tag, anchor.category_name)
    risk_facets = _build_case_risk_facets(
        scenario_family=scenario_family,
        category_name=anchor.category_name,
        risk_targets=risk_targets,
        comfort_targets=comfort_targets,
        anchor_frame=anchor_frame,
    )
    case_id = _case_id(anchor)
    return {
        "case_id": case_id,
        "dataset": "nuplan",
        "source_db": str(anchor.db_path),
        "log_name": anchor.log_name,
        "vehicle_name": anchor.vehicle_name,
        "location": anchor.location,
        "map_version": anchor.map_version,
        "scene_name": anchor.scene_name,
        "scenario_tag": anchor.scenario_tag,
        "scenario_family": scenario_family,
        "scenario_description": SCENARIO_FAMILY_DESCRIPTIONS.get(scenario_family, ""),
        "category_name": anchor.category_name,
        "difficulty_label": _difficulty_label(risk_targets, comfort_targets),
        "risk_facets": risk_facets,
        "anchor_timestamp_us": anchor.timestamp_us,
        "anchor_lidar_pc_token": _hex(anchor.lidar_pc_token),
        "anchor_track_token": _hex(anchor.track_token),
        "history_frame_count": len([frame for frame in frames if int(frame["timestamp_us"]) < anchor.timestamp_us]),
        "future_frame_count": len(future_frames),
        "anchor_frame": anchor_frame,
        "frames": frames,
        "future_frames": future_frames,
        "risk_targets": risk_targets,
        "comfort_targets": comfort_targets,
    }


def _downsample_frame_rows(rows: Sequence[Tuple[Any, ...]], anchor_timestamp_us: int, min_step_us: int) -> List[Tuple[Any, ...]]:
    if not rows:
        return []
    if min_step_us <= 0:
        return list(rows)
    selected: List[Tuple[Any, ...]] = []
    last_ts: Optional[int] = None
    anchor_row = min(rows, key=lambda row: abs(int(row[1]) - anchor_timestamp_us))
    for row in rows:
        timestamp_us = int(row[1])
        if row == anchor_row or last_ts is None or timestamp_us - last_ts >= min_step_us:
            selected.append(row)
            last_ts = timestamp_us
    if anchor_row not in selected:
        selected.append(anchor_row)
        selected.sort(key=lambda row: int(row[1]))
    return selected


def _read_primary_actor_states(
    conn: sqlite3.Connection,
    lidar_tokens: Sequence[bytes],
    track_token: bytes,
) -> Dict[bytes, Dict[str, Any]]:
    if not lidar_tokens:
        return {}
    placeholders = ",".join("?" for _ in lidar_tokens)
    rows = conn.execute(
        f"""
        SELECT
            lb.lidar_pc_token,
            c.name,
            lb.x,
            lb.y,
            lb.z,
            lb.width,
            lb.length,
            lb.height,
            lb.vx,
            lb.vy,
            lb.vz,
            lb.yaw
        FROM lidar_box lb
        JOIN track tr ON tr.token = lb.track_token
        JOIN category c ON c.token = tr.category_token
        WHERE lb.track_token = ?
            AND lb.lidar_pc_token IN ({placeholders})
        """,
        [track_token, *lidar_tokens],
    ).fetchall()
    result: Dict[bytes, Dict[str, Any]] = {}
    for row in rows:
        result[bytes(row[0])] = {
            "category_name": str(row[1]),
            "x": float(row[2]),
            "y": float(row[3]),
            "z": float(row[4] or 0.0),
            "width": float(row[5] or 0.0),
            "length": float(row[6] or 0.0),
            "height": float(row[7] or 0.0),
            "vx": float(row[8] or 0.0),
            "vy": float(row[9] or 0.0),
            "vz": float(row[10] or 0.0),
            "yaw": float(row[11] or 0.0),
        }
    return result


def _read_tags_by_lidar(conn: sqlite3.Connection, lidar_tokens: Sequence[bytes]) -> Dict[bytes, List[str]]:
    if not lidar_tokens:
        return {}
    placeholders = ",".join("?" for _ in lidar_tokens)
    rows = conn.execute(
        f"""
        SELECT lidar_pc_token, type
        FROM scenario_tag
        WHERE lidar_pc_token IN ({placeholders})
        ORDER BY type
        """,
        list(lidar_tokens),
    ).fetchall()
    result: Dict[bytes, List[str]] = defaultdict(list)
    for token, tag in rows:
        result[bytes(token)].append(str(tag))
    return dict(result)


def _read_traffic_lights_by_lidar(conn: sqlite3.Connection, lidar_tokens: Sequence[bytes]) -> Dict[bytes, Dict[str, int]]:
    if not lidar_tokens:
        return {}
    placeholders = ",".join("?" for _ in lidar_tokens)
    rows = conn.execute(
        f"""
        SELECT lidar_pc_token, status, COUNT(*)
        FROM traffic_light_status
        WHERE lidar_pc_token IN ({placeholders})
        GROUP BY lidar_pc_token, status
        """,
        list(lidar_tokens),
    ).fetchall()
    result: Dict[bytes, Dict[str, int]] = defaultdict(dict)
    for token, status, count in rows:
        result[bytes(token)][str(status)] = int(count)
    return dict(result)


def _format_actor_state(actor: Dict[str, Any], ego: Dict[str, float]) -> Dict[str, Any]:
    x_ego, y_ego = _global_to_ego(
        x=float(actor["x"]),
        y=float(actor["y"]),
        ego_x=float(ego["x"]),
        ego_y=float(ego["y"]),
        ego_yaw=float(ego["yaw"]),
    )
    distance = math.hypot(float(actor["x"]) - float(ego["x"]), float(actor["y"]) - float(ego["y"]))
    rel_vx = float(actor["vx"]) - float(ego["vx"])
    rel_vy = float(actor["vy"]) - float(ego["vy"])
    ttc = _linear_ttc(
        rel_x=float(actor["x"]) - float(ego["x"]),
        rel_y=float(actor["y"]) - float(ego["y"]),
        rel_vx=rel_vx,
        rel_vy=rel_vy,
    )
    return {
        **actor,
        "x_ego": x_ego,
        "y_ego": y_ego,
        "distance_m": distance,
        "relative_speed_mps": math.hypot(rel_vx, rel_vy),
        "ttc_s": ttc,
    }


def _compute_risk_targets(future_frames: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    actor_frames = [frame for frame in future_frames if frame.get("primary_actor")]
    red_light_count = sum(1 for frame in future_frames if int(frame.get("traffic_light_status", {}).get("red", 0)) > 0)
    if not actor_frames:
        return {
            "actor_visible_frame_count": 0,
            "min_distance_m": None,
            "min_distance_timestamp_us": None,
            "min_ttc_s": None,
            "collision_proxy": False,
            "red_light_frame_count": red_light_count,
            "risk_severity_score": 0.0,
            "near_miss_severity": "unknown",
        }
    distances = [float(frame["primary_actor"]["distance_m"]) for frame in actor_frames]
    ttcs = [
        float(frame["primary_actor"]["ttc_s"])
        for frame in actor_frames
        if frame["primary_actor"].get("ttc_s") is not None
    ]
    min_distance_frame = min(actor_frames, key=lambda frame: float(frame["primary_actor"]["distance_m"]))
    min_ttc = min(ttcs) if ttcs else None
    severity_score = _risk_severity_score(min(distances), min_ttc, red_light_count)
    return {
        "actor_visible_frame_count": len(actor_frames),
        "min_distance_m": min(distances),
        "min_distance_timestamp_us": int(min_distance_frame["timestamp_us"]),
        "min_ttc_s": min_ttc,
        "collision_proxy": min(distances) < 2.0,
        "red_light_frame_count": red_light_count,
        "risk_severity_score": severity_score,
        "near_miss_severity": _near_miss_severity(severity_score),
    }


def _compute_comfort_targets(future_frames: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    states = [_logged_ego_state_from_frame(frame) for frame in future_frames]
    comfort = _compute_comfort_metrics(states)
    comfort["comfort_violation"] = _comfort_violation(comfort)
    return comfort


def _logged_ego_state_from_frame(frame: Dict[str, Any]) -> Dict[str, Any]:
    ego = frame["ego"]
    return {
        "timestamp_us": int(frame["timestamp_us"]),
        "x": float(ego["x"]),
        "y": float(ego["y"]),
        "yaw": float(ego["yaw"]),
        "vx": float(ego.get("vx", 0.0)),
        "vy": float(ego.get("vy", 0.0)),
        "speed_mps": float(ego.get("speed_mps", math.hypot(float(ego.get("vx", 0.0)), float(ego.get("vy", 0.0))))),
        "acceleration_x": float(ego.get("acceleration_x", 0.0)),
        "acceleration_y": float(ego.get("acceleration_y", 0.0)),
        "angular_rate_z": float(ego.get("angular_rate_z", 0.0)),
    }


def _build_case_risk_facets(
    scenario_family: str,
    category_name: str,
    risk_targets: Dict[str, Any],
    comfort_targets: Dict[str, Any],
    anchor_frame: Dict[str, Any],
) -> Dict[str, Any]:
    anchor_speed = float(anchor_frame.get("ego", {}).get("speed_mps", 0.0))
    return {
        "scenario_family": scenario_family,
        "actor_category": category_name,
        "distance_band": _distance_band(risk_targets.get("min_distance_m")),
        "ttc_band": _ttc_band(risk_targets.get("min_ttc_s")),
        "red_light_context": bool(int(risk_targets.get("red_light_frame_count") or 0) > 0),
        "comfort_band": "violating" if bool(comfort_targets.get("comfort_violation")) else "nominal",
        "ego_speed_band": _speed_band(anchor_speed),
        "near_miss_severity": risk_targets.get("near_miss_severity", "unknown"),
    }


def _difficulty_label(risk_targets: Dict[str, Any], comfort_targets: Dict[str, Any]) -> str:
    score = float(risk_targets.get("risk_severity_score") or 0.0)
    if bool(comfort_targets.get("comfort_violation")):
        score = max(score, 0.55)
    if score >= 0.75:
        return "hard"
    if score >= 0.45:
        return "medium"
    return "easy"


def _complete_predicted_states(states: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    completed = [dict(state) for state in states]
    completed.sort(key=lambda state: int(state.get("timestamp_us", 0)))
    for idx, state in enumerate(completed):
        state["timestamp_us"] = int(state.get("timestamp_us", 0))
        state["x"] = float(state.get("x", 0.0))
        state["y"] = float(state.get("y", 0.0))
        state["yaw"] = float(state.get("yaw", 0.0))
        if "vx" not in state or "vy" not in state:
            if idx == 0 and len(completed) > 1:
                dt = max((int(completed[1].get("timestamp_us", 0)) - state["timestamp_us"]) / 1_000_000.0, 1e-6)
                state["vx"] = (float(completed[1].get("x", 0.0)) - state["x"]) / dt
                state["vy"] = (float(completed[1].get("y", 0.0)) - state["y"]) / dt
            elif idx > 0:
                prev = completed[idx - 1]
                dt = max((state["timestamp_us"] - int(prev.get("timestamp_us", 0))) / 1_000_000.0, 1e-6)
                state["vx"] = (state["x"] - float(prev.get("x", 0.0))) / dt
                state["vy"] = (state["y"] - float(prev.get("y", 0.0))) / dt
            else:
                state["vx"] = 0.0
                state["vy"] = 0.0
        state["vx"] = float(state.get("vx", 0.0))
        state["vy"] = float(state.get("vy", 0.0))
        state["speed_mps"] = float(state.get("speed_mps", math.hypot(state["vx"], state["vy"])))
    return completed


def _compute_comfort_metrics(states: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    completed = _complete_predicted_states(states)
    if not completed:
        return {
            "max_acceleration_mps2": None,
            "max_jerk_mps3": None,
            "max_yaw_rate_rps": None,
        }

    acceleration_norms: List[float] = []
    yaw_rates: List[float] = []
    for state in completed:
        if "acceleration_x" in state or "acceleration_y" in state:
            acceleration_norms.append(
                math.hypot(float(state.get("acceleration_x", 0.0)), float(state.get("acceleration_y", 0.0)))
            )
        if "angular_rate_z" in state:
            yaw_rates.append(abs(float(state.get("angular_rate_z", 0.0))))

    for prev, curr in zip(completed, completed[1:]):
        dt = max((int(curr["timestamp_us"]) - int(prev["timestamp_us"])) / 1_000_000.0, 1e-6)
        acceleration_norms.append(
            math.hypot(float(curr["vx"]) - float(prev["vx"]), float(curr["vy"]) - float(prev["vy"])) / dt
        )
        yaw_rates.append(abs(_angle_diff(float(curr["yaw"]), float(prev["yaw"]))) / dt)

    jerk_values: List[float] = []
    for idx in range(1, len(acceleration_norms)):
        if idx < len(completed):
            dt = max(
                (int(completed[idx]["timestamp_us"]) - int(completed[idx - 1]["timestamp_us"])) / 1_000_000.0,
                1e-6,
            )
        else:
            dt = 1.0
        jerk_values.append(abs(acceleration_norms[idx] - acceleration_norms[idx - 1]) / dt)

    return {
        "max_acceleration_mps2": max(acceleration_norms) if acceleration_norms else 0.0,
        "max_jerk_mps3": max(jerk_values) if jerk_values else 0.0,
        "max_yaw_rate_rps": max(yaw_rates) if yaw_rates else 0.0,
    }


def _comfort_violation(comfort: Dict[str, Any]) -> bool:
    return any(
        float(comfort.get(key) or 0.0) > threshold
        for key, threshold in COMFORT_THRESHOLDS.items()
    )


def _risk_severity_score(min_distance_m: Optional[float], min_ttc_s: Optional[float], red_light_count: int) -> float:
    distance_score = 0.0
    if min_distance_m is not None:
        distance = float(min_distance_m)
        if distance <= 2.0:
            distance_score = 1.0
        elif distance <= 5.0:
            distance_score = 0.75
        elif distance <= 10.0:
            distance_score = 0.45
        else:
            distance_score = 0.20

    ttc_score = 0.0
    if min_ttc_s is not None:
        ttc = float(min_ttc_s)
        if ttc <= 1.5:
            ttc_score = 1.0
        elif ttc <= 3.0:
            ttc_score = 0.70
        elif ttc <= 5.0:
            ttc_score = 0.40
        else:
            ttc_score = 0.20

    light_score = 0.20 if red_light_count > 0 else 0.0
    return min(1.0, 0.55 * distance_score + 0.35 * ttc_score + light_score)


def _near_miss_severity(score: float) -> str:
    if score >= 0.75:
        return "critical"
    if score >= 0.45:
        return "moderate"
    if score > 0.0:
        return "low"
    return "unknown"


def _distance_band(value: Any) -> str:
    if value is None:
        return "unknown"
    distance = float(value)
    if distance <= 2.0:
        return "collision_proxy"
    if distance <= 5.0:
        return "critical_range"
    if distance <= 10.0:
        return "near_range"
    return "far_range"


def _ttc_band(value: Any) -> str:
    if value is None:
        return "unknown"
    ttc = float(value)
    if ttc <= 1.5:
        return "critical_ttc"
    if ttc <= 3.0:
        return "urgent_ttc"
    if ttc <= 5.0:
        return "moderate_ttc"
    return "long_ttc"


def _speed_band(speed_mps: float) -> str:
    if speed_mps >= 12.0:
        return "high_speed"
    if speed_mps >= 5.0:
        return "urban_speed"
    return "low_speed"


def _abs_optional(predicted: Any, logged: Any) -> Optional[float]:
    if predicted is None or logged is None:
        return None
    return abs(float(predicted) - float(logged))


def _ttc_similarity(predicted: Any, logged: Any) -> float:
    if predicted is None and logged is None:
        return 1.0
    if predicted is None or logged is None:
        return 0.5
    return max(0.0, 1.0 - abs(float(predicted) - float(logged)) / 5.0)


def _comfort_similarity(acc_error: Any, jerk_error: Any, yaw_rate_error: Any) -> float:
    scores = []
    if acc_error is not None:
        scores.append(max(0.0, 1.0 - float(acc_error) / 4.0))
    if jerk_error is not None:
        scores.append(max(0.0, 1.0 - float(jerk_error) / 5.0))
    if yaw_rate_error is not None:
        scores.append(max(0.0, 1.0 - float(yaw_rate_error) / 1.0))
    return mean(scores) if scores else 1.0


def _evaluate_replay_case(
    case: Dict[str, Any],
    prediction: Optional[Dict[str, Any]],
    collision_distance_m: float,
) -> Dict[str, Any]:
    future_frames = list(case.get("future_frames", []))
    gt_by_timestamp = {int(frame["timestamp_us"]): frame for frame in future_frames}
    predicted_states = _complete_predicted_states(list(prediction.get("future_ego_states", []))) if prediction else []
    aligned: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for state in predicted_states:
        timestamp_us = int(state.get("timestamp_us", 0))
        frame = gt_by_timestamp.get(timestamp_us)
        if frame is not None:
            aligned.append((frame, state))

    if not prediction:
        failure_tags = ["missing_prediction"]
    elif len(aligned) < len(future_frames):
        failure_tags = ["partial_horizon"]
    else:
        failure_tags = []

    if not aligned:
        return {
            "case_id": case["case_id"],
            "scenario_tag": case["scenario_tag"],
            "scenario_family": case["scenario_family"],
            "difficulty_label": case.get("difficulty_label", ""),
            "location": case["location"],
            "scene_name": case["scene_name"],
            "full_horizon": False,
            "horizon_recall": 0.0,
            "ego_ade_m": None,
            "ego_fde_m": None,
            "logged_min_distance_m": case["risk_targets"].get("min_distance_m"),
            "predicted_min_distance_m": None,
            "min_distance_error_m": None,
            "logged_min_ttc_s": case["risk_targets"].get("min_ttc_s"),
            "predicted_min_ttc_s": None,
            "min_ttc_error_s": None,
            "red_light_context_recall": 0.0 if int(case["risk_targets"].get("red_light_frame_count") or 0) else 1.0,
            "logged_max_acceleration_mps2": case.get("comfort_targets", {}).get("max_acceleration_mps2"),
            "predicted_max_acceleration_mps2": None,
            "max_acceleration_error_mps2": None,
            "logged_max_jerk_mps3": case.get("comfort_targets", {}).get("max_jerk_mps3"),
            "predicted_max_jerk_mps3": None,
            "max_jerk_error_mps3": None,
            "logged_max_yaw_rate_rps": case.get("comfort_targets", {}).get("max_yaw_rate_rps"),
            "predicted_max_yaw_rate_rps": None,
            "max_yaw_rate_error_rps": None,
            "comfort_violation_mismatch": None,
            "logged_collision_proxy": bool(case["risk_targets"].get("collision_proxy")),
            "predicted_collision_proxy": None,
            "risk_fidelity_score": 0.0,
            "failure_tags": failure_tags,
        }

    ego_errors: List[float] = []
    predicted_distances: List[float] = []
    predicted_ttcs: List[float] = []
    for frame, state in aligned:
        gt_ego = frame["ego"]
        error = math.hypot(float(state["x"]) - float(gt_ego["x"]), float(state["y"]) - float(gt_ego["y"]))
        ego_errors.append(error)
        actor = frame.get("primary_actor")
        if actor:
            predicted_distances.append(
                math.hypot(float(state["x"]) - float(actor["x"]), float(state["y"]) - float(actor["y"]))
            )
            ttc = _linear_ttc(
                rel_x=float(actor["x"]) - float(state["x"]),
                rel_y=float(actor["y"]) - float(state["y"]),
                rel_vx=float(actor.get("vx", 0.0)) - float(state.get("vx", 0.0)),
                rel_vy=float(actor.get("vy", 0.0)) - float(state.get("vy", 0.0)),
            )
            if ttc is not None:
                predicted_ttcs.append(ttc)

    logged_min_distance = case["risk_targets"].get("min_distance_m")
    predicted_min_distance = min(predicted_distances) if predicted_distances else None
    min_distance_error = (
        abs(float(predicted_min_distance) - float(logged_min_distance))
        if predicted_min_distance is not None and logged_min_distance is not None
        else None
    )
    logged_min_ttc = case["risk_targets"].get("min_ttc_s")
    predicted_min_ttc = min(predicted_ttcs) if predicted_ttcs else None
    min_ttc_error = (
        abs(float(predicted_min_ttc) - float(logged_min_ttc))
        if predicted_min_ttc is not None and logged_min_ttc is not None
        else None
    )
    logged_red_light_count = int(case["risk_targets"].get("red_light_frame_count") or 0)
    predicted_red_light_count = sum(
        1 for frame, _ in aligned if int(frame.get("traffic_light_status", {}).get("red", 0)) > 0
    )
    red_light_context_recall = (
        _safe_ratio(predicted_red_light_count, logged_red_light_count) if logged_red_light_count else 1.0
    )
    predicted_comfort = _compute_comfort_metrics(predicted_states)
    predicted_comfort["comfort_violation"] = _comfort_violation(predicted_comfort)
    logged_comfort = dict(case.get("comfort_targets", {}))
    max_acceleration_error = _abs_optional(
        predicted_comfort.get("max_acceleration_mps2"),
        logged_comfort.get("max_acceleration_mps2"),
    )
    max_jerk_error = _abs_optional(
        predicted_comfort.get("max_jerk_mps3"),
        logged_comfort.get("max_jerk_mps3"),
    )
    max_yaw_rate_error = _abs_optional(
        predicted_comfort.get("max_yaw_rate_rps"),
        logged_comfort.get("max_yaw_rate_rps"),
    )
    comfort_violation_mismatch = (
        bool(predicted_comfort.get("comfort_violation")) != bool(logged_comfort.get("comfort_violation"))
    )
    logged_collision = bool(case["risk_targets"].get("collision_proxy"))
    predicted_collision = (
        bool(predicted_min_distance is not None and predicted_min_distance < collision_distance_m)
        if predicted_min_distance is not None
        else None
    )
    if predicted_collision is not None and predicted_collision != logged_collision:
        failure_tags.append("collision_proxy_mismatch")
    if min_distance_error is not None and min_distance_error > 3.0:
        failure_tags.append("risk_distance_error")
    if min_ttc_error is not None and min_ttc_error > 2.0:
        failure_tags.append("ttc_error")
    if red_light_context_recall < 1.0:
        failure_tags.append("red_light_context_miss")
    if comfort_violation_mismatch:
        failure_tags.append("comfort_violation_mismatch")

    full_horizon = len(aligned) == len(future_frames)
    horizon_recall = len(aligned) / max(len(future_frames), 1)
    risk_distance_score = 0.0 if min_distance_error is None else max(0.0, 1.0 - min_distance_error / 10.0)
    collision_score = 1.0 if predicted_collision == logged_collision else 0.0
    ttc_score = _ttc_similarity(predicted_min_ttc, logged_min_ttc)
    comfort_score = _comfort_similarity(max_acceleration_error, max_jerk_error, max_yaw_rate_error)
    risk_fidelity = (
        0.35 * risk_distance_score
        + 0.20 * collision_score
        + 0.15 * ttc_score
        + 0.10 * red_light_context_recall
        + 0.10 * horizon_recall
        + 0.10 * comfort_score
    )
    return {
        "case_id": case["case_id"],
        "scenario_tag": case["scenario_tag"],
        "scenario_family": case["scenario_family"],
        "difficulty_label": case.get("difficulty_label", ""),
        "location": case["location"],
        "scene_name": case["scene_name"],
        "full_horizon": full_horizon,
        "horizon_recall": horizon_recall,
        "ego_ade_m": mean(ego_errors),
        "ego_fde_m": ego_errors[-1],
        "logged_min_distance_m": logged_min_distance,
        "predicted_min_distance_m": predicted_min_distance,
        "min_distance_error_m": min_distance_error,
        "logged_min_ttc_s": logged_min_ttc,
        "predicted_min_ttc_s": predicted_min_ttc,
        "min_ttc_error_s": min_ttc_error,
        "red_light_context_recall": red_light_context_recall,
        "logged_max_acceleration_mps2": logged_comfort.get("max_acceleration_mps2"),
        "predicted_max_acceleration_mps2": predicted_comfort.get("max_acceleration_mps2"),
        "max_acceleration_error_mps2": max_acceleration_error,
        "logged_max_jerk_mps3": logged_comfort.get("max_jerk_mps3"),
        "predicted_max_jerk_mps3": predicted_comfort.get("max_jerk_mps3"),
        "max_jerk_error_mps3": max_jerk_error,
        "logged_max_yaw_rate_rps": logged_comfort.get("max_yaw_rate_rps"),
        "predicted_max_yaw_rate_rps": predicted_comfort.get("max_yaw_rate_rps"),
        "max_yaw_rate_error_rps": max_yaw_rate_error,
        "comfort_violation_mismatch": comfort_violation_mismatch,
        "logged_collision_proxy": logged_collision,
        "predicted_collision_proxy": predicted_collision,
        "risk_fidelity_score": risk_fidelity,
        "failure_tags": failure_tags,
    }


def _build_nuplan_evaluation_summary(
    profile_name: str,
    benchmark_path: Path,
    predictions_path: Path,
    case_metrics: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    finite_ade = _finite_values(case_metrics, "ego_ade_m")
    finite_fde = _finite_values(case_metrics, "ego_fde_m")
    finite_distance_errors = _finite_values(case_metrics, "min_distance_error_m")
    finite_ttc_errors = _finite_values(case_metrics, "min_ttc_error_s")
    finite_acceleration_errors = _finite_values(case_metrics, "max_acceleration_error_mps2")
    finite_jerk_errors = _finite_values(case_metrics, "max_jerk_error_mps3")
    finite_yaw_rate_errors = _finite_values(case_metrics, "max_yaw_rate_error_rps")
    finite_risk_scores = _finite_values(case_metrics, "risk_fidelity_score")
    overview = {
        "profile_name": profile_name,
        "case_count": len(case_metrics),
        "full_horizon_count": sum(1 for row in case_metrics if row["full_horizon"]),
        "full_horizon_rate": _safe_ratio(sum(1 for row in case_metrics if row["full_horizon"]), len(case_metrics)),
        "mean_horizon_recall": mean([float(row["horizon_recall"]) for row in case_metrics]) if case_metrics else 0.0,
        "mean_ego_ade_m": mean(finite_ade) if finite_ade else None,
        "mean_ego_fde_m": mean(finite_fde) if finite_fde else None,
        "mean_min_distance_error_m": mean(finite_distance_errors) if finite_distance_errors else None,
        "mean_min_ttc_error_s": mean(finite_ttc_errors) if finite_ttc_errors else None,
        "mean_red_light_context_recall": mean(
            [float(row["red_light_context_recall"]) for row in case_metrics]
        )
        if case_metrics
        else 0.0,
        "mean_max_acceleration_error_mps2": mean(finite_acceleration_errors) if finite_acceleration_errors else None,
        "mean_max_jerk_error_mps3": mean(finite_jerk_errors) if finite_jerk_errors else None,
        "mean_max_yaw_rate_error_rps": mean(finite_yaw_rate_errors) if finite_yaw_rate_errors else None,
        "mean_risk_fidelity_score": mean(finite_risk_scores) if finite_risk_scores else 0.0,
        "collision_proxy_mismatch_count": sum(
            1 for row in case_metrics if "collision_proxy_mismatch" in row["failure_tags"]
        ),
        "comfort_violation_mismatch_count": sum(
            1 for row in case_metrics if "comfort_violation_mismatch" in row["failure_tags"]
        ),
        "ttc_error_count": sum(1 for row in case_metrics if "ttc_error" in row["failure_tags"]),
        "risk_distance_error_count": sum(1 for row in case_metrics if "risk_distance_error" in row["failure_tags"]),
    }
    return {
        "metadata": {
            "schema": "nuplan_replay_evaluation_v1",
            "benchmark_path": str(benchmark_path),
            "predictions_path": str(predictions_path),
        },
        "overview": overview,
        "scenario_family_breakdown": _breakdown(case_metrics, "scenario_family"),
        "scenario_tag_breakdown": _breakdown(case_metrics, "scenario_tag"),
        "difficulty_breakdown": _breakdown(case_metrics, "difficulty_label"),
        "case_metrics": list(case_metrics),
    }


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _unique_strings(values: Sequence[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _profile_label(profile_name: str) -> str:
    return str(profile_name).replace("_", "-")


def _load_nuplan_case_study_evaluations(evaluation_dirs: Sequence[Path]) -> List[Dict[str, Any]]:
    payloads: List[Dict[str, Any]] = []
    for eval_dir in evaluation_dirs:
        eval_dir = Path(eval_dir)
        metrics_path = eval_dir / "nuplan_replay_metrics.json"
        if not metrics_path.exists():
            raise FileNotFoundError(f"Missing nuPlan replay metrics: {metrics_path}")

        metrics = _read_json(metrics_path)
        profile_name = str(metrics.get("overview", {}).get("profile_name") or eval_dir.name)
        predictions_path = _resolve_existing_path(
            str(metrics.get("metadata", {}).get("predictions_path") or ""),
            base_dir=eval_dir,
        )
        if predictions_path is None:
            raise FileNotFoundError(f"Missing nuPlan replay predictions for evaluation: {eval_dir}")

        predictions_payload = _read_json(predictions_path)
        payloads.append(
            {
                "profile_name": profile_name,
                "evaluation_dir": str(eval_dir),
                "metrics": {
                    str(row.get("case_id")): dict(row)
                    for row in metrics.get("case_metrics", [])
                    if row.get("case_id")
                },
                "predictions": {
                    str(row.get("case_id")): dict(row)
                    for row in predictions_payload.get("predictions", [])
                    if row.get("case_id")
                },
            }
        )
    return payloads


def _resolve_existing_path(path_text: str, base_dir: Path) -> Optional[Path]:
    if not path_text:
        return None
    path = Path(path_text)
    candidates = [path]
    if not path.is_absolute():
        candidates.extend(
            [
                base_dir / path,
                base_dir.parent / path,
                Path.cwd() / path,
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _build_nuplan_case_study_rows(
    benchmark: Dict[str, Any],
    evaluation_payloads: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for case in benchmark.get("cases", []):
        case_id = str(case.get("case_id") or "")
        profiles: List[Dict[str, Any]] = []
        risk_scores: List[float] = []
        ade_values: List[float] = []
        distance_errors: List[float] = []
        for payload in evaluation_payloads:
            metric = payload["metrics"].get(case_id)
            prediction = payload["predictions"].get(case_id)
            if not metric or not prediction:
                continue
            risk_score = _safe_float(metric.get("risk_fidelity_score"), 0.0)
            ade_m = _safe_float(metric.get("ego_ade_m"), math.nan)
            distance_error = _safe_float(metric.get("min_distance_error_m"), math.nan)
            profiles.append(
                {
                    "profile_name": str(payload["profile_name"]),
                    "metric": metric,
                    "prediction": prediction,
                }
            )
            risk_scores.append(risk_score)
            if math.isfinite(ade_m):
                ade_values.append(ade_m)
            if math.isfinite(distance_error):
                distance_errors.append(distance_error)

        if not profiles:
            continue
        rows.append(
            {
                "case": dict(case),
                "profiles": profiles,
                "risk_gap": max(risk_scores) - min(risk_scores) if risk_scores else 0.0,
                "mean_ade_m": mean(ade_values) if ade_values else 0.0,
                "max_distance_error_m": max(distance_errors) if distance_errors else 0.0,
            }
        )
    return rows


def _select_nuplan_case_studies(case_rows: Sequence[Dict[str, Any]], max_cases: int) -> List[Dict[str, Any]]:
    if max_cases <= 0:
        return []
    ordered = sorted(
        case_rows,
        key=lambda row: (
            float(row.get("risk_gap") or 0.0),
            float(row.get("max_distance_error_m") or 0.0),
            float(row.get("mean_ade_m") or 0.0),
            str(row.get("case", {}).get("scenario_tag") or ""),
        ),
        reverse=True,
    )

    selected: List[Dict[str, Any]] = []
    seen_families: set[str] = set()
    for row in ordered:
        family = str(row.get("case", {}).get("scenario_family") or "")
        if family in seen_families:
            continue
        selected.append(row)
        seen_families.add(family)
        if len(selected) >= max_cases:
            return selected

    for row in ordered:
        if row in selected:
            continue
        selected.append(row)
        if len(selected) >= max_cases:
            break
    return selected


def _render_nuplan_case_study_figure(rows: Sequence[Dict[str, Any]], figure_path: Path) -> List[Dict[str, Any]]:
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    profile_names = _unique_strings(
        [
            str(profile["profile_name"])
            for row in rows
            for profile in row.get("profiles", [])
            if str(profile.get("profile_name") or "") != "logged_ego"
        ]
    )
    color_map = {
        profile_name: NUPLAN_PROFILE_COLORS[idx % len(NUPLAN_PROFILE_COLORS)]
        for idx, profile_name in enumerate(profile_names)
    }

    if not rows:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.axis("off")
        ax.text(0.5, 0.5, "No nuPlan replay case studies available", ha="center", va="center")
        fig.savefig(figure_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return []

    cols = 2 if len(rows) > 1 else 1
    rows_n = int(math.ceil(len(rows) / cols))
    fig = plt.figure(figsize=(8.5 * cols, 6.4 * rows_n), constrained_layout=True)
    grid = fig.add_gridspec(rows_n, cols)
    axes = [fig.add_subplot(grid[idx // cols, idx % cols]) for idx in range(rows_n * cols)]

    summary_rows: List[Dict[str, Any]] = []
    for ax, row in zip(axes, rows):
        _draw_nuplan_case_panel(ax, row, color_map=color_map)
        case = dict(row["case"])
        summary_rows.append(
            {
                "case_id": str(case.get("case_id") or ""),
                "scene_name": str(case.get("scene_name") or ""),
                "scenario_tag": str(case.get("scenario_tag") or ""),
                "scenario_family": str(case.get("scenario_family") or ""),
                "difficulty_label": str(case.get("difficulty_label") or ""),
                "location": str(case.get("location") or ""),
                "min_distance_m": case.get("risk_targets", {}).get("min_distance_m"),
                "min_ttc_s": case.get("risk_targets", {}).get("min_ttc_s"),
                "profiles": [
                    {
                        "profile_name": str(profile["profile_name"]),
                        "risk_fidelity_score": _safe_float(profile["metric"].get("risk_fidelity_score"), 0.0),
                        "ego_ade_m": _safe_float(profile["metric"].get("ego_ade_m"), 0.0),
                        "min_distance_error_m": _safe_float(
                            profile["metric"].get("min_distance_error_m"),
                            0.0,
                        ),
                        "min_ttc_error_s": _safe_float(profile["metric"].get("min_ttc_error_s"), 0.0),
                    }
                    for profile in row.get("profiles", [])
                ],
            }
        )

    for ax in axes[len(rows):]:
        ax.axis("off")

    legend_handles = [
        plt.Line2D([0], [0], color=NUPLAN_HISTORY_COLOR, linewidth=2.0, label="ego history"),
        plt.Line2D([0], [0], color=NUPLAN_LOGGED_EGO_COLOR, linewidth=2.2, label="logged ego future"),
        plt.Line2D([0], [0], color=NUPLAN_ACTOR_COLOR, linewidth=2.0, label="primary actor"),
    ]
    for profile_name in profile_names:
        legend_handles.append(
            plt.Line2D(
                [0],
                [0],
                color=color_map[profile_name],
                linestyle="--",
                linewidth=2.0,
                label=_profile_label(profile_name),
            )
        )
    fig.legend(handles=legend_handles, loc="upper center", ncol=min(5, len(legend_handles)), frameon=False)
    fig.suptitle("nuPlan Replay Regression Case Studies", fontsize=16, fontweight="bold")
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return summary_rows


def _draw_nuplan_case_panel(ax: plt.Axes, row: Dict[str, Any], color_map: Dict[str, str]) -> None:
    case = dict(row["case"])
    points: List[Tuple[float, float]] = []

    ax.set_facecolor(NUPLAN_PLOT_BACKGROUND)
    ax.grid(True, color=NUPLAN_GRID_COLOR, linewidth=0.7, alpha=0.8)
    ax.axhline(0.0, color="#9c958c", linewidth=0.8)
    ax.axvline(0.0, color="#9c958c", linewidth=0.8)
    ax.scatter([0.0], [0.0], marker="*", s=90, color="#111111", zorder=8)

    history_xy = _case_ego_xy(case, include_future=False)
    if history_xy:
        points.extend(history_xy)
        ax.plot(
            [x for x, _ in history_xy],
            [y for _, y in history_xy],
            color=NUPLAN_HISTORY_COLOR,
            linewidth=2.0,
        )

    future_xy = _case_ego_xy(case, include_future=True)
    if future_xy:
        points.extend(future_xy)
        ax.plot(
            [x for x, _ in future_xy],
            [y for _, y in future_xy],
            color=NUPLAN_LOGGED_EGO_COLOR,
            linewidth=2.2,
        )
        ax.scatter([x for x, _ in future_xy], [y for _, y in future_xy], color=NUPLAN_LOGGED_EGO_COLOR, s=18)

    actor_xy = _case_actor_xy(case)
    if actor_xy:
        points.extend(actor_xy)
        ax.plot(
            [x for x, _ in actor_xy],
            [y for _, y in actor_xy],
            color=NUPLAN_ACTOR_COLOR,
            linewidth=2.0,
        )
        ax.scatter([x for x, _ in actor_xy], [y for _, y in actor_xy], color=NUPLAN_ACTOR_COLOR, s=22)

    min_distance_ts = case.get("risk_targets", {}).get("min_distance_timestamp_us")
    if min_distance_ts is not None:
        min_frame = next(
            (
                frame
                for frame in case.get("future_frames", [])
                if int(frame.get("timestamp_us", -1)) == int(min_distance_ts)
            ),
            None,
        )
        if min_frame and min_frame.get("primary_actor"):
            actor = min_frame["primary_actor"]
            actor_point = _global_to_anchor_ego(case, float(actor["x"]), float(actor["y"]))
            ax.scatter([actor_point[0]], [actor_point[1]], marker="x", color="#7a1f1f", s=80, linewidths=2.0)

    for profile in row.get("profiles", []):
        profile_name = str(profile["profile_name"])
        if profile_name == "logged_ego":
            continue
        pred_xy = _prediction_xy(case, profile.get("prediction", {}))
        if not pred_xy:
            continue
        points.extend(pred_xy)
        color = color_map.get(profile_name, NUPLAN_PROFILE_COLORS[0])
        ax.plot(
            [x for x, _ in pred_xy],
            [y for _, y in pred_xy],
            color=color,
            linestyle="--",
            linewidth=2.0,
        )
        ax.scatter([x for x, _ in pred_xy], [y for _, y in pred_xy], color=color, s=16)

    _set_equal_limits(ax, points)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Forward in anchor ego frame (m)")
    ax.set_ylabel("Left in anchor ego frame (m)")
    ax.set_title(
        "{0} | {1} | {2}".format(
            case.get("scenario_family", ""),
            case.get("difficulty_label", ""),
            case.get("location", ""),
        )
    )
    ax.text(
        0.02,
        0.02,
        _case_study_panel_text(case, row.get("profiles", [])),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        family="monospace",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#fffdfa", "edgecolor": "#d8d0c4", "alpha": 0.92},
    )


def _case_ego_xy(case: Dict[str, Any], include_future: bool) -> List[Tuple[float, float]]:
    frames = case.get("future_frames", []) if include_future else [
        frame for frame in case.get("frames", []) if float(frame.get("dt_from_anchor_s", 0.0)) < 0.0
    ]
    points: List[Tuple[float, float]] = []
    for frame in frames:
        ego = frame.get("ego", {})
        points.append(_global_to_anchor_ego(case, float(ego.get("x", 0.0)), float(ego.get("y", 0.0))))
    return points


def _case_actor_xy(case: Dict[str, Any]) -> List[Tuple[float, float]]:
    points: List[Tuple[float, float]] = []
    for frame in case.get("future_frames", []):
        actor = frame.get("primary_actor")
        if not actor:
            continue
        points.append(_global_to_anchor_ego(case, float(actor.get("x", 0.0)), float(actor.get("y", 0.0))))
    return points


def _prediction_xy(case: Dict[str, Any], prediction: Dict[str, Any]) -> List[Tuple[float, float]]:
    points: List[Tuple[float, float]] = []
    for state in prediction.get("future_ego_states", []):
        points.append(_global_to_anchor_ego(case, float(state.get("x", 0.0)), float(state.get("y", 0.0))))
    return points


def _global_to_anchor_ego(case: Dict[str, Any], x: float, y: float) -> Tuple[float, float]:
    anchor_ego = case.get("anchor_frame", {}).get("ego", {})
    return _global_to_ego(
        x=x,
        y=y,
        ego_x=float(anchor_ego.get("x", 0.0)),
        ego_y=float(anchor_ego.get("y", 0.0)),
        ego_yaw=float(anchor_ego.get("yaw", 0.0)),
    )


def _set_equal_limits(ax: plt.Axes, points: Sequence[Tuple[float, float]]) -> None:
    if not points:
        ax.set_xlim(-20.0, 20.0)
        ax.set_ylim(-20.0, 20.0)
        return
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    x_mid = (min(xs) + max(xs)) / 2.0
    y_mid = (min(ys) + max(ys)) / 2.0
    span = max(max(xs) - min(xs), max(ys) - min(ys), 12.0)
    pad = span * 0.18
    half = span / 2.0 + pad
    ax.set_xlim(x_mid - half, x_mid + half)
    ax.set_ylim(y_mid - half, y_mid + half)


def _case_study_panel_text(case: Dict[str, Any], profiles: Sequence[Dict[str, Any]]) -> str:
    risk_targets = case.get("risk_targets", {})
    lines = [
        f"case={str(case.get('case_id', ''))[:18]}",
        f"tag={case.get('scenario_tag', '')}",
        "min_dist={0}, min_ttc={1}".format(
            _format_optional_float(risk_targets.get("min_distance_m")),
            _format_optional_float(risk_targets.get("min_ttc_s")),
        ),
    ]
    for profile in profiles:
        metric = profile.get("metric", {})
        lines.append(
            "{0}: rf={1}, ade={2}, de={3}".format(
                _profile_label(str(profile.get("profile_name", ""))),
                _format_optional_float(metric.get("risk_fidelity_score")),
                _format_optional_float(metric.get("ego_ade_m")),
                _format_optional_float(metric.get("min_distance_error_m")),
            )
        )
    return "\n".join(lines)


def _render_nuplan_case_studies_markdown(metadata: Dict[str, Any]) -> str:
    lines = [
        "# nuPlan Replay Case Studies",
        "",
        f"- Cases: `{metadata['case_count']}`",
        "- Figure: `nuplan_replay_case_studies.png`",
        "",
        "| Case | Family | Difficulty | Location | Risk Target | Profile Summary |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for case in metadata.get("cases", []):
        profile_summary = "; ".join(
            "{0}: rf={1:.3f}, ade={2:.3f}".format(
                _profile_label(str(profile.get("profile_name", ""))),
                float(profile.get("risk_fidelity_score") or 0.0),
                float(profile.get("ego_ade_m") or 0.0),
            )
            for profile in case.get("profiles", [])
        )
        risk_target = "dist={0}, ttc={1}".format(
            _format_optional_float(case.get("min_distance_m")),
            _format_optional_float(case.get("min_ttc_s")),
        )
        lines.append(
            "| `{0}` | `{1}` | `{2}` | `{3}` | `{4}` | {5} |".format(
                str(case.get("case_id", ""))[:18],
                case.get("scenario_family", ""),
                case.get("difficulty_label", ""),
                case.get("location", ""),
                risk_target,
                profile_summary,
            )
        )
    lines.append("")
    return "\n".join(lines)


def _write_nuplan_evaluation_outputs(summary: Dict[str, Any], output_dir: Path) -> None:
    (output_dir / "nuplan_replay_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_case_metrics_csv(summary["case_metrics"], output_dir / "nuplan_replay_case_metrics.csv")
    markdown = _render_nuplan_evaluation_markdown(summary)
    (output_dir / "nuplan_replay_metrics_summary.md").write_text(markdown, encoding="utf-8")
    (output_dir / "nuplan_replay_metrics_summary.html").write_text(_markdown_to_basic_html(markdown), encoding="utf-8")


def _write_case_metrics_csv(rows: Sequence[Dict[str, Any]], output_path: Path) -> None:
    fieldnames = [
        "case_id",
        "scenario_tag",
        "scenario_family",
        "difficulty_label",
        "location",
        "scene_name",
        "full_horizon",
        "horizon_recall",
        "ego_ade_m",
        "ego_fde_m",
        "logged_min_distance_m",
        "predicted_min_distance_m",
        "min_distance_error_m",
        "logged_min_ttc_s",
        "predicted_min_ttc_s",
        "min_ttc_error_s",
        "red_light_context_recall",
        "logged_max_acceleration_mps2",
        "predicted_max_acceleration_mps2",
        "max_acceleration_error_mps2",
        "logged_max_jerk_mps3",
        "predicted_max_jerk_mps3",
        "max_jerk_error_mps3",
        "logged_max_yaw_rate_rps",
        "predicted_max_yaw_rate_rps",
        "max_yaw_rate_error_rps",
        "comfort_violation_mismatch",
        "logged_collision_proxy",
        "predicted_collision_proxy",
        "risk_fidelity_score",
        "failure_tags",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            payload = dict(row)
            payload["failure_tags"] = ";".join(row.get("failure_tags", []))
            writer.writerow({key: payload.get(key) for key in fieldnames})


def _render_nuplan_evaluation_markdown(summary: Dict[str, Any]) -> str:
    overview = summary["overview"]
    lines = [
        "# nuPlan Replay Evaluation",
        "",
        f"- Profile: `{overview['profile_name']}`",
        f"- Cases: `{overview['case_count']}`",
        f"- Full horizon: `{overview['full_horizon_count']}/{overview['case_count']}`",
        f"- Mean horizon recall: `{overview['mean_horizon_recall']:.3f}`",
        f"- Mean risk fidelity: `{overview['mean_risk_fidelity_score']:.3f}`",
        f"- Collision proxy mismatches: `{overview['collision_proxy_mismatch_count']}`",
        f"- Comfort violation mismatches: `{overview['comfort_violation_mismatch_count']}`",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Mean ego ADE | `{_format_optional_float(overview['mean_ego_ade_m'])}` |",
        f"| Mean ego FDE | `{_format_optional_float(overview['mean_ego_fde_m'])}` |",
        f"| Mean min-distance error | `{_format_optional_float(overview['mean_min_distance_error_m'])}` |",
        f"| Mean min-TTC error | `{_format_optional_float(overview['mean_min_ttc_error_s'])}` |",
        f"| Mean red-light context recall | `{overview['mean_red_light_context_recall']:.3f}` |",
        f"| Mean max-acceleration error | `{_format_optional_float(overview['mean_max_acceleration_error_mps2'])}` |",
        f"| Mean max-jerk error | `{_format_optional_float(overview['mean_max_jerk_error_mps3'])}` |",
        f"| Mean max-yaw-rate error | `{_format_optional_float(overview['mean_max_yaw_rate_error_rps'])}` |",
        "",
        "## Scenario Family Breakdown",
        "",
        "| Family | Cases | Full Horizon | Risk Fidelity | Ego ADE | Distance Error | TTC Error |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in summary["scenario_family_breakdown"]:
        lines.append(
            f"| `{row['scenario_family']}` | `{row['case_count']}` | "
            f"`{row['full_horizon_count']}/{row['case_count']}` | "
            f"`{row['mean_risk_fidelity_score']:.3f}` | "
            f"`{_format_optional_float(row['mean_ego_ade_m'])}` | "
            f"`{_format_optional_float(row['mean_min_distance_error_m'])}` | "
            f"`{_format_optional_float(row['mean_min_ttc_error_s'])}` |"
        )
    lines.append("")
    return "\n".join(lines)


def _write_nuplan_study_summary(manifest: Dict[str, Any], output_dir: Path) -> None:
    lines = [
        "# nuPlan Replay Study",
        "",
        f"- Benchmark cases: `{manifest['benchmark']['metadata']['case_count']}`",
        f"- Scanned DBs: `{manifest['benchmark']['metadata']['db_count_scanned']}`",
        f"- Unified cases: `{manifest.get('unified_cases', {}).get('case_count', 0)}`",
        "",
        "| Profile | Full Horizon | Ego ADE | Min-Distance Error | TTC Error | Risk Fidelity |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for item in manifest["evaluations"]:
        overview = item["overview"]
        lines.append(
            f"| `{item['profile_name']}` | "
            f"`{overview['full_horizon_count']}/{overview['case_count']}` | "
            f"`{_format_optional_float(overview['mean_ego_ade_m'])}` | "
            f"`{_format_optional_float(overview['mean_min_distance_error_m'])}` | "
            f"`{_format_optional_float(overview['mean_min_ttc_error_s'])}` | "
            f"`{overview['mean_risk_fidelity_score']:.3f}` |"
        )
    text = "\n".join(lines) + "\n"
    (output_dir / "nuplan_replay_study_summary.md").write_text(text, encoding="utf-8")
    (output_dir / "nuplan_replay_study_summary.html").write_text(_markdown_to_basic_html(text), encoding="utf-8")


def _write_nuplan_study_artifact_manifest(output_dir: Path, evaluations: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    artifact_specs = [
        (output_dir / "nuplan_replay_benchmark.json", "benchmark", "replay_benchmark"),
        (output_dir / "unified_risk_cases.json", "benchmark", "unified_case_collection"),
        (output_dir / "nuplan_replay_study_manifest.json", "summary", "study_manifest"),
        (output_dir / "nuplan_replay_study_summary.md", "summary", "study_summary"),
        (output_dir / "nuplan_replay_study_summary.html", "summary", "study_summary"),
        (output_dir / "comparison/nuplan_replay_comparison.json", "comparison", "metrics_comparison"),
        (output_dir / "comparison/nuplan_replay_leaderboard.csv", "comparison", "leaderboard"),
        (output_dir / "comparison/nuplan_replay_comparison_summary.md", "comparison", "comparison_summary"),
        (output_dir / "comparison/nuplan_replay_comparison_summary.html", "comparison", "comparison_summary"),
        (output_dir / "case_studies/nuplan_replay_case_studies.png", "evidence", "case_study_figure"),
        (output_dir / "case_studies/nuplan_replay_case_studies.json", "evidence", "case_study_summary"),
        (output_dir / "case_studies/nuplan_replay_case_studies.md", "evidence", "case_study_summary"),
        (output_dir / "case_studies/nuplan_replay_case_studies.html", "evidence", "case_study_summary"),
    ]
    for item in evaluations:
        profile_name = str(item.get("profile_name") or "profile")
        artifact_specs.extend(
            [
                (output_dir / f"{profile_name}_rollouts.json", "prediction", "ego_rollouts"),
                (Path(str(item["evaluation_dir"])) / "nuplan_replay_metrics.json", "evaluation", "metrics"),
                (Path(str(item["evaluation_dir"])) / "nuplan_replay_case_metrics.csv", "evaluation", "case_metrics"),
                (Path(str(item["evaluation_dir"])) / "nuplan_replay_metrics_summary.md", "evaluation", "metrics_summary"),
                (Path(str(item["evaluation_dir"])) / "nuplan_replay_metrics_summary.html", "evaluation", "metrics_summary"),
            ]
        )

    artifacts = [
        build_artifact_entry(path=path, role=role, kind=kind, output_root=output_dir)
        for path, role, kind in artifact_specs
    ]
    return write_artifact_manifest(
        output_dir=output_dir,
        artifacts=artifacts,
        metadata={"benchmark_layer": "nuplan_replay_regression"},
    )


def _profile_row_from_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    overview = summary["overview"]
    return {
        "profile_name": overview["profile_name"],
        "case_count": overview["case_count"],
        "full_horizon_count": overview["full_horizon_count"],
        "full_horizon_rate": overview["full_horizon_rate"],
        "mean_ego_ade_m": overview["mean_ego_ade_m"],
        "mean_ego_fde_m": overview["mean_ego_fde_m"],
        "mean_min_distance_error_m": overview["mean_min_distance_error_m"],
        "mean_min_ttc_error_s": overview["mean_min_ttc_error_s"],
        "mean_red_light_context_recall": overview["mean_red_light_context_recall"],
        "mean_max_acceleration_error_mps2": overview["mean_max_acceleration_error_mps2"],
        "mean_max_jerk_error_mps3": overview["mean_max_jerk_error_mps3"],
        "mean_max_yaw_rate_error_rps": overview["mean_max_yaw_rate_error_rps"],
        "mean_risk_fidelity_score": overview["mean_risk_fidelity_score"],
        "collision_proxy_mismatch_count": overview["collision_proxy_mismatch_count"],
        "comfort_violation_mismatch_count": overview["comfort_violation_mismatch_count"],
        "evaluation_dir": summary.get("metadata", {}).get("evaluation_dir", ""),
    }


def _comparison_matrix(summaries: Sequence[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    values = sorted(
        {
            str(row.get(key) or "unknown")
            for summary in summaries
            for row in summary.get("case_metrics", [])
        }
    )
    rows: List[Dict[str, Any]] = []
    for value in values:
        cells = []
        for summary in summaries:
            profile = summary["overview"]["profile_name"]
            group_rows = [row for row in summary.get("case_metrics", []) if str(row.get(key) or "unknown") == value]
            cells.append({"profile_name": profile, **_aggregate_metric_rows(group_rows)})
        rows.append({key: value, "cells": cells})
    return rows


def _case_comparison_matrix(summaries: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    case_ids = sorted(
        {
            str(row["case_id"])
            for summary in summaries
            for row in summary.get("case_metrics", [])
        }
    )
    rows = []
    for case_id in case_ids:
        cells = []
        metadata: Dict[str, Any] = {}
        for summary in summaries:
            profile = summary["overview"]["profile_name"]
            match = next((row for row in summary.get("case_metrics", []) if str(row["case_id"]) == case_id), None)
            if match and not metadata:
                metadata = {
                    "scenario_tag": match.get("scenario_tag", ""),
                    "scenario_family": match.get("scenario_family", ""),
                    "difficulty_label": match.get("difficulty_label", ""),
                    "location": match.get("location", ""),
                    "scene_name": match.get("scene_name", ""),
                }
            cells.append(
                {
                    "profile_name": profile,
                    "risk_fidelity_score": match.get("risk_fidelity_score") if match else None,
                    "ego_ade_m": match.get("ego_ade_m") if match else None,
                    "min_distance_error_m": match.get("min_distance_error_m") if match else None,
                    "min_ttc_error_s": match.get("min_ttc_error_s") if match else None,
                    "failure_tags": match.get("failure_tags", []) if match else ["missing_prediction"],
                }
            )
        rows.append({"case_id": case_id, **metadata, "cells": cells})
    return rows


def _aggregate_metric_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    risk_scores = _finite_values(rows, "risk_fidelity_score")
    ego_ades = _finite_values(rows, "ego_ade_m")
    distance_errors = _finite_values(rows, "min_distance_error_m")
    ttc_errors = _finite_values(rows, "min_ttc_error_s")
    return {
        "case_count": len(rows),
        "full_horizon_count": sum(1 for row in rows if row.get("full_horizon")),
        "mean_risk_fidelity_score": mean(risk_scores) if risk_scores else 0.0,
        "mean_ego_ade_m": mean(ego_ades) if ego_ades else None,
        "mean_min_distance_error_m": mean(distance_errors) if distance_errors else None,
        "mean_min_ttc_error_s": mean(ttc_errors) if ttc_errors else None,
    }


def _write_nuplan_comparison_outputs(comparison: Dict[str, Any], output_dir: Path) -> None:
    (output_dir / "nuplan_replay_comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    _write_nuplan_comparison_csv(comparison, output_dir / "nuplan_replay_leaderboard.csv")
    markdown = _render_nuplan_comparison_markdown(comparison)
    (output_dir / "nuplan_replay_comparison_summary.md").write_text(markdown, encoding="utf-8")
    (output_dir / "nuplan_replay_comparison_summary.html").write_text(_markdown_to_basic_html(markdown), encoding="utf-8")


def _write_nuplan_comparison_csv(comparison: Dict[str, Any], output_path: Path) -> None:
    fieldnames = [
        "profile_name",
        "case_count",
        "full_horizon_count",
        "full_horizon_rate",
        "mean_ego_ade_m",
        "mean_ego_fde_m",
        "mean_min_distance_error_m",
        "mean_min_ttc_error_s",
        "mean_red_light_context_recall",
        "mean_max_acceleration_error_mps2",
        "mean_max_jerk_error_mps3",
        "mean_max_yaw_rate_error_rps",
        "mean_risk_fidelity_score",
        "collision_proxy_mismatch_count",
        "comfort_violation_mismatch_count",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in comparison["profiles"]:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _render_nuplan_comparison_markdown(comparison: Dict[str, Any]) -> str:
    lines = [
        "# nuPlan Replay Comparison",
        "",
        f"- Profiles: `{comparison['overview']['profile_count']}`",
        f"- Cases: `{comparison['overview']['case_count']}`",
        "",
        "| Profile | Full Horizon | Ego ADE | Distance Error | TTC Error | Red-Light Recall | Risk Fidelity |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in comparison["profiles"]:
        lines.append(
            f"| `{row['profile_name']}` | "
            f"`{row['full_horizon_count']}/{row['case_count']}` | "
            f"`{_format_optional_float(row['mean_ego_ade_m'])}` | "
            f"`{_format_optional_float(row['mean_min_distance_error_m'])}` | "
            f"`{_format_optional_float(row['mean_min_ttc_error_s'])}` | "
            f"`{row['mean_red_light_context_recall']:.3f}` | "
            f"`{row['mean_risk_fidelity_score']:.3f}` |"
        )
    lines.extend(["", "## Difficulty Matrix", "", "| Difficulty | Profile Metrics |", "| --- | --- |"])
    for row in comparison["difficulty_matrix"]:
        cell_text = "; ".join(
            f"{cell['profile_name']}: risk={cell['mean_risk_fidelity_score']:.3f}, "
            f"ade={_format_optional_float(cell['mean_ego_ade_m'])}"
            for cell in row["cells"]
        )
        lines.append(f"| `{row['difficulty_label']}` | {cell_text} |")
    lines.append("")
    return "\n".join(lines)


def _breakdown(rows: Sequence[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key) or "unknown")].append(row)
    result: List[Dict[str, Any]] = []
    for value, group in sorted(grouped.items()):
        risk_scores = _finite_values(group, "risk_fidelity_score")
        ego_ades = _finite_values(group, "ego_ade_m")
        distance_errors = _finite_values(group, "min_distance_error_m")
        ttc_errors = _finite_values(group, "min_ttc_error_s")
        result.append(
            {
                key: value,
                "case_count": len(group),
                "full_horizon_count": sum(1 for row in group if row["full_horizon"]),
                "mean_risk_fidelity_score": mean(risk_scores) if risk_scores else 0.0,
                "mean_ego_ade_m": mean(ego_ades) if ego_ades else None,
                "mean_min_distance_error_m": mean(distance_errors) if distance_errors else None,
                "mean_min_ttc_error_s": mean(ttc_errors) if ttc_errors else None,
            }
        )
    return result


def _finite_values(rows: Sequence[Dict[str, Any]], key: str) -> List[float]:
    values: List[float] = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        value_f = float(value)
        if math.isfinite(value_f):
            values.append(value_f)
    return values


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _format_optional_float(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


def _nuplan_replay_profile_description(profile_name: str) -> str:
    descriptions = {
        "logged_ego": "Logged future ego trajectory used as an oracle reference.",
        "history_kinematic": (
            "Automatic history-conditioned kinematic rollout estimated from pre-anchor ego motion."
        ),
        "constant_velocity": "Constant-velocity rollout from the anchor ego state.",
        "stopped": "Stationary ego rollout at the anchor pose.",
    }
    return descriptions.get(profile_name, "")


def _history_kinematic_context(case: Dict[str, Any]) -> Dict[str, float]:
    anchor = case["anchor_frame"]
    anchor_ts = int(anchor["timestamp_us"])
    anchor_ego = anchor["ego"]
    anchor_yaw = float(anchor_ego.get("yaw", 0.0))
    history_frames = [
        frame
        for frame in case.get("frames", [])
        if int(frame.get("timestamp_us", 0)) <= anchor_ts
    ]
    history_frames.sort(key=lambda frame: int(frame.get("timestamp_us", 0)))
    if not history_frames or int(history_frames[-1].get("timestamp_us", 0)) != anchor_ts:
        history_frames.append(anchor)

    speed = float(anchor_ego.get("speed_mps", math.hypot(float(anchor_ego.get("vx", 0.0)), float(anchor_ego.get("vy", 0.0)))))
    acceleration = _longitudinal_acceleration(anchor_ego)
    yaw_rate = float(anchor_ego.get("angular_rate_z", 0.0))

    if len(history_frames) >= 2:
        prev = history_frames[-2]
        curr = history_frames[-1]
        prev_ts = int(prev["timestamp_us"])
        curr_ts = int(curr["timestamp_us"])
        dt = max((curr_ts - prev_ts) / 1_000_000.0, 1e-6)
        prev_ego = prev["ego"]
        curr_ego = curr["ego"]
        dx = float(curr_ego["x"]) - float(prev_ego["x"])
        dy = float(curr_ego["y"]) - float(prev_ego["y"])
        measured_speed = math.hypot(dx, dy) / dt
        speed = measured_speed if math.isfinite(measured_speed) else speed
        yaw_rate = _angle_diff(float(curr_ego.get("yaw", 0.0)), float(prev_ego.get("yaw", 0.0))) / dt

    if len(history_frames) >= 3:
        prev_prev = history_frames[-3]
        prev = history_frames[-2]
        curr = history_frames[-1]
        prev_speed = _frame_pair_speed(prev_prev, prev)
        curr_speed = _frame_pair_speed(prev, curr)
        dt = max((int(curr["timestamp_us"]) - int(prev["timestamp_us"])) / 1_000_000.0, 1e-6)
        acceleration = (curr_speed - prev_speed) / dt

    return {
        "anchor_timestamp_us": float(anchor_ts),
        "x": float(anchor_ego["x"]),
        "y": float(anchor_ego["y"]),
        "yaw": anchor_yaw,
        "speed_mps": max(0.0, speed),
        "acceleration_mps2": _clamp(acceleration, -KINEMATIC_ACCELERATION_LIMIT_MPS2, KINEMATIC_ACCELERATION_LIMIT_MPS2),
        "yaw_rate_rps": _clamp(yaw_rate, -KINEMATIC_YAW_RATE_LIMIT_RPS, KINEMATIC_YAW_RATE_LIMIT_RPS),
    }


def _history_kinematic_state_at(context: Dict[str, float], timestamp_us: int) -> Dict[str, float]:
    anchor_ts = int(context.get("anchor_timestamp_us", timestamp_us))
    horizon_s = max((int(timestamp_us) - anchor_ts) / 1_000_000.0, 0.0)
    x = float(context.get("x", 0.0))
    y = float(context.get("y", 0.0))
    yaw = float(context.get("yaw", 0.0))
    speed = max(0.0, float(context.get("speed_mps", 0.0)))
    acceleration = float(context.get("acceleration_mps2", 0.0))
    yaw_rate = float(context.get("yaw_rate_rps", 0.0))

    remaining = horizon_s
    while remaining > 1e-9:
        step = min(0.1, remaining)
        next_speed = max(0.0, speed + acceleration * step)
        distance = 0.5 * (speed + next_speed) * step
        mid_yaw = yaw + 0.5 * yaw_rate * step
        x += math.cos(mid_yaw) * distance
        y += math.sin(mid_yaw) * distance
        yaw = _normalize_angle(yaw + yaw_rate * step)
        speed = next_speed
        remaining -= step

    return {
        "timestamp_us": float(timestamp_us),
        "x": x,
        "y": y,
        "yaw": yaw,
        "vx": speed * math.cos(yaw),
        "vy": speed * math.sin(yaw),
        "acceleration_x": acceleration * math.cos(yaw),
        "acceleration_y": acceleration * math.sin(yaw),
        "angular_rate_z": yaw_rate,
    }


def _frame_pair_speed(prev: Dict[str, Any], curr: Dict[str, Any]) -> float:
    dt = max((int(curr["timestamp_us"]) - int(prev["timestamp_us"])) / 1_000_000.0, 1e-6)
    prev_ego = prev["ego"]
    curr_ego = curr["ego"]
    return math.hypot(float(curr_ego["x"]) - float(prev_ego["x"]), float(curr_ego["y"]) - float(prev_ego["y"])) / dt


def _longitudinal_acceleration(ego: Dict[str, Any]) -> float:
    yaw = float(ego.get("yaw", 0.0))
    ax = float(ego.get("acceleration_x", 0.0))
    ay = float(ego.get("acceleration_y", 0.0))
    return ax * math.cos(yaw) + ay * math.sin(yaw)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def _normalize_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def _markdown_to_basic_html(markdown: str) -> str:
    escaped_lines = []
    for line in markdown.splitlines():
        safe_line = html.escape(line)
        if line.startswith("# "):
            escaped_lines.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            escaped_lines.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("- "):
            escaped_lines.append(f"<p>{safe_line}</p>")
        else:
            escaped_lines.append(f"<p>{safe_line}</p>")
    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head><meta charset=\"utf-8\"><title>nuPlan Replay Evaluation</title></head>",
            "<body>",
            *escaped_lines,
            "</body></html>",
        ]
    )


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _case_id(anchor: NuPlanAnchor) -> str:
    raw = "|".join(
        [
            anchor.db_path.stem,
            str(anchor.timestamp_us),
            anchor.scenario_tag,
            _hex(anchor.track_token),
        ]
    )
    return "nuplan_" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def _hex(token: bytes) -> str:
    return bytes(token).hex()


def _yaw_from_quaternion(qw: float, qx: float, qy: float, qz: float) -> float:
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def _global_to_ego(x: float, y: float, ego_x: float, ego_y: float, ego_yaw: float) -> Tuple[float, float]:
    dx = x - ego_x
    dy = y - ego_y
    c = math.cos(-ego_yaw)
    s = math.sin(-ego_yaw)
    return c * dx - s * dy, s * dx + c * dy


def _linear_ttc(rel_x: float, rel_y: float, rel_vx: float, rel_vy: float) -> Optional[float]:
    speed_sq = rel_vx * rel_vx + rel_vy * rel_vy
    if speed_sq < 1e-6:
        return None
    ttc = -((rel_x * rel_vx + rel_y * rel_vy) / speed_sq)
    if ttc <= 0.0 or ttc > 20.0:
        return None
    return ttc


def _angle_diff(a: float, b: float) -> float:
    return math.atan2(math.sin(a - b), math.cos(a - b))


def _scenario_family(scenario_tag: str, category_name: str) -> str:
    tag = scenario_tag.lower()
    category = category_name.lower()
    if "pedestrian" in tag or "pedestrian" in category:
        return "vru_interaction"
    if "high_speed" in tag:
        return "high_speed_interaction"
    if "long_vehicle" in tag:
        return "large_vehicle_interaction"
    if "construction" in tag or "trafficcone" in tag:
        return "static_obstacle_context"
    if "intersection" in tag or "traffic_light" in tag:
        return "intersection_context"
    return "general_interaction"

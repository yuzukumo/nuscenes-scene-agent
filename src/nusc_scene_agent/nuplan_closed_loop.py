from __future__ import annotations

import csv
import html
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from nusc_scene_agent.artifact_manifest import build_artifact_entry, write_artifact_manifest
from nusc_scene_agent.nuplan_replay import (
    DEFAULT_NUPLAN_SPLIT,
    generate_nuplan_replay_benchmark,
)


DEFAULT_NUPLAN_CLOSED_LOOP_OUTPUT = Path("outputs/nuplan_closed_loop_study_v1")
DEFAULT_NUPLAN_CLOSED_LOOP_PROFILES = ["logged_ego_oracle", "history_kinematic", "idm_like_following"]

NUPLAN_CLOSED_LOOP_SCHEMA = "nuplan_closed_loop_study_v1"
NUPLAN_CLOSED_LOOP_METRICS_SCHEMA = "nuplan_closed_loop_metrics_v1"
NUPLAN_CLOSED_LOOP_PROTOCOL = "ego-only replay simulation with non-reactive logged actors"
NUPLAN_CLOSED_LOOP_METRIC_PROTOCOL_VERSION = "2.0"
COLLISION_DISTANCE_M = 2.0
TARGET_FOLLOWING_DISTANCE_M = 8.0
TIME_GAP_S = 1.2
MAX_ACCELERATION_MPS2 = 3.0
MAX_DECELERATION_MPS2 = -4.0
MAX_YAW_RATE_RPS = 0.8
PLOT_BACKGROUND = "#f7f4ee"
LOGGED_EGO_COLOR = "#1f6f99"
HISTORY_COLOR = "#303030"
ACTOR_COLOR = "#c44e38"
PROFILE_COLORS = ["#2a9d8f", "#c46a00", "#6f4e7c", "#6b8e23", "#7a1f1f"]


def run_nuplan_closed_loop_study(
    split_dir: Path = DEFAULT_NUPLAN_SPLIT,
    output_dir: Path = DEFAULT_NUPLAN_CLOSED_LOOP_OUTPUT,
    *,
    max_dbs: int = 64,
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
    benchmark_path = output_dir / "nuplan_closed_loop_benchmark.json"
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
    benchmark = _read_json(benchmark_path)
    profile_names = list(profiles or DEFAULT_NUPLAN_CLOSED_LOOP_PROFILES)
    evaluations = []
    for profile_name in profile_names:
        evaluation = evaluate_closed_loop_profile(benchmark, profile_name=profile_name)
        evaluation_dir = output_dir / f"{profile_name}_closed_loop"
        _write_profile_outputs(evaluation, evaluation_dir)
        evaluations.append(
            {
                "profile_name": profile_name,
                "evaluation_dir": str(evaluation_dir),
                "overview": evaluation["overview"],
            }
        )

    comparison = build_closed_loop_comparison(evaluations, output_dir=output_dir / "comparison")
    case_studies = render_closed_loop_case_studies(
        benchmark=benchmark,
        evaluation_dirs=[Path(item["evaluation_dir"]) for item in evaluations],
        output_dir=output_dir / "case_studies",
        max_cases=min(4, max_cases),
    )
    manifest = {
        "schema": NUPLAN_CLOSED_LOOP_SCHEMA,
        "evaluation_protocol": NUPLAN_CLOSED_LOOP_PROTOCOL,
        "metric_protocol_version": NUPLAN_CLOSED_LOOP_METRIC_PROTOCOL_VERSION,
        "progress_ratio_semantics": {
            "progress_ratio": "bounded completion ratio in [0, 1]",
            "raw_progress_ratio": "simulated path length divided by logged path length, capped at 1.5",
            "closed_loop_score": "uses raw_progress_ratio to penalize both under- and over-progress",
        },
        "actor_dynamics": "replayed_logged_states",
        "traffic_light_dynamics": "replayed_logged_context",
        "planner_feedback_to_other_agents": False,
        "output_dir": str(output_dir),
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
    }
    (output_dir / "nuplan_closed_loop_study_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    _write_study_summary(manifest, output_dir)
    artifact_manifest = _write_artifact_manifest(output_dir, evaluations)
    manifest["artifact_manifest"] = {
        "path": str(output_dir / "artifact_manifest.json"),
        "overview": artifact_manifest.get("overview", {}),
    }
    (output_dir / "nuplan_closed_loop_study_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return manifest


def evaluate_closed_loop_profile(benchmark: Mapping[str, Any], profile_name: str) -> Dict[str, Any]:
    case_metrics = []
    rollouts = []
    for case in benchmark.get("cases", []):
        rollout = _simulate_closed_loop_case(dict(case), profile_name=profile_name)
        rollouts.append(rollout)
        case_metrics.append(_evaluate_closed_loop_case(dict(case), rollout))
    overview = _build_overview(profile_name, case_metrics)
    return {
        "schema": NUPLAN_CLOSED_LOOP_METRICS_SCHEMA,
        "evaluation_protocol": NUPLAN_CLOSED_LOOP_PROTOCOL,
        "metric_protocol_version": NUPLAN_CLOSED_LOOP_METRIC_PROTOCOL_VERSION,
        "progress_ratio_semantics": {
            "progress_ratio": "bounded completion ratio in [0, 1]",
            "raw_progress_ratio": "simulated path length divided by logged path length, capped at 1.5",
            "closed_loop_score": "uses raw_progress_ratio to penalize both under- and over-progress",
        },
        "actor_dynamics": "replayed_logged_states",
        "planner_feedback_to_other_agents": False,
        "overview": overview,
        "case_metrics": case_metrics,
        "rollouts": rollouts,
        "scenario_family_breakdown": _breakdown(case_metrics, "scenario_family"),
        "difficulty_breakdown": _breakdown(case_metrics, "difficulty_label"),
    }


def build_closed_loop_comparison(evaluations: Sequence[Mapping[str, Any]], output_dir: Path) -> Dict[str, Any]:
    profiles = []
    for item in evaluations:
        metrics = _read_json(Path(str(item["evaluation_dir"])) / "closed_loop_metrics.json")
        profiles.append(metrics["overview"])
    case_count = max((int(profile.get("case_count") or 0) for profile in profiles), default=0)
    payload = {
        "schema": "nuplan_closed_loop_comparison_v1",
        "overview": {
            "profile_count": len(profiles),
            "case_count": case_count,
        },
        "profiles": profiles,
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "closed_loop_comparison.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_comparison_csv(payload, output_dir / "closed_loop_leaderboard.csv")
    markdown = _render_comparison_markdown(payload)
    (output_dir / "closed_loop_comparison_summary.md").write_text(markdown, encoding="utf-8")
    (output_dir / "closed_loop_comparison_summary.html").write_text(_markdown_to_basic_html(markdown), encoding="utf-8")
    return payload


def render_closed_loop_case_studies(
    benchmark: Mapping[str, Any],
    evaluation_dirs: Sequence[Path],
    output_dir: Path,
    max_cases: int = 4,
) -> Dict[str, Any]:
    case_by_id = {str(case["case_id"]): dict(case) for case in benchmark.get("cases", [])}
    profiles = []
    for eval_dir in evaluation_dirs:
        payload = _read_json(Path(eval_dir) / "closed_loop_metrics.json")
        profiles.append(
            {
                "profile_name": str(payload["overview"]["profile_name"]),
                "rollouts": {str(row["case_id"]): row for row in payload.get("rollouts", [])},
                "metrics": {str(row["case_id"]): row for row in payload.get("case_metrics", [])},
            }
        )

    rows = []
    for case_id, case in case_by_id.items():
        profile_rows = []
        for profile in profiles:
            metric = profile["metrics"].get(case_id)
            rollout = profile["rollouts"].get(case_id)
            if not metric or not rollout:
                continue
            profile_rows.append(
                {
                    "profile_name": profile["profile_name"],
                    "metric": metric,
                    "rollout": rollout,
                }
            )
        if profile_rows:
            rows.append({"case": case, "profiles": profile_rows})
    rows.sort(key=lambda row: _case_study_priority(row), reverse=True)
    selected = rows[: max(int(max_cases), 0)]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / "nuplan_closed_loop_case_studies.png"
    summary_rows = _render_case_study_figure(selected, figure_path)
    payload = {
        "schema": "nuplan_closed_loop_case_studies_v1",
        "case_count": len(summary_rows),
        "figure_path": str(figure_path),
        "cases": summary_rows,
    }
    (output_dir / "nuplan_closed_loop_case_studies.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    markdown = _render_case_study_markdown(payload)
    (output_dir / "nuplan_closed_loop_case_studies.md").write_text(markdown, encoding="utf-8")
    (output_dir / "nuplan_closed_loop_case_studies.html").write_text(_markdown_to_basic_html(markdown), encoding="utf-8")
    return {
        "output_dir": str(output_dir),
        "case_count": len(summary_rows),
        "figure_path": str(figure_path),
    }


def _simulate_closed_loop_case(case: Dict[str, Any], profile_name: str) -> Dict[str, Any]:
    frames = list(case.get("future_frames", []))
    if not frames:
        return {"case_id": case.get("case_id", ""), "profile_name": profile_name, "states": []}
    state = _state_from_frame(case["anchor_frame"])
    history_context = _history_context(case)
    states: List[Dict[str, Any]] = []
    prev_ts = int(case["anchor_frame"]["timestamp_us"])
    for idx, frame in enumerate(frames):
        timestamp_us = int(frame["timestamp_us"])
        dt = max((timestamp_us - prev_ts) / 1_000_000.0, 0.0)
        if idx == 0 or profile_name == "logged_ego_oracle":
            state = _state_from_frame(frame)
        else:
            command = _planner_command(
                profile_name=profile_name,
                state=state,
                frame=frame,
                history_context=history_context,
            )
            state = _integrate_state(state, timestamp_us=timestamp_us, dt=dt, command=command)
        states.append(
            {
                "timestamp_us": timestamp_us,
                "x": state["x"],
                "y": state["y"],
                "yaw": state["yaw"],
                "vx": state["vx"],
                "vy": state["vy"],
                "speed_mps": state["speed_mps"],
                "acceleration_mps2": state["acceleration_mps2"],
                "yaw_rate_rps": state["yaw_rate_rps"],
            }
        )
        prev_ts = timestamp_us
    return {
        "case_id": case["case_id"],
        "profile_name": profile_name,
        "states": states,
    }


def _planner_command(
    profile_name: str,
    state: Mapping[str, float],
    frame: Mapping[str, Any],
    history_context: Mapping[str, float],
) -> Dict[str, float]:
    if profile_name == "history_kinematic":
        return {
            "acceleration_mps2": float(history_context.get("acceleration_mps2", 0.0)),
            "yaw_rate_rps": float(history_context.get("yaw_rate_rps", 0.0)),
        }
    if profile_name == "idm_like_following":
        actor = frame.get("primary_actor")
        acceleration = float(history_context.get("acceleration_mps2", 0.0))
        if actor:
            distance = math.hypot(float(actor["x"]) - float(state["x"]), float(actor["y"]) - float(state["y"]))
            actor_speed = math.hypot(float(actor.get("vx", 0.0)), float(actor.get("vy", 0.0)))
            desired_gap = TARGET_FOLLOWING_DISTANCE_M + float(state.get("speed_mps", 0.0)) * TIME_GAP_S
            if distance < desired_gap:
                acceleration = max(MAX_DECELERATION_MPS2, -1.5 * (desired_gap - distance) / max(desired_gap, 1.0))
            elif actor_speed > float(state.get("speed_mps", 0.0)):
                acceleration = min(MAX_ACCELERATION_MPS2, 0.8)
        return {
            "acceleration_mps2": acceleration,
            "yaw_rate_rps": float(history_context.get("yaw_rate_rps", 0.0)),
        }
    if profile_name == "constant_velocity":
        return {"acceleration_mps2": 0.0, "yaw_rate_rps": 0.0}
    if profile_name == "stopped":
        return {"acceleration_mps2": MAX_DECELERATION_MPS2, "yaw_rate_rps": 0.0}
    raise ValueError(f"Unknown closed-loop profile: {profile_name}")


def _evaluate_closed_loop_case(case: Dict[str, Any], rollout: Mapping[str, Any]) -> Dict[str, Any]:
    future_frames = list(case.get("future_frames", []))
    logged_by_ts = {int(frame["timestamp_us"]): frame for frame in future_frames}
    states = list(rollout.get("states", []))
    aligned = [(logged_by_ts[int(state["timestamp_us"])], state) for state in states if int(state["timestamp_us"]) in logged_by_ts]
    if not aligned:
        return _missing_case_metric(case)

    ego_errors = [
        math.hypot(float(state["x"]) - float(frame["ego"]["x"]), float(state["y"]) - float(frame["ego"]["y"]))
        for frame, state in aligned
    ]
    predicted_distances = []
    predicted_ttcs = []
    collision = False
    for frame, state in aligned:
        actor = frame.get("primary_actor")
        if not actor:
            continue
        distance = math.hypot(float(state["x"]) - float(actor["x"]), float(state["y"]) - float(actor["y"]))
        predicted_distances.append(distance)
        collision = collision or distance < COLLISION_DISTANCE_M
        ttc = _linear_ttc(
            rel_x=float(actor["x"]) - float(state["x"]),
            rel_y=float(actor["y"]) - float(state["y"]),
            rel_vx=float(actor.get("vx", 0.0)) - float(state.get("vx", 0.0)),
            rel_vy=float(actor.get("vy", 0.0)) - float(state.get("vy", 0.0)),
        )
        if ttc is not None:
            predicted_ttcs.append(ttc)

    logged_min_distance = case.get("risk_targets", {}).get("min_distance_m")
    predicted_min_distance = min(predicted_distances) if predicted_distances else None
    min_distance_error = _abs_optional(predicted_min_distance, logged_min_distance)
    logged_min_ttc = case.get("risk_targets", {}).get("min_ttc_s")
    predicted_min_ttc = min(predicted_ttcs) if predicted_ttcs else None
    min_ttc_error = _abs_optional(predicted_min_ttc, logged_min_ttc)
    comfort = _closed_loop_comfort(states)
    logged_collision = bool(case.get("risk_targets", {}).get("collision_proxy"))
    failure_tags = []
    if collision != logged_collision:
        failure_tags.append("collision_proxy_mismatch")
    if min_distance_error is not None and min_distance_error > 3.0:
        failure_tags.append("risk_distance_error")
    if min_ttc_error is not None and min_ttc_error > 2.0:
        failure_tags.append("ttc_error")
    if bool(comfort.get("comfort_violation")):
        failure_tags.append("comfort_violation")
    if ego_errors and ego_errors[-1] > 8.0:
        failure_tags.append("closed_loop_drift")

    horizon_recall = len(aligned) / max(len(future_frames), 1)
    progress = _closed_loop_progress(states)
    logged_progress = _logged_progress(future_frames)
    raw_progress_ratio = (
        min(1.5, progress / max(logged_progress, 1e-6))
        if logged_progress > 0
        else 1.0
    )
    progress_ratio = min(1.0, raw_progress_ratio)
    distance_score = 0.0 if min_distance_error is None else max(0.0, 1.0 - min_distance_error / 10.0)
    ttc_score = _ttc_similarity(predicted_min_ttc, logged_min_ttc)
    collision_score = 1.0 if collision == logged_collision else 0.0
    comfort_score = 0.0 if bool(comfort.get("comfort_violation")) else 1.0
    progress_score = max(0.0, 1.0 - abs(1.0 - raw_progress_ratio))
    closed_loop_score = (
        0.25 * distance_score
        + 0.20 * collision_score
        + 0.15 * ttc_score
        + 0.15 * comfort_score
        + 0.15 * progress_score
        + 0.10 * horizon_recall
    )
    return {
        "case_id": case["case_id"],
        "scenario_tag": case["scenario_tag"],
        "scenario_family": case["scenario_family"],
        "difficulty_label": case.get("difficulty_label", ""),
        "location": case.get("location", ""),
        "scene_name": case.get("scene_name", ""),
        "full_horizon": len(aligned) == len(future_frames),
        "horizon_recall": horizon_recall,
        "ego_ade_m": mean(ego_errors),
        "ego_fde_m": ego_errors[-1],
        "closed_loop_progress_m": progress,
        "logged_progress_m": logged_progress,
        "progress_ratio": progress_ratio,
        "raw_progress_ratio": raw_progress_ratio,
        "logged_min_distance_m": logged_min_distance,
        "predicted_min_distance_m": predicted_min_distance,
        "min_distance_error_m": min_distance_error,
        "logged_min_ttc_s": logged_min_ttc,
        "predicted_min_ttc_s": predicted_min_ttc,
        "min_ttc_error_s": min_ttc_error,
        "logged_collision_proxy": logged_collision,
        "predicted_collision_proxy": collision,
        "max_acceleration_mps2": comfort["max_acceleration_mps2"],
        "max_jerk_mps3": comfort["max_jerk_mps3"],
        "max_yaw_rate_rps": comfort["max_yaw_rate_rps"],
        "comfort_violation": comfort["comfort_violation"],
        "closed_loop_score": closed_loop_score,
        "failure_tags": failure_tags,
    }


def _missing_case_metric(case: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "case_id": case.get("case_id", ""),
        "scenario_tag": case.get("scenario_tag", ""),
        "scenario_family": case.get("scenario_family", ""),
        "difficulty_label": case.get("difficulty_label", ""),
        "location": case.get("location", ""),
        "scene_name": case.get("scene_name", ""),
        "full_horizon": False,
        "horizon_recall": 0.0,
        "ego_ade_m": None,
        "ego_fde_m": None,
        "closed_loop_progress_m": 0.0,
        "logged_progress_m": _logged_progress(list(case.get("future_frames", []))),
        "progress_ratio": 0.0,
        "raw_progress_ratio": 0.0,
        "logged_min_distance_m": case.get("risk_targets", {}).get("min_distance_m"),
        "predicted_min_distance_m": None,
        "min_distance_error_m": None,
        "logged_min_ttc_s": case.get("risk_targets", {}).get("min_ttc_s"),
        "predicted_min_ttc_s": None,
        "min_ttc_error_s": None,
        "logged_collision_proxy": bool(case.get("risk_targets", {}).get("collision_proxy")),
        "predicted_collision_proxy": None,
        "max_acceleration_mps2": None,
        "max_jerk_mps3": None,
        "max_yaw_rate_rps": None,
        "comfort_violation": None,
        "closed_loop_score": 0.0,
        "failure_tags": ["missing_rollout"],
    }


def _build_overview(profile_name: str, rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    scores = _finite_values(rows, "closed_loop_score")
    ade = _finite_values(rows, "ego_ade_m")
    fde = _finite_values(rows, "ego_fde_m")
    distance_errors = _finite_values(rows, "min_distance_error_m")
    ttc_errors = _finite_values(rows, "min_ttc_error_s")
    progress_ratios = _finite_values(rows, "progress_ratio")
    raw_progress_ratios = _finite_values(rows, "raw_progress_ratio")
    return {
        "profile_name": profile_name,
        "case_count": len(rows),
        "full_horizon_count": sum(1 for row in rows if bool(row.get("full_horizon"))),
        "full_horizon_rate": _safe_ratio(sum(1 for row in rows if bool(row.get("full_horizon"))), len(rows)),
        "mean_ego_ade_m": mean(ade) if ade else None,
        "mean_ego_fde_m": mean(fde) if fde else None,
        "mean_min_distance_error_m": mean(distance_errors) if distance_errors else None,
        "mean_min_ttc_error_s": mean(ttc_errors) if ttc_errors else None,
        "mean_progress_ratio": mean(progress_ratios) if progress_ratios else None,
        "mean_raw_progress_ratio": mean(raw_progress_ratios) if raw_progress_ratios else None,
        "mean_closed_loop_score": mean(scores) if scores else 0.0,
        "collision_proxy_mismatch_count": sum(
            1 for row in rows if "collision_proxy_mismatch" in set(row.get("failure_tags", []))
        ),
        "comfort_violation_count": sum(1 for row in rows if "comfort_violation" in set(row.get("failure_tags", []))),
        "closed_loop_drift_count": sum(1 for row in rows if "closed_loop_drift" in set(row.get("failure_tags", []))),
    }


def _history_context(case: Mapping[str, Any]) -> Dict[str, float]:
    anchor = case["anchor_frame"]
    frames = [frame for frame in case.get("frames", []) if int(frame.get("timestamp_us", 0)) <= int(anchor["timestamp_us"])]
    frames.sort(key=lambda frame: int(frame.get("timestamp_us", 0)))
    acceleration = _longitudinal_acceleration(anchor["ego"])
    yaw_rate = float(anchor["ego"].get("angular_rate_z", 0.0))
    if len(frames) >= 2:
        prev = frames[-2]
        curr = frames[-1]
        dt = max((int(curr["timestamp_us"]) - int(prev["timestamp_us"])) / 1_000_000.0, 1e-6)
        yaw_rate = _angle_diff(float(curr["ego"].get("yaw", 0.0)), float(prev["ego"].get("yaw", 0.0))) / dt
    if len(frames) >= 3:
        prev_prev = frames[-3]
        prev = frames[-2]
        curr = frames[-1]
        acceleration = (_frame_pair_speed(prev, curr) - _frame_pair_speed(prev_prev, prev)) / max(
            (int(curr["timestamp_us"]) - int(prev["timestamp_us"])) / 1_000_000.0,
            1e-6,
        )
    return {
        "acceleration_mps2": _clamp(acceleration, MAX_DECELERATION_MPS2, MAX_ACCELERATION_MPS2),
        "yaw_rate_rps": _clamp(yaw_rate, -MAX_YAW_RATE_RPS, MAX_YAW_RATE_RPS),
    }


def _state_from_frame(frame: Mapping[str, Any]) -> Dict[str, float]:
    ego = frame["ego"]
    speed = float(ego.get("speed_mps", math.hypot(float(ego.get("vx", 0.0)), float(ego.get("vy", 0.0)))))
    yaw = float(ego.get("yaw", 0.0))
    return {
        "x": float(ego["x"]),
        "y": float(ego["y"]),
        "yaw": yaw,
        "vx": float(ego.get("vx", speed * math.cos(yaw))),
        "vy": float(ego.get("vy", speed * math.sin(yaw))),
        "speed_mps": speed,
        "acceleration_mps2": _longitudinal_acceleration(ego),
        "yaw_rate_rps": float(ego.get("angular_rate_z", 0.0)),
    }


def _integrate_state(
    state: Mapping[str, float],
    *,
    timestamp_us: int,
    dt: float,
    command: Mapping[str, float],
) -> Dict[str, float]:
    acceleration = _clamp(float(command.get("acceleration_mps2", 0.0)), MAX_DECELERATION_MPS2, MAX_ACCELERATION_MPS2)
    yaw_rate = _clamp(float(command.get("yaw_rate_rps", 0.0)), -MAX_YAW_RATE_RPS, MAX_YAW_RATE_RPS)
    x = float(state["x"])
    y = float(state["y"])
    yaw = float(state["yaw"])
    speed = max(0.0, float(state["speed_mps"]))
    remaining = max(float(dt), 0.0)
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
        "speed_mps": speed,
        "acceleration_mps2": acceleration,
        "yaw_rate_rps": yaw_rate,
    }


def _closed_loop_comfort(states: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    accelerations = [abs(float(state.get("acceleration_mps2", 0.0))) for state in states]
    yaw_rates = [abs(float(state.get("yaw_rate_rps", 0.0))) for state in states]
    jerk_values = []
    for prev, curr in zip(states, states[1:]):
        dt = max((int(curr["timestamp_us"]) - int(prev["timestamp_us"])) / 1_000_000.0, 1e-6)
        jerk_values.append(abs(float(curr.get("acceleration_mps2", 0.0)) - float(prev.get("acceleration_mps2", 0.0))) / dt)
    max_acc = max(accelerations) if accelerations else 0.0
    max_jerk = max(jerk_values) if jerk_values else 0.0
    max_yaw_rate = max(yaw_rates) if yaw_rates else 0.0
    return {
        "max_acceleration_mps2": max_acc,
        "max_jerk_mps3": max_jerk,
        "max_yaw_rate_rps": max_yaw_rate,
        "comfort_violation": max_acc > 4.0 or max_jerk > 5.0 or max_yaw_rate > 1.0,
    }


def _closed_loop_progress(states: Sequence[Mapping[str, Any]]) -> float:
    if len(states) < 2:
        return 0.0
    return sum(
        math.hypot(float(curr["x"]) - float(prev["x"]), float(curr["y"]) - float(prev["y"]))
        for prev, curr in zip(states, states[1:])
    )


def _logged_progress(frames: Sequence[Mapping[str, Any]]) -> float:
    if len(frames) < 2:
        return 0.0
    return sum(
        math.hypot(
            float(curr["ego"]["x"]) - float(prev["ego"]["x"]),
            float(curr["ego"]["y"]) - float(prev["ego"]["y"]),
        )
        for prev, curr in zip(frames, frames[1:])
    )


def _write_profile_outputs(payload: Mapping[str, Any], output_dir: Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "closed_loop_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_case_metrics_csv(payload.get("case_metrics", []), output_dir / "closed_loop_case_metrics.csv")
    markdown = _render_profile_markdown(payload)
    (output_dir / "closed_loop_metrics_summary.md").write_text(markdown, encoding="utf-8")
    (output_dir / "closed_loop_metrics_summary.html").write_text(_markdown_to_basic_html(markdown), encoding="utf-8")


def _write_case_metrics_csv(rows: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    fieldnames = [
        "case_id",
        "scenario_family",
        "scenario_tag",
        "difficulty_label",
        "location",
        "full_horizon",
        "ego_ade_m",
        "ego_fde_m",
        "closed_loop_progress_m",
        "logged_progress_m",
        "progress_ratio",
        "raw_progress_ratio",
        "min_distance_error_m",
        "min_ttc_error_s",
        "predicted_collision_proxy",
        "comfort_violation",
        "closed_loop_score",
        "failure_tags",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            payload = {key: row.get(key) for key in fieldnames}
            payload["failure_tags"] = ";".join(str(tag) for tag in row.get("failure_tags", []))
            writer.writerow(payload)


def _write_comparison_csv(payload: Mapping[str, Any], output_path: Path) -> None:
    fieldnames = [
        "profile_name",
        "case_count",
        "full_horizon_count",
        "mean_ego_ade_m",
        "mean_ego_fde_m",
        "mean_min_distance_error_m",
        "mean_min_ttc_error_s",
        "mean_progress_ratio",
        "mean_raw_progress_ratio",
        "mean_closed_loop_score",
        "collision_proxy_mismatch_count",
        "comfort_violation_count",
        "closed_loop_drift_count",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in payload.get("profiles", []):
            writer.writerow({key: row.get(key) for key in fieldnames})


def _write_study_summary(manifest: Mapping[str, Any], output_dir: Path) -> None:
    lines = [
        "# nuPlan Closed-Loop Replay Study",
        "",
        f"- Benchmark cases: `{manifest.get('benchmark', {}).get('metadata', {}).get('case_count', 0)}`",
        f"- Profiles: `{len(manifest.get('evaluations', []))}`",
        "",
        "| Profile | Cases | Ego ADE | Distance Error | Progress Ratio | Raw Progress Ratio | Closed-Loop Score |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in manifest.get("evaluations", []):
        overview = dict(item.get("overview") or {})
        lines.append(
            f"| `{overview.get('profile_name', '')}` | `{overview.get('case_count', 0)}` | "
            f"`{_format_optional_float(overview.get('mean_ego_ade_m'))}` | "
            f"`{_format_optional_float(overview.get('mean_min_distance_error_m'))}` | "
            f"`{_format_optional_float(overview.get('mean_progress_ratio'))}` | "
            f"`{_format_optional_float(overview.get('mean_raw_progress_ratio'))}` | "
            f"`{float(overview.get('mean_closed_loop_score') or 0.0):.3f}` |"
        )
    markdown = "\n".join(lines) + "\n"
    (output_dir / "nuplan_closed_loop_study_summary.md").write_text(markdown, encoding="utf-8")
    (output_dir / "nuplan_closed_loop_study_summary.html").write_text(_markdown_to_basic_html(markdown), encoding="utf-8")


def _write_artifact_manifest(output_dir: Path, evaluations: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    specs = [
        (output_dir / "nuplan_closed_loop_benchmark.json", "benchmark", "closed_loop_benchmark"),
        (output_dir / "nuplan_closed_loop_study_manifest.json", "summary", "study_manifest"),
        (output_dir / "nuplan_closed_loop_study_summary.md", "summary", "study_summary"),
        (output_dir / "nuplan_closed_loop_study_summary.html", "summary", "study_summary"),
        (output_dir / "comparison/closed_loop_comparison.json", "comparison", "metrics_comparison"),
        (output_dir / "comparison/closed_loop_leaderboard.csv", "comparison", "leaderboard"),
        (output_dir / "comparison/closed_loop_comparison_summary.md", "comparison", "comparison_summary"),
        (output_dir / "comparison/closed_loop_comparison_summary.html", "comparison", "comparison_summary"),
        (output_dir / "case_studies/nuplan_closed_loop_case_studies.png", "evidence", "case_study_figure"),
        (output_dir / "case_studies/nuplan_closed_loop_case_studies.json", "evidence", "case_study_summary"),
        (output_dir / "case_studies/nuplan_closed_loop_case_studies.md", "evidence", "case_study_summary"),
        (output_dir / "case_studies/nuplan_closed_loop_case_studies.html", "evidence", "case_study_summary"),
    ]
    for item in evaluations:
        eval_dir = Path(str(item["evaluation_dir"]))
        specs.extend(
            [
                (eval_dir / "closed_loop_metrics.json", "evaluation", "metrics"),
                (eval_dir / "closed_loop_case_metrics.csv", "evaluation", "case_metrics"),
                (eval_dir / "closed_loop_metrics_summary.md", "evaluation", "metrics_summary"),
                (eval_dir / "closed_loop_metrics_summary.html", "evaluation", "metrics_summary"),
            ]
        )
    artifacts = [build_artifact_entry(path, role=role, kind=kind, output_root=output_dir) for path, role, kind in specs]
    return write_artifact_manifest(
        output_dir=output_dir,
        artifacts=artifacts,
        metadata={"benchmark_layer": "nuplan_closed_loop_replay"},
    )


def _render_profile_markdown(payload: Mapping[str, Any]) -> str:
    overview = dict(payload.get("overview") or {})
    lines = [
        "# nuPlan Closed-Loop Metrics",
        "",
        f"- Profile: `{overview.get('profile_name', '')}`",
        f"- Cases: `{overview.get('case_count', 0)}`",
        f"- Full horizon: `{overview.get('full_horizon_count', 0)}/{overview.get('case_count', 0)}`",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Ego ADE | `{_format_optional_float(overview.get('mean_ego_ade_m'))}` |",
        f"| Ego FDE | `{_format_optional_float(overview.get('mean_ego_fde_m'))}` |",
        f"| Distance error | `{_format_optional_float(overview.get('mean_min_distance_error_m'))}` |",
        f"| TTC error | `{_format_optional_float(overview.get('mean_min_ttc_error_s'))}` |",
        f"| Progress ratio | `{_format_optional_float(overview.get('mean_progress_ratio'))}` |",
        f"| Raw progress ratio | `{_format_optional_float(overview.get('mean_raw_progress_ratio'))}` |",
        f"| Closed-loop score | `{float(overview.get('mean_closed_loop_score') or 0.0):.3f}` |",
        "",
    ]
    return "\n".join(lines)


def _render_comparison_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# nuPlan Closed-Loop Comparison",
        "",
        f"- Profiles: `{payload.get('overview', {}).get('profile_count', 0)}`",
        f"- Cases: `{payload.get('overview', {}).get('case_count', 0)}`",
        "",
        "| Profile | Cases | Ego ADE | Distance Error | Progress Ratio | Raw Progress Ratio | Closed-Loop Score |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in payload.get("profiles", []):
        lines.append(
            f"| `{row.get('profile_name', '')}` | `{row.get('case_count', 0)}` | "
            f"`{_format_optional_float(row.get('mean_ego_ade_m'))}` | "
            f"`{_format_optional_float(row.get('mean_min_distance_error_m'))}` | "
            f"`{_format_optional_float(row.get('mean_progress_ratio'))}` | "
            f"`{_format_optional_float(row.get('mean_raw_progress_ratio'))}` | "
            f"`{float(row.get('mean_closed_loop_score') or 0.0):.3f}` |"
        )
    lines.append("")
    return "\n".join(lines)


def _render_case_study_figure(rows: Sequence[Mapping[str, Any]], output_path: Path) -> List[Dict[str, Any]]:
    if not rows:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No closed-loop case studies available", ha="center", va="center")
        ax.axis("off")
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return []
    fig, axes = plt.subplots(len(rows), 1, figsize=(10, max(4, 3.2 * len(rows))))
    if len(rows) == 1:
        axes = [axes]
    summary_rows = []
    for ax, row in zip(axes, rows):
        case = dict(row["case"])
        profiles = list(row["profiles"])
        ax.set_facecolor(PLOT_BACKGROUND)
        history_xy = [(float(frame["ego"]["x"]), float(frame["ego"]["y"])) for frame in case.get("frames", []) if float(frame.get("dt_from_anchor_s", 0.0)) < 0.0]
        future_xy = [(float(frame["ego"]["x"]), float(frame["ego"]["y"])) for frame in case.get("future_frames", [])]
        actor_xy = [
            (float(frame["primary_actor"]["x"]), float(frame["primary_actor"]["y"]))
            for frame in case.get("future_frames", [])
            if frame.get("primary_actor")
        ]
        _plot_xy(ax, history_xy, HISTORY_COLOR, "history", linestyle="--")
        _plot_xy(ax, future_xy, LOGGED_EGO_COLOR, "logged future")
        _plot_xy(ax, actor_xy, ACTOR_COLOR, "risk actor")
        color_map = {
            str(profile["profile_name"]): PROFILE_COLORS[idx % len(PROFILE_COLORS)]
            for idx, profile in enumerate(profile for profile in profiles if str(profile["profile_name"]) != "logged_ego_oracle")
        }
        for profile in profiles:
            name = str(profile["profile_name"])
            if name == "logged_ego_oracle":
                continue
            rollout_xy = [(float(state["x"]), float(state["y"])) for state in profile["rollout"].get("states", [])]
            _plot_xy(ax, rollout_xy, color_map.get(name, PROFILE_COLORS[0]), name)
        title = "{0} | {1} | {2}".format(
            case.get("case_id", ""),
            case.get("scenario_family", ""),
            case.get("difficulty_label", ""),
        )
        ax.set_title(title, loc="left", fontsize=9)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=7)
        summary_rows.append(
            {
                "case_id": case.get("case_id", ""),
                "scenario_family": case.get("scenario_family", ""),
                "difficulty_label": case.get("difficulty_label", ""),
                "profiles": [
                    {
                        "profile_name": str(profile["profile_name"]),
                        "closed_loop_score": float(profile["metric"].get("closed_loop_score") or 0.0),
                        "ego_ade_m": profile["metric"].get("ego_ade_m"),
                    }
                    for profile in profiles
                ],
            }
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return summary_rows


def _plot_xy(ax: Any, xy: Sequence[Tuple[float, float]], color: str, label: str, linestyle: str = "-") -> None:
    if not xy:
        return
    xs = [x for x, _ in xy]
    ys = [y for _, y in xy]
    ax.plot(xs, ys, color=color, linewidth=2.0, linestyle=linestyle, label=label)
    ax.scatter(xs[-1:], ys[-1:], color=color, s=20)


def _render_case_study_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# nuPlan Closed-Loop Case Studies",
        "",
        f"- Cases: `{payload.get('case_count', 0)}`",
        f"- Figure: `{payload.get('figure_path', '')}`",
        "",
        "| Case | Family | Difficulty | Profile Summary |",
        "| --- | --- | --- | --- |",
    ]
    for case in payload.get("cases", []):
        profile_summary = "; ".join(
            "{0}: score={1:.3f}, ade={2}".format(
                profile.get("profile_name", ""),
                float(profile.get("closed_loop_score") or 0.0),
                _format_optional_float(profile.get("ego_ade_m")),
            )
            for profile in case.get("profiles", [])
        )
        lines.append(
            f"| `{case.get('case_id', '')}` | `{case.get('scenario_family', '')}` | "
            f"`{case.get('difficulty_label', '')}` | {profile_summary} |"
        )
    lines.append("")
    return "\n".join(lines)


def _case_study_priority(row: Mapping[str, Any]) -> float:
    scores = [
        float(profile.get("metric", {}).get("closed_loop_score") or 0.0)
        for profile in row.get("profiles", [])
        if str(profile.get("profile_name") or "") != "logged_ego_oracle"
    ]
    return 1.0 - min(scores or [1.0])


def _breakdown(rows: Sequence[Mapping[str, Any]], key: str) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key) or "unknown")].append(row)
    result = []
    for value, group in sorted(grouped.items()):
        scores = _finite_values(group, "closed_loop_score")
        ades = _finite_values(group, "ego_ade_m")
        result.append(
            {
                key: value,
                "case_count": len(group),
                "mean_closed_loop_score": mean(scores) if scores else 0.0,
                "mean_ego_ade_m": mean(ades) if ades else None,
            }
        )
    return result


def _finite_values(rows: Sequence[Mapping[str, Any]], key: str) -> List[float]:
    values = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        value_f = float(value)
        if math.isfinite(value_f):
            values.append(value_f)
    return values


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _format_optional_float(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


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


def _linear_ttc(rel_x: float, rel_y: float, rel_vx: float, rel_vy: float) -> Optional[float]:
    speed_sq = rel_vx * rel_vx + rel_vy * rel_vy
    if speed_sq < 1e-6:
        return None
    ttc = -((rel_x * rel_vx + rel_y * rel_vy) / speed_sq)
    if ttc <= 0.0 or ttc > 20.0:
        return None
    return ttc


def _frame_pair_speed(prev: Mapping[str, Any], curr: Mapping[str, Any]) -> float:
    dt = max((int(curr["timestamp_us"]) - int(prev["timestamp_us"])) / 1_000_000.0, 1e-6)
    prev_ego = prev["ego"]
    curr_ego = curr["ego"]
    return math.hypot(float(curr_ego["x"]) - float(prev_ego["x"]), float(curr_ego["y"]) - float(prev_ego["y"])) / dt


def _longitudinal_acceleration(ego: Mapping[str, Any]) -> float:
    yaw = float(ego.get("yaw", 0.0))
    ax = float(ego.get("acceleration_x", 0.0))
    ay = float(ego.get("acceleration_y", 0.0))
    return ax * math.cos(yaw) + ay * math.sin(yaw)


def _angle_diff(a: float, b: float) -> float:
    return math.atan2(math.sin(a - b), math.cos(a - b))


def _normalize_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


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
            "<head><meta charset=\"utf-8\"><title>nuPlan Closed-Loop Replay</title></head>",
            "<body>",
            *escaped_lines,
            "</body></html>",
        ]
    )

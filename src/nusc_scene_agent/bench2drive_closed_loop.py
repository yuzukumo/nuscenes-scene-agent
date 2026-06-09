from __future__ import annotations

import csv
import html
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from nusc_scene_agent.bench2drive_e2e import (
    DEFAULT_BENCH2DRIVE_OUTPUT,
    DEFAULT_BENCH2DRIVE_TENSOR_MANIFEST,
    VisionE2EModelConfig,
    _Bench2DriveVisionDataset,
    _batch_to_device,
    _build_vision_e2e_model,
    _read_manifest_rows,
    _require_torch_stack,
)
from nusc_scene_agent.carla_closed_loop import PurePursuitConfig, pure_pursuit_control
from nusc_scene_agent.geometry import normalize_angle


DEFAULT_BENCH2DRIVE_CLOSED_LOOP_OUTPUT = Path("outputs/bench2drive_vision_closed_loop_trajectory_transformer_final")
BENCH2DRIVE_CLOSED_LOOP_SCHEMA = "bench2drive_vision_closed_loop_v1"

MAX_STEER_RAD = 0.65
WHEEL_BASE_M = 2.85
MAX_ACCEL_MPS2 = 3.0
MAX_BRAKE_MPS2 = 5.0


@dataclass(frozen=True)
class ClosedLoopControlConfig:
    dt_s: float = 0.5
    horizon_s: float = 10.0
    target_speed_mps: float = 5.5
    min_target_speed_mps: float = 1.0
    brake_probability_threshold: float = 0.85
    lookahead_m: float = 9.0
    speed_kp: float = 0.45
    max_steer_rad: float = MAX_STEER_RAD
    max_accel_mps2: float = MAX_ACCEL_MPS2
    max_brake_mps2: float = MAX_BRAKE_MPS2
    wheel_base_m: float = WHEEL_BASE_M


def run_bench2drive_vision_closed_loop(
    manifest_path: Path = DEFAULT_BENCH2DRIVE_TENSOR_MANIFEST,
    checkpoint_path: Path = DEFAULT_BENCH2DRIVE_OUTPUT / "vision_e2e_planner_best.pt",
    output_dir: Path = DEFAULT_BENCH2DRIVE_CLOSED_LOOP_OUTPUT,
    *,
    split: str = "val",
    max_cases: int = 64,
    max_frames_per_clip: int = 20,
    image_size: int = 160,
    device: str = "",
    video_fps: int = 6,
    case_selection: str = "balanced",
    control_config: Optional[ClosedLoopControlConfig] = None,
) -> Dict[str, Any]:
    torch, _, _ = _require_torch_stack(require_pillow=True)
    started_at = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _load_closed_loop_rows(Path(manifest_path), split=split)
    cases = _select_closed_loop_cases(
        rows,
        max_cases=max_cases,
        max_frames_per_clip=max_frames_per_clip,
        case_selection=case_selection,
    )
    if not cases:
        raise ValueError(f"No closed-loop cases found in {manifest_path} for split={split!r}")

    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    model_config = VisionE2EModelConfig(**dict(checkpoint.get("model_config") or {}))
    model = _build_vision_e2e_model(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(target_device)
    model.eval()

    config = control_config or ClosedLoopControlConfig()
    case_reports = []
    for case_rows in cases:
        report = _run_closed_loop_case(
            torch=torch,
            model=model,
            case_rows=case_rows,
            device=target_device,
            image_size=image_size,
            config=config,
        )
        case_dir = output_dir / "cases" / report["case_id"]
        media_outputs = _write_case_outputs(report, case_dir, video_fps=video_fps)
        case_reports.append(
            {
                **report["metrics"],
                "case_id": report["case_id"],
                "clip_name": report["clip_name"],
                "scenario_family": report["scenario_family"],
                "case_dir": str(case_dir),
                "figure_path": str(case_dir / "closed_loop_rollout.png"),
                "video_path": media_outputs.get("video_path", ""),
                "gif_path": media_outputs.get("gif_path", ""),
                "media_path": media_outputs.get("media_path", ""),
            }
        )

    comparison = _build_closed_loop_comparison(case_reports)
    _write_closed_loop_outputs(
        output_dir=output_dir,
        manifest_path=Path(manifest_path),
        checkpoint_path=Path(checkpoint_path),
        split=split,
        image_size=image_size,
        device=str(target_device),
        config=config,
        case_reports=case_reports,
        comparison=comparison,
        runtime_s=time.time() - started_at,
    )
    return {
        "schema": BENCH2DRIVE_CLOSED_LOOP_SCHEMA,
        "output_dir": str(output_dir),
        "manifest_path": str(manifest_path),
        "checkpoint_path": str(checkpoint_path),
        "split": split,
        "case_selection": case_selection,
        "case_count": len(case_reports),
        "comparison": comparison,
        "cases": case_reports,
    }


def _load_closed_loop_rows(manifest_path: Path, *, split: str) -> List[Dict[str, Any]]:
    rows = [dict(row) for row in _read_manifest_rows(manifest_path)]
    if split != "all":
        rows = [row for row in rows if str(row.get("split") or "") == split]
    rows = [
        row
        for row in rows
        if row.get("future_waypoints_ego")
        and row.get("route_features")
        and (row.get("tensor_cache_path") or row.get("camera_cache_paths") or row.get("archive"))
    ]
    rows.sort(key=lambda row: (str(row.get("clip_name") or ""), int(row.get("frame_id") or 0)))
    return rows


def _select_closed_loop_cases(
    rows: Sequence[Mapping[str, Any]],
    *,
    max_cases: int,
    max_frames_per_clip: int,
    case_selection: str,
) -> List[List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("clip_name") or ""), []).append(dict(row))
    candidates = [
        clip_rows[: max(int(max_frames_per_clip), 2)]
        for _, clip_rows in sorted(grouped.items())
        if len(clip_rows) >= 2
    ]
    if str(case_selection) == "qualitative":
        candidates.sort(key=lambda clip_rows: _qualitative_selection_priority(clip_rows), reverse=True)
    elif str(case_selection) == "stress":
        candidates.sort(key=lambda clip_rows: _stress_selection_priority(clip_rows), reverse=True)
    else:
        candidates.sort(key=lambda clip_rows: _balanced_selection_priority(clip_rows), reverse=True)
    if max_cases > 0:
        candidates = candidates[: int(max_cases)]
    return candidates


def _balanced_selection_priority(rows: Sequence[Mapping[str, Any]]) -> float:
    object_count = mean(float(row.get("object_count") or 0.0) for row in rows)
    brake_count = sum(1 for row in rows if bool(row.get("should_brake")))
    family = str(rows[0].get("scenario_family") or "")
    family_bonus = 2.0 if any(token in family for token in ["Accident", "Hazard", "Pedestrian", "Blocked"]) else 0.0
    return object_count + 0.75 * brake_count + family_bonus


def _stress_selection_priority(rows: Sequence[Mapping[str, Any]]) -> float:
    return _balanced_selection_priority(rows) + 0.5 * len(rows)


def _qualitative_selection_priority(rows: Sequence[Mapping[str, Any]]) -> float:
    speed_values = [max(float(dict(row.get("ego_state") or {}).get("speed") or 0.0), 0.0) for row in rows]
    mean_speed = mean(speed_values) if speed_values else 0.0
    speed_stability = -mean(abs(value - mean_speed) for value in speed_values) if speed_values else 0.0
    brake_count = sum(1 for row in rows if bool(row.get("should_brake")))
    route_extent = _logged_route_extent(rows)
    return route_extent + 0.35 * mean_speed + 0.1 * speed_stability - 2.0 * brake_count


def _logged_route_extent(rows: Sequence[Mapping[str, Any]]) -> float:
    if len(rows) < 2:
        return 0.0
    first = dict(rows[0])
    states = [_logged_state_from_row(row, first) for row in rows]
    return _polyline_length(states)


def _run_closed_loop_case(
    *,
    torch: Any,
    model: Any,
    case_rows: Sequence[Mapping[str, Any]],
    device: Any,
    image_size: int,
    config: ClosedLoopControlConfig,
) -> Dict[str, Any]:
    first = dict(case_rows[0])
    initial_speed = float(dict(first.get("ego_state") or {}).get("speed") or 0.0)
    state = {
        "x": 0.0,
        "y": 0.0,
        "yaw": 0.0,
        "speed_mps": max(initial_speed, 0.0),
        "acceleration_mps2": 0.0,
    }
    route_global = _logged_route_from_rows(case_rows)
    closed_loop_states = []
    logged_states = []
    prediction_rows = []
    dataset = _Bench2DriveVisionDataset(case_rows, image_size=int(image_size))
    previous_accel = 0.0
    max_steps = min(len(case_rows), max(int(config.horizon_s / max(config.dt_s, 1e-6)), 1))
    with torch.no_grad():
        for step_idx in range(max_steps):
            row = dict(case_rows[step_idx])
            sample = dataset[step_idx]
            batch = {
                "images": sample["images"].unsqueeze(0),
                "route": sample["route"].unsqueeze(0),
                "future": sample["future"].unsqueeze(0),
                "control": sample["control"].unsqueeze(0),
                "brake": sample["brake"].unsqueeze(0),
            }
            images, route, _, _, _ = _batch_to_device(batch, device)
            prediction = model(images, route)
            pred_waypoints = prediction["future"].detach().cpu().reshape(-1, 2).tolist()
            pred_control = prediction["control"].detach().cpu().reshape(-1).tolist()
            brake_prob = float(torch.sigmoid(prediction["brake_logits"]).detach().cpu().reshape(-1)[0])
            target_speed = _target_speed_from_prediction(pred_waypoints, brake_prob, config)
            control = _control_from_predicted_waypoints(
                state,
                pred_waypoints,
                target_speed,
                config,
                predicted_control=pred_control,
                brake_probability=brake_prob,
            )
            new_state = _integrate_closed_loop_state(state, control, dt_s=config.dt_s, config=config)
            yaw_rate = normalize_angle(new_state["yaw"] - state["yaw"]) / max(config.dt_s, 1e-6)
            jerk = (new_state["acceleration_mps2"] - previous_accel) / max(config.dt_s, 1e-6)
            previous_accel = new_state["acceleration_mps2"]
            state = new_state
            logged = _logged_state_from_row(row, first)
            logged_states.append(logged)
            closed_loop_states.append(
                {
                    "step": step_idx,
                    "t_s": step_idx * config.dt_s,
                    **state,
                    "yaw_rate_rps": yaw_rate,
                    "jerk_mps3": jerk,
                    "steer": float(control["steer"]),
                    "throttle": float(control["throttle"]),
                    "brake": float(control["brake"]),
                    "target_speed_mps": float(target_speed),
                    "brake_probability": brake_prob,
                }
            )
            prediction_rows.append(
                {
                    "step": step_idx,
                    "frame_id": int(row.get("frame_id") or 0),
                    "predicted_waypoints_ego": pred_waypoints,
                    "logged_future_waypoints_ego": row.get("future_waypoints_ego") or [],
                    "brake_probability": brake_prob,
                    "predicted_control": pred_control,
                    "control": control,
                }
            )
    metrics = _evaluate_closed_loop_rollout(
        closed_loop_states=closed_loop_states,
        logged_states=logged_states,
        route_global=route_global,
    )
    return {
        "schema": "bench2drive_vision_closed_loop_case_v1",
        "case_id": _safe_case_id(first),
        "clip_name": str(first.get("clip_name") or ""),
        "scenario_family": str(first.get("scenario_family") or ""),
        "frame_count": len(closed_loop_states),
        "control_config": dict(config.__dict__),
        "logged_route": route_global,
        "logged_states": logged_states,
        "closed_loop_states": closed_loop_states,
        "predictions": prediction_rows,
        "metrics": metrics,
    }


def _logged_route_from_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, float]]:
    if not rows:
        return []
    first = dict(rows[0])
    route = []
    for row in rows:
        route.append(_logged_state_from_row(row, first))
    future = first.get("future_waypoints_ego") or []
    for idx, waypoint in enumerate(future):
        x, y = _bench2drive_local_to_control_xy(float(waypoint[0]), float(waypoint[1]))
        route.append(
            {
                "step": len(route) + idx,
                "x": x,
                "y": y,
                "yaw": 0.0,
                "speed_mps": 0.0,
            }
        )
    return route


def _logged_state_from_row(row: Mapping[str, Any], first_row: Mapping[str, Any]) -> Dict[str, float]:
    ego = dict(row.get("ego_state") or {})
    first_ego = dict(first_row.get("ego_state") or {})
    x0 = float(first_ego.get("x") or 0.0)
    y0 = float(first_ego.get("y") or 0.0)
    theta0 = float(first_ego.get("theta") or 0.0)
    bench_x, bench_y = _global_to_initial_ego(
        x=float(ego.get("x") or 0.0),
        y=float(ego.get("y") or 0.0),
        x0=x0,
        y0=y0,
        theta0=theta0,
    )
    x, y = _bench2drive_local_to_control_xy(bench_x, bench_y)
    yaw = normalize_angle(float(ego.get("theta") or 0.0) - theta0)
    return {
        "x": x,
        "y": y,
        "yaw": yaw,
        "speed_mps": float(ego.get("speed") or 0.0),
    }


def _global_to_initial_ego(*, x: float, y: float, x0: float, y0: float, theta0: float) -> Tuple[float, float]:
    dx = x - x0
    dy = y - y0
    cos_t = math.cos(theta0)
    sin_t = math.sin(theta0)
    return cos_t * dx + sin_t * dy, -sin_t * dx + cos_t * dy


def _bench2drive_local_to_control_xy(x: float, y: float) -> Tuple[float, float]:
    return -float(y), float(x)


def _target_speed_from_prediction(
    pred_waypoints: Sequence[Sequence[float]],
    brake_probability: float,
    config: ClosedLoopControlConfig,
) -> float:
    if brake_probability >= config.brake_probability_threshold:
        return 0.0
    if len(pred_waypoints) >= 2:
        control_points = [_bench2drive_local_to_control_xy(float(point[0]), float(point[1])) for point in pred_waypoints]
        distances = [math.hypot(end[0] - start[0], end[1] - start[1]) for start, end in zip(control_points[:-1], control_points[1:])]
        if distances:
            speed = mean(distances) / max(config.dt_s, 1e-6)
            blended_speed = max(speed, 0.85 * config.target_speed_mps)
            return max(min(blended_speed, config.target_speed_mps), config.min_target_speed_mps)
    return config.target_speed_mps


def _control_from_predicted_waypoints(
    state: Mapping[str, Any],
    pred_waypoints: Sequence[Sequence[float]],
    target_speed_mps: float,
    config: ClosedLoopControlConfig,
    *,
    predicted_control: Optional[Sequence[float]] = None,
    brake_probability: float = 0.0,
) -> Dict[str, float]:
    if not pred_waypoints:
        return {"steer": 0.0, "throttle": 0.0, "brake": 1.0, "target_speed_mps": 0.0}
    route = _predicted_waypoints_to_route(state, pred_waypoints)
    if len(route) < 2 or _polyline_length(route) < 0.5:
        return {"steer": 0.0, "throttle": 0.0, "brake": 1.0, "target_speed_mps": 0.0}
    controller_config = PurePursuitConfig(
        lookahead_m=config.lookahead_m,
        speed_kp=config.speed_kp,
        max_steer_rad=config.max_steer_rad,
        max_accel_mps2=config.max_accel_mps2,
        max_brake_mps2=config.max_brake_mps2,
        wheel_base_m=config.wheel_base_m,
    )
    control = pure_pursuit_control(
        state=state,
        route=route,
        target_speed_mps=target_speed_mps,
        config=controller_config,
    )
    if predicted_control is not None and len(predicted_control) >= 3:
        model_steer = max(min(float(predicted_control[0]), 1.0), -1.0)
        model_throttle = max(min(float(predicted_control[1]), 1.0), 0.0)
        model_brake = max(min(float(predicted_control[2]), 1.0), 0.0)
        strong_brake = brake_probability >= config.brake_probability_threshold and model_brake >= 0.25
        if strong_brake:
            model_throttle = 0.0
            model_brake = max(model_brake, brake_probability)
        elif model_brake < 0.25:
            model_brake = 0.0
        control["steer"] = 0.75 * float(control["steer"]) + 0.25 * model_steer
        control["throttle"] = 0.65 * float(control["throttle"]) + 0.35 * model_throttle
        control["brake"] = max(0.65 * float(control["brake"]) + 0.35 * model_brake, 0.0)
        if control["brake"] > 0.2:
            control["throttle"] *= max(0.0, 1.0 - control["brake"])
    return control


def _predicted_waypoints_to_route(
    state: Mapping[str, Any],
    pred_waypoints: Sequence[Sequence[float]],
) -> List[Dict[str, float]]:
    route = [{"x": float(state["x"]), "y": float(state["y"])}]
    yaw = float(state["yaw"])
    cos_t = math.cos(yaw)
    sin_t = math.sin(yaw)
    for waypoint in pred_waypoints:
        local_x, local_y = _bench2drive_local_to_control_xy(float(waypoint[0]), float(waypoint[1]))
        route.append(
            {
                "x": float(state["x"]) + cos_t * local_x - sin_t * local_y,
                "y": float(state["y"]) + sin_t * local_x + cos_t * local_y,
            }
        )
    return route


def _integrate_closed_loop_state(
    state: Mapping[str, Any],
    control: Mapping[str, Any],
    *,
    dt_s: float,
    config: ClosedLoopControlConfig,
) -> Dict[str, float]:
    steer_rad = float(control.get("steer") or 0.0) * config.max_steer_rad
    acceleration = (
        float(control.get("throttle") or 0.0) * config.max_accel_mps2
        - float(control.get("brake") or 0.0) * config.max_brake_mps2
    )
    speed = max(float(state.get("speed_mps") or 0.0) + acceleration * dt_s, 0.0)
    yaw_rate = speed / max(config.wheel_base_m, 1e-6) * math.tan(steer_rad)
    yaw = normalize_angle(float(state.get("yaw") or 0.0) + yaw_rate * dt_s)
    x = float(state.get("x") or 0.0) + speed * math.cos(yaw) * dt_s
    y = float(state.get("y") or 0.0) + speed * math.sin(yaw) * dt_s
    return {
        "x": x,
        "y": y,
        "yaw": yaw,
        "speed_mps": speed,
        "acceleration_mps2": acceleration,
    }


def _evaluate_closed_loop_rollout(
    *,
    closed_loop_states: Sequence[Mapping[str, Any]],
    logged_states: Sequence[Mapping[str, Any]],
    route_global: Sequence[Mapping[str, Any]],
) -> Dict[str, float]:
    count = min(len(closed_loop_states), len(logged_states))
    if count <= 0:
        return {
            "frame_count": 0.0,
            "closed_loop_ade_m": 0.0,
            "closed_loop_fde_m": 0.0,
            "mean_lateral_error_m": 0.0,
            "max_lateral_error_m": 0.0,
            "route_completion": 0.0,
            "comfort_violation_count": 0.0,
            "closed_loop_score": 0.0,
        }
    distances = [
        math.hypot(
            float(closed_loop_states[idx]["x"]) - float(logged_states[idx]["x"]),
            float(closed_loop_states[idx]["y"]) - float(logged_states[idx]["y"]),
        )
        for idx in range(count)
    ]
    lateral_errors = [_distance_to_route(state, route_global) for state in closed_loop_states]
    completion = _route_completion(closed_loop_states, route_global)
    comfort_violations = sum(
        1
        for state in closed_loop_states
        if abs(float(state.get("acceleration_mps2") or 0.0)) > 5.5
        or abs(float(state.get("jerk_mps3") or 0.0)) > 14.0
        or abs(float(state.get("yaw_rate_rps") or 0.0)) > 1.2
    )
    ade = mean(distances)
    fde = distances[-1]
    mean_lat = mean(lateral_errors) if lateral_errors else 0.0
    max_lat = max(lateral_errors) if lateral_errors else 0.0
    score = _closed_loop_score(
        completion=completion,
        ade=ade,
        fde=fde,
        max_lateral_error=max_lat,
        comfort_violation_count=comfort_violations,
    )
    return {
        "frame_count": float(count),
        "closed_loop_ade_m": ade,
        "closed_loop_fde_m": fde,
        "mean_lateral_error_m": mean_lat,
        "max_lateral_error_m": max_lat,
        "route_completion": completion,
        "comfort_violation_count": float(comfort_violations),
        "closed_loop_score": score,
    }


def _distance_to_route(state: Mapping[str, Any], route: Sequence[Mapping[str, Any]]) -> float:
    if len(route) < 2:
        return 0.0
    px = float(state.get("x") or 0.0)
    py = float(state.get("y") or 0.0)
    best = float("inf")
    for start, end in zip(route[:-1], route[1:]):
        sx, sy = float(start["x"]), float(start["y"])
        ex, ey = float(end["x"]), float(end["y"])
        vx, vy = ex - sx, ey - sy
        denom = vx * vx + vy * vy
        if denom <= 1e-9:
            continue
        t = max(min(((px - sx) * vx + (py - sy) * vy) / denom, 1.0), 0.0)
        proj_x = sx + t * vx
        proj_y = sy + t * vy
        best = min(best, math.hypot(px - proj_x, py - proj_y))
    return best if math.isfinite(best) else 0.0


def _route_completion(states: Sequence[Mapping[str, Any]], route: Sequence[Mapping[str, Any]]) -> float:
    if len(route) < 2 or not states:
        return 0.0
    route_length = _polyline_length(route)
    if route_length <= 1e-6:
        return 0.0
    progress = max(_project_progress(state, route) for state in states)
    return max(min(progress / route_length, 1.0), 0.0)


def _project_progress(state: Mapping[str, Any], route: Sequence[Mapping[str, Any]]) -> float:
    px = float(state.get("x") or 0.0)
    py = float(state.get("y") or 0.0)
    best_distance = float("inf")
    best_progress = 0.0
    cumulative = 0.0
    for start, end in zip(route[:-1], route[1:]):
        sx, sy = float(start["x"]), float(start["y"])
        ex, ey = float(end["x"]), float(end["y"])
        vx, vy = ex - sx, ey - sy
        seg_len = math.hypot(vx, vy)
        denom = seg_len * seg_len
        if denom <= 1e-9:
            continue
        t = max(min(((px - sx) * vx + (py - sy) * vy) / denom, 1.0), 0.0)
        proj_x = sx + t * vx
        proj_y = sy + t * vy
        distance = math.hypot(px - proj_x, py - proj_y)
        if distance < best_distance:
            best_distance = distance
            best_progress = cumulative + t * seg_len
        cumulative += seg_len
    return best_progress


def _polyline_length(route: Sequence[Mapping[str, Any]]) -> float:
    return sum(
        math.hypot(float(end["x"]) - float(start["x"]), float(end["y"]) - float(start["y"]))
        for start, end in zip(route[:-1], route[1:])
    )


def _closed_loop_score(
    *,
    completion: float,
    ade: float,
    fde: float,
    max_lateral_error: float,
    comfort_violation_count: int,
) -> float:
    tracking_penalty = math.exp(-0.08 * ade - 0.04 * fde)
    lateral_penalty = max(0.0, 1.0 - max(0.0, max_lateral_error - 2.0) / 6.0)
    comfort_penalty = max(0.5, 1.0 - 0.02 * comfort_violation_count)
    return max(min(completion * tracking_penalty * lateral_penalty * comfort_penalty, 1.0), 0.0)


def _build_closed_loop_comparison(case_reports: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    metric_keys = [
        "closed_loop_ade_m",
        "closed_loop_fde_m",
        "mean_lateral_error_m",
        "max_lateral_error_m",
        "route_completion",
        "closed_loop_score",
    ]
    return {
        "case_count": len(case_reports),
        "metrics": {
            f"mean_{key}": _safe_mean(case_reports, key)
            for key in metric_keys
        },
        "scenario_family_breakdown": _scenario_family_breakdown(case_reports),
    }


def _scenario_family_breakdown(case_reports: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for report in case_reports:
        grouped.setdefault(str(report.get("scenario_family") or "unknown"), []).append(report)
    return {
        family: {
            "case_count": len(rows),
            "mean_closed_loop_ade_m": _safe_mean(rows, "closed_loop_ade_m"),
            "mean_closed_loop_score": _safe_mean(rows, "closed_loop_score"),
            "mean_route_completion": _safe_mean(rows, "route_completion"),
        }
        for family, rows in sorted(grouped.items())
    }


def _write_case_outputs(report: Mapping[str, Any], output_dir: Path, *, video_fps: int) -> Dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "closed_loop_case.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_case_csv(report, output_dir / "closed_loop_states.csv")
    _render_case_rollout_figure(report, output_dir / "closed_loop_rollout.png")
    return _render_case_rollout_video(report, output_dir / "closed_loop_rollout.mp4", fps=video_fps)


def _write_case_csv(report: Mapping[str, Any], output_path: Path) -> None:
    fieldnames = [
        "step",
        "t_s",
        "x",
        "y",
        "yaw",
        "speed_mps",
        "steer",
        "throttle",
        "brake",
        "target_speed_mps",
        "brake_probability",
    ]
    with Path(output_path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in report.get("closed_loop_states", []):
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _render_case_rollout_figure(report: Mapping[str, Any], output_path: Path) -> None:
    route = list(report.get("logged_route") or [])
    logged = list(report.get("logged_states") or [])
    rollout = list(report.get("closed_loop_states") or [])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=160)
    ax = axes[0]
    if route:
        ax.plot([p["x"] for p in route], [p["y"] for p in route], color="#404040", lw=2.0, label="logged route")
    if logged:
        ax.plot([p["x"] for p in logged], [p["y"] for p in logged], color="#3b6ea8", lw=1.8, label="logged ego")
    if rollout:
        ax.plot([p["x"] for p in rollout], [p["y"] for p in rollout], color="#c45c2c", lw=1.8, label="closed-loop model")
        ax.scatter([rollout[0]["x"]], [rollout[0]["y"]], s=32, color="#2a9d8f", label="start")
    ax.set_title(f"{report.get('case_id', '')}")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1]
    if rollout:
        steps = [row["step"] for row in rollout]
        ax.plot(steps, [row["speed_mps"] for row in rollout], label="speed")
        ax.plot(steps, [row["brake_probability"] for row in rollout], label="brake probability")
        ax.plot(steps, [row["steer"] for row in rollout], label="steer")
    ax.set_title("Closed-loop signals")
    ax.set_xlabel("step")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _render_case_rollout_video(report: Mapping[str, Any], output_path: Path, *, fps: int) -> Dict[str, str]:
    output_path = Path(output_path)
    try:
        import imageio.v2 as imageio
    except Exception:
        gif_path = output_path.with_suffix(".gif")
        _render_case_rollout_gif(report, gif_path, fps=fps)
        return _media_result(output_path, gif_path)
    frames = _render_case_video_frames(report)
    if not frames:
        return _media_result(output_path, output_path.with_suffix(".gif"))
    try:
        imageio.mimsave(str(output_path), frames, fps=max(int(fps), 1), macro_block_size=8)
    except Exception:
        gif_path = output_path.with_suffix(".gif")
        _render_case_rollout_gif(report, gif_path, fps=fps)
    return _media_result(output_path, output_path.with_suffix(".gif"))


def _render_case_rollout_gif(report: Mapping[str, Any], output_path: Path, *, fps: int) -> None:
    output_path = Path(output_path)
    frames = _render_case_video_frames(report)
    if not frames:
        return
    try:
        import imageio.v2 as imageio
    except Exception:
        _save_frames_as_gif_with_pillow(frames, output_path, fps=fps)
        return
    imageio.mimsave(str(output_path), frames, duration=1.0 / max(int(fps), 1))


def _save_frames_as_gif_with_pillow(frames: Sequence[Any], output_path: Path, *, fps: int) -> None:
    from PIL import Image

    pil_frames = [Image.fromarray(frame) for frame in frames]
    if not pil_frames:
        return
    duration_ms = int(round(1000.0 / max(int(fps), 1)))
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )


def _media_result(video_path: Path, gif_path: Path) -> Dict[str, str]:
    video_path = Path(video_path)
    gif_path = Path(gif_path)
    media_path = video_path if video_path.exists() else gif_path if gif_path.exists() else Path("")
    return {
        "video_path": str(video_path) if video_path.exists() else "",
        "gif_path": str(gif_path) if gif_path.exists() else "",
        "media_path": str(media_path) if str(media_path) else "",
    }


def _render_case_video_frames(report: Mapping[str, Any]) -> List[Any]:
    frames = []
    route = list(report.get("logged_route") or [])
    logged = list(report.get("logged_states") or [])
    rollout = list(report.get("closed_loop_states") or [])
    if not rollout:
        return frames
    x_values = [float(p["x"]) for p in route + logged + rollout]
    y_values = [float(p["y"]) for p in route + logged + rollout]
    x_min, x_max = min(x_values) - 4.0, max(x_values) + 4.0
    y_min, y_max = min(y_values) - 4.0, max(y_values) + 4.0
    for idx in range(len(rollout)):
        fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
        if route:
            ax.plot([p["x"] for p in route], [p["y"] for p in route], color="#666666", lw=1.5, label="logged route")
        if logged:
            ax.plot([p["x"] for p in logged[: idx + 1]], [p["y"] for p in logged[: idx + 1]], color="#3b6ea8", lw=1.6, label="logged")
        ax.plot([p["x"] for p in rollout[: idx + 1]], [p["y"] for p in rollout[: idx + 1]], color="#c45c2c", lw=1.8, label="model")
        ax.scatter([rollout[idx]["x"]], [rollout[idx]["y"]], color="#c45c2c", s=36)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.25)
        ax.set_title(f"{report.get('case_id', '')} step {idx}")
        ax.legend(fontsize=7, loc="best")
        fig.tight_layout()
        frames.append(_figure_to_array(fig))
        plt.close(fig)
    return frames


def _figure_to_array(fig: Any) -> Any:
    import numpy as np

    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    if hasattr(fig.canvas, "tostring_rgb"):
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        return data.reshape((height, width, 3))
    data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    return data.reshape((height, width, 4))[:, :, :3].copy()


def _write_closed_loop_outputs(
    *,
    output_dir: Path,
    manifest_path: Path,
    checkpoint_path: Path,
    split: str,
    image_size: int,
    device: str,
    config: ClosedLoopControlConfig,
    case_reports: Sequence[Mapping[str, Any]],
    comparison: Mapping[str, Any],
    runtime_s: float,
) -> None:
    payload = {
        "schema": BENCH2DRIVE_CLOSED_LOOP_SCHEMA,
        "manifest_path": str(manifest_path),
        "checkpoint_path": str(checkpoint_path),
        "output_dir": str(output_dir),
        "split": split,
        "image_size": int(image_size),
        "device": device,
        "control_config": dict(config.__dict__),
        "case_count": len(case_reports),
        "comparison": comparison,
        "cases": list(case_reports),
        "runtime_s": round(float(runtime_s), 3),
    }
    (output_dir / "closed_loop_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_summary_csv(case_reports, output_dir / "closed_loop_case_metrics.csv")
    markdown = _render_closed_loop_markdown(payload)
    (output_dir / "closed_loop_report.md").write_text(markdown, encoding="utf-8")
    (output_dir / "closed_loop_report.html").write_text(_markdown_to_basic_html(markdown), encoding="utf-8")
    _render_overview_figure(case_reports, output_dir / "closed_loop_overview.png")


def _write_summary_csv(rows: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    fieldnames = [
        "case_id",
        "clip_name",
        "scenario_family",
        "frame_count",
        "closed_loop_ade_m",
        "closed_loop_fde_m",
        "mean_lateral_error_m",
        "max_lateral_error_m",
        "route_completion",
        "comfort_violation_count",
        "closed_loop_score",
        "case_dir",
        "video_path",
        "gif_path",
        "media_path",
    ]
    with Path(output_path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _render_closed_loop_markdown(payload: Mapping[str, Any]) -> str:
    comparison = dict(payload.get("comparison") or {})
    metrics = dict(comparison.get("metrics") or {})
    lines = [
        "# Bench2Drive Vision Closed-Loop Evaluation",
        "",
        f"- Cases: `{payload.get('case_count', 0)}`",
        f"- Split: `{payload.get('split', '')}`",
        f"- Checkpoint: `{payload.get('checkpoint_path', '')}`",
        f"- Runtime: `{payload.get('runtime_s', 0.0)}` seconds",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for key in [
        "mean_closed_loop_ade_m",
        "mean_closed_loop_fde_m",
        "mean_mean_lateral_error_m",
        "mean_route_completion",
        "mean_closed_loop_score",
    ]:
        lines.append(f"| `{key}` | `{_format_float(metrics.get(key))}` |")
    lines.extend(["", "| Case | Scenario | ADE | Completion | Score | Media |", "| --- | --- | ---: | ---: | ---: | --- |"])
    for row in payload.get("cases", []):
        lines.append(
            "| `{0}` | `{1}` | `{2}` | `{3}` | `{4}` | `{5}` |".format(
                row.get("case_id", ""),
                row.get("scenario_family", ""),
                _format_float(row.get("closed_loop_ade_m")),
                _format_float(row.get("route_completion")),
                _format_float(row.get("closed_loop_score")),
                row.get("media_path") or row.get("video_path") or row.get("gif_path") or "",
            )
        )
    lines.append("")
    return "\n".join(lines)


def _render_overview_figure(rows: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    if not rows:
        return
    labels = [str(row.get("case_id") or "")[-24:] for row in rows]
    ade = [float(row.get("closed_loop_ade_m") or 0.0) for row in rows]
    score = [float(row.get("closed_loop_score") or 0.0) for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=160)
    axes[0].barh(labels, ade, color="#3b6ea8")
    axes[0].set_title("Closed-loop ADE")
    axes[0].set_xlabel("meters")
    axes[0].grid(axis="x", alpha=0.25)
    axes[1].barh(labels, score, color="#c45c2c")
    axes[1].set_title("Closed-loop score")
    axes[1].set_xlim(0.0, 1.0)
    axes[1].grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _safe_case_id(row: Mapping[str, Any]) -> str:
    clip = str(row.get("clip_name") or "case")
    frame = int(row.get("frame_id") or 0)
    safe_clip = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in clip)
    return f"{safe_clip}_{frame:05d}"


def _safe_mean(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return mean(values) if values else 0.0


def _format_float(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


def _markdown_to_basic_html(markdown: str) -> str:
    lines = ["<!doctype html>", "<html><body>"]
    for line in markdown.splitlines():
        if line.startswith("# "):
            lines.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("|"):
            lines.append(f"<pre>{html.escape(line)}</pre>")
        elif line.startswith("- "):
            lines.append(f"<p>{html.escape(line)}</p>")
        elif not line:
            lines.append("<br>")
        else:
            lines.append(f"<p>{html.escape(line)}</p>")
    lines.append("</body></html>")
    return "\n".join(lines)

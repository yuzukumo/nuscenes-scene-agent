from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Mapping, Optional, Sequence


CARLA_VIDEO_AUDIT_SCHEMA = "carla_vision_video_audit_v1"


def audit_carla_vision_rollouts(
    report_path: Path = Path("outputs/carla_semantic_demo_final/carla_semantic_demo_report.json"),
    output_path: Optional[Path] = None,
    *,
    min_resolution_width: int = 1920,
    min_resolution_height: int = 1080,
    min_fps: float = 8.0,
    min_frames: int = 80,
    min_traffic_manager_vehicles: int = 1,
    max_scripted_vehicles: int = 0,
    max_collision_count: int = 0,
    min_route_completion: float = 0.05,
    max_mean_lateral_error_m: float = 2.5,
    max_lateral_error_m: float = 6.0,
    max_safety_override_ratio: float = 0.35,
    max_nearest_actor_distance_m: float = 60.0,
    nearby_actor_distance_m: float = 30.0,
    min_nearby_actor_ratio: float = 0.30,
    require_semantic_match: bool = False,
    require_model_control: bool = True,
    require_hevc: bool = False,
) -> Dict[str, Any]:
    report_path = Path(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    scenario_reports = _scenario_reports(report)
    scenarios = [
        _audit_scenario(
            scenario,
            min_resolution_width=int(min_resolution_width),
            min_resolution_height=int(min_resolution_height),
            min_fps=float(min_fps),
            min_frames=int(min_frames),
            min_traffic_manager_vehicles=int(min_traffic_manager_vehicles),
            max_scripted_vehicles=int(max_scripted_vehicles),
            max_collision_count=int(max_collision_count),
            min_route_completion=float(min_route_completion),
            max_mean_lateral_error_m=float(max_mean_lateral_error_m),
            max_lateral_error_m=float(max_lateral_error_m),
            max_safety_override_ratio=float(max_safety_override_ratio),
            max_nearest_actor_distance_m=float(max_nearest_actor_distance_m),
            nearby_actor_distance_m=float(nearby_actor_distance_m),
            min_nearby_actor_ratio=float(min_nearby_actor_ratio),
            require_semantic_match=bool(require_semantic_match),
            require_model_control=bool(require_model_control),
            require_hevc=bool(require_hevc),
        )
        for scenario in scenario_reports
    ]
    failures = [item for scenario in scenarios for item in scenario["failures"]]
    warnings = [item for scenario in scenarios for item in scenario["warnings"]]
    payload = {
        "schema": CARLA_VIDEO_AUDIT_SCHEMA,
        "report_path": str(report_path),
        "status": "pass" if not failures else "fail",
        "scenario_count": len(scenarios),
        "failure_count": len(failures),
        "warning_count": len(warnings),
        "failures": failures,
        "warnings": warnings,
        "thresholds": {
            "min_resolution_width": int(min_resolution_width),
            "min_resolution_height": int(min_resolution_height),
            "min_fps": float(min_fps),
            "min_frames": int(min_frames),
            "min_traffic_manager_vehicles": int(min_traffic_manager_vehicles),
            "max_scripted_vehicles": int(max_scripted_vehicles),
            "max_collision_count": int(max_collision_count),
            "min_route_completion": float(min_route_completion),
            "max_mean_lateral_error_m": float(max_mean_lateral_error_m),
            "max_lateral_error_m": float(max_lateral_error_m),
            "max_safety_override_ratio": float(max_safety_override_ratio),
            "max_nearest_actor_distance_m": float(max_nearest_actor_distance_m),
            "nearby_actor_distance_m": float(nearby_actor_distance_m),
            "min_nearby_actor_ratio": float(min_nearby_actor_ratio),
            "require_semantic_match": bool(require_semantic_match),
            "require_model_control": bool(require_model_control),
            "require_hevc": bool(require_hevc),
        },
        "summary": _audit_summary(scenarios),
        "scenarios": scenarios,
    }
    target_path = Path(output_path) if output_path is not None else _default_audit_output_path(report_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    target_path.with_suffix(".md").write_text(_render_audit_markdown(payload), encoding="utf-8")
    return payload


def _scenario_reports(report: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    if str(report.get("schema") or "") == "carla_vision_closed_loop_batch_v1":
        return [dict(item) for item in list(report.get("scenarios") or [])]
    return [dict(report)]


def _default_audit_output_path(report_path: Path) -> Path:
    if Path(report_path).name == "carla_semantic_demo_report.json":
        return Path(report_path).with_name("carla_semantic_demo_audit.json")
    return Path(report_path).with_name("carla_vision_video_audit.json")


def _audit_scenario(
    scenario: Mapping[str, Any],
    *,
    min_resolution_width: int,
    min_resolution_height: int,
    min_fps: float,
    min_frames: int,
    min_traffic_manager_vehicles: int,
    max_scripted_vehicles: int,
    max_collision_count: int,
    min_route_completion: float,
    max_mean_lateral_error_m: float,
    max_lateral_error_m: float,
    max_safety_override_ratio: float,
    max_nearest_actor_distance_m: float,
    nearby_actor_distance_m: float,
    min_nearby_actor_ratio: float,
    require_semantic_match: bool,
    require_model_control: bool,
    require_hevc: bool,
) -> Dict[str, Any]:
    name = str(scenario.get("name") or _nested(scenario, ["scenario", "name"]) or "scenario")
    metrics = dict(scenario.get("metrics") or {})
    attribution = dict(scenario.get("control_attribution") or {})
    media = dict(scenario.get("media") or {})
    video_path = Path(str(media.get("video_path") or ""))
    states_path = Path(str(media.get("states_csv") or ""))
    video = _inspect_video(video_path)
    states = _inspect_states_csv(
        states_path,
        nearby_actor_distance_m=nearby_actor_distance_m,
        visible_actor_distance_m=max_nearest_actor_distance_m,
    )
    failures: List[str] = []
    warnings: List[str] = []
    nearest_actor_distance = float(
        states.get("min_natural_traffic_distance_m")
        if states.get("min_natural_traffic_distance_m") is not None
        else states.get("min_scenario_actor_distance_m")
        or 0.0
    )
    nearby_actor_ratio = float(
        states.get("nearby_natural_traffic_ratio")
        if states.get("nearby_natural_traffic_ratio") is not None
        else states.get("nearby_actor_ratio")
        or 0.0
    )

    if not video["exists"]:
        failures.append(f"{name}:missing_video")
    if video["opened"] is False:
        failures.append(f"{name}:unreadable_video")
    if int(video.get("width") or 0) < int(min_resolution_width) or int(video.get("height") or 0) < int(min_resolution_height):
        failures.append(f"{name}:video_resolution_below_threshold")
    if float(video.get("fps") or 0.0) < float(min_fps):
        failures.append(f"{name}:video_fps_below_threshold")
    if int(video.get("frame_count") or 0) < int(min_frames):
        failures.append(f"{name}:video_frame_count_below_threshold")
    codec = str(video.get("codec_name") or video.get("fourcc") or "").lower()
    if codec not in {"hevc", "hvc1", "hev1", "x265"}:
        message = f"{name}:video_codec_not_confirmed_as_hevc"
        if require_hevc:
            failures.append(message)
        else:
            warnings.append(message)
    if not states["exists"]:
        failures.append(f"{name}:missing_states_csv")
    if int(states.get("row_count") or 0) != int(video.get("frame_count") or 0):
        warnings.append(f"{name}:states_video_frame_count_mismatch")
    model_waypoint_ratio = float(
        attribution.get("model_waypoint_controller_ratio")
        if attribution.get("model_waypoint_controller_ratio") is not None
        else attribution.get("direct_model_control_ratio")
        or 0.0
    )
    if require_model_control and model_waypoint_ratio < 0.99:
        failures.append(f"{name}:ego_not_model_controlled_for_full_rollout")
    if bool(attribution.get("ego_uses_carla_autopilot")):
        failures.append(f"{name}:ego_uses_carla_autopilot")
    if bool(attribution.get("ego_uses_map_route_tracking")):
        failures.append(f"{name}:ego_uses_carla_route_controller")
    if int(attribution.get("traffic_manager_vehicle_count") or 0) < int(min_traffic_manager_vehicles):
        failures.append(f"{name}:insufficient_natural_traffic")
    if int(attribution.get("scripted_vehicle_count") or 0) > int(max_scripted_vehicles):
        failures.append(f"{name}:scripted_vehicle_count_above_threshold")
    if int(metrics.get("collision_count") or 0) > int(max_collision_count):
        failures.append(f"{name}:collision_count_above_threshold")
    if float(metrics.get("mean_lateral_error_m") or 0.0) > float(max_mean_lateral_error_m):
        failures.append(f"{name}:mean_lateral_error_above_threshold")
    max_lateral = float(states.get("max_lateral_error_m") or metrics.get("max_lateral_error_m") or 0.0)
    if max_lateral > float(max_lateral_error_m):
        failures.append(f"{name}:max_lateral_error_above_threshold")
    lane_ratio = (
        states.get("projected_ego_on_driving_lane_ratio")
        if states.get("projected_ego_on_driving_lane_ratio") is not None
        else states.get("ego_on_driving_lane_ratio")
    )
    strict_straight_lane_ratio = states.get("straight_projected_ego_on_driving_lane_ratio")
    lane_check_ratio = (
        strict_straight_lane_ratio if strict_straight_lane_ratio is not None else lane_ratio
    )
    if lane_check_ratio is not None and float(lane_check_ratio or 0.0) < 0.95:
        failures.append(f"{name}:ego_left_driving_lane")
    if float(metrics.get("route_completion") or 0.0) < float(min_route_completion):
        warnings.append(f"{name}:low_route_completion")
    if float(attribution.get("safety_override_ratio") or 0.0) > float(max_safety_override_ratio):
        warnings.append(f"{name}:high_safety_override_ratio")
    semantic_evidence = _semantic_evidence_for_scenario(
        name=name,
        scenario=scenario,
        metrics=metrics,
        attribution=attribution,
        states=states,
        nearby_actor_ratio=nearby_actor_ratio,
    )
    semantic_target = str(semantic_evidence.get("target") or "")
    if int(attribution.get("traffic_manager_vehicle_count") or 0) > 0 and semantic_target != "pedestrian_yield":
        if nearest_actor_distance > float(max_nearest_actor_distance_m):
            failures.append(f"{name}:natural_traffic_too_far_from_ego")
        if nearby_actor_ratio < float(min_nearby_actor_ratio):
            failures.append(f"{name}:insufficient_nearby_natural_traffic")
    if require_semantic_match and not bool(semantic_evidence.get("passed")):
        failures.extend(f"{name}:{failure}" for failure in list(semantic_evidence.get("failures") or []))

    return {
        "name": name,
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "warnings": warnings,
        "metrics": {
            "route_completion": metrics.get("route_completion"),
            "driving_score": metrics.get("driving_score"),
            "collision_count": metrics.get("collision_count"),
            "mean_lateral_error_m": metrics.get("mean_lateral_error_m"),
            "mean_speed_mps": metrics.get("mean_speed_mps"),
            "min_scenario_actor_distance_m": metrics.get("min_scenario_actor_distance_m"),
            "right_turn_command_ratio": states.get("right_turn_command_ratio")
            if states.get("right_turn_command_ratio") is not None
            else metrics.get("right_turn_command_ratio"),
            "max_walker_crossing_completion": states.get("max_walker_crossing_completion")
            if states.get("max_walker_crossing_completion") is not None
            else metrics.get("max_walker_crossing_completion"),
            "post_crossing_max_speed_mps": states.get("post_crossing_max_speed_mps")
            if states.get("post_crossing_max_speed_mps") is not None
            else metrics.get("post_crossing_max_speed_mps"),
            "ego_on_driving_lane_ratio": states.get("ego_on_driving_lane_ratio"),
            "projected_ego_on_driving_lane_ratio": states.get("projected_ego_on_driving_lane_ratio"),
            "straight_projected_ego_on_driving_lane_ratio": strict_straight_lane_ratio,
            "lane_check_ratio": lane_check_ratio,
            "turn_maneuver_frame_count": states.get("turn_maneuver_frame_count"),
            "max_driving_lane_center_distance_m": states.get("max_driving_lane_center_distance_m"),
        },
        "control_attribution": {
            "model_waypoint_controller_ratio": model_waypoint_ratio,
            "network_control_blend_ratio": attribution.get("network_control_blend_ratio"),
            "lane_departure_guard_ratio": attribution.get("lane_departure_guard_ratio"),
            "control_pipeline": attribution.get("control_pipeline"),
            "direct_model_control_ratio": attribution.get("direct_model_control_ratio"),
            "safety_override_ratio": attribution.get("safety_override_ratio"),
            "ego_uses_carla_autopilot": attribution.get("ego_uses_carla_autopilot", False),
            "ego_uses_map_route_tracking": attribution.get("ego_uses_map_route_tracking", False),
            "traffic_manager_vehicle_count": attribution.get("traffic_manager_vehicle_count"),
            "scripted_vehicle_count": attribution.get("scripted_vehicle_count"),
            "crosswalk_walker_count": attribution.get("crosswalk_walker_count"),
            "controlled_walker_count": attribution.get("controlled_walker_count"),
        },
        "video": video,
        "states": states,
        "traffic_visibility": {
            "max_nearest_actor_distance_m": max_nearest_actor_distance_m,
            "nearby_actor_distance_m": nearby_actor_distance_m,
            "nearby_actor_ratio": nearby_actor_ratio,
            "min_actor_distance_m": nearest_actor_distance,
            "mean_actor_distance_m": states.get("mean_natural_traffic_distance_m")
            if states.get("mean_natural_traffic_distance_m") is not None
            else states.get("mean_scenario_actor_distance_m"),
            "visible_actor_ratio": states.get("visible_natural_traffic_ratio")
            if states.get("visible_natural_traffic_ratio") is not None
            else states.get("visible_actor_ratio"),
        },
        "semantic_evidence": semantic_evidence,
    }


def _inspect_video(path: Path) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "opened": False,
        "frame_count": 0,
        "width": 0,
        "height": 0,
        "fps": 0.0,
        "fourcc": "",
        "first_frame_read": False,
    }
    if not path.exists():
        return payload
    try:
        import cv2  # type: ignore
    except Exception as exc:
        payload["error"] = f"opencv_unavailable:{exc}"
        return payload
    capture = cv2.VideoCapture(str(path))
    try:
        payload["opened"] = bool(capture.isOpened())
        if not capture.isOpened():
            return payload
        payload["frame_count"] = int(round(float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)))
        payload["width"] = int(round(float(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0.0)))
        payload["height"] = int(round(float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0.0)))
        payload["fps"] = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        fourcc_int = int(capture.get(cv2.CAP_PROP_FOURCC) or 0)
        payload["fourcc"] = "".join(chr((fourcc_int >> (8 * idx)) & 255) for idx in range(4)).strip()
        ok, frame = capture.read()
        payload["first_frame_read"] = bool(ok)
        if ok:
            payload["first_frame_shape"] = list(frame.shape)
    finally:
        capture.release()
    payload.update(_ffprobe_video_codec(path))
    return payload


def _ffprobe_video_codec(path: Path) -> Dict[str, Any]:
    ffprobe_path = shutil.which("ffprobe") or _env_binary("ffprobe")
    if not ffprobe_path:
        return {}
    command = [
        ffprobe_path,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,codec_tag_string",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=10.0)
    except Exception:
        return {}
    if result.returncode != 0:
        return {}
    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError:
        return {}
    streams = list(payload.get("streams") or [])
    if not streams:
        return {}
    stream = dict(streams[0])
    return {
        "codec_name": str(stream.get("codec_name") or ""),
        "codec_tag_string": str(stream.get("codec_tag_string") or ""),
    }


def _env_binary(name: str) -> str:
    candidate = Path(sys.executable).with_name(name)
    return str(candidate) if candidate.exists() else ""


def _inspect_states_csv(
    path: Path,
    *,
    nearby_actor_distance_m: float = 30.0,
    visible_actor_distance_m: float = 60.0,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "row_count": 0,
        "columns": [],
    }
    if not path.exists():
        return payload
    rows = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        payload["columns"] = list(reader.fieldnames or [])
        for row in reader:
            rows.append(row)
    payload["row_count"] = len(rows)
    if rows:
        payload["mean_speed_mps"] = _mean_float(rows, "speed_mps")
        payload["max_lateral_error_m"] = max(abs(_float(row.get("lateral_error_m"))) for row in rows)
        payload["mean_lateral_error_m"] = mean(abs(_float(row.get("lateral_error_m"))) for row in rows)
        payload["model_control_ratio"] = _mean_bool(rows, "ego_control_mode", {"e2e_waypoint_control", "e2e_direct"})
        payload["safety_override_ratio"] = sum(1 for row in rows if _float(row.get("safety_brake")) > 0.05) / len(rows)
        payload["unique_phases"] = sorted({str(row.get("scenario_phase") or "") for row in rows if row.get("scenario_phase")})
        if "ego_on_driving_lane" in payload["columns"]:
            payload["ego_on_driving_lane_ratio"] = sum(
                1 for row in rows if _float(row.get("ego_on_driving_lane")) >= 0.5
            ) / len(rows)
            lane_distances = [
                _float(row.get("ego_nearest_driving_lane_center_distance_m"))
                for row in rows
                if _optional_nonnegative_float(row.get("ego_nearest_driving_lane_center_distance_m")) is not None
            ]
            if lane_distances:
                payload["max_driving_lane_center_distance_m"] = max(lane_distances)
                payload["mean_driving_lane_center_distance_m"] = mean(lane_distances)
            projected_lane_rows = 0
            projected_lane_valid_rows = 0
            for row in rows:
                lane_distance = _optional_nonnegative_float(row.get("ego_nearest_driving_lane_center_distance_m"))
                if lane_distance is None:
                    continue
                lane_width = _float(row.get("ego_driving_lane_width_m"))
                distance_limit = max(float(lane_width) * 0.60, 2.05) if lane_width > 0.0 else 2.05
                projected_lane_valid_rows += 1
                if float(lane_distance) <= distance_limit:
                    projected_lane_rows += 1
            if projected_lane_valid_rows:
                payload["projected_ego_on_driving_lane_ratio"] = (
                    projected_lane_rows / projected_lane_valid_rows
                )
            if "command" in payload["columns"]:
                turn_rows = [
                    row for row in rows if int(_float(row.get("command")) or 4) in {1, 2}
                ]
                straight_rows = [
                    row for row in rows if int(_float(row.get("command")) or 4) not in {1, 2}
                ]
                payload["turn_maneuver_frame_count"] = len(turn_rows)
                payload["straight_frame_count"] = len(straight_rows)
                straight_ratio = _projected_lane_ratio(straight_rows)
                if straight_ratio is not None:
                    payload["straight_projected_ego_on_driving_lane_ratio"] = straight_ratio
        payload["right_turn_command_ratio"] = sum(
            1 for row in rows if int(_float(row.get("command")) or 4) == 2
        ) / len(rows)
        payload["right_turn_command_frame_count"] = sum(
            1 for row in rows if int(_float(row.get("command")) or 4) == 2
        )
        walker_completion = [_float(row.get("walker_crossing_completion")) for row in rows]
        payload["max_walker_crossing_completion"] = max(walker_completion) if walker_completion else 0.0
        walker_lateral = [
            _float(row.get("walker_route_lateral_m"))
            for row in rows
            if _optional_nonnegative_float(row.get("walker_route_progress_m")) is not None
        ]
        payload["walker_route_lateral_span_m"] = max(walker_lateral) - min(walker_lateral) if walker_lateral else 0.0
        post_crossing_speeds = [
            _float(row.get("speed_mps")) for row in rows if _float(row.get("walker_crossing_completion")) >= 0.95
        ]
        payload["post_crossing_max_speed_mps"] = max(post_crossing_speeds) if post_crossing_speeds else 0.0
        crossing_rows = [
            row
            for row in rows
            if str(row.get("scenario_phase") or "") == "pedestrian_crossing"
            and 0.05 <= _float(row.get("walker_crossing_completion")) < 0.95
        ]
        payload["pedestrian_crossing_min_speed_mps"] = (
            min(_float(row.get("speed_mps")) for row in crossing_rows) if crossing_rows else None
        )
        first_complete_idx = next(
            (
                idx
                for idx, row in enumerate(rows)
                if _float(row.get("walker_crossing_completion")) >= 0.95
            ),
            None,
        )
        if first_complete_idx is not None:
            start_progress = _float(rows[first_complete_idx].get("route_progress_m"))
            post_progress = [_float(row.get("route_progress_m")) for row in rows[first_complete_idx:]]
            payload["post_crossing_route_progress_delta_m"] = (
                max(post_progress) - start_progress if post_progress else 0.0
            )
        else:
            payload["post_crossing_route_progress_delta_m"] = 0.0
        final_rows = rows[-20:]
        payload["final_mean_speed_mps"] = _mean_float(final_rows, "speed_mps") if final_rows else 0.0
        scenario_distances = [
            value
            for row in rows
            for value in [_optional_nonnegative_float(row.get("scenario_actor_distance_m"))]
            if value is not None
        ]
        if scenario_distances:
            payload["min_scenario_actor_distance_m"] = min(scenario_distances)
            payload["mean_scenario_actor_distance_m"] = mean(scenario_distances)
            payload["nearby_actor_ratio"] = (
                sum(1 for value in scenario_distances if value <= float(nearby_actor_distance_m)) / len(rows)
            )
            payload["visible_actor_ratio"] = (
                sum(1 for value in scenario_distances if value <= float(visible_actor_distance_m)) / len(rows)
            )
        natural_distances = [
            value
            for row in rows
            for value in [_optional_nonnegative_float(row.get("natural_traffic_nearest_distance_m"))]
            if value is not None
        ]
        if natural_distances:
            payload["min_natural_traffic_distance_m"] = min(natural_distances)
            payload["mean_natural_traffic_distance_m"] = mean(natural_distances)
            payload["nearby_natural_traffic_ratio"] = (
                sum(1 for value in natural_distances if value <= float(nearby_actor_distance_m)) / len(rows)
            )
            payload["visible_natural_traffic_ratio"] = (
                sum(1 for value in natural_distances if value <= float(visible_actor_distance_m)) / len(rows)
            )
            payload["mean_visible_natural_traffic_count"] = _mean_float(rows, "natural_traffic_visible_actor_count")
            payload["mean_front_natural_traffic_count"] = _mean_float(rows, "natural_traffic_front_actor_count")
            payload["mean_adjacent_natural_traffic_count"] = _mean_float(rows, "natural_traffic_adjacent_actor_count")
            payload["mean_same_lane_front_natural_traffic_count"] = _mean_float(
                rows,
                "natural_traffic_same_lane_front_actor_count",
            )
    return payload


def _semantic_evidence_for_scenario(
    *,
    name: str,
    scenario: Mapping[str, Any],
    metrics: Mapping[str, Any],
    attribution: Mapping[str, Any],
    states: Mapping[str, Any],
    nearby_actor_ratio: float,
) -> Dict[str, Any]:
    expectation = _semantic_expectation_for_scenario(name=name, scenario=scenario)
    if expectation == "generic":
        return {"target": expectation, "passed": True, "failures": [], "notes": ["generic_diagnostic"]}
    checks = {
        "nearby_actor_ratio": float(nearby_actor_ratio),
        "min_actor_distance_m": states.get("min_natural_traffic_distance_m")
        if states.get("min_natural_traffic_distance_m") is not None
        else states.get("min_scenario_actor_distance_m"),
        "mean_front_actor_count": float(states.get("mean_front_natural_traffic_count") or 0.0),
        "mean_same_lane_front_actor_count": float(states.get("mean_same_lane_front_natural_traffic_count") or 0.0),
        "mean_adjacent_actor_count": float(states.get("mean_adjacent_natural_traffic_count") or 0.0),
        "crosswalk_walker_count": int(attribution.get("crosswalk_walker_count") or 0),
        "safety_override_ratio": float(attribution.get("safety_override_ratio") or 0.0),
        "collision_count": int(metrics.get("collision_count") or 0),
        "right_turn_command_ratio": float(
            states.get("right_turn_command_ratio")
            if states.get("right_turn_command_ratio") is not None
            else metrics.get("right_turn_command_ratio")
            or 0.0
        ),
        "right_turn_command_frame_count": int(states.get("right_turn_command_frame_count") or 0),
        "max_walker_crossing_completion": float(
            states.get("max_walker_crossing_completion")
            if states.get("max_walker_crossing_completion") is not None
            else metrics.get("max_walker_crossing_completion")
            or 0.0
        ),
        "walker_route_lateral_span_m": float(
            states.get("walker_route_lateral_span_m")
            if states.get("walker_route_lateral_span_m") is not None
            else metrics.get("walker_route_lateral_span_m")
            or 0.0
        ),
        "post_crossing_max_speed_mps": float(
            states.get("post_crossing_max_speed_mps")
            if states.get("post_crossing_max_speed_mps") is not None
            else metrics.get("post_crossing_max_speed_mps")
            or 0.0
        ),
        "pedestrian_crossing_min_speed_mps": states.get("pedestrian_crossing_min_speed_mps"),
        "post_crossing_route_progress_delta_m": float(states.get("post_crossing_route_progress_delta_m") or 0.0),
        "final_mean_speed_mps": float(
            states.get("final_mean_speed_mps")
            if states.get("final_mean_speed_mps") is not None
            else metrics.get("final_mean_speed_mps")
            or 0.0
        ),
    }
    failures: List[str] = []
    if expectation == "pedestrian_yield":
        if int(checks["crosswalk_walker_count"]) <= 0:
            failures.append("semantic_pedestrian_yield_missing_crosswalk_walker")
        if float(checks["safety_override_ratio"]) < 0.03:
            failures.append("semantic_pedestrian_yield_missing_brake_or_yield_response")
        if int(checks["right_turn_command_frame_count"]) <= 0:
            failures.append("semantic_pedestrian_yield_missing_right_turn_route_context")
        if float(checks["max_walker_crossing_completion"]) < 0.95:
            failures.append("semantic_pedestrian_yield_pedestrian_did_not_complete_crossing")
        if float(checks["walker_route_lateral_span_m"]) < 3.0:
            failures.append("semantic_pedestrian_yield_insufficient_pedestrian_crossing_span")
        if checks["pedestrian_crossing_min_speed_mps"] is None or float(checks["pedestrian_crossing_min_speed_mps"]) > 0.30:
            failures.append("semantic_pedestrian_yield_ego_did_not_stop_during_crossing")
        if max(float(checks["post_crossing_max_speed_mps"]), float(checks["final_mean_speed_mps"])) < 0.75:
            failures.append("semantic_pedestrian_yield_ego_did_not_resume_after_yield")
        if float(checks["post_crossing_route_progress_delta_m"]) < 0.5:
            failures.append("semantic_pedestrian_yield_ego_did_not_move_after_clearance")
    elif expectation == "dense_follow":
        if float(checks["nearby_actor_ratio"]) < 0.45:
            failures.append("semantic_dense_follow_insufficient_nearby_traffic")
        if max(float(checks["mean_front_actor_count"]), float(checks["mean_same_lane_front_actor_count"])) < 0.20:
            failures.append("semantic_dense_follow_missing_front_vehicle_context")
    elif expectation == "adjacent_lane_interaction":
        if float(checks["nearby_actor_ratio"]) < 0.30:
            failures.append("semantic_adjacent_lane_insufficient_nearby_traffic")
        if float(checks["mean_adjacent_actor_count"]) < 0.15:
            failures.append("semantic_adjacent_lane_missing_adjacent_vehicle_context")
    else:
        failures.append("semantic_unknown_expectation")
    return {
        "target": expectation,
        "passed": not failures,
        "failures": failures,
        "checks": checks,
    }


def _semantic_expectation_for_scenario(*, name: str, scenario: Mapping[str, Any]) -> str:
    text = " ".join(
        [
            str(name or ""),
            str(scenario.get("type") or ""),
            str(_nested(scenario, ["scenario", "type"]) or ""),
            str(_nested(scenario, ["scenario", "name"]) or ""),
        ]
    ).lower()
    if any(token in text for token in ["pedestrian", "crosswalk", "yield"]):
        return "pedestrian_yield"
    if any(token in text for token in ["adjacent", "cut_in", "cut-in", "intrusion", "lane_interaction"]):
        return "adjacent_lane_interaction"
    if any(token in text for token in ["dense", "follow", "overtake", "traffic_follow"]):
        return "dense_follow"
    return "generic"


def _audit_summary(scenarios: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        "passed_scenarios": sum(1 for row in scenarios if row.get("status") == "pass"),
        "failed_scenarios": sum(1 for row in scenarios if row.get("status") != "pass"),
        "total_video_frames": sum(int(dict(row.get("video") or {}).get("frame_count") or 0) for row in scenarios),
        "total_state_rows": sum(int(dict(row.get("states") or {}).get("row_count") or 0) for row in scenarios),
        "total_collisions": sum(int(dict(row.get("metrics") or {}).get("collision_count") or 0) for row in scenarios),
        "total_traffic_manager_vehicles": sum(
            int(dict(row.get("control_attribution") or {}).get("traffic_manager_vehicle_count") or 0)
            for row in scenarios
        ),
        "mean_nearby_actor_ratio": mean(
            float(dict(row.get("traffic_visibility") or {}).get("nearby_actor_ratio") or 0.0) for row in scenarios
        )
        if scenarios
        else 0.0,
    }


def _render_audit_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# CARLA Vision Rollout Video Audit",
        "",
        f"- Status: `{payload.get('status')}`",
        f"- Scenarios: `{payload.get('scenario_count')}`",
        f"- Failures: `{payload.get('failure_count')}`",
        f"- Warnings: `{payload.get('warning_count')}`",
        "",
        "| Scenario | Status | Frames | Video | TM vehicles | Collisions | Model-control ratio | Safety ratio |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for scenario in list(payload.get("scenarios") or []):
        row = dict(scenario)
        video = dict(row.get("video") or {})
        attribution = dict(row.get("control_attribution") or {})
        metrics = dict(row.get("metrics") or {})
        lines.append(
            "| {name} | `{status}` | {frames} | {width}x{height} {codec} | {tm} | {collisions} | {model:.3f} | {safety:.3f} |".format(
                name=row.get("name"),
                status=row.get("status"),
                frames=int(video.get("frame_count") or 0),
                width=int(video.get("width") or 0),
                height=int(video.get("height") or 0),
                codec=str(video.get("fourcc") or ""),
                tm=int(attribution.get("traffic_manager_vehicle_count") or 0),
                collisions=int(metrics.get("collision_count") or 0),
                model=float(attribution.get("direct_model_control_ratio") or 0.0),
                safety=float(attribution.get("safety_override_ratio") or 0.0),
            )
        )
    if payload.get("failures"):
        lines.extend(["", "## Failures", ""])
        lines.extend(f"- `{item}`" for item in list(payload.get("failures") or []))
    if payload.get("warnings"):
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- `{item}`" for item in list(payload.get("warnings") or []))
    return "\n".join(lines) + "\n"


def _nested(payload: Mapping[str, Any], keys: Sequence[str]) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _projected_lane_ratio(rows: Sequence[Mapping[str, Any]]) -> Optional[float]:
    valid = 0
    on_lane = 0
    for row in rows:
        lane_distance = _optional_nonnegative_float(row.get("ego_nearest_driving_lane_center_distance_m"))
        if lane_distance is None:
            continue
        lane_width = _float(row.get("ego_driving_lane_width_m"))
        distance_limit = max(float(lane_width) * 0.60, 2.05) if lane_width > 0.0 else 2.05
        valid += 1
        if float(lane_distance) <= distance_limit:
            on_lane += 1
    return on_lane / valid if valid else None


def _optional_nonnegative_float(value: Any) -> Optional[float]:
    if value is None or str(value).strip() == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result >= 0.0 else None


def _mean_float(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    return mean(_float(row.get(key)) for row in rows) if rows else 0.0


def _mean_bool(rows: Sequence[Mapping[str, Any]], key: str, positive_values: set[str]) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if str(row.get(key) or "") in positive_values) / len(rows)

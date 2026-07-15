from __future__ import annotations

import json
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Mapping, Optional, Sequence

from nusc_scene_agent.bench2drive_e2e import DEFAULT_BENCH2DRIVE_OUTPUT
from nusc_scene_agent.carla_closed_loop import DEFAULT_CARLA_ROOT
from nusc_scene_agent.carla_video_audit import audit_carla_vision_rollouts
from nusc_scene_agent.carla_vision_closed_loop import run_carla_vision_closed_loop


CARLA_SEMANTIC_DEMO_MINING_SCHEMA = "carla_semantic_demo_mining_v1"
DEFAULT_CARLA_SEMANTIC_DEMO_OUTPUT = Path("outputs/carla_semantic_demo_final")
DEFAULT_CARLA_SEMANTIC_DEMO_TRIALS_OUTPUT = Path("outputs/carla_semantic_demo_trials")
CARLA_SEMANTIC_DEMO_REPORT_NAME = "carla_semantic_demo_report.json"
CARLA_SEMANTIC_DEMO_AUDIT_NAME = "carla_semantic_demo_audit.json"


@dataclass(frozen=True)
class CarlaSemanticDemoAttempt:
    name: str
    scenario_type: str
    spawn_index: int
    destination_index: int
    target_speed_mps: float
    horizon_s: Optional[float]
    parameters: Mapping[str, Any]


@dataclass(frozen=True)
class CarlaSemanticDemoTarget:
    target_id: str
    attempts: Sequence[CarlaSemanticDemoAttempt]


def mine_carla_semantic_demos(
    *,
    carla_root: Path = DEFAULT_CARLA_ROOT,
    checkpoint_path: Path = DEFAULT_BENCH2DRIVE_OUTPUT / "vision_e2e_planner_best.pt",
    output_dir: Path = DEFAULT_CARLA_SEMANTIC_DEMO_OUTPUT,
    trials_output_dir: Path = DEFAULT_CARLA_SEMANTIC_DEMO_TRIALS_OUTPUT,
    host: str = "127.0.0.1",
    port_start: int = 2040,
    town: str = "Town10HD_Opt",
    fps: int = 10,
    horizon_s: float = 18.0,
    image_size: int = 160,
    camera_width: int = 640,
    camera_height: int = 360,
    video_width: int = 1920,
    video_height: int = 1080,
    camera_fov: float = 90.0,
    carla_quality_level: str = "Epic",
    brake_probability_threshold: float = 0.82,
    device: str = "cuda",
    auto_launch: bool = True,
    cuda_visible_devices: str = "",
    traffic_manager_port_start: int = 8040,
    launch_timeout_s: float = 180.0,
    rpc_timeout_s: float = 90.0,
    keep_server: bool = False,
    reuse_carla_server: bool = True,
    video_fps: int = 10,
    video_encoder: str = "hevc_nvenc",
    video_nvenc_preset: str = "p4",
    video_quality: int = 23,
    render_gif: bool = False,
    enable_scenario_safety_override: bool = True,
    enable_lane_departure_guard: bool = False,
    condition_ego_route_traffic_lights: bool = True,
    route_sampling_resolution_m: float = 2.0,
    route_min_length_m: float = 55.0,
    route_max_length_m: float = 160.0,
    route_preferred_length_m: float = 95.0,
    max_attempts_per_target: int = 4,
    targets: Optional[Sequence[Mapping[str, Any]]] = None,
    min_resolution_width: int = 1920,
    min_resolution_height: int = 1080,
    min_fps: float = 8.0,
    min_frames: int = 80,
    min_traffic_manager_vehicles: int = 1,
    min_route_completion: float = 0.05,
    max_mean_lateral_error_m: float = 2.5,
    max_lateral_error_m: float = 6.0,
    max_safety_override_ratio: float = 0.45,
    max_nearest_actor_distance_m: float = 60.0,
    nearby_actor_distance_m: float = 30.0,
    min_nearby_actor_ratio: float = 0.30,
    require_hevc: bool = False,
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    trials_output_dir = Path(trials_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trials_output_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.time()

    target_specs = _normalize_targets(targets)
    target_results = []
    selected_reports: List[Dict[str, Any]] = []
    attempt_counter = 0

    for target in target_specs:
        target_dir = trials_output_dir / _safe_filename(target.target_id)
        target_dir.mkdir(parents=True, exist_ok=True)
        attempts = list(target.attempts)[: max(int(max_attempts_per_target), 1)]
        passed_report: Optional[Dict[str, Any]] = None
        passed_audit: Optional[Dict[str, Any]] = None
        attempt_rows = []
        for attempt_idx, attempt in enumerate(attempts, start=1):
            attempt_counter += 1
            attempt_output = target_dir / f"attempt_{attempt_idx:02d}_{_safe_filename(attempt.name)}"
            if attempt_output.exists():
                shutil.rmtree(attempt_output)
            report: Optional[Dict[str, Any]] = None
            audit: Optional[Dict[str, Any]] = None
            error = ""
            attempt_port = int(port_start) if reuse_carla_server else int(port_start) + attempt_counter - 1
            attempt_tm_port = (
                int(traffic_manager_port_start)
                if reuse_carla_server
                else int(traffic_manager_port_start) + attempt_counter - 1
            )
            try:
                report = run_carla_vision_closed_loop(
                    carla_root=Path(carla_root),
                    checkpoint_path=Path(checkpoint_path),
                    output_dir=attempt_output,
                    host=str(host),
                    port=attempt_port,
                    town=str(town),
                    spawn_index=int(attempt.spawn_index),
                    destination_index=int(attempt.destination_index),
                    route_sampling_resolution_m=float(route_sampling_resolution_m),
                    route_min_length_m=float(route_min_length_m),
                    route_max_length_m=float(route_max_length_m),
                    route_preferred_length_m=float(route_preferred_length_m),
                    fps=int(fps),
                    horizon_s=float(attempt.horizon_s if attempt.horizon_s is not None else horizon_s),
                    image_size=int(image_size),
                    camera_width=int(camera_width),
                    camera_height=int(camera_height),
                    video_width=int(video_width),
                    video_height=int(video_height),
                    camera_fov=float(camera_fov),
                    carla_quality_level=str(carla_quality_level),
                    scenario_type=str(attempt.scenario_type),
                    scenario_name=str(attempt.name),
                    scenario_params=dict(attempt.parameters),
                    target_speed_mps=float(attempt.target_speed_mps),
                    brake_probability_threshold=float(brake_probability_threshold),
                    enable_scenario_safety_override=bool(enable_scenario_safety_override),
                    enable_lane_departure_guard=bool(enable_lane_departure_guard),
                    condition_ego_route_traffic_lights=bool(condition_ego_route_traffic_lights),
                    device=str(device),
                    auto_launch=bool(auto_launch),
                    cuda_visible_devices=str(cuda_visible_devices),
                    traffic_manager_port=attempt_tm_port,
                    launch_timeout_s=float(launch_timeout_s),
                    rpc_timeout_s=float(rpc_timeout_s),
                    keep_server=bool(keep_server or reuse_carla_server),
                    video_fps=int(video_fps),
                    video_encoder=str(video_encoder),
                    video_nvenc_preset=str(video_nvenc_preset),
                    video_quality=int(video_quality),
                    render_gif=bool(render_gif),
                )
                audit = audit_carla_vision_rollouts(
                    report_path=attempt_output / "carla_vision_closed_loop_report.json",
                    output_path=attempt_output / CARLA_SEMANTIC_DEMO_AUDIT_NAME,
                    min_resolution_width=int(min_resolution_width),
                    min_resolution_height=int(min_resolution_height),
                    min_fps=float(min_fps),
                    min_frames=int(min_frames),
                    min_traffic_manager_vehicles=int(min_traffic_manager_vehicles),
                    max_scripted_vehicles=0,
                    max_collision_count=0,
                    min_route_completion=float(min_route_completion),
                    max_mean_lateral_error_m=float(max_mean_lateral_error_m),
                    max_lateral_error_m=float(max_lateral_error_m),
                    max_safety_override_ratio=float(max_safety_override_ratio),
                    max_nearest_actor_distance_m=float(max_nearest_actor_distance_m),
                    nearby_actor_distance_m=float(nearby_actor_distance_m),
                    min_nearby_actor_ratio=float(min_nearby_actor_ratio),
                    require_semantic_match=True,
                    require_model_control=True,
                    require_hevc=bool(require_hevc),
                )
            except Exception as exc:
                error = str(exc)
            status = str(audit.get("status") if audit else "error")
            attempt_rows.append(
                {
                    "attempt_index": attempt_idx,
                    "name": attempt.name,
                    "scenario_type": attempt.scenario_type,
                    "spawn_index": attempt.spawn_index,
                    "destination_index": attempt.destination_index,
                    "output_dir": str(attempt_output),
                    "status": status,
                    "error": error,
                    "failures": list(audit.get("failures") or []) if audit else [],
                    "warnings": list(audit.get("warnings") or []) if audit else [],
                }
            )
            if status == "pass" and report is not None:
                passed_report = report
                passed_audit = audit
                break

        promoted_report = None
        if passed_report is not None:
            promoted_dir = output_dir / _safe_filename(target.target_id)
            promoted_report = _promote_carla_report(
                report=passed_report,
                source_dir=Path(str(passed_report.get("output_dir") or "")),
                target_dir=promoted_dir,
            )
            selected_reports.append(promoted_report)

        target_results.append(
            {
                "target_id": target.target_id,
                "status": "pass" if promoted_report is not None else "fail",
                "attempt_count": len(attempt_rows),
                "attempts": attempt_rows,
                "selected_output_dir": str(promoted_report.get("output_dir") or "") if promoted_report else "",
                "selected_audit": _audit_brief(passed_audit) if passed_audit else {},
            }
        )

    final_report = _write_semantic_demo_report(
        output_dir=output_dir,
        carla_root=Path(carla_root),
        checkpoint_path=Path(checkpoint_path),
        town=str(town),
        reports=selected_reports,
        target_results=target_results,
        started_at=started_at,
    )
    final_audit = {}
    if selected_reports:
        final_audit = audit_carla_vision_rollouts(
            report_path=output_dir / CARLA_SEMANTIC_DEMO_REPORT_NAME,
            output_path=output_dir / CARLA_SEMANTIC_DEMO_AUDIT_NAME,
            min_resolution_width=int(min_resolution_width),
            min_resolution_height=int(min_resolution_height),
            min_fps=float(min_fps),
            min_frames=int(min_frames),
            min_traffic_manager_vehicles=int(min_traffic_manager_vehicles),
            max_scripted_vehicles=0,
            max_collision_count=0,
            min_route_completion=float(min_route_completion),
            max_mean_lateral_error_m=float(max_mean_lateral_error_m),
            max_lateral_error_m=float(max_lateral_error_m),
            max_safety_override_ratio=float(max_safety_override_ratio),
            max_nearest_actor_distance_m=float(max_nearest_actor_distance_m),
            nearby_actor_distance_m=float(nearby_actor_distance_m),
            min_nearby_actor_ratio=float(min_nearby_actor_ratio),
            require_semantic_match=True,
            require_model_control=True,
            require_hevc=bool(require_hevc),
        )

    mining_report = {
        "schema": CARLA_SEMANTIC_DEMO_MINING_SCHEMA,
        "status": _mining_status(target_results),
        "output_dir": str(output_dir),
        "trials_output_dir": str(trials_output_dir),
        "report_path": str(output_dir / CARLA_SEMANTIC_DEMO_REPORT_NAME),
        "audit_path": str(output_dir / CARLA_SEMANTIC_DEMO_AUDIT_NAME) if final_audit else "",
        "target_count": len(target_results),
        "passed_target_count": sum(1 for row in target_results if row.get("status") == "pass"),
        "attempt_count": sum(int(row.get("attempt_count") or 0) for row in target_results),
        "reuse_carla_server": bool(reuse_carla_server),
        "require_hevc": bool(require_hevc),
        "targets": target_results,
        "final_report": {
            "scenario_count": final_report.get("scenario_count"),
            "aggregate": final_report.get("aggregate", {}),
        },
        "final_audit": _audit_brief(final_audit) if final_audit else {},
        "runtime_s": round(time.time() - started_at, 3),
    }
    (output_dir / "carla_semantic_demo_mining_report.json").write_text(
        json.dumps(mining_report, indent=2),
        encoding="utf-8",
    )
    (output_dir / "carla_semantic_demo_mining_report.md").write_text(
        _render_mining_markdown(mining_report),
        encoding="utf-8",
    )
    if reuse_carla_server and auto_launch and not keep_server:
        _terminate_carla_processes_for_port(int(port_start))
    return mining_report


def _normalize_targets(raw_targets: Optional[Sequence[Mapping[str, Any]]]) -> List[CarlaSemanticDemoTarget]:
    if raw_targets:
        return [_target_from_mapping(row) for row in raw_targets]
    return _default_semantic_demo_targets()


def _target_from_mapping(raw: Mapping[str, Any]) -> CarlaSemanticDemoTarget:
    target_id = str(raw.get("target_id") or raw.get("id") or raw.get("name") or "target")
    attempts = []
    for idx, raw_attempt in enumerate(list(raw.get("attempts") or raw.get("scenarios") or []), start=1):
        attempts.append(_attempt_from_mapping(raw_attempt, fallback_name=f"{target_id}_attempt_{idx}"))
    if not attempts:
        attempts.append(_attempt_from_mapping(raw, fallback_name=target_id))
    return CarlaSemanticDemoTarget(target_id=target_id, attempts=attempts)


def _attempt_from_mapping(raw: Mapping[str, Any], *, fallback_name: str) -> CarlaSemanticDemoAttempt:
    standard = {"name", "scenario_type", "type", "spawn_index", "destination_index", "target_speed_mps", "parameters", "params"}
    parameters = dict(raw.get("parameters") or raw.get("params") or {})
    parameters.update({str(key): value for key, value in raw.items() if key not in standard})
    return CarlaSemanticDemoAttempt(
        name=str(raw.get("name") or fallback_name),
        scenario_type=str(raw.get("scenario_type") or raw.get("type") or fallback_name),
        spawn_index=int(raw.get("spawn_index") or 0),
        destination_index=int(raw.get("destination_index") if raw.get("destination_index") is not None else -1),
        target_speed_mps=float(raw.get("target_speed_mps") if raw.get("target_speed_mps") is not None else 4.0),
        horizon_s=None if raw.get("horizon_s") is None else float(raw.get("horizon_s")),
        parameters=parameters,
    )


def _default_semantic_demo_targets() -> List[CarlaSemanticDemoTarget]:
    right_turn_attempts = []
    for idx, spawn_index in enumerate([67, 95, 107, 58, 124, 115, 0, 32, 88, 116, 52, 71], start=1):
        right_turn_attempts.append(
            CarlaSemanticDemoAttempt(
                name=f"right_turn_pedestrian_yield_{idx:02d}",
                scenario_type="pedestrian_crossing",
                spawn_index=spawn_index,
                destination_index=-1,
                target_speed_mps=5.0,
                horizon_s=60.0,
                parameters={
                    "route_maneuver": "right",
                    "require_crosswalk_near_maneuver": True,
                    "search_start_candidates": True,
                    "start_candidate_count": 32,
                    "start_candidate_radius_m": 130.0,
                    "prefer_rightmost_start_lane": True,
                    "require_rightmost_start_lane": False,
                    "disallow_lane_change_before_maneuver": True,
                    "maneuver_min_progress_m": 26.0,
                    "maneuver_max_progress_m": 55.0,
                    "min_post_maneuver_length_m": 22.0,
                    "progress_m": "auto",
                    "crosswalk_maneuver_offset_m": 5.5,
                    "crosswalk_search_window_m": 42.0,
                    "crosswalk_max_route_distance_m": 12.0,
                    "lateral_offset_m": -5.2 if idx % 2 else 5.2,
                    "actor_speed_mps": 2.45,
                    "pedestrian_count": 6,
                    "pedestrian_second_wave_count": 4,
                    "pedestrian_group_spacing_m": 0.82,
                    "pedestrian_group_depth_m": 0.48,
                    "pedestrian_trigger_mode": "time",
                    "pedestrian_trigger_time_s": 9.0,
                    "pedestrian_start_advance_s": 0.0,
                    "pedestrian_start_delay_step_s": 0.22,
                    "pedestrian_second_wave_delay_s": 2.8,
                    "pedestrian_ego_path_corridor_m": 1.85,
                    "pedestrian_ego_path_approach_margin_m": 0.25,
                    "pedestrian_yield_trigger_distance_m": 8.0,
                    "pedestrian_yield_trigger_gap_m": 8.0,
                    "trigger_distance_before_crossing_m": 9.0,
                    "pedestrian_trigger_buffer_m": 0.0,
                    "event_end_after_crossing_m": 70.0,
                    "require_crosswalk": True,
                    "ambient_safety_mode": "emergency_only",
                    "ambient_vehicle_count": 36,
                    "ambient_target_speed_mps": 5.8,
                    "ambient_speed_variation_mps": 1.0,
                    "ambient_lane_change_percentage": 18.0,
                    "ambient_max_lateral_m": 80.0,
                    "ambient_preferred_progress_m": 54.0,
                    "ambient_min_progress_m": 20.0,
                    "ambient_max_progress_m": 140.0,
                },
            )
        )
    return [
        CarlaSemanticDemoTarget(
            target_id="right_turn_pedestrian_yield",
            attempts=right_turn_attempts,
        ),
    ]


def _promote_carla_report(*, report: Mapping[str, Any], source_dir: Path, target_dir: Path) -> Dict[str, Any]:
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.copytree(source_dir, target_dir)
    promoted = json.loads(json.dumps(report))
    _rewrite_paths(promoted, source_dir=source_dir, target_dir=target_dir)
    promoted["output_dir"] = str(target_dir)
    report_path = target_dir / "carla_vision_closed_loop_report.json"
    report_path.write_text(json.dumps(promoted, indent=2), encoding="utf-8")
    return promoted


def _rewrite_paths(value: Any, *, source_dir: Path, target_dir: Path) -> Any:
    source_text = str(source_dir)
    target_text = str(target_dir)
    if isinstance(value, dict):
        for key, item in list(value.items()):
            if isinstance(item, str):
                value[key] = item.replace(source_text, target_text)
            else:
                _rewrite_paths(item, source_dir=source_dir, target_dir=target_dir)
    elif isinstance(value, list):
        for item in value:
            _rewrite_paths(item, source_dir=source_dir, target_dir=target_dir)
    return value


def _write_semantic_demo_report(
    *,
    output_dir: Path,
    carla_root: Path,
    checkpoint_path: Path,
    town: str,
    reports: Sequence[Mapping[str, Any]],
    target_results: Sequence[Mapping[str, Any]],
    started_at: float,
) -> Dict[str, Any]:
    scenario_rows = []
    for report in reports:
        scenario = dict(report.get("scenario") or {})
        metrics = dict(report.get("metrics") or {})
        media = dict(report.get("media") or {})
        attribution = dict(report.get("control_attribution") or {})
        scenario_rows.append(
            {
                "name": str(scenario.get("name") or ""),
                "type": str(scenario.get("type") or ""),
                "actor_count": int(scenario.get("actor_count") or 0),
                "output_dir": str(report.get("output_dir") or ""),
                "metrics": metrics,
                "control_attribution": attribution,
                "media": media,
                "route_length_m": float(report.get("route_length_m") or 0.0),
            }
        )
    summary = {
        "schema": "carla_vision_closed_loop_batch_v1",
        "output_dir": str(output_dir),
        "carla_root": str(carla_root),
        "checkpoint_path": str(checkpoint_path),
        "town": str(town),
        "requested_town": str(town),
        "scenario_count": len(scenario_rows),
        "semantic_targets": [
            {
                "target_id": row.get("target_id"),
                "status": row.get("status"),
                "selected_output_dir": row.get("selected_output_dir"),
            }
            for row in target_results
        ],
        "aggregate": _aggregate_scenarios(scenario_rows),
        "scenarios": scenario_rows,
        "runtime_s": round(time.time() - float(started_at), 3),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / CARLA_SEMANTIC_DEMO_REPORT_NAME).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "carla_semantic_demo_report.md").write_text(_render_demo_markdown(summary), encoding="utf-8")
    return summary


def _aggregate_scenarios(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    metrics = [dict(row.get("metrics") or {}) for row in rows]
    attributions = [dict(row.get("control_attribution") or {}) for row in rows]
    return {
        "mean_route_completion": mean([float(row.get("route_completion") or 0.0) for row in metrics]) if metrics else 0.0,
        "mean_driving_score": mean([float(row.get("driving_score") or 0.0) for row in metrics]) if metrics else 0.0,
        "total_collisions": sum(int(row.get("collision_count") or 0) for row in metrics),
        "total_frames": sum(int(row.get("frame_count") or 0) for row in metrics),
        "total_traffic_manager_vehicles": sum(int(row.get("traffic_manager_vehicle_count") or 0) for row in attributions),
        "total_scripted_vehicles": sum(int(row.get("scripted_vehicle_count") or 0) for row in attributions),
        "total_crosswalk_pedestrians": sum(int(row.get("crosswalk_walker_count") or 0) for row in attributions),
    }


def _audit_brief(audit: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if not audit:
        return {}
    return {
        "status": audit.get("status"),
        "scenario_count": audit.get("scenario_count"),
        "failure_count": audit.get("failure_count"),
        "warning_count": audit.get("warning_count"),
        "failures": list(audit.get("failures") or []),
        "warnings": list(audit.get("warnings") or []),
        "summary": dict(audit.get("summary") or {}),
    }


def _mining_status(target_results: Sequence[Mapping[str, Any]]) -> str:
    if not target_results:
        return "fail"
    passed = sum(1 for row in target_results if row.get("status") == "pass")
    if passed == len(target_results):
        return "pass"
    if passed > 0:
        return "partial"
    return "fail"


def _render_demo_markdown(summary: Mapping[str, Any]) -> str:
    aggregate = dict(summary.get("aggregate") or {})
    lines = [
        "# CARLA Semantic Demo Report",
        "",
        f"- Town: `{summary.get('town', '')}`",
        f"- Scenarios: `{summary.get('scenario_count', 0)}`",
        f"- Mean route completion: `{float(aggregate.get('mean_route_completion') or 0.0):.3f}`",
        f"- Mean driving score: `{float(aggregate.get('mean_driving_score') or 0.0):.3f}`",
        f"- Total collisions: `{int(aggregate.get('total_collisions') or 0)}`",
        f"- Traffic Manager vehicles: `{int(aggregate.get('total_traffic_manager_vehicles') or 0)}`",
        f"- Scripted vehicles: `{int(aggregate.get('total_scripted_vehicles') or 0)}`",
        f"- Crosswalk pedestrians: `{int(aggregate.get('total_crosswalk_pedestrians') or 0)}`",
        "",
        "| Scenario | Type | Frames | Completion | Driving Score | Collisions | TM Vehicles | MP4 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in list(summary.get("scenarios") or []):
        metrics = dict(row.get("metrics") or {})
        attribution = dict(row.get("control_attribution") or {})
        media = dict(row.get("media") or {})
        lines.append(
            "| {name} | {stype} | {frames} | {completion:.3f} | {score:.3f} | {collisions} | {tm} | `{mp4}` |".format(
                name=str(row.get("name") or ""),
                stype=str(row.get("type") or ""),
                frames=int(metrics.get("frame_count") or 0),
                completion=float(metrics.get("route_completion") or 0.0),
                score=float(metrics.get("driving_score") or 0.0),
                collisions=int(metrics.get("collision_count") or 0),
                tm=int(attribution.get("traffic_manager_vehicle_count") or 0),
                mp4=str(media.get("video_path") or ""),
            )
        )
    lines.append("")
    return "\n".join(lines)


def _render_mining_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# CARLA Semantic Demo Mining",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Targets: `{report.get('passed_target_count')}/{report.get('target_count')}`",
        f"- Attempts: `{report.get('attempt_count')}`",
        f"- Report: `{report.get('report_path', '')}`",
        f"- Audit: `{report.get('audit_path', '')}`",
        "",
        "| Target | Status | Attempts | Selected Output |",
        "| --- | --- | ---: | --- |",
    ]
    for row in list(report.get("targets") or []):
        lines.append(
            "| {target} | `{status}` | {attempts} | `{output}` |".format(
                target=str(row.get("target_id") or ""),
                status=str(row.get("status") or ""),
                attempts=int(row.get("attempt_count") or 0),
                output=str(row.get("selected_output_dir") or ""),
            )
        )
    lines.append("")
    return "\n".join(lines)


def _safe_filename(value: str) -> str:
    text = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value).strip().lower())
    return text.strip("_") or "item"


def _terminate_carla_processes_for_port(port: int) -> None:
    pattern = f"carla-rpc-port={int(port)}"
    try:
        subprocess.run(["pkill", "-f", pattern], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass

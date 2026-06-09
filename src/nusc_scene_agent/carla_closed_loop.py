from __future__ import annotations

import importlib
import math
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from nusc_scene_agent.geometry import normalize_angle


DEFAULT_CARLA_ROOT = Path("external/carla/latest")

MAX_STEER_RAD = 0.65
WHEEL_BASE_M = 2.85
MAX_ACCEL_MPS2 = 3.0
MAX_BRAKE_MPS2 = 5.0


@dataclass(frozen=True)
class PurePursuitConfig:
    lookahead_m: float = 8.0
    speed_kp: float = 0.35
    max_steer_rad: float = MAX_STEER_RAD
    max_accel_mps2: float = MAX_ACCEL_MPS2
    max_brake_mps2: float = MAX_BRAKE_MPS2
    wheel_base_m: float = WHEEL_BASE_M


def inspect_carla_runtime(carla_root: Path = DEFAULT_CARLA_ROOT) -> Dict[str, Any]:
    carla_root = Path(carla_root)
    launch_script = carla_root / "CarlaUE4.sh"
    binary = carla_root / "CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping"
    whls = sorted((carla_root / "PythonAPI/carla/dist").glob("carla-*.whl"))
    python_tag = "cp{0}{1}".format(sys.version_info.major, sys.version_info.minor)
    matching_whls = [path for path in whls if python_tag in path.name]
    map_root = carla_root / "CarlaUE4/Content/Carla/Maps"
    maps = _discover_carla_maps(map_root)
    return {
        "schema": "carla_runtime_inventory_v1",
        "carla_root": str(carla_root),
        "exists": carla_root.exists(),
        "launch_script": str(launch_script),
        "launch_script_exists": launch_script.exists(),
        "binary": str(binary),
        "binary_exists": binary.exists(),
        "python_tag": python_tag,
        "python_api_wheels": [str(path) for path in whls],
        "matching_python_api_wheel": str(matching_whls[0]) if matching_whls else "",
        "map_count": len(maps),
        "maps": maps,
        "headless_launch_command": format_carla_launch_command(
            build_carla_launch_command(carla_root, render_offscreen=True, null_rhi=False)
        ),
        "no_render_launch_command": format_carla_launch_command(
            build_carla_launch_command(carla_root, render_offscreen=False, null_rhi=True)
        ),
    }


def build_carla_launch_command(
    carla_root: Path = DEFAULT_CARLA_ROOT,
    *,
    render_offscreen: bool = True,
    null_rhi: bool = False,
    port: int = 2000,
    quality_level: str = "Low",
    fps: int = 20,
) -> List[str]:
    command = [
        str(Path(carla_root) / "CarlaUE4.sh"),
        "-nosound",
        f"-carla-rpc-port={int(port)}",
        f"-quality-level={quality_level}",
    ]
    if render_offscreen:
        command.append("-RenderOffScreen")
    if null_rhi:
        command.append("-NullRHI")
    if fps > 0:
        command.extend(["-benchmark", f"-fps={int(fps)}"])
    return command


def format_carla_launch_command(command: Sequence[str], *, cuda_visible_devices: str = "") -> str:
    prefix = f"CUDA_VISIBLE_DEVICES={cuda_visible_devices} " if cuda_visible_devices else ""
    return prefix + " ".join(_shell_quote(part) for part in command)


def run_carla_connection_smoke(
    carla_root: Path = DEFAULT_CARLA_ROOT,
    *,
    host: str = "127.0.0.1",
    port: int = 2000,
    timeout_s: float = 10.0,
    town: str = "",
    load_town: bool = False,
) -> Dict[str, Any]:
    _add_carla_python_paths(Path(carla_root))
    carla = importlib.import_module("carla")
    client = carla.Client(host, int(port))
    client.set_timeout(float(timeout_s))
    world = client.load_world(town) if town and load_town else client.get_world()
    carla_map = world.get_map()
    available_maps = [str(item) for item in client.get_available_maps()]
    spawn_points = carla_map.get_spawn_points()
    return {
        "schema": "carla_connection_smoke_v1",
        "host": host,
        "port": int(port),
        "world_map": str(carla_map.name),
        "available_map_count": len(available_maps),
        "available_maps": available_maps[:32],
        "spawn_point_count": len(spawn_points),
    }


def pure_pursuit_control(
    *,
    state: Mapping[str, Any],
    route: Sequence[Mapping[str, Any]],
    target_speed_mps: float,
    config: PurePursuitConfig | None = None,
) -> Dict[str, float]:
    config = config or PurePursuitConfig()
    lookahead = _lookahead_waypoint(state, route, config.lookahead_m)
    dx = float(lookahead["x"]) - float(state["x"])
    dy = float(lookahead["y"]) - float(state["y"])
    target_angle = math.atan2(dy, dx)
    heading_error = normalize_angle(target_angle - float(state["yaw"]))
    steer_rad = math.atan2(
        2.0 * config.wheel_base_m * math.sin(heading_error),
        max(config.lookahead_m, 1e-3),
    )
    steer = max(min(steer_rad / config.max_steer_rad, 1.0), -1.0)
    speed_error = float(target_speed_mps) - float(state["speed_mps"])
    accel_command = config.speed_kp * speed_error
    throttle = max(min(accel_command / config.max_accel_mps2, 1.0), 0.0)
    brake = max(min(-accel_command / config.max_brake_mps2, 1.0), 0.0)
    return {
        "steer": steer,
        "throttle": throttle,
        "brake": brake,
        "target_speed_mps": float(target_speed_mps),
    }


def project_to_route(point_xy: Tuple[float, float], route: Sequence[Mapping[str, Any]]) -> Tuple[float, float, int]:
    if len(route) < 2:
        return 0.0, 0.0, 0
    px, py = float(point_xy[0]), float(point_xy[1])
    best_distance = float("inf")
    best_progress = 0.0
    best_lateral = 0.0
    best_segment = 0
    cumulative = 0.0
    for idx, (start, end) in enumerate(zip(route[:-1], route[1:])):
        sx, sy = float(start["x"]), float(start["y"])
        ex, ey = float(end["x"]), float(end["y"])
        vx, vy = ex - sx, ey - sy
        seg_len2 = vx * vx + vy * vy
        if seg_len2 <= 1e-9:
            continue
        t = max(min(((px - sx) * vx + (py - sy) * vy) / seg_len2, 1.0), 0.0)
        proj_x = sx + t * vx
        proj_y = sy + t * vy
        dx = px - proj_x
        dy = py - proj_y
        distance = math.hypot(dx, dy)
        seg_len = math.sqrt(seg_len2)
        cross = vx * (py - sy) - vy * (px - sx)
        lateral = math.copysign(distance, cross)
        if distance < best_distance:
            best_distance = distance
            best_progress = cumulative + t * seg_len
            best_lateral = lateral
            best_segment = idx
        cumulative += seg_len
    return best_progress, best_lateral, best_segment


def _lookahead_waypoint(
    state: Mapping[str, Any],
    route: Sequence[Mapping[str, Any]],
    lookahead_m: float,
) -> Mapping[str, Any]:
    if not route:
        return {"x": float(state.get("x") or 0.0), "y": float(state.get("y") or 0.0)}
    progress, _, _ = project_to_route((float(state["x"]), float(state["y"])), route)
    target_progress = progress + lookahead_m
    cumulative = 0.0
    for start, end in zip(route[:-1], route[1:]):
        sx, sy = float(start["x"]), float(start["y"])
        ex, ey = float(end["x"]), float(end["y"])
        seg_len = math.hypot(ex - sx, ey - sy)
        if cumulative + seg_len >= target_progress:
            ratio = (target_progress - cumulative) / max(seg_len, 1e-6)
            return {
                "x": sx + ratio * (ex - sx),
                "y": sy + ratio * (ey - sy),
            }
        cumulative += seg_len
    return route[-1]


def _add_carla_python_paths(carla_root: Path) -> None:
    carla_root = Path(carla_root)
    python_tag = "cp{0}{1}".format(sys.version_info.major, sys.version_info.minor)
    whls = sorted((carla_root / "PythonAPI/carla/dist").glob(f"carla-*{python_tag}*.whl"))
    candidate_paths = []
    if whls:
        candidate_paths.append(_extract_carla_wheel(whls[0]))
    candidate_paths.append(carla_root / "PythonAPI/carla")
    for path in candidate_paths:
        text = str(path)
        if path.exists() and text not in sys.path:
            sys.path.insert(0, text)


def _extract_carla_wheel(wheel_path: Path) -> Path:
    wheel_path = Path(wheel_path)
    output_dir = wheel_path.parent / "extracted" / wheel_path.stem
    marker = output_dir / ".extracted"
    if marker.exists():
        return output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(wheel_path) as archive:
        archive.extractall(output_dir)
    marker.write_text(str(wheel_path), encoding="utf-8")
    return output_dir


def _discover_carla_maps(map_root: Path) -> List[str]:
    if not map_root.exists():
        return []
    maps = []
    for path in sorted(map_root.rglob("*.umap")):
        name = path.stem
        if "_BuiltData" in name or "_Tile_" in name:
            continue
        if not (name.startswith("Town") or name in {"EmptyMap", "OpenDriveMap"}):
            continue
        maps.append(name)
    return sorted(set(maps))


def _shell_quote(value: str) -> str:
    if not value:
        return "''"
    if all(ch.isalnum() or ch in "/._-=:" for ch in value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"

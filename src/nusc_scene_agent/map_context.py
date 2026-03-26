from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from nuscenes.map_expansion.map_api import NuScenesMap
from shapely.geometry import MultiPolygon, Polygon

from nusc_scene_agent.data_utils import normalize_map_layout
from nusc_scene_agent.geometry import ego_xy_to_global, global_xy_to_anchor_ego
from nusc_scene_agent.models import RetrievalCandidate


MAP_PATCH_LAYERS = [
    "drivable_area",
    "lane",
    "lane_connector",
    "ped_crossing",
    "walkway",
]

_MAP_CACHE: Dict[Tuple[str, str], Optional[NuScenesMap]] = {}


def _metadata_value(conn: sqlite3.Connection, key: str) -> Optional[str]:
    row = conn.execute("SELECT value FROM metadata WHERE key = ?", (key,)).fetchone()
    if not row:
        return None
    return str(row[0])


def _get_map_api(dataroot: Path, location: str) -> Optional[NuScenesMap]:
    cache_key = (str(dataroot.resolve()), location)
    if cache_key in _MAP_CACHE:
        return _MAP_CACHE[cache_key]

    normalize_map_layout(dataroot)
    try:
        _MAP_CACHE[cache_key] = NuScenesMap(dataroot=str(dataroot), map_name=location)
    except Exception:
        _MAP_CACHE[cache_key] = None
    return _MAP_CACHE[cache_key]


def _active_point_layers(query_behaviors: List[str]) -> List[str]:
    layers = ["drivable_area", "lane"]
    if "crossing" in query_behaviors:
        layers.extend(["ped_crossing", "walkway"])
    return layers


def _point_layers(map_api: NuScenesMap, x: float, y: float, layer_names: List[str]) -> Dict[str, str]:
    return {layer: map_api.record_on_point(x, y, layer) for layer in layer_names}


def _record_polygons(map_api: NuScenesMap, layer_name: str, token: str) -> List[Polygon]:
    if not token:
        return []

    record = map_api.get(layer_name, token)
    if layer_name == "drivable_area":
        polygon_tokens = list(record["polygon_tokens"])
    else:
        polygon_tokens = [record["polygon_token"]]

    polygons: List[Polygon] = []
    for polygon_token in polygon_tokens:
        polygon = map_api.extract_polygon(polygon_token)
        if polygon.is_empty:
            continue
        if isinstance(polygon, Polygon):
            polygons.append(polygon)
        elif isinstance(polygon, MultiPolygon):
            polygons.extend([geom for geom in polygon.geoms if not geom.is_empty])
    return polygons


def _polygons_to_anchor(
    polygons: List[Polygon],
    anchor_xy: np.ndarray,
    anchor_yaw: float,
) -> List[np.ndarray]:
    transformed: List[np.ndarray] = []
    for polygon in polygons:
        exterior = np.asarray(polygon.exterior.coords, dtype=float)
        if exterior.size == 0:
            continue
        transformed.append(global_xy_to_anchor_ego(exterior[:, :2], anchor_xy, anchor_yaw))
    return transformed


def _collect_patch_geometries(
    map_api: NuScenesMap,
    anchor_xy: np.ndarray,
    anchor_yaw: float,
    patch_box: Tuple[float, float, float, float],
) -> Dict[str, List[np.ndarray]]:
    patch_records = map_api.get_records_in_patch(patch_box, layer_names=MAP_PATCH_LAYERS, mode="intersect")
    geometries: Dict[str, List[np.ndarray]] = {}

    for layer_name in MAP_PATCH_LAYERS:
        tokens = patch_records.get(layer_name, [])
        max_tokens = 12 if layer_name == "drivable_area" else 24
        layer_polygons: List[np.ndarray] = []
        for token in tokens[:max_tokens]:
            polygons = _record_polygons(map_api, layer_name, token)
            layer_polygons.extend(_polygons_to_anchor(polygons, anchor_xy, anchor_yaw))
        if layer_polygons:
            geometries[layer_name] = layer_polygons

    return geometries


def build_case_map_context(
    conn: sqlite3.Connection,
    candidate: RetrievalCandidate,
    timeline: pd.DataFrame,
    ego_window: pd.DataFrame,
    query_behaviors: Optional[List[str]] = None,
    include_patch_geometries: bool = True,
    patch_radius_m: float = 35.0,
) -> Tuple[Dict[str, object], Dict[str, List[np.ndarray]]]:
    dataroot_value = _metadata_value(conn, "dataroot")
    if not dataroot_value:
        return {"available": False, "reason": "missing_dataroot"}, {}

    map_api = _get_map_api(Path(dataroot_value), candidate.location)
    if map_api is None:
        return {"available": False, "reason": "missing_map"}, {}

    anchor_rows = ego_window.loc[ego_window["sample_idx"] == candidate.sample_idx]
    if anchor_rows.empty:
        return {"available": False, "reason": "missing_anchor_sample"}, {}

    anchor_row = anchor_rows.iloc[0]
    anchor_xy = np.asarray([anchor_row["ego_x"], anchor_row["ego_y"]], dtype=float)
    anchor_yaw = float(anchor_row["ego_yaw"])
    point_layers = _active_point_layers(query_behaviors or [])
    patch_box = (
        float(anchor_xy[0] - patch_radius_m),
        float(anchor_xy[1] - patch_radius_m),
        float(anchor_xy[0] + patch_radius_m),
        float(anchor_xy[1] + patch_radius_m),
    )

    ego_layers_anchor = _point_layers(map_api, float(anchor_xy[0]), float(anchor_xy[1]), point_layers)
    ego_closest_lane = map_api.get_closest_lane(float(anchor_xy[0]), float(anchor_xy[1]), radius=8.0)

    actor_layer_sequence: List[Dict[str, str]] = []
    actor_lane_ids: List[str] = []
    actor_global_points: List[List[float]] = []
    actor_layers_anchor: Dict[str, str] = {}
    actor_anchor_lane = ""

    ordered_timeline = timeline.sort_values("sample_idx")
    for _, row in ordered_timeline.iterrows():
        actor_global_xy = ego_xy_to_global(
            [row["x_ego"], row["y_ego"]],
            [row["ego_x"], row["ego_y"]],
            float(row["ego_yaw"]),
        )
        actor_layers = _point_layers(map_api, float(actor_global_xy[0]), float(actor_global_xy[1]), point_layers)
        actor_lane = map_api.get_closest_lane(float(actor_global_xy[0]), float(actor_global_xy[1]), radius=8.0)

        actor_global_points.append([float(actor_global_xy[0]), float(actor_global_xy[1])])
        actor_layer_sequence.append(actor_layers)
        actor_lane_ids.append(actor_lane)

        if row["sample_token"] == candidate.sample_token:
            actor_layers_anchor = actor_layers
            actor_anchor_lane = actor_lane

    unique_crosswalks = sorted({item["ped_crossing"] for item in actor_layer_sequence if item.get("ped_crossing")})
    unique_walkways = sorted({item["walkway"] for item in actor_layer_sequence if item.get("walkway")})
    unique_drivable = sorted({item["drivable_area"] for item in actor_layer_sequence if item.get("drivable_area")})
    actor_lane_ids = [lane_id for lane_id in actor_lane_ids if lane_id]

    map_context: Dict[str, object] = {
        "available": True,
        "patch_radius_m": float(patch_radius_m),
        "ego_layers_anchor": dict(ego_layers_anchor),
        "actor_layers_anchor": dict(actor_layers_anchor),
        "ego_closest_lane": str(ego_closest_lane or ""),
        "actor_closest_lane_anchor": str(actor_anchor_lane or ""),
        "ego_in_lane_anchor": bool(
            ego_layers_anchor.get("lane") or ego_layers_anchor.get("lane_connector") or ego_closest_lane
        ),
        "ego_on_drivable_anchor": bool(ego_layers_anchor.get("drivable_area")),
        "actor_on_crosswalk_any": bool(unique_crosswalks),
        "actor_on_walkway_any": bool(unique_walkways),
        "actor_on_drivable_any": bool(unique_drivable),
        "actor_uses_ego_lane_any": bool(ego_closest_lane and ego_closest_lane in actor_lane_ids),
        "shares_lane_at_anchor": bool(ego_closest_lane and actor_anchor_lane and ego_closest_lane == actor_anchor_lane),
        "actor_crosswalk_count": int(len(unique_crosswalks)),
        "actor_walkway_count": int(len(unique_walkways)),
        "actor_drivable_count": int(len(unique_drivable)),
        "actor_global_track_xy": actor_global_points,
        "patch_box_global": [float(value) for value in patch_box],
    }

    patch_geometries = {}
    if include_patch_geometries:
        patch_geometries = _collect_patch_geometries(map_api, anchor_xy, anchor_yaw, patch_box)
    return map_context, patch_geometries

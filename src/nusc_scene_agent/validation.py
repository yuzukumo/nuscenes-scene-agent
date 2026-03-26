from __future__ import annotations

import sqlite3
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from nusc_scene_agent.map_context import build_case_map_context
from nusc_scene_agent.models import ParsedQuery, RetrievalCandidate, ValidatedCase


def detect_crossing_like(track: pd.DataFrame) -> bool:
    if len(track) < 2:
        return False
    ordered = track.sort_values("sample_idx")
    y_values = ordered["y_ego"].to_numpy(dtype=float)
    x_values = ordered["x_ego"].to_numpy(dtype=float)
    sign_change = np.any(y_values[:-1] * y_values[1:] <= 0.0)
    lateral_span = float(np.max(y_values) - np.min(y_values))
    front_presence = bool(np.any((x_values > -5.0) & (x_values < 20.0)))
    return sign_change and lateral_span >= 2.0 and front_presence


def detect_map_supported_crossing_like(track: pd.DataFrame, map_context: Dict[str, object]) -> bool:
    if track.empty:
        return False
    if not (map_context.get("actor_on_crosswalk_any") or map_context.get("actor_on_walkway_any")):
        return False

    ordered = track.sort_values("sample_idx")
    x_values = ordered["x_ego"].to_numpy(dtype=float)
    y_values = ordered["y_ego"].to_numpy(dtype=float)
    front_presence = bool(np.any((x_values > -5.0) & (x_values < 20.0)))
    path_crossing = bool(np.min(x_values) <= 0.0 <= np.max(x_values))
    near_lane = float(np.min(np.abs(y_values))) <= 8.0
    lateral_motion = float(np.max(y_values) - np.min(y_values)) >= 0.5
    return front_presence and path_crossing and near_lane and lateral_motion


def detect_cut_in_like(track: pd.DataFrame, positions: List[str] | None = None) -> bool:
    if len(track) < 3:
        return False
    ordered = track.sort_values("sample_idx")
    positions = positions or []

    if not positions:
        in_corridor = ordered[(ordered["x_ego"] > 0.0) & (ordered["x_ego"] < 25.0) & (ordered["y_ego"].abs() < 2.0)]
        if in_corridor.empty:
            return False

        corridor_idx = int(in_corridor.iloc[0]["sample_idx"])
        earlier = ordered[ordered["sample_idx"] < corridor_idx]
        if earlier.empty:
            return False

        started_from_side = bool((earlier["y_ego"].abs() >= 2.5).any())
        lateral_motion = float(ordered["y_ego"].abs().max() - ordered["y_ego"].abs().min()) >= 2.0
        closing_motion = float(ordered["x_ego"].diff().fillna(0.0).mean()) < 0.0
        return started_from_side and lateral_motion and closing_motion

    if "left" in positions:
        side_values = ordered["y_ego"].to_numpy(dtype=float)
        valid_side = side_values >= 0.0
    elif "right" in positions:
        side_values = -ordered["y_ego"].to_numpy(dtype=float)
        valid_side = side_values >= 0.0
    else:
        side_values = ordered["y_ego"].abs().to_numpy(dtype=float)
        valid_side = side_values >= 0.0

    ordered = ordered.loc[valid_side].copy()
    if ordered.empty:
        return False
    side_values = side_values[valid_side]

    front_presence = bool(((ordered["x_ego"] > -2.0) & (ordered["x_ego"] < 25.0)).any())
    merge_zone_mask = side_values <= 3.5
    if not merge_zone_mask.any():
        return False

    merge_zone_idx = int(np.argmax(merge_zone_mask))
    earlier_side = side_values[:merge_zone_idx]
    if earlier_side.size == 0:
        return False

    started_from_side = float(np.max(earlier_side)) >= 5.0
    lateral_motion = float(np.max(side_values) - np.min(side_values)) >= 3.0
    ends_near_lane = float(np.min(side_values[merge_zone_idx:])) <= 3.5
    closing_motion = float(ordered["x_ego"].diff().fillna(0.0).mean()) < -0.5
    return front_presence and started_from_side and lateral_motion and ends_near_lane and closing_motion


def detect_map_supported_cut_in_like(track: pd.DataFrame, map_context: Dict[str, object]) -> bool:
    if len(track) < 3:
        return False
    if not (map_context.get("actor_uses_ego_lane_any") or map_context.get("shares_lane_at_anchor")):
        return False

    ordered = track.sort_values("sample_idx")
    abs_y = ordered["y_ego"].abs().to_numpy(dtype=float)
    x_values = ordered["x_ego"].to_numpy(dtype=float)
    front_presence = bool(np.any((x_values > -2.0) & (x_values < 25.0)))
    started_from_side = float(np.max(abs_y[: max(2, len(abs_y) // 2)])) >= 3.5
    ends_closer_to_center = float(abs_y[0] - abs_y[-1]) >= 0.3
    near_lane_boundary = float(np.min(abs_y)) <= 4.2
    closing_motion = float(np.mean(np.diff(x_values))) < -0.5
    return front_presence and started_from_side and ends_closer_to_center and near_lane_boundary and closing_motion


def detect_oncoming_like(track: pd.DataFrame) -> bool:
    if track.empty:
        return False
    ordered = track.sort_values("sample_idx")
    heading_gap = np.abs(np.pi - np.abs(ordered["heading_delta"].to_numpy(dtype=float)))
    closing = float(ordered["rel_vx"].mean()) < -1.0
    front_presence = float((ordered["x_ego"] > 0.0).mean()) >= 0.5
    return front_presence and closing and float(np.nanmean(heading_gap)) <= 1.0


def detect_stopped_lead_like(track: pd.DataFrame) -> bool:
    if track.empty:
        return False
    ordered = track.sort_values("sample_idx")
    front_corridor = bool(
        ((ordered["x_ego"] > 0.0) & (ordered["x_ego"] < 25.0) & (ordered["y_ego"].abs() < 3.0)).any()
    )
    stationary_frames = int((ordered["speed"] <= 1.0).sum())
    return front_corridor and stationary_frames >= max(2, len(ordered) // 2)


def _map_behavior_score(query: ParsedQuery, map_context: Dict[str, object]) -> float:
    if not map_context.get("available"):
        return 0.0

    if "crossing" in query.behaviors:
        if map_context.get("actor_on_crosswalk_any"):
            return 1.0
        if map_context.get("actor_on_walkway_any"):
            return 0.6
        return 0.0

    if "cut_in" in query.behaviors:
        if map_context.get("actor_uses_ego_lane_any"):
            return 1.0
        if map_context.get("actor_on_drivable_any") and map_context.get("ego_on_drivable_anchor"):
            return 0.5
        return 0.0

    if "stopped_lead" in query.behaviors:
        if map_context.get("shares_lane_at_anchor"):
            return 1.0
        if map_context.get("actor_on_drivable_any") and map_context.get("ego_on_drivable_anchor"):
            return 0.5
        return 0.0

    if "oncoming" in query.behaviors:
        if map_context.get("shares_lane_at_anchor"):
            return 1.0
        if map_context.get("actor_uses_ego_lane_any"):
            return 0.6
        return 0.0

    return 1.0 if map_context.get("actor_on_drivable_any") else 0.0


def _load_validation_window(
    conn: sqlite3.Connection, candidate: RetrievalCandidate, window_radius: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    lower = max(candidate.sample_idx - window_radius, 0)
    upper = candidate.sample_idx + window_radius

    timeline = pd.read_sql_query(
        """
        SELECT
            a.*,
            s.timestamp_us,
            s.location,
            s.ego_x,
            s.ego_y,
            s.ego_yaw
        FROM agents a
        JOIN samples s ON s.sample_token = a.sample_token
        WHERE a.scene_token = ?
          AND a.instance_token = ?
          AND a.sample_idx BETWEEN ? AND ?
        ORDER BY a.sample_idx ASC
        """,
        conn,
        params=(candidate.scene_token, candidate.instance_token, lower, upper),
    )

    context_agents = pd.read_sql_query(
        """
        SELECT *
        FROM agents
        WHERE sample_token = ?
        ORDER BY distance ASC
        LIMIT 48
        """,
        conn,
        params=(candidate.sample_token,),
    )

    ego_window = pd.read_sql_query(
        """
        SELECT *
        FROM samples
        WHERE scene_token = ?
          AND sample_idx BETWEEN ? AND ?
        ORDER BY sample_idx ASC
        """,
        conn,
        params=(candidate.scene_token, lower, upper),
    )

    return timeline, context_agents, ego_window


def validate_candidate(
    conn: sqlite3.Connection,
    query: ParsedQuery,
    candidate: RetrievalCandidate,
    include_map_geometries: bool = True,
    window_radius: int = 4,
) -> ValidatedCase:
    timeline, context_agents, ego_window = _load_validation_window(conn, candidate, window_radius)
    if timeline.empty:
        return ValidatedCase(
            query=query,
            candidate=candidate,
            validation_score=0.0,
            passed=False,
            behavior_matches={},
            evidence={},
            notes=["No actor timeline was found in the validation window."],
            timeline=timeline,
            context_agents=context_agents,
            ego_window=ego_window,
        )

    anchor_time = float(
        timeline.loc[timeline["sample_token"] == candidate.sample_token, "timestamp_us"].iloc[0]
    )
    timeline = timeline.copy()
    timeline["t_sec"] = (timeline["timestamp_us"].astype(float) - anchor_time) / 1_000_000.0
    map_context, map_geometries = build_case_map_context(
        conn,
        candidate,
        timeline,
        ego_window,
        query_behaviors=query.behaviors,
        include_patch_geometries=include_map_geometries,
    )

    behavior_matches: Dict[str, bool] = {}
    if "crossing" in query.behaviors:
        behavior_matches["crossing"] = detect_crossing_like(timeline) or detect_map_supported_crossing_like(
            timeline, map_context
        )
    if "cut_in" in query.behaviors:
        behavior_matches["cut_in"] = detect_cut_in_like(timeline, positions=query.positions) or detect_map_supported_cut_in_like(
            timeline, map_context
        )
    if "oncoming" in query.behaviors:
        behavior_matches["oncoming"] = detect_oncoming_like(timeline)
    if "stopped_lead" in query.behaviors:
        behavior_matches["stopped_lead"] = detect_stopped_lead_like(timeline)

    min_distance = float(timeline["distance"].min())
    finite_ttc = timeline["ttc"].dropna()
    min_ttc = float(finite_ttc.min()) if not finite_ttc.empty else float("inf")
    max_closing_speed = float(max(0.0, -timeline["rel_vx"].min()))
    lateral_span = float(timeline["y_ego"].max() - timeline["y_ego"].min())
    duration_s = float(timeline["t_sec"].max() - timeline["t_sec"].min()) if len(timeline) > 1 else 0.0

    proximity_score = max(0.0, 1.0 - min_distance / max(query.near_distance_m, 1.0))
    ttc_score = 0.0
    if np.isfinite(min_ttc):
        ttc_score = max(0.0, 1.0 - min_ttc / max(query.max_ttc_s, 0.5))
    persistence_score = min(len(timeline) / 4.0, 1.0)
    behavior_score = 1.0 if not behavior_matches else sum(behavior_matches.values()) / len(behavior_matches)
    map_score = _map_behavior_score(query, map_context)
    validation_score = round(
        (
            0.35 * proximity_score
            + 0.20 * ttc_score
            + 0.15 * persistence_score
            + 0.20 * behavior_score
            + 0.10 * map_score
        )
        * 100.0,
        2,
    )

    notes: List[str] = [
        "Min distance: {0:.2f} m".format(min_distance),
        "Timeline length: {0} frames over {1:.2f} s".format(len(timeline), duration_s),
    ]
    if np.isfinite(min_ttc):
        notes.append("Minimum TTC: {0:.2f} s".format(min_ttc))
    if behavior_matches:
        notes.extend(
            [
                "{0}: {1}".format(name, "matched" if matched else "not matched")
                for name, matched in sorted(behavior_matches.items())
            ]
        )
    if map_context.get("available"):
        notes.append("Map ego lane: {0}".format("yes" if map_context.get("ego_in_lane_anchor") else "no"))
        if "crossing" in query.behaviors:
            notes.append("Map crosswalk hit: {0}".format("yes" if map_context.get("actor_on_crosswalk_any") else "no"))
        if "cut_in" in query.behaviors or "oncoming" in query.behaviors or "stopped_lead" in query.behaviors:
            notes.append("Map shared lane: {0}".format("yes" if map_context.get("shares_lane_at_anchor") else "no"))

    passed = proximity_score >= 0.15 and (all(behavior_matches.values()) if behavior_matches else True)
    evidence = {
        "min_distance_m": round(min_distance, 3),
        "min_ttc_s": None if not np.isfinite(min_ttc) else round(min_ttc, 3),
        "max_closing_speed_mps": round(max_closing_speed, 3),
        "lateral_span_m": round(lateral_span, 3),
        "timeline_frames": int(len(timeline)),
        "duration_s": round(duration_s, 3),
        "anchor_sample_token": candidate.sample_token,
        "anchor_sample_idx": candidate.sample_idx,
        "scene_name": candidate.scene_name,
        "location": candidate.location,
        "map_available": bool(map_context.get("available")),
        "map_score": round(map_score, 3),
        "ego_in_lane_anchor": bool(map_context.get("ego_in_lane_anchor")),
        "ego_on_drivable_anchor": bool(map_context.get("ego_on_drivable_anchor")),
        "actor_on_crosswalk_any": bool(map_context.get("actor_on_crosswalk_any")),
        "actor_on_walkway_any": bool(map_context.get("actor_on_walkway_any")),
        "actor_uses_ego_lane_any": bool(map_context.get("actor_uses_ego_lane_any")),
        "shares_lane_at_anchor": bool(map_context.get("shares_lane_at_anchor")),
    }

    return ValidatedCase(
        query=query,
        candidate=candidate,
        validation_score=validation_score,
        passed=passed,
        behavior_matches=behavior_matches,
        evidence=evidence,
        notes=notes,
        timeline=timeline,
        context_agents=context_agents,
        ego_window=ego_window,
        map_context=map_context,
        map_geometries=map_geometries,
    )

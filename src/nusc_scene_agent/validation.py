from __future__ import annotations

import math
import sqlite3
from dataclasses import asdict, dataclass
from typing import Dict, List, Mapping, Tuple

import numpy as np
import pandas as pd

from nusc_scene_agent.map_context import build_case_map_context
from nusc_scene_agent.models import ParsedQuery, RetrievalCandidate, ValidatedCase


VALIDATION_PROTOCOL_VERSION = "2.2"
VALIDATION_SCORE_WEIGHTS = {
    "proximity": 0.35,
    "ttc": 0.20,
    "persistence": 0.15,
    "behavior": 0.20,
    "map": 0.10,
}
VALIDATION_SCORE_PROFILES = {
    "default": dict(VALIDATION_SCORE_WEIGHTS),
    "equal": {name: 1.0 / len(VALIDATION_SCORE_WEIGHTS) for name in VALIDATION_SCORE_WEIGHTS},
}
VALIDATION_PASS_THRESHOLDS = {
    "proximity_score": 0.15,
}
HEURISTIC_THRESHOLDS = {
    "crossing_lateral_span_m": 2.0,
    "crossing_front_min_x_m": -5.0,
    "crossing_front_max_x_m": 20.0,
    "map_crossing_near_lane_abs_y_m": 8.0,
    "map_crossing_lateral_motion_m": 0.5,
    "cut_in_front_min_x_m": -2.0,
    "cut_in_front_max_x_m": 25.0,
    "cut_in_lane_corridor_abs_y_m": 2.0,
    "cut_in_default_side_start_abs_y_m": 2.5,
    "cut_in_default_lateral_motion_m": 2.0,
    "cut_in_default_closing_dx_m": 0.0,
    "cut_in_side_start_abs_y_m": 5.0,
    "cut_in_lateral_motion_m": 3.0,
    "cut_in_merge_zone_abs_y_m": 3.5,
    "cut_in_closing_dx_m": -0.5,
    "cut_in_closing_min_consecutive_frames": 2,
    "map_cut_in_side_start_abs_y_m": 3.5,
    "map_cut_in_end_delta_abs_y_m": 0.3,
    "map_cut_in_lane_boundary_abs_y_m": 4.2,
    "stopped_lead_front_max_x_m": 25.0,
    "stopped_lead_lane_abs_y_m": 3.0,
    "stopped_lead_speed_mps": 1.0,
}
SIGNED_HEURISTIC_THRESHOLDS = {
    "crossing_front_min_x_m",
    "cut_in_front_min_x_m",
}
CLOSING_DELTA_THRESHOLDS = {
    "cut_in_default_closing_dx_m",
    "cut_in_closing_dx_m",
}
INTEGER_HEURISTIC_THRESHOLDS = {
    "cut_in_closing_min_consecutive_frames",
}


def _finite_float(value: object, label: str) -> float:
    try:
        resolved = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("{0} must be a finite number.".format(label)) from exc
    if not math.isfinite(resolved):
        raise ValueError("{0} must be a finite number.".format(label))
    return resolved


def _validate_heuristic_thresholds(thresholds: Mapping[str, float]) -> Dict[str, float]:
    resolved: Dict[str, float] = {}
    for name, raw_value in thresholds.items():
        value = _finite_float(raw_value, "Validation heuristic threshold {0}".format(name))
        if name in INTEGER_HEURISTIC_THRESHOLDS:
            if value < 1.0 or not value.is_integer():
                raise ValueError(
                    "Validation heuristic threshold {0} must be a positive integer.".format(name)
                )
        elif name in CLOSING_DELTA_THRESHOLDS:
            if value > 0.0:
                raise ValueError(
                    "Validation heuristic threshold {0} must be non-positive.".format(name)
                )
        elif name not in SIGNED_HEURISTIC_THRESHOLDS and value < 0.0:
            raise ValueError(
                "Validation heuristic threshold {0} must be non-negative.".format(name)
            )
        resolved[name] = value

    for lower_name, upper_name in [
        ("crossing_front_min_x_m", "crossing_front_max_x_m"),
        ("cut_in_front_min_x_m", "cut_in_front_max_x_m"),
    ]:
        if resolved[lower_name] >= resolved[upper_name]:
            raise ValueError(
                "Validation heuristic threshold {0} must be smaller than {1}.".format(
                    lower_name, upper_name
                )
            )
    return resolved


def _resolve_heuristic_thresholds(overrides: Mapping[str, float] | None = None) -> Dict[str, float]:
    thresholds = {name: float(value) for name, value in HEURISTIC_THRESHOLDS.items()}
    for name, value in dict(overrides or {}).items():
        if name not in thresholds:
            raise ValueError("Unknown validation heuristic threshold: {0}".format(name))
        thresholds[name] = _finite_float(
            value,
            "Validation heuristic threshold {0}".format(name),
        )
    return _validate_heuristic_thresholds(thresholds)


@dataclass(frozen=True)
class ValidationConfig:
    name: str = "full_system"
    enable_map_context: bool = True
    enable_event_localization: bool = True
    enable_actor_grounding: bool = True
    score_profile: str = "default"
    score_weight_overrides: Dict[str, float] | None = None
    threshold_overrides: Dict[str, float] | None = None
    pass_threshold_overrides: Dict[str, float] | None = None

    def resolved_score_weights(self) -> Dict[str, float]:
        if self.score_profile not in VALIDATION_SCORE_PROFILES:
            raise ValueError(
                "Unknown validation score profile: {0}. Available profiles: {1}".format(
                    self.score_profile, ", ".join(sorted(VALIDATION_SCORE_PROFILES))
                )
            )
        weights = dict(VALIDATION_SCORE_PROFILES[self.score_profile])
        for name, value in dict(self.score_weight_overrides or {}).items():
            if name not in weights:
                raise ValueError("Unknown validation score weight: {0}".format(name))
            value = _finite_float(value, "Validation score weight {0}".format(name))
            if value < 0.0:
                raise ValueError(
                    "Validation score weights must be finite and non-negative: {0}".format(name)
                )
            weights[name] = value
        if not all(math.isfinite(value) and value >= 0.0 for value in weights.values()):
            raise ValueError("Validation score weights must be finite and non-negative.")
        total = sum(weights.values())
        if total <= 0.0:
            raise ValueError("Validation score weights must have a positive sum.")
        return {name: value / total for name, value in weights.items()}

    def resolved_heuristic_thresholds(self) -> Dict[str, float]:
        return _resolve_heuristic_thresholds(self.threshold_overrides)

    def resolved_pass_thresholds(self) -> Dict[str, float]:
        thresholds = {name: float(value) for name, value in VALIDATION_PASS_THRESHOLDS.items()}
        for name, value in dict(self.pass_threshold_overrides or {}).items():
            if name not in thresholds:
                raise ValueError("Unknown validation pass threshold: {0}".format(name))
            thresholds[name] = _finite_float(
                value,
                "Validation pass threshold {0}".format(name),
            )
        for name, value in thresholds.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    "Validation pass threshold {0} must be in [0, 1].".format(name)
                )
        return thresholds

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["score_weights"] = self.resolved_score_weights()
        payload["heuristic_thresholds"] = self.resolved_heuristic_thresholds()
        payload["pass_thresholds"] = self.resolved_pass_thresholds()
        return payload


def detect_crossing_like(track: pd.DataFrame, thresholds: Mapping[str, float] | None = None) -> bool:
    thresholds = _resolve_heuristic_thresholds(thresholds)
    if len(track) < 2:
        return False
    ordered = track.sort_values("sample_idx")
    y_values = ordered["y_ego"].to_numpy(dtype=float)
    x_values = ordered["x_ego"].to_numpy(dtype=float)
    sign_change = np.any(y_values[:-1] * y_values[1:] <= 0.0)
    lateral_span = float(np.max(y_values) - np.min(y_values))
    front_presence = bool(
        np.any(
            (x_values > thresholds["crossing_front_min_x_m"])
            & (x_values < thresholds["crossing_front_max_x_m"])
        )
    )
    return sign_change and lateral_span >= thresholds["crossing_lateral_span_m"] and front_presence


def detect_map_supported_crossing_like(
    track: pd.DataFrame,
    map_context: Dict[str, object],
    thresholds: Mapping[str, float] | None = None,
) -> bool:
    thresholds = _resolve_heuristic_thresholds(thresholds)
    if track.empty:
        return False
    if not (map_context.get("actor_on_crosswalk_any") or map_context.get("actor_on_walkway_any")):
        return False

    ordered = track.sort_values("sample_idx")
    x_values = ordered["x_ego"].to_numpy(dtype=float)
    y_values = ordered["y_ego"].to_numpy(dtype=float)
    front_presence = bool(
        np.any(
            (x_values > thresholds["crossing_front_min_x_m"])
            & (x_values < thresholds["crossing_front_max_x_m"])
        )
    )
    path_crossing = bool(np.min(x_values) <= 0.0 <= np.max(x_values))
    near_lane = float(np.min(np.abs(y_values))) <= thresholds["map_crossing_near_lane_abs_y_m"]
    lateral_motion = (
        float(np.max(y_values) - np.min(y_values)) >= thresholds["map_crossing_lateral_motion_m"]
    )
    return front_presence and path_crossing and near_lane and lateral_motion


def _has_local_closing_motion(
    x_values: np.ndarray,
    threshold_m: float,
    min_consecutive_frames: int | None = None,
) -> bool:
    min_consecutive_frames = int(
        min_consecutive_frames
        or HEURISTIC_THRESHOLDS["cut_in_closing_min_consecutive_frames"]
    )
    if len(x_values) <= min_consecutive_frames:
        return False

    diffs = np.diff(np.asarray(x_values, dtype=float))
    finite = np.isfinite(diffs)
    if not finite.any():
        return False

    closing = finite & (diffs < float(threshold_m))
    run_length = 0
    for value in closing:
        if bool(value):
            run_length += 1
            if run_length >= min_consecutive_frames:
                return True
        else:
            run_length = 0
    return False


def detect_cut_in_like(
    track: pd.DataFrame,
    positions: List[str] | None = None,
    thresholds: Mapping[str, float] | None = None,
) -> bool:
    if len(track) < 3:
        return False
    thresholds = _resolve_heuristic_thresholds(thresholds)
    ordered = track.sort_values("sample_idx")
    positions = positions or []

    if not positions:
        in_corridor = ordered[
            (ordered["x_ego"] > 0.0)
            & (ordered["x_ego"] < thresholds["cut_in_front_max_x_m"])
            & (ordered["y_ego"].abs() < thresholds["cut_in_lane_corridor_abs_y_m"])
        ]
        if in_corridor.empty:
            return False

        corridor_idx = int(in_corridor.iloc[0]["sample_idx"])
        earlier = ordered[ordered["sample_idx"] < corridor_idx]
        if earlier.empty:
            return False

        started_from_side = bool(
            (earlier["y_ego"].abs() >= thresholds["cut_in_default_side_start_abs_y_m"]).any()
        )
        lateral_motion = (
            float(ordered["y_ego"].abs().max() - ordered["y_ego"].abs().min())
            >= thresholds["cut_in_default_lateral_motion_m"]
        )
        closing_motion = _has_local_closing_motion(
            ordered["x_ego"].to_numpy(dtype=float),
            thresholds["cut_in_default_closing_dx_m"],
            int(thresholds["cut_in_closing_min_consecutive_frames"]),
        )
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

    front_presence = bool(
        (
            (ordered["x_ego"] > thresholds["cut_in_front_min_x_m"])
            & (ordered["x_ego"] < thresholds["cut_in_front_max_x_m"])
        ).any()
    )
    merge_zone_mask = side_values <= thresholds["cut_in_merge_zone_abs_y_m"]
    if not merge_zone_mask.any():
        return False

    merge_zone_idx = int(np.argmax(merge_zone_mask))
    earlier_side = side_values[:merge_zone_idx]
    if earlier_side.size == 0:
        return False

    started_from_side = float(np.max(earlier_side)) >= thresholds["cut_in_side_start_abs_y_m"]
    lateral_motion = float(np.max(side_values) - np.min(side_values)) >= thresholds[
        "cut_in_lateral_motion_m"
    ]
    ends_near_lane = float(np.min(side_values[merge_zone_idx:])) <= thresholds["cut_in_merge_zone_abs_y_m"]
    closing_motion = _has_local_closing_motion(
        ordered["x_ego"].to_numpy(dtype=float),
        thresholds["cut_in_closing_dx_m"],
        int(thresholds["cut_in_closing_min_consecutive_frames"]),
    )
    return front_presence and started_from_side and lateral_motion and ends_near_lane and closing_motion


def detect_map_supported_cut_in_like(
    track: pd.DataFrame,
    map_context: Dict[str, object],
    thresholds: Mapping[str, float] | None = None,
) -> bool:
    if len(track) < 3:
        return False
    if not (map_context.get("actor_uses_ego_lane_any") or map_context.get("shares_lane_at_anchor")):
        return False

    thresholds = _resolve_heuristic_thresholds(thresholds)
    ordered = track.sort_values("sample_idx")
    abs_y = ordered["y_ego"].abs().to_numpy(dtype=float)
    x_values = ordered["x_ego"].to_numpy(dtype=float)
    front_presence = bool(
        np.any(
            (x_values > thresholds["cut_in_front_min_x_m"])
            & (x_values < thresholds["cut_in_front_max_x_m"])
        )
    )
    closing_motion = _has_local_closing_motion(
        x_values,
        thresholds["cut_in_closing_dx_m"],
        int(thresholds["cut_in_closing_min_consecutive_frames"]),
    )
    ego_lane = str(map_context.get("ego_closest_lane") or "")
    lane_sequence = [str(item or "") for item in list(map_context.get("actor_lane_sequence") or [])]
    if ego_lane and lane_sequence:
        midpoint = max(1, len(lane_sequence) // 2)
        started_outside_ego_lane = any(item and item != ego_lane for item in lane_sequence[:midpoint])
        ended_in_ego_lane = ego_lane in lane_sequence[midpoint:]
        if started_outside_ego_lane and ended_in_ego_lane:
            return front_presence and closing_motion

    started_from_side = (
        float(np.max(abs_y[: max(2, len(abs_y) // 2)]))
        >= thresholds["map_cut_in_side_start_abs_y_m"]
    )
    ends_closer_to_center = float(abs_y[0] - abs_y[-1]) >= thresholds[
        "map_cut_in_end_delta_abs_y_m"
    ]
    near_lane_boundary = float(np.min(abs_y)) <= thresholds["map_cut_in_lane_boundary_abs_y_m"]
    return front_presence and started_from_side and ends_closer_to_center and near_lane_boundary and closing_motion


def detect_oncoming_like(track: pd.DataFrame) -> bool:
    if track.empty:
        return False
    ordered = track.sort_values("sample_idx")
    heading_gap = np.abs(np.pi - np.abs(ordered["heading_delta"].to_numpy(dtype=float)))
    closing = float(ordered["rel_vx"].mean()) < -1.0
    front_presence = float((ordered["x_ego"] > 0.0).mean()) >= 0.5
    return front_presence and closing and float(np.nanmean(heading_gap)) <= 1.0


def detect_stopped_lead_like(track: pd.DataFrame, thresholds: Mapping[str, float] | None = None) -> bool:
    thresholds = _resolve_heuristic_thresholds(thresholds)
    if track.empty:
        return False
    ordered = track.sort_values("sample_idx")
    front_corridor = bool(
        (
            (ordered["x_ego"] > 0.0)
            & (ordered["x_ego"] < thresholds["stopped_lead_front_max_x_m"])
            & (ordered["y_ego"].abs() < thresholds["stopped_lead_lane_abs_y_m"])
        ).any()
    )
    stationary_frames = int((ordered["speed"] <= thresholds["stopped_lead_speed_mps"]).sum())
    return front_corridor and stationary_frames >= max(2, len(ordered) // 2)


def _primary_behavior(query: ParsedQuery) -> str:
    if query.behaviors:
        return str(query.behaviors[0])
    return "proximity"


def _event_mask(track: pd.DataFrame, query: ParsedQuery) -> np.ndarray:
    ordered = track.sort_values("sample_idx")
    x_values = ordered["x_ego"].to_numpy(dtype=float)
    y_values = ordered["y_ego"].to_numpy(dtype=float)
    distance = ordered["distance"].to_numpy(dtype=float)
    rel_vx = ordered["rel_vx"].to_numpy(dtype=float)
    speed = ordered["speed"].to_numpy(dtype=float)
    heading_delta = ordered["heading_delta"].to_numpy(dtype=float)

    behavior = _primary_behavior(query)
    if behavior == "crossing":
        return ((x_values >= -6.0) & (x_values <= 18.0) & (np.abs(y_values) <= 8.0)) | (
            np.abs(x_values) <= 2.5
        )
    if behavior == "cut_in":
        return (x_values >= -6.0) & (x_values <= 24.0) & (np.abs(y_values) <= 4.5)
    if behavior == "oncoming":
        heading_gap = np.abs(np.pi - np.abs(heading_delta))
        return (x_values >= 0.0) & (rel_vx < 0.0) & (heading_gap <= 1.1)
    if behavior == "stopped_lead":
        return (x_values >= 0.0) & (x_values <= 25.0) & (np.abs(y_values) <= 3.5) & (speed <= 1.0)
    return distance <= float(query.near_distance_m)


def _peak_index(track: pd.DataFrame) -> int:
    ordered = track.sort_values("sample_idx").reset_index(drop=True)
    if ordered.empty:
        return -1
    ttc = ordered["ttc"].replace([np.inf, -np.inf], np.nan)
    if ttc.notna().any():
        return int(ttc.astype(float).idxmin())
    return int(ordered["distance"].astype(float).idxmin())


def _contiguous_segments(mask: np.ndarray) -> List[Tuple[int, int]]:
    if len(mask) == 0:
        return []
    segments: List[Tuple[int, int]] = []
    start = None
    for idx, value in enumerate(mask):
        if value and start is None:
            start = idx
        if start is not None and (idx == len(mask) - 1 or not mask[idx + 1]):
            segments.append((start, idx))
            start = None
    return segments


def localize_event(track: pd.DataFrame, query: ParsedQuery, candidate: RetrievalCandidate) -> Dict[str, object]:
    ordered = track.sort_values("sample_idx").reset_index(drop=True)
    if ordered.empty:
        return {}

    mask = _event_mask(ordered, query)
    peak_idx = _peak_index(ordered)
    if peak_idx < 0:
        return {}

    segments = _contiguous_segments(mask)
    selected_segment = None
    for start, end in segments:
        if start <= peak_idx <= end:
            selected_segment = (start, end)
            break
    if selected_segment is None:
        selected_segment = (max(0, peak_idx - 1), min(len(ordered) - 1, peak_idx + 1))

    start_idx, end_idx = selected_segment
    peak_row = ordered.iloc[peak_idx]
    start_row = ordered.iloc[start_idx]
    end_row = ordered.iloc[end_idx]

    start_sample_idx = int(start_row["sample_idx"])
    end_sample_idx = int(end_row["sample_idx"])
    peak_sample_idx = int(peak_row["sample_idx"])
    start_t_sec = float(start_row["t_sec"]) if "t_sec" in ordered.columns else 0.0
    end_t_sec = float(end_row["t_sec"]) if "t_sec" in ordered.columns else 0.0
    peak_t_sec = float(peak_row["t_sec"]) if "t_sec" in ordered.columns else 0.0

    return {
        "primary_behavior": _primary_behavior(query),
        "start_sample_idx": start_sample_idx,
        "end_sample_idx": end_sample_idx,
        "peak_sample_idx": peak_sample_idx,
        "anchor_sample_idx": int(candidate.sample_idx),
        "start_t_sec": round(start_t_sec, 3),
        "end_t_sec": round(end_t_sec, 3),
        "peak_t_sec": round(peak_t_sec, 3),
        "duration_s": round(max(0.0, end_t_sec - start_t_sec), 3),
        "frame_count": int(end_idx - start_idx + 1),
        "anchor_within_window": bool(start_sample_idx <= int(candidate.sample_idx) <= end_sample_idx),
        "peak_distance_m": round(float(peak_row["distance"]), 3),
        "peak_ttc_s": (
            None
            if pd.isna(peak_row["ttc"]) or not np.isfinite(float(peak_row["ttc"]))
            else round(float(peak_row["ttc"]), 3)
        ),
    }


def build_actor_grounding(track: pd.DataFrame, candidate: RetrievalCandidate, query: ParsedQuery) -> Dict[str, object]:
    ordered = track.sort_values("sample_idx").reset_index(drop=True)
    if ordered.empty:
        return {}
    return {
        "role": "primary_actor",
        "instance_token": candidate.instance_token,
        "category_name": candidate.category_name,
        "category_group": candidate.category_group,
        "anchor_sample_token": candidate.sample_token,
        "anchor_sample_idx": int(candidate.sample_idx),
        "track_start_sample_idx": int(ordered.iloc[0]["sample_idx"]),
        "track_end_sample_idx": int(ordered.iloc[-1]["sample_idx"]),
        "track_frame_count": int(len(ordered)),
        "grounded_positions": list(query.positions),
        "grounded_behaviors": list(query.behaviors),
        "grounded_risk_terms": list(query.risk_terms),
    }


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
    validation_config: ValidationConfig | None = None,
) -> ValidatedCase:
    validation_config = validation_config or ValidationConfig()
    heuristic_thresholds = validation_config.resolved_heuristic_thresholds()
    pass_thresholds = validation_config.resolved_pass_thresholds()
    score_weights = validation_config.resolved_score_weights()
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
            actor_grounding={},
            event_localization={},
        )

    anchor_time = float(
        timeline.loc[timeline["sample_token"] == candidate.sample_token, "timestamp_us"].iloc[0]
    )
    timeline = timeline.copy()
    timeline["t_sec"] = (timeline["timestamp_us"].astype(float) - anchor_time) / 1_000_000.0
    if validation_config.enable_map_context:
        map_context, map_geometries = build_case_map_context(
            conn,
            candidate,
            timeline,
            ego_window,
            query_behaviors=query.behaviors,
            include_patch_geometries=include_map_geometries,
        )
    else:
        map_context, map_geometries = {"available": False, "reason": "ablation:no_map_context"}, {}

    behavior_matches: Dict[str, bool] = {}
    if "crossing" in query.behaviors:
        behavior_matches["crossing"] = detect_crossing_like(
            timeline, heuristic_thresholds
        ) or detect_map_supported_crossing_like(
            timeline, map_context, heuristic_thresholds
        )
    if "cut_in" in query.behaviors:
        behavior_matches["cut_in"] = detect_cut_in_like(
            timeline, positions=query.positions, thresholds=heuristic_thresholds
        ) or detect_map_supported_cut_in_like(
            timeline, map_context, heuristic_thresholds
        )
    if "oncoming" in query.behaviors:
        behavior_matches["oncoming"] = detect_oncoming_like(timeline)
    if "stopped_lead" in query.behaviors:
        behavior_matches["stopped_lead"] = detect_stopped_lead_like(timeline, heuristic_thresholds)

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
    validation_quality_score = round(
        (
            score_weights["proximity"] * proximity_score
            + score_weights["ttc"] * ttc_score
            + score_weights["persistence"] * persistence_score
            + score_weights["behavior"] * behavior_score
            + score_weights["map"] * map_score
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
    if not validation_config.enable_map_context:
        notes.append("Map context: ablated")
    if map_context.get("available"):
        notes.append("Map ego lane: {0}".format("yes" if map_context.get("ego_in_lane_anchor") else "no"))
        if "crossing" in query.behaviors:
            notes.append("Map crosswalk hit: {0}".format("yes" if map_context.get("actor_on_crosswalk_any") else "no"))
        if "cut_in" in query.behaviors or "oncoming" in query.behaviors or "stopped_lead" in query.behaviors:
            notes.append("Map shared lane: {0}".format("yes" if map_context.get("shares_lane_at_anchor") else "no"))

    context_agent_count = int(len(context_agents))
    if context_agent_count > 48:
        notes.append("Context agents loaded without truncation: {0}".format(context_agent_count))

    proximity_threshold = float(pass_thresholds["proximity_score"])
    proximity_passed = proximity_score >= proximity_threshold
    behavior_passed = all(behavior_matches.values()) if behavior_matches else True
    reason_codes: List[str] = []
    if not proximity_passed:
        reason_codes.append("insufficient_proximity")
    reason_codes.extend(
        "behavior_mismatch:{0}".format(name)
        for name, matched in sorted(behavior_matches.items())
        if not matched
    )
    passed = not reason_codes
    proximity_gate_score = min(proximity_score / max(proximity_threshold, 1e-6), 1.0)
    behavior_gate_score = 1.0 if behavior_passed else 0.0
    gate_score = round(min(proximity_gate_score, behavior_gate_score), 6)
    gate_decision = {
        "protocol_version": VALIDATION_PROTOCOL_VERSION,
        "status": "pass" if passed else "fail",
        "reason_codes": reason_codes,
        "gate_score": gate_score,
    }
    evidence = {
        "min_distance_m": round(min_distance, 3),
        "min_ttc_s": None if not np.isfinite(min_ttc) else round(min_ttc, 3),
        "max_closing_speed_mps": round(max_closing_speed, 3),
        "lateral_span_m": round(lateral_span, 3),
        "timeline_frames": int(len(timeline)),
        "context_agent_count": context_agent_count,
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
        "validation_profile": str(validation_config.name),
        "validation_protocol_version": VALIDATION_PROTOCOL_VERSION,
        "validation_quality_score": validation_quality_score,
        "validation_score_profile": str(validation_config.score_profile),
        "validation_score_semantics": "continuous quality score in [0, 100]; acceptance is a separate gate",
        "acceptance_score": validation_quality_score if passed else 0.0,
        "validation_gate_score": gate_score,
        "validation_gate_decision": gate_decision,
        "validation_component_scores": {
            "proximity": round(float(proximity_score), 6),
            "ttc": round(float(ttc_score), 6),
            "persistence": round(float(persistence_score), 6),
            "behavior": round(float(behavior_score), 6),
            "map": round(float(map_score), 6),
        },
        "validation_pass_components": {
            "proximity_passed": bool(proximity_passed),
            "behavior_passed": bool(behavior_passed),
            "overall_passed": bool(passed),
        },
        "validation_score_weights": score_weights,
        "validation_pass_thresholds": pass_thresholds,
        "heuristic_thresholds": heuristic_thresholds,
        "map_context_enabled": bool(validation_config.enable_map_context),
        "event_localization_enabled": bool(validation_config.enable_event_localization),
        "actor_grounding_enabled": bool(validation_config.enable_actor_grounding),
    }
    event_localization = (
        localize_event(timeline, query, candidate) if validation_config.enable_event_localization else {}
    )
    actor_grounding = (
        build_actor_grounding(timeline, candidate, query) if validation_config.enable_actor_grounding else {}
    )
    if event_localization:
        evidence.update(
            {
                "event_start_sample_idx": int(event_localization["start_sample_idx"]),
                "event_end_sample_idx": int(event_localization["end_sample_idx"]),
                "event_peak_sample_idx": int(event_localization["peak_sample_idx"]),
                "event_duration_s": float(event_localization["duration_s"]),
            }
        )
        notes.append(
            "Event window: sample {0} to {1} (peak {2})".format(
                event_localization["start_sample_idx"],
                event_localization["end_sample_idx"],
                event_localization["peak_sample_idx"],
            )
        )
    elif not validation_config.enable_event_localization:
        notes.append("Event localization: ablated")
    if not validation_config.enable_actor_grounding:
        notes.append("Actor grounding: ablated")

    return ValidatedCase(
        query=query,
        candidate=candidate,
        validation_score=validation_quality_score,
        passed=passed,
        behavior_matches=behavior_matches,
        evidence=evidence,
        notes=notes,
        timeline=timeline,
        context_agents=context_agents,
        ego_window=ego_window,
        map_context=map_context,
        map_geometries=map_geometries,
        actor_grounding=actor_grounding,
        event_localization=event_localization,
        gate_score=gate_score,
        gate_decision=gate_decision,
        quality_score=validation_quality_score,
    )

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import pandas as pd

from nusc_scene_agent.models import ParsedQuery, RetrievalCandidate


VEHICLE_LIKE = {"vehicle", "bus", "truck"}
VRU_LIKE = {"pedestrian", "bicycle", "motorcycle"}
SQLITE_IN_CLAUSE_BATCH = 900
RETRIEVAL_SCORE_WEIGHTS = {
    "distance": 5.0,
    "ttc": 3.0,
    "category": 1.5,
    "oncoming": 3.0,
    "crossing_lateral": 2.0,
    "crossing_front": 1.0,
    "cut_in_side": 2.5,
    "cut_in_front": 1.0,
    "cut_in_temporal": 1.0,
    "stopped_lead": 4.0,
    "lidar_visible": 0.3,
    "radar_visible": 0.2,
}
EQUAL_RETRIEVAL_SCORE_WEIGHTS = {
    key: 1.0 for key in RETRIEVAL_SCORE_WEIGHTS
}
RETRIEVAL_SCORE_PROFILES = {
    "default": RETRIEVAL_SCORE_WEIGHTS,
    "equal": EQUAL_RETRIEVAL_SCORE_WEIGHTS,
}


@dataclass(frozen=True)
class RetrievalScoreConfig:
    profile_name: str = "default"
    weights: dict[str, float] | None = None
    candidate_scan_limit: int = 50000
    deduplication_key: str = "sample_instance"

    def resolved_candidate_scan_limit(self) -> int:
        try:
            value = int(self.candidate_scan_limit)
        except (TypeError, ValueError) as exc:
            raise ValueError("candidate_scan_limit must be a non-negative integer.") from exc
        if value != self.candidate_scan_limit or value < 0:
            raise ValueError(
                "candidate_scan_limit must be a non-negative integer; use 0 for a full scan."
            )
        return value

    def resolved_weights(self) -> dict[str, float]:
        if self.profile_name not in RETRIEVAL_SCORE_PROFILES:
            raise ValueError(
                "Unknown retrieval score profile: {0}. Available profiles: {1}".format(
                    self.profile_name, ", ".join(sorted(RETRIEVAL_SCORE_PROFILES))
                )
            )
        base = dict(RETRIEVAL_SCORE_PROFILES[self.profile_name])
        if self.weights:
            for key, value in self.weights.items():
                if key not in base:
                    raise ValueError("Unknown retrieval score weight: {0}".format(key))
                value = float(value)
                if not math.isfinite(value) or value < 0.0:
                    raise ValueError(
                        "Retrieval score weights must be finite and non-negative: {0}".format(key)
                    )
                base[key] = value
        if not all(math.isfinite(value) and value >= 0.0 for value in base.values()):
            raise ValueError("Retrieval score weights must be finite and non-negative.")
        if sum(base.values()) <= 0.0:
            raise ValueError("Retrieval score weights must have a positive sum.")
        return base

    def to_dict(self) -> dict[str, object]:
        return {
            "profile_name": self.profile_name,
            "weights": self.resolved_weights(),
            "candidate_scan_limit": self.resolved_candidate_scan_limit(),
            "deduplication_key": self.deduplication_key,
            "score_protocol": "vectorized_feature_score_v2",
        }


def _float_value(value: object, default: float = 0.0) -> float:
    if value is None or pd.isna(value):
        return default
    return float(value)


def _cut_in_temporal_bonus(row: pd.Series, query: ParsedQuery) -> float:
    y_ego = _float_value(row.get("y_ego"))
    abs_y = abs(y_ego)
    prev_max_y = _float_value(row.get("prev_max_y"), default=y_ego)
    prev_min_y = _float_value(row.get("prev_min_y"), default=y_ego)
    future_min_abs_y = _float_value(row.get("future_min_abs_y"), default=abs_y)

    if "left" in query.positions:
        if y_ego < 0.0:
            return 0.0
        side_collapse = max(0.0, prev_max_y - y_ego) if y_ego >= 0.0 else 0.0
    elif "right" in query.positions:
        if y_ego > 0.0:
            return 0.0
        side_collapse = max(0.0, y_ego - prev_min_y) if y_ego <= 0.0 else 0.0
    else:
        left_collapse = max(0.0, prev_max_y - y_ego) if y_ego >= 0.0 else 0.0
        right_collapse = max(0.0, y_ego - prev_min_y) if y_ego <= 0.0 else 0.0
        side_collapse = max(left_collapse, right_collapse)

    future_lane_entry = max(0.0, abs_y - future_min_abs_y)
    collapse_score = max(0.0, min(1.0, side_collapse / 6.0))
    future_entry_score = max(0.0, min(1.0, future_lane_entry / 3.0))
    lane_entry_bonus = 1.0 if future_min_abs_y <= 2.5 else 0.0
    return 3.5 * collapse_score + 2.0 * future_entry_score + lane_entry_bonus


def _numeric_column(frame: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    if name not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce").fillna(default).astype(float)


def _cut_in_temporal_bonus_frame(frame: pd.DataFrame, query: ParsedQuery) -> pd.Series:
    y_ego = _numeric_column(frame, "y_ego")
    abs_y = y_ego.abs()
    prev_max_y_raw = pd.to_numeric(frame["prev_max_y"], errors="coerce") if "prev_max_y" in frame.columns else y_ego
    prev_min_y_raw = pd.to_numeric(frame["prev_min_y"], errors="coerce") if "prev_min_y" in frame.columns else y_ego
    future_min_abs_raw = (
        pd.to_numeric(frame["future_min_abs_y"], errors="coerce") if "future_min_abs_y" in frame.columns else abs_y
    )
    prev_max_y = prev_max_y_raw.where(prev_max_y_raw.notna(), y_ego).astype(float)
    prev_min_y = prev_min_y_raw.where(prev_min_y_raw.notna(), y_ego).astype(float)
    future_min_abs_y = future_min_abs_raw.where(future_min_abs_raw.notna(), abs_y).astype(float)

    if "left" in query.positions:
        valid_side = y_ego >= 0.0
        side_collapse = (prev_max_y - y_ego).clip(lower=0.0).where(valid_side, 0.0)
    elif "right" in query.positions:
        valid_side = y_ego <= 0.0
        side_collapse = (y_ego - prev_min_y).clip(lower=0.0).where(valid_side, 0.0)
    else:
        left_collapse = (prev_max_y - y_ego).clip(lower=0.0).where(y_ego >= 0.0, 0.0)
        right_collapse = (y_ego - prev_min_y).clip(lower=0.0).where(y_ego <= 0.0, 0.0)
        side_collapse = pd.concat([left_collapse, right_collapse], axis=1).max(axis=1)

    future_lane_entry = (abs_y - future_min_abs_y).clip(lower=0.0)
    collapse_score = (side_collapse / 6.0).clip(lower=0.0, upper=1.0)
    future_entry_score = (future_lane_entry / 3.0).clip(lower=0.0, upper=1.0)
    lane_entry_bonus = (future_min_abs_y <= 2.5).astype(float)
    return 3.5 * collapse_score + 2.0 * future_entry_score + lane_entry_bonus


def _score_row(row: pd.Series, query: ParsedQuery, weights: dict[str, float] | None = None) -> float:
    weights = weights or RETRIEVAL_SCORE_WEIGHTS
    near_distance = max(query.near_distance_m, 1.0)
    score = max(0.0, 1.0 - float(row["distance"]) / near_distance) * weights["distance"]

    ttc = row["ttc"]
    if pd.notna(ttc):
        score += max(0.0, 1.0 - float(ttc) / max(query.max_ttc_s, 0.5)) * weights["ttc"]

    if query.category_groups and row["category_group"] in query.category_groups:
        score += weights["category"]

    if "oncoming" in query.behaviors:
        heading_gap = abs(abs(float(row["heading_delta"])) - math.pi)
        heading_score = max(0.0, 1.0 - heading_gap / 1.2)
        closing_score = max(0.0, min(1.0, -float(row["rel_vx"]) / 8.0))
        score += weights["oncoming"] * heading_score * closing_score

    if "crossing" in query.behaviors:
        lateral_score = max(0.0, min(1.0, abs(float(row["y_ego"])) / 6.0))
        front_bonus = 1.0 if -5.0 <= float(row["x_ego"]) <= 20.0 else 0.0
        if row["category_group"] in VRU_LIKE:
            score += weights["crossing_lateral"] * lateral_score + weights["crossing_front"] * front_bonus

    if "cut_in" in query.behaviors:
        side_score = max(0.0, min(1.0, abs(float(row["y_ego"])) / 4.0))
        front_score = 1.0 if -10.0 <= float(row["x_ego"]) <= 30.0 else 0.0
        if row["category_group"] in VEHICLE_LIKE:
            score += weights["cut_in_side"] * side_score + weights["cut_in_front"] * front_score
            score += weights["cut_in_temporal"] * _cut_in_temporal_bonus(row, query)

    if "stopped_lead" in query.behaviors and int(row["is_stationary"]) == 1:
        if float(row["x_ego"]) > 0.0 and abs(float(row["y_ego"])) < 3.0:
            score += weights["stopped_lead"]

    if int(row["num_lidar_pts"]) > 0:
        score += weights["lidar_visible"]
    if int(row["num_radar_pts"]) > 0:
        score += weights["radar_visible"]

    return score


def _score_frame(frame: pd.DataFrame, query: ParsedQuery, weights: dict[str, float] | None = None) -> pd.Series:
    weights = weights or RETRIEVAL_SCORE_WEIGHTS
    distance = _numeric_column(frame, "distance")
    ttc = pd.to_numeric(frame["ttc"], errors="coerce") if "ttc" in frame.columns else pd.Series(np.nan, index=frame.index)
    x_ego = _numeric_column(frame, "x_ego")
    y_ego = _numeric_column(frame, "y_ego")
    heading_delta = _numeric_column(frame, "heading_delta")
    rel_vx = _numeric_column(frame, "rel_vx")
    is_stationary = _numeric_column(frame, "is_stationary")
    num_lidar_pts = _numeric_column(frame, "num_lidar_pts")
    num_radar_pts = _numeric_column(frame, "num_radar_pts")
    category_group = frame["category_group"] if "category_group" in frame.columns else pd.Series("", index=frame.index)

    near_distance = max(float(query.near_distance_m), 1.0)
    score = (1.0 - distance / near_distance).clip(lower=0.0) * weights["distance"]

    valid_ttc = ttc.notna()
    ttc_score = (1.0 - ttc / max(float(query.max_ttc_s), 0.5)).clip(lower=0.0) * weights["ttc"]
    score = score + ttc_score.where(valid_ttc, 0.0)

    if query.category_groups:
        score = score + category_group.isin(query.category_groups).astype(float) * weights["category"]

    if "oncoming" in query.behaviors:
        heading_gap = (heading_delta.abs() - math.pi).abs()
        heading_score = (1.0 - heading_gap / 1.2).clip(lower=0.0)
        closing_score = (-rel_vx / 8.0).clip(lower=0.0, upper=1.0)
        score = score + weights["oncoming"] * heading_score * closing_score

    if "crossing" in query.behaviors:
        lateral_score = (y_ego.abs() / 6.0).clip(lower=0.0, upper=1.0)
        front_bonus = ((x_ego >= -5.0) & (x_ego <= 20.0)).astype(float)
        vru_mask = category_group.isin(VRU_LIKE).astype(float)
        score = score + vru_mask * (
            weights["crossing_lateral"] * lateral_score
            + weights["crossing_front"] * front_bonus
        )

    if "cut_in" in query.behaviors:
        side_score = (y_ego.abs() / 4.0).clip(lower=0.0, upper=1.0)
        front_score = ((x_ego >= -10.0) & (x_ego <= 30.0)).astype(float)
        vehicle_mask = category_group.isin(VEHICLE_LIKE).astype(float)
        score = score + vehicle_mask * (
            weights["cut_in_side"] * side_score
            + weights["cut_in_front"] * front_score
            + weights["cut_in_temporal"] * _cut_in_temporal_bonus_frame(frame, query)
        )

    if "stopped_lead" in query.behaviors:
        stopped_mask = ((is_stationary.astype(int) == 1) & (x_ego > 0.0) & (y_ego.abs() < 3.0)).astype(float)
        score = score + stopped_mask * weights["stopped_lead"]

    score = score + (num_lidar_pts.astype(int) > 0).astype(float) * weights["lidar_visible"]
    score = score + (num_radar_pts.astype(int) > 0).astype(float) * weights["radar_visible"]
    return score.astype(float)


def _load_cut_in_temporal_features(conn: sqlite3.Connection, ann_tokens: Sequence[str]) -> pd.DataFrame:
    if not ann_tokens:
        return pd.DataFrame(columns=["ann_token", "prev_max_y", "prev_min_y", "future_min_abs_y"])

    frames: List[pd.DataFrame] = []
    for start in range(0, len(ann_tokens), SQLITE_IN_CLAUSE_BATCH):
        batch = list(ann_tokens[start : start + SQLITE_IN_CLAUSE_BATCH])
        placeholders = ", ".join(["?"] * len(batch))
        sql = """
            SELECT
                anchor.ann_token,
                MAX(prev.y_ego) AS prev_max_y,
                MIN(prev.y_ego) AS prev_min_y,
                MIN(ABS(future.y_ego)) AS future_min_abs_y
            FROM agents anchor
            LEFT JOIN agents prev
              ON prev.scene_token = anchor.scene_token
             AND prev.instance_token = anchor.instance_token
             AND prev.sample_idx BETWEEN anchor.sample_idx - 4 AND anchor.sample_idx - 1
            LEFT JOIN agents future
              ON future.scene_token = anchor.scene_token
             AND future.instance_token = anchor.instance_token
             AND future.sample_idx BETWEEN anchor.sample_idx AND anchor.sample_idx + 3
            WHERE anchor.ann_token IN ({0})
            GROUP BY anchor.ann_token
        """.format(placeholders)
        frames.append(pd.read_sql_query(sql, conn, params=batch))

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["ann_token", "prev_max_y", "prev_min_y", "future_min_abs_y"]
    )


def retrieve_candidates(
    conn: sqlite3.Connection,
    query: ParsedQuery,
    top_k: int = 5,
    candidate_pool: int = 30,
    score_config: RetrievalScoreConfig | None = None,
) -> List[RetrievalCandidate]:
    score_config = score_config or RetrievalScoreConfig()
    score_weights = score_config.resolved_weights()
    where_clauses = ["a.distance <= ?"]
    params: List[object] = [max(query.near_distance_m * 1.8, 20.0)]

    if query.category_groups:
        placeholders = ", ".join(["?"] * len(query.category_groups))
        where_clauses.append("a.category_group IN ({0})".format(placeholders))
        params.extend(query.category_groups)

    if "front" in query.positions:
        where_clauses.append("a.is_front = 1")
    if "rear" in query.positions:
        where_clauses.append("a.is_rear = 1")
    if "left" in query.positions:
        where_clauses.append("a.is_left = 1")
    if "right" in query.positions:
        where_clauses.append("a.is_right = 1")

    if "stopped_lead" in query.behaviors:
        where_clauses.append("a.is_stationary = 1")

    if "oncoming" in query.behaviors:
        where_clauses.append("a.is_front = 1")

    scan_limit = score_config.resolved_candidate_scan_limit()
    sql = """
        SELECT
            a.ann_token,
            a.sample_token,
            a.scene_token,
            a.scene_name,
            a.sample_idx,
            a.instance_token,
            a.category_name,
            a.category_group,
            a.distance,
            a.ttc,
            a.x_ego,
            a.y_ego,
            a.speed,
            a.rel_vx,
            a.rel_vy,
            a.heading_delta,
            a.is_stationary,
            a.num_lidar_pts,
            a.num_radar_pts,
            s.location,
            s.scene_description
        FROM agents a
        JOIN samples s ON s.sample_token = a.sample_token
        WHERE {0}
    """.format(" AND ".join(where_clauses))
    if scan_limit > 0:
        # The indexed coarse order keeps the exact vectorized scoring stage bounded.
        # The limit is recorded in every candidate so experiments can distinguish a
        # bounded retrieval run from a full scan.
        sql += " ORDER BY a.distance ASC, COALESCE(a.ttc, 1e9) ASC, a.ann_token ASC LIMIT ?"
        params.append(scan_limit)

    frame = pd.read_sql_query(sql, conn, params=params)
    if frame.empty:
        return []

    if "cut_in" in query.behaviors:
        temporal = _load_cut_in_temporal_features(conn, frame["ann_token"].tolist())
        if not temporal.empty:
            frame = frame.merge(temporal, on="ann_token", how="left")

    frame["retrieval_score"] = _score_frame(frame, query, weights=score_weights)
    frame = frame.sort_values(
        ["retrieval_score", "distance", "ttc", "ann_token"],
        ascending=[False, True, True, True],
        kind="mergesort",
    )
    frame = frame.drop_duplicates(subset=["sample_token", "instance_token"], keep="first")
    frame = frame.head(max(candidate_pool, top_k))
    frame["retrieval_rank"] = range(1, len(frame) + 1)
    frame["retrieval_rank_source"] = (
        "rule_score_sql_prefilter" if scan_limit > 0 else "rule_score_full_scan"
    )
    return [RetrievalCandidate.from_record(record) for record in frame.to_dict(orient="records")]

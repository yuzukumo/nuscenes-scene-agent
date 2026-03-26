from __future__ import annotations

import math
import sqlite3
from typing import List, Sequence

import pandas as pd

from nusc_scene_agent.models import ParsedQuery, RetrievalCandidate


VEHICLE_LIKE = {"vehicle", "bus", "truck"}
VRU_LIKE = {"pedestrian", "bicycle", "motorcycle"}
SQLITE_IN_CLAUSE_BATCH = 900


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


def _score_row(row: pd.Series, query: ParsedQuery) -> float:
    near_distance = max(query.near_distance_m, 1.0)
    score = max(0.0, 1.0 - float(row["distance"]) / near_distance) * 5.0

    ttc = row["ttc"]
    if pd.notna(ttc):
        score += max(0.0, 1.0 - float(ttc) / max(query.max_ttc_s, 0.5)) * 3.0

    if query.category_groups and row["category_group"] in query.category_groups:
        score += 1.5

    if "oncoming" in query.behaviors:
        heading_gap = abs(abs(float(row["heading_delta"])) - math.pi)
        heading_score = max(0.0, 1.0 - heading_gap / 1.2)
        closing_score = max(0.0, min(1.0, -float(row["rel_vx"]) / 8.0))
        score += 3.0 * heading_score * closing_score

    if "crossing" in query.behaviors:
        lateral_score = max(0.0, min(1.0, abs(float(row["y_ego"])) / 6.0))
        front_bonus = 1.0 if -5.0 <= float(row["x_ego"]) <= 20.0 else 0.0
        if row["category_group"] in VRU_LIKE:
            score += 2.0 * lateral_score + front_bonus

    if "cut_in" in query.behaviors:
        side_score = max(0.0, min(1.0, abs(float(row["y_ego"])) / 4.0))
        front_score = 1.0 if -10.0 <= float(row["x_ego"]) <= 30.0 else 0.0
        if row["category_group"] in VEHICLE_LIKE:
            score += 2.5 * side_score + front_score
            score += _cut_in_temporal_bonus(row, query)

    if "stopped_lead" in query.behaviors and int(row["is_stationary"]) == 1:
        if float(row["x_ego"]) > 0.0 and abs(float(row["y_ego"])) < 3.0:
            score += 4.0

    if int(row["num_lidar_pts"]) > 0:
        score += 0.3
    if int(row["num_radar_pts"]) > 0:
        score += 0.2

    return score


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
) -> List[RetrievalCandidate]:
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

    frame = pd.read_sql_query(sql, conn, params=params)
    if frame.empty:
        return []

    if "cut_in" in query.behaviors:
        temporal = _load_cut_in_temporal_features(conn, frame["ann_token"].tolist())
        if not temporal.empty:
            frame = frame.merge(temporal, on="ann_token", how="left")

    frame["retrieval_score"] = frame.apply(lambda row: _score_row(row, query), axis=1)
    frame = frame.sort_values("retrieval_score", ascending=False)
    frame = frame.drop_duplicates(subset=["instance_token"], keep="first")
    frame = frame.head(max(candidate_pool, top_k))
    return [RetrievalCandidate.from_record(record) for record in frame.to_dict(orient="records")]

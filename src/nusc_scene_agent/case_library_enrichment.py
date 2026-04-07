from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Dict, List

from nusc_scene_agent.models import ParsedQuery, RetrievalCandidate
from nusc_scene_agent.query_parser import parse_query
from nusc_scene_agent.validation import validate_candidate


def _unique(items: List[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for item in items:
        value = str(item).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _reconstruct_query(entry: Dict[str, object]) -> ParsedQuery:
    source_queries = list(entry.get("source_queries") or [])
    text = str(source_queries[0]) if source_queries else str(entry.get("case_key") or "query")
    query = parse_query(text)
    category_group = str(entry.get("category_group") or "")
    if category_group:
        query.category_groups = _unique(list(query.category_groups) + [category_group])
    behaviors = [str(item) for item in (entry.get("all_behaviors") or [])]
    if behaviors:
        query.behaviors = _unique(list(query.behaviors) + behaviors)
    return query


def _load_candidate(conn: sqlite3.Connection, entry: Dict[str, object]) -> RetrievalCandidate:
    sample_token = str(entry["sample_token"])
    instance_token = str(entry["instance_token"])
    frame = conn.execute(
        """
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
            0.0 AS retrieval_score,
            a.num_lidar_pts,
            a.num_radar_pts,
            s.location,
            s.scene_description
        FROM agents a
        JOIN samples s ON s.sample_token = a.sample_token
        WHERE a.sample_token = ?
          AND a.instance_token = ?
        LIMIT 1
        """,
        (sample_token, instance_token),
    ).fetchone()
    if frame is None:
        raise KeyError("Candidate not found for sample {0} instance {1}".format(sample_token, instance_token))
    return RetrievalCandidate.from_record(dict(frame))


def enrich_case_library(case_library_path: Path, db_path: Path, output_path: Path) -> Dict[str, object]:
    case_library_path = case_library_path.resolve()
    db_path = db_path.resolve()
    output_path = output_path.resolve()

    entries = json.loads(case_library_path.read_text(encoding="utf-8"))
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    enriched: List[Dict[str, object]] = []
    try:
        for entry in entries:
            query = _reconstruct_query(entry)
            candidate = _load_candidate(conn, entry)
            validated = validate_candidate(conn, query, candidate, include_map_geometries=False)
            updated = dict(entry)
            updated["actor_track_start_sample_idx"] = validated.actor_grounding.get("track_start_sample_idx")
            updated["actor_track_end_sample_idx"] = validated.actor_grounding.get("track_end_sample_idx")
            updated["actor_track_frame_count"] = validated.actor_grounding.get("track_frame_count")
            updated["event_primary_behavior"] = validated.event_localization.get("primary_behavior", "")
            updated["event_start_sample_idx"] = validated.event_localization.get("start_sample_idx")
            updated["event_end_sample_idx"] = validated.event_localization.get("end_sample_idx")
            updated["event_peak_sample_idx"] = validated.event_localization.get("peak_sample_idx")
            updated["event_duration_s"] = validated.event_localization.get("duration_s")
            enriched.append(updated)
    finally:
        conn.close()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(enriched, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "source_case_count": len(entries),
        "enriched_case_count": len(enriched),
        "output_path": str(output_path),
    }

from __future__ import annotations

import csv
import json
import math
import sqlite3
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from nusc_scene_agent.models import ParsedQuery, RetrievalCandidate
from nusc_scene_agent.llm_client import LLMConfig
from nusc_scene_agent.llm_query_planner import resolve_query
from nusc_scene_agent.retrieval import retrieve_candidates


CATEGORY_GROUPS = ["vehicle", "bus", "truck", "pedestrian", "bicycle", "motorcycle"]
POSITIONS = ["front", "rear", "left", "right"]
BEHAVIORS = ["crossing", "cut_in", "oncoming", "stopped_lead"]
RISK_TERMS = ["risky", "urgent"]


@dataclass(frozen=True)
class MultimodalRetrievalConfig:
    language_weight: float = 0.35
    bev_geometry_weight: float = 0.30
    motion_weight: float = 0.25
    sensor_weight: float = 0.10
    base_score_weight: float = 0.08
    distance_scale_m: float = 40.0
    ttc_scale_s: float = 8.0

    def normalized_weights(self) -> Dict[str, float]:
        weights = {
            "language": max(0.0, float(self.language_weight)),
            "bev_geometry": max(0.0, float(self.bev_geometry_weight)),
            "motion": max(0.0, float(self.motion_weight)),
            "sensor": max(0.0, float(self.sensor_weight)),
        }
        total = sum(weights.values()) or 1.0
        return {key: value / total for key, value in weights.items()}


@dataclass(frozen=True)
class MultimodalScore:
    rank: int
    candidate_key: str
    ann_token: str
    scene_name: str
    sample_idx: int
    category_group: str
    distance_m: float
    ttc_s: float
    base_retrieval_score: float
    fused_score: float
    modality_scores: Dict[str, float]
    modality_weights: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def score_candidates_with_multimodal_model(
    query: ParsedQuery,
    candidates: Sequence[RetrievalCandidate],
    config: MultimodalRetrievalConfig | None = None,
) -> List[MultimodalScore]:
    config = config or MultimodalRetrievalConfig()
    weights = config.normalized_weights()
    scored: List[MultimodalScore] = []
    max_base_score = max([abs(float(candidate.retrieval_score)) for candidate in candidates] + [1.0])

    for candidate in candidates:
        modality_scores = {
            "language": _cosine(_query_language_vector(query), _candidate_language_vector(candidate)),
            "bev_geometry": _cosine(
                _query_bev_geometry_vector(query),
                _candidate_bev_geometry_vector(candidate, config),
            ),
            "motion": _cosine(_query_motion_vector(query), _candidate_motion_vector(candidate, config)),
            "sensor": _cosine(_query_sensor_vector(query), _candidate_sensor_vector(candidate)),
        }
        fused = sum(weights[name] * modality_scores[name] for name in weights)
        fused += float(config.base_score_weight) * max(0.0, float(candidate.retrieval_score) / max_base_score)
        scored.append(
            MultimodalScore(
                rank=0,
                candidate_key="{0}:{1}".format(candidate.scene_token, candidate.ann_token),
                ann_token=candidate.ann_token,
                scene_name=candidate.scene_name,
                sample_idx=int(candidate.sample_idx),
                category_group=candidate.category_group,
                distance_m=round(float(candidate.distance), 4),
                ttc_s=round(float(candidate.ttc), 4) if math.isfinite(float(candidate.ttc)) else float("inf"),
                base_retrieval_score=round(float(candidate.retrieval_score), 6),
                fused_score=round(float(fused), 6),
                modality_scores={key: round(float(value), 6) for key, value in modality_scores.items()},
                modality_weights={key: round(float(value), 6) for key, value in weights.items()},
            )
        )

    scored.sort(
        key=lambda item: (
            item.fused_score,
            item.modality_scores.get("bev_geometry", 0.0),
            item.modality_scores.get("motion", 0.0),
        ),
        reverse=True,
    )
    return [
        MultimodalScore(
            rank=rank,
            candidate_key=item.candidate_key,
            ann_token=item.ann_token,
            scene_name=item.scene_name,
            sample_idx=item.sample_idx,
            category_group=item.category_group,
            distance_m=item.distance_m,
            ttc_s=item.ttc_s,
            base_retrieval_score=item.base_retrieval_score,
            fused_score=item.fused_score,
            modality_scores=item.modality_scores,
            modality_weights=item.modality_weights,
        )
        for rank, item in enumerate(scored, start=1)
    ]


def rerank_candidates_with_multimodal_model(
    query: ParsedQuery,
    candidates: Sequence[RetrievalCandidate],
    config: MultimodalRetrievalConfig | None = None,
) -> List[RetrievalCandidate]:
    score_by_token = {
        score.ann_token: score
        for score in score_candidates_with_multimodal_model(query, candidates, config=config)
    }
    candidate_by_token = {candidate.ann_token: candidate for candidate in candidates}
    reranked: List[RetrievalCandidate] = []
    for score in sorted(score_by_token.values(), key=lambda item: item.rank):
        candidate = candidate_by_token[score.ann_token]
        reranked.append(replace(candidate, retrieval_score=float(score.fused_score)))
    return reranked


def write_multimodal_retrieval_report(
    query: ParsedQuery,
    scores: Sequence[MultimodalScore],
    output_dir: Path,
    config: MultimodalRetrievalConfig | None = None,
) -> Dict[str, Any]:
    config = config or MultimodalRetrievalConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "multimodal_scene_retrieval_report_v1",
        "query": query.to_dict(),
        "config": asdict(config),
        "candidate_count": len(scores),
        "scores": [score.to_dict() for score in scores],
    }
    (output_dir / "multimodal_retrieval_report.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_score_csv(scores, output_dir / "multimodal_retrieval_scores.csv")
    (output_dir / "multimodal_retrieval_report.md").write_text(
        _render_markdown_report(payload),
        encoding="utf-8",
    )
    return {
        "schema": payload["schema"],
        "output_dir": str(output_dir),
        "json": str(output_dir / "multimodal_retrieval_report.json"),
        "csv": str(output_dir / "multimodal_retrieval_scores.csv"),
        "markdown": str(output_dir / "multimodal_retrieval_report.md"),
        "candidate_count": len(scores),
    }


def run_multimodal_retrieval_report(
    db_path: Path,
    query_text: str,
    output_dir: Path,
    top_k: int = 20,
    candidate_pool: int = 64,
    query_mode: str = "rule",
    llm_config: LLMConfig | None = None,
    config: MultimodalRetrievalConfig | None = None,
) -> Dict[str, Any]:
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError("Missing SQLite index: {0}".format(db_path))
    query = resolve_query(query_text, mode=query_mode, config=llm_config)
    conn = sqlite3.connect(str(db_path))
    try:
        candidates = retrieve_candidates(
            conn,
            query=query,
            top_k=top_k,
            candidate_pool=max(candidate_pool, top_k),
        )
    finally:
        conn.close()
    scores = score_candidates_with_multimodal_model(query, candidates, config=config)
    selected_scores = scores[:top_k]
    return write_multimodal_retrieval_report(query, selected_scores, Path(output_dir), config=config)


def _query_language_vector(query: ParsedQuery) -> List[float]:
    return (
        _multi_hot(query.category_groups, CATEGORY_GROUPS)
        + _multi_hot(query.positions, POSITIONS)
        + _multi_hot(query.behaviors, BEHAVIORS)
        + _multi_hot(query.risk_terms, RISK_TERMS)
    )


def _candidate_language_vector(candidate: RetrievalCandidate) -> List[float]:
    positions = []
    if candidate.x_ego >= 0.0:
        positions.append("front")
    else:
        positions.append("rear")
    if candidate.y_ego >= 1.5:
        positions.append("left")
    if candidate.y_ego <= -1.5:
        positions.append("right")
    behaviors = []
    if abs(candidate.y_ego) >= 2.0 and 0.0 <= candidate.x_ego <= 25.0:
        behaviors.append("crossing")
    if abs(candidate.y_ego) >= 2.0 and candidate.category_group in {"vehicle", "bus", "truck"}:
        behaviors.append("cut_in")
    if candidate.x_ego >= 0.0 and abs(abs(candidate.heading_delta) - math.pi) <= 1.2:
        behaviors.append("oncoming")
    if candidate.x_ego >= 0.0 and abs(candidate.y_ego) <= 3.0 and candidate.speed <= 0.8:
        behaviors.append("stopped_lead")
    risk_terms = []
    if candidate.distance <= 15.0:
        risk_terms.append("risky")
    if math.isfinite(candidate.ttc) and candidate.ttc <= 3.0:
        risk_terms.append("urgent")
    return (
        _multi_hot([candidate.category_group], CATEGORY_GROUPS)
        + _multi_hot(positions, POSITIONS)
        + _multi_hot(behaviors, BEHAVIORS)
        + _multi_hot(risk_terms, RISK_TERMS)
    )


def _query_bev_geometry_vector(query: ParsedQuery) -> List[float]:
    front = 1.0 if "front" in query.positions else 0.0
    rear = 1.0 if "rear" in query.positions else 0.0
    left = 1.0 if "left" in query.positions else 0.0
    right = 1.0 if "right" in query.positions else 0.0
    near = 1.0
    urgent_ttc = 1.0 if "urgent" in query.risk_terms else 0.5
    lane_center = 1.0 if "stopped_lead" in query.behaviors else 0.0
    lateral = 1.0 if {"crossing", "cut_in"}.intersection(query.behaviors) else 0.0
    return [front, rear, left, right, near, urgent_ttc, lane_center, lateral]


def _candidate_bev_geometry_vector(candidate: RetrievalCandidate, config: MultimodalRetrievalConfig) -> List[float]:
    x = float(candidate.x_ego)
    y = float(candidate.y_ego)
    distance = max(0.0, float(candidate.distance))
    ttc = float(candidate.ttc)
    return [
        1.0 if x >= 0.0 else 0.0,
        1.0 if x < 0.0 else 0.0,
        _sigmoid((y - 1.0) / 2.0),
        _sigmoid((-y - 1.0) / 2.0),
        max(0.0, 1.0 - distance / max(float(config.distance_scale_m), 1.0)),
        max(0.0, 1.0 - ttc / max(float(config.ttc_scale_s), 1.0)) if math.isfinite(ttc) else 0.0,
        max(0.0, 1.0 - abs(y) / 3.5),
        min(1.0, abs(y) / 6.0),
    ]


def _query_motion_vector(query: ParsedQuery) -> List[float]:
    return [
        1.0 if "stopped_lead" in query.behaviors else 0.0,
        1.0 if "oncoming" in query.behaviors else 0.0,
        1.0 if "crossing" in query.behaviors else 0.0,
        1.0 if "cut_in" in query.behaviors else 0.0,
        1.0 if "urgent" in query.risk_terms else 0.5,
    ]


def _candidate_motion_vector(candidate: RetrievalCandidate, config: MultimodalRetrievalConfig) -> List[float]:
    ttc = float(candidate.ttc)
    return [
        max(0.0, 1.0 - abs(float(candidate.speed)) / 1.5),
        max(0.0, 1.0 - abs(abs(float(candidate.heading_delta)) - math.pi) / 1.2),
        min(1.0, abs(float(candidate.rel_vy)) / 4.0),
        min(1.0, abs(float(candidate.y_ego)) / 4.0) * min(1.0, max(0.0, float(candidate.x_ego)) / 25.0),
        max(0.0, 1.0 - ttc / max(float(config.ttc_scale_s), 1.0)) if math.isfinite(ttc) else 0.0,
    ]


def _query_sensor_vector(query: ParsedQuery) -> List[float]:
    vru_focus = 1.0 if set(query.category_groups).intersection({"pedestrian", "bicycle", "motorcycle"}) else 0.0
    vehicle_focus = 1.0 if set(query.category_groups).intersection({"vehicle", "bus", "truck"}) else 0.0
    return [1.0, vru_focus, vehicle_focus]


def _candidate_sensor_vector(candidate: RetrievalCandidate) -> List[float]:
    lidar = min(1.0, max(0, int(candidate.num_lidar_pts)) / 12.0)
    radar = min(1.0, max(0, int(candidate.num_radar_pts)) / 6.0)
    visibility = max(lidar, 0.75 * radar)
    return [visibility, lidar, radar]


def _multi_hot(values: Sequence[str], vocabulary: Sequence[str]) -> List[float]:
    value_set = {str(value) for value in values}
    return [1.0 if item in value_set else 0.0 for item in vocabulary]


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    dot = sum(float(a) * float(b) for a, b in zip(left, right))
    left_norm = math.sqrt(sum(float(a) * float(a) for a in left))
    right_norm = math.sqrt(sum(float(b) * float(b) for b in right))
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return max(0.0, min(1.0, dot / (left_norm * right_norm)))


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-float(value)))


def _write_score_csv(scores: Sequence[MultimodalScore], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "rank",
            "candidate_key",
            "scene_name",
            "sample_idx",
            "category_group",
            "distance_m",
            "ttc_s",
            "base_retrieval_score",
            "fused_score",
            "language_score",
            "bev_geometry_score",
            "motion_score",
            "sensor_score",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for score in scores:
            writer.writerow(
                {
                    "rank": score.rank,
                    "candidate_key": score.candidate_key,
                    "scene_name": score.scene_name,
                    "sample_idx": score.sample_idx,
                    "category_group": score.category_group,
                    "distance_m": score.distance_m,
                    "ttc_s": score.ttc_s,
                    "base_retrieval_score": score.base_retrieval_score,
                    "fused_score": score.fused_score,
                    "language_score": score.modality_scores.get("language", 0.0),
                    "bev_geometry_score": score.modality_scores.get("bev_geometry", 0.0),
                    "motion_score": score.modality_scores.get("motion", 0.0),
                    "sensor_score": score.modality_scores.get("sensor", 0.0),
                }
            )


def _render_markdown_report(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Multimodal Scene Retrieval Report",
        "",
        "Schema: `{0}`".format(payload.get("schema")),
        "",
        "Candidate count: `{0}`".format(payload.get("candidate_count")),
        "",
        "| Rank | Scene | Category | Distance | TTC | Fused | Language | BEV | Motion | Sensor |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for score in list(payload.get("scores") or [])[:20]:
        modality = dict(score.get("modality_scores") or {})
        lines.append(
            "| {rank} | {scene} | {category} | {distance:.3f} | {ttc} | {fused:.3f} | {language:.3f} | {bev:.3f} | {motion:.3f} | {sensor:.3f} |".format(
                rank=int(score.get("rank") or 0),
                scene=str(score.get("scene_name") or ""),
                category=str(score.get("category_group") or ""),
                distance=float(score.get("distance_m") or 0.0),
                ttc=_format_float(score.get("ttc_s")),
                fused=float(score.get("fused_score") or 0.0),
                language=float(modality.get("language") or 0.0),
                bev=float(modality.get("bev_geometry") or 0.0),
                motion=float(modality.get("motion") or 0.0),
                sensor=float(modality.get("sensor") or 0.0),
            )
        )
    lines.append("")
    return "\n".join(lines)


def _format_float(value: Any) -> str:
    try:
        numeric = float(value)
    except Exception:  # noqa: BLE001
        return ""
    if not math.isfinite(numeric):
        return "inf"
    return "{0:.3f}".format(numeric)

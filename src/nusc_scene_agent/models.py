from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ParsedQuery:
    original_text: str
    normalized_text: str
    category_groups: List[str]
    positions: List[str]
    behaviors: List[str]
    near_distance_m: float
    max_ttc_s: float
    risk_terms: List[str] = field(default_factory=list)
    specific_keywords: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievalCandidate:
    ann_token: str
    sample_token: str
    scene_token: str
    scene_name: str
    sample_idx: int
    instance_token: str
    category_name: str
    category_group: str
    location: str
    distance: float
    ttc: float
    x_ego: float
    y_ego: float
    speed: float
    rel_vx: float
    rel_vy: float
    heading_delta: float
    retrieval_score: float
    scene_description: str = ""
    num_lidar_pts: int = 0
    num_radar_pts: int = 0

    @classmethod
    def from_record(cls, record: Dict[str, Any]) -> "RetrievalCandidate":
        return cls(
            ann_token=str(record["ann_token"]),
            sample_token=str(record["sample_token"]),
            scene_token=str(record["scene_token"]),
            scene_name=str(record["scene_name"]),
            sample_idx=int(record["sample_idx"]),
            instance_token=str(record["instance_token"]),
            category_name=str(record["category_name"]),
            category_group=str(record["category_group"]),
            location=str(record["location"]),
            distance=float(record["distance"]),
            ttc=float(record["ttc"]) if record["ttc"] is not None else float("inf"),
            x_ego=float(record["x_ego"]),
            y_ego=float(record["y_ego"]),
            speed=float(record["speed"]),
            rel_vx=float(record["rel_vx"]),
            rel_vy=float(record["rel_vy"]),
            heading_delta=float(record["heading_delta"]),
            retrieval_score=float(record["retrieval_score"]),
            scene_description=str(record.get("scene_description") or ""),
            num_lidar_pts=int(record.get("num_lidar_pts") or 0),
            num_radar_pts=int(record.get("num_radar_pts") or 0),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ValidatedCase:
    query: ParsedQuery
    candidate: RetrievalCandidate
    validation_score: float
    passed: bool
    behavior_matches: Dict[str, bool]
    evidence: Dict[str, Any]
    notes: List[str]
    timeline: Any
    context_agents: Any
    ego_window: Any
    map_context: Dict[str, Any] = field(default_factory=dict)
    map_geometries: Dict[str, Any] = field(default_factory=dict)
    figure_path: Optional[str] = None
    report_dir: Optional[str] = None

    def summary_dict(self) -> Dict[str, Any]:
        payload = {
            "query": self.query.to_dict(),
            "candidate": self.candidate.to_dict(),
            "validation_score": self.validation_score,
            "passed": self.passed,
            "behavior_matches": dict(self.behavior_matches),
            "evidence": dict(self.evidence),
            "notes": list(self.notes),
        }
        if self.map_context:
            payload["map_context"] = dict(self.map_context)
        if self.figure_path:
            payload["figure_path"] = self.figure_path
        if self.report_dir:
            payload["report_dir"] = self.report_dir
        return payload

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import yaml

from nusc_scene_agent.models import ParsedQuery


ACTOR_ALIASES = {
    "car": "vehicle",
    "vehicle": "vehicle",
    "pedestrian": "pedestrian",
    "person": "pedestrian",
    "bicycle": "bicycle",
    "bike": "bicycle",
    "cyclist": "bicycle",
    "motorcycle": "motorcycle",
    "motorbike": "motorcycle",
    "bus": "bus",
    "truck": "truck",
    "trailer": "truck",
}


def _unique(items: Sequence[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for item in items:
        if not item:
            continue
        value = str(item)
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _normalize_actor(actor: str) -> str:
    key = str(actor).strip().lower()
    return ACTOR_ALIASES.get(key, key)


@dataclass
class BenchmarkQuerySpec:
    id: str
    description: str
    natural_language: str
    top_k: int = 3
    candidate_pool: Optional[int] = None
    apply_query_overrides: bool = True
    tags: List[str] = field(default_factory=list)
    actors: List[str] = field(default_factory=list)
    positions: List[str] = field(default_factory=list)
    behaviors: List[str] = field(default_factory=list)
    risk_terms: List[str] = field(default_factory=list)
    map_constraints: Dict[str, object] = field(default_factory=dict)
    thresholds: Dict[str, float] = field(default_factory=dict)
    reference_case_keys: List[str] = field(default_factory=list)
    reference_scene_names: List[str] = field(default_factory=list)
    reference_instance_tokens: List[str] = field(default_factory=list)
    reference_event_sample_range: List[int] = field(default_factory=list)
    reference_peak_sample_idx: Optional[int] = None
    expect_match: Optional[bool] = None
    benchmark_group: str = ""
    variant_type: str = ""

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "BenchmarkQuerySpec":
        query_payload = dict(payload.get("query") or {})
        description = str(payload.get("description") or query_payload.get("natural_language") or "")
        natural_language = str(query_payload.get("natural_language") or description)
        return cls(
            id=str(payload.get("id") or description or "query"),
            description=description,
            natural_language=natural_language,
            top_k=int(payload.get("top_k") or 3),
            apply_query_overrides=bool(
                query_payload.get("apply_query_overrides")
                if query_payload.get("apply_query_overrides") is not None
                else payload.get("apply_query_overrides")
                if payload.get("apply_query_overrides") is not None
                else True
            ),
            candidate_pool=(
                int(payload["candidate_pool"])
                if payload.get("candidate_pool") is not None
                else int(query_payload["candidate_pool"])
                if query_payload.get("candidate_pool") is not None
                else None
            ),
            tags=_unique(payload.get("tags") or []),
            actors=_unique([_normalize_actor(item) for item in (query_payload.get("actors") or payload.get("actors") or [])]),
            positions=_unique(query_payload.get("positions") or payload.get("positions") or []),
            behaviors=_unique(query_payload.get("behaviors") or payload.get("behaviors") or []),
            risk_terms=_unique(query_payload.get("risk_terms") or payload.get("risk_terms") or []),
            map_constraints=dict(query_payload.get("map_constraints") or payload.get("map_constraints") or {}),
            thresholds=dict(query_payload.get("thresholds") or payload.get("thresholds") or {}),
            reference_case_keys=_unique(
                payload.get("reference_case_keys") or query_payload.get("reference_case_keys") or []
            ),
            reference_scene_names=_unique(
                payload.get("reference_scene_names") or query_payload.get("reference_scene_names") or []
            ),
            reference_instance_tokens=_unique(
                payload.get("reference_instance_tokens") or query_payload.get("reference_instance_tokens") or []
            ),
            reference_event_sample_range=[
                int(item)
                for item in (
                    payload.get("reference_event_sample_range")
                    or query_payload.get("reference_event_sample_range")
                    or []
                )
            ][:2],
            reference_peak_sample_idx=(
                int(payload["reference_peak_sample_idx"])
                if payload.get("reference_peak_sample_idx") is not None
                else int(query_payload["reference_peak_sample_idx"])
                if query_payload.get("reference_peak_sample_idx") is not None
                else None
            ),
            expect_match=(
                bool(payload.get("expect_match"))
                if payload.get("expect_match") is not None
                else bool(query_payload.get("expect_match"))
                if query_payload.get("expect_match") is not None
                else None
            ),
            benchmark_group=str(payload.get("benchmark_group") or query_payload.get("benchmark_group") or ""),
            variant_type=str(payload.get("variant_type") or query_payload.get("variant_type") or ""),
        )


def load_benchmark_config(config_path: Path) -> List[BenchmarkQuerySpec]:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    queries = payload.get("queries", [])
    return [BenchmarkQuerySpec.from_dict(dict(item)) for item in queries]


def apply_benchmark_spec(query: ParsedQuery, spec: BenchmarkQuerySpec) -> ParsedQuery:
    specific_keywords = _unique(list(query.specific_keywords) + list(spec.tags))
    if spec.map_constraints:
        specific_keywords.extend(
            ["map:" + key for key, enabled in sorted(spec.map_constraints.items()) if bool(enabled)]
        )
        specific_keywords = _unique(specific_keywords)

    if not spec.apply_query_overrides:
        return replace(
            query,
            original_text=spec.natural_language,
            normalized_text=query.normalized_text,
            specific_keywords=specific_keywords,
        )

    category_groups = _unique(list(query.category_groups) + list(spec.actors))
    positions = _unique(list(query.positions) + list(spec.positions))
    behaviors = _unique(list(query.behaviors) + list(spec.behaviors))
    risk_terms = _unique(list(query.risk_terms) + list(spec.risk_terms))

    near_distance_m = float(spec.thresholds.get("near_distance_m", query.near_distance_m))
    max_ttc_s = float(spec.thresholds.get("max_ttc_s", query.max_ttc_s))

    return replace(
        query,
        original_text=spec.natural_language,
        normalized_text=query.normalized_text,
        category_groups=category_groups,
        positions=positions,
        behaviors=behaviors,
        near_distance_m=near_distance_m,
        max_ttc_s=max_ttc_s,
        risk_terms=risk_terms,
        specific_keywords=specific_keywords,
    )

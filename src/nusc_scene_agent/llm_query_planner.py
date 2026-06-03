from __future__ import annotations

import re
from dataclasses import replace
from typing import Dict, List, Optional, Sequence

from nusc_scene_agent.llm_client import LLMConfig, llm_json
from nusc_scene_agent.models import ParsedQuery
from nusc_scene_agent.query_parser import parse_query


ALLOWED_CATEGORY_GROUPS = ["vehicle", "bus", "truck", "pedestrian", "bicycle", "motorcycle"]
ALLOWED_POSITIONS = ["front", "left", "right", "rear"]
ALLOWED_BEHAVIORS = ["crossing", "cut_in", "oncoming", "stopped_lead"]
ALLOWED_RISK_TERMS = ["risky", "urgent"]
VEHICLE_FAMILY = {"vehicle", "bus", "truck"}
VRU_FAMILY = {"pedestrian", "bicycle", "motorcycle"}

ACTOR_ALIASES = {
    "car": "vehicle",
    "vehicle": "vehicle",
    "pedestrian": "pedestrian",
    "person": "pedestrian",
    "walker": "pedestrian",
    "bicycle": "bicycle",
    "bike": "bicycle",
    "cyclist": "bicycle",
    "motorcycle": "motorcycle",
    "motorbike": "motorcycle",
    "bus": "bus",
    "truck": "truck",
    "trailer": "truck",
    "lorry": "truck",
}

POSITION_ALIASES = {
    "front": "front",
    "ahead": "front",
    "forward": "front",
    "left": "left",
    "right": "right",
    "rear": "rear",
    "behind": "rear",
    "back": "rear",
}

BEHAVIOR_ALIASES = {
    "cross": "crossing",
    "crosses": "crossing",
    "crossing": "crossing",
    "jaywalk": "crossing",
    "cut in": "cut_in",
    "cut-in": "cut_in",
    "cuts in": "cut_in",
    "merge": "cut_in",
    "oncoming": "oncoming",
    "opposite direction": "oncoming",
    "stopped": "stopped_lead",
    "stationary": "stopped_lead",
    "parked": "stopped_lead",
    "blocking": "stopped_lead",
}

RISK_ALIASES = {
    "risky": "risky",
    "dangerous": "risky",
    "danger": "risky",
    "urgent": "urgent",
    "collision": "urgent",
    "crash": "urgent",
}


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower().replace("-", " "))


def _unique(items: Sequence[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for item in items:
        value = str(item).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _normalize_category_groups(items: Sequence[str]) -> List[str]:
    normalized: List[str] = []
    for item in items:
        key = _normalize_text(item)
        value = ACTOR_ALIASES.get(key, key)
        if value in ALLOWED_CATEGORY_GROUPS and value not in normalized:
            normalized.append(value)
    return normalized


def _normalize_positions(items: Sequence[str]) -> List[str]:
    normalized: List[str] = []
    for item in items:
        key = _normalize_text(item)
        value = POSITION_ALIASES.get(key, key)
        if value in ALLOWED_POSITIONS and value not in normalized:
            normalized.append(value)
    return normalized


def _normalize_behaviors(items: Sequence[str]) -> List[str]:
    normalized: List[str] = []
    for item in items:
        key = _normalize_text(item)
        value = BEHAVIOR_ALIASES.get(key, key)
        if value in ALLOWED_BEHAVIORS and value not in normalized:
            normalized.append(value)
    return normalized


def _normalize_risk_terms(items: Sequence[str]) -> List[str]:
    normalized: List[str] = []
    for item in items:
        key = _normalize_text(item)
        value = RISK_ALIASES.get(key, key)
        if value in ALLOWED_RISK_TERMS and value not in normalized:
            normalized.append(value)
    return normalized


def _build_query_from_payload(text: str, payload: Dict[str, object]) -> ParsedQuery:
    category_groups = _normalize_category_groups(payload.get("category_groups") or [])
    positions = _normalize_positions(payload.get("positions") or [])
    behaviors = _normalize_behaviors(payload.get("behaviors") or [])
    risk_terms = _normalize_risk_terms(payload.get("risk_terms") or [])
    specific_keywords = _unique([str(item) for item in (payload.get("specific_keywords") or [])])

    if "crossing" in behaviors and not category_groups:
        category_groups = ["pedestrian", "bicycle", "motorcycle"]
    if "cut_in" in behaviors and not category_groups:
        category_groups = ["vehicle", "bus", "truck"]
    if "stopped_lead" in behaviors and "front" not in positions:
        positions.append("front")

    near_distance_m = float(payload.get("near_distance_m") or (15.0 if risk_terms else 25.0))
    max_ttc_s = float(payload.get("max_ttc_s") or (4.0 if "urgent" in risk_terms else 6.0))
    near_distance_m = min(max(near_distance_m, 5.0), 60.0)
    max_ttc_s = min(max(max_ttc_s, 2.0), 10.0)

    return ParsedQuery(
        original_text=text,
        normalized_text=_normalize_text(text),
        category_groups=category_groups,
        positions=positions,
        behaviors=behaviors,
        near_distance_m=near_distance_m,
        max_ttc_s=max_ttc_s,
        risk_terms=risk_terms,
        specific_keywords=_unique(list(specific_keywords) + ["planner:llm"]),
    )


def _has_structured_signal(query: ParsedQuery) -> bool:
    return bool(query.category_groups or query.positions or query.behaviors or query.risk_terms)


def _actor_family(actor: str) -> str:
    if actor in VEHICLE_FAMILY:
        return "vehicle"
    if actor in VRU_FAMILY:
        return "vru"
    return actor


def _hypothesis_signature(query: ParsedQuery) -> tuple:
    return (
        tuple(query.category_groups),
        tuple(query.positions),
        tuple(query.behaviors),
        round(float(query.near_distance_m), 3),
        round(float(query.max_ttc_s), 3),
        tuple(query.risk_terms),
    )


def _merge_category_groups(rule_groups: Sequence[str], llm_groups: Sequence[str]) -> List[str]:
    rule = _unique(rule_groups)
    llm = _unique(llm_groups)
    if not rule:
        return llm
    if not llm:
        return rule

    exact_overlap = [item for item in llm if item in rule]
    if exact_overlap:
        return exact_overlap

    rule_families = {_actor_family(item) for item in rule}
    llm_families = {_actor_family(item) for item in llm}
    if len(rule_families) == 1 and rule_families == llm_families:
        if "vehicle" in rule and any(item in {"bus", "truck"} for item in llm):
            return llm
        if len(llm) <= len(rule):
            return llm
        return rule
    return rule


def _merge_positions(rule_positions: Sequence[str], llm_positions: Sequence[str], behaviors: Sequence[str]) -> List[str]:
    rule = _unique(rule_positions)
    llm = _unique(llm_positions)
    if not rule:
        return llm
    if not llm:
        return rule

    exact_overlap = [item for item in llm if item in rule]
    if exact_overlap:
        return exact_overlap

    lateral_rule = [item for item in rule if item in {"left", "right"}]
    lateral_llm = [item for item in llm if item in {"left", "right"}]
    if lateral_llm and not lateral_rule:
        return lateral_llm
    if lateral_rule and not lateral_llm:
        return lateral_rule
    if "front" in llm and "front" not in rule and any(item in {"crossing", "oncoming", "stopped_lead"} for item in behaviors):
        return ["front"]
    return rule


def _merge_behaviors(rule_behaviors: Sequence[str], llm_behaviors: Sequence[str]) -> List[str]:
    rule = _unique(rule_behaviors)
    llm = _unique(llm_behaviors)
    if not rule:
        return llm
    if not llm:
        return rule

    exact_overlap = [item for item in llm if item in rule]
    if exact_overlap:
        return exact_overlap
    return rule


def _merge_risk_terms(rule_risk_terms: Sequence[str], llm_risk_terms: Sequence[str]) -> List[str]:
    rule = _unique(rule_risk_terms)
    llm = _unique(llm_risk_terms)
    if not rule:
        return llm
    if not llm:
        return rule

    exact_overlap = [item for item in llm if item in rule]
    if exact_overlap:
        return exact_overlap
    return rule


def _tag_query(query: ParsedQuery, marker: str) -> ParsedQuery:
    return replace(query, specific_keywords=_unique(list(query.specific_keywords) + [marker]))


def plan_query_with_llm(text: str, config: LLMConfig) -> ParsedQuery:
    system_prompt = (
        "You are a planner for a nuScenes risky scene retrieval system. "
        "Convert the user's natural-language query into a compact JSON object. "
        "Use only the allowed labels.\n"
        "Allowed category_groups: vehicle, bus, truck, pedestrian, bicycle, motorcycle.\n"
        "Allowed positions: front, left, right, rear.\n"
        "Allowed behaviors: crossing, cut_in, oncoming, stopped_lead.\n"
        "Allowed risk_terms: risky, urgent.\n"
        "Return valid JSON only with keys: category_groups, positions, behaviors, risk_terms, near_distance_m, max_ttc_s, specific_keywords.\n"
        "Do not include unsupported labels."
    )
    user_prompt = "Query: {0}".format(text)
    payload = llm_json(
        config=config,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.0,
    )
    return _build_query_from_payload(text, payload)


def merge_queries(rule_query: ParsedQuery, llm_query: ParsedQuery) -> ParsedQuery:
    behaviors = _merge_behaviors(rule_query.behaviors, llm_query.behaviors)
    category_groups = _merge_category_groups(rule_query.category_groups, llm_query.category_groups)
    positions = _merge_positions(rule_query.positions, llm_query.positions, behaviors)
    risk_terms = _merge_risk_terms(rule_query.risk_terms, llm_query.risk_terms)
    if rule_query.behaviors and llm_query.behaviors and not set(rule_query.behaviors).intersection(llm_query.behaviors):
        risk_terms = list(rule_query.risk_terms)
    specific_keywords = _unique(list(rule_query.specific_keywords) + list(llm_query.specific_keywords) + ["planner:hybrid"])

    near_distance_m = max(float(rule_query.near_distance_m), float(llm_query.near_distance_m))
    max_ttc_s = max(float(rule_query.max_ttc_s), float(llm_query.max_ttc_s))

    return ParsedQuery(
        original_text=rule_query.original_text,
        normalized_text=rule_query.normalized_text,
        category_groups=category_groups,
        positions=positions,
        behaviors=behaviors,
        near_distance_m=near_distance_m,
        max_ttc_s=max_ttc_s,
        risk_terms=risk_terms,
        specific_keywords=specific_keywords,
    )


def resolve_hybrid_queries(text: str, config: LLMConfig) -> List[ParsedQuery]:
    rule_query = _tag_query(parse_query(text), "planner:rule")
    try:
        llm_query = _tag_query(plan_query_with_llm(text, config), "planner:llm_only")
    except Exception:  # noqa: BLE001
        return [_tag_query(rule_query, "planner:llm_error")]
    if not _has_structured_signal(llm_query):
        return [_tag_query(rule_query, "planner:llm_fallback")]

    merged_query = _tag_query(merge_queries(rule_query, llm_query), "planner:hybrid_merge")
    hypotheses: List[ParsedQuery] = []
    seen = set()
    for query in [rule_query, llm_query, merged_query]:
        signature = _hypothesis_signature(query)
        if signature in seen:
            continue
        seen.add(signature)
        hypotheses.append(query)
    return hypotheses


def resolve_query(text: str, mode: str = "rule", config: Optional[LLMConfig] = None) -> ParsedQuery:
    rule_query = parse_query(text)
    if mode == "rule":
        return rule_query
    if config is None:
        raise ValueError("LLM query mode requires an Ollama base URL and model.")
    try:
        llm_query = plan_query_with_llm(text, config)
    except Exception:  # noqa: BLE001
        fallback = parse_query(text)
        fallback.specific_keywords = _unique(list(fallback.specific_keywords) + ["planner:llm_error"])
        return fallback
    if not _has_structured_signal(llm_query):
        fallback = parse_query(text)
        fallback.specific_keywords = _unique(list(fallback.specific_keywords) + ["planner:llm_fallback"])
        return fallback
    if mode == "llm":
        return llm_query
    if mode == "hybrid":
        return merge_queries(rule_query, llm_query)
    raise ValueError("Unsupported query mode: {0}".format(mode))

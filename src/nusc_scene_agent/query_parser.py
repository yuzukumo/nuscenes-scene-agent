from __future__ import annotations

import re
from typing import List

from nusc_scene_agent.models import ParsedQuery


CATEGORY_RULES = [
    (["vulnerable road user", "vru"], ["pedestrian", "bicycle", "motorcycle"]),
    (["pedestrian", "walker", "person", "someone"], ["pedestrian"]),
    (["bike rider", "bicycle rider", "bike", "bicycle", "cyclist"], ["bicycle"]),
    (["motorcycle", "motorbike"], ["motorcycle"]),
    (["city bus", "bus"], ["bus"]),
    (["truck", "lorry", "trailer", "construction vehicle"], ["truck"]),
    (["lead vehicle", "lead car", "car", "vehicle"], ["vehicle"]),
]

POSITION_RULES = [
    (["straight ahead", "directly ahead", "dead ahead", "ahead", "front"], "front"),
    (["driver's side", "drivers side", "left"], "left"),
    (["passenger side", "right"], "right"),
    (["rear", "behind", "back"], "rear"),
]

BEHAVIOR_RULES = [
    (["steps into", "step into", "darts out", "dart out", "cuts across", "crossing", "cross", "crosses", "jaywalk"], "crossing"),
    (["drifts over", "drift over", "merges over", "merge over", "noses in", "merge", "cut in", "cut-in", "cuts in"], "cut_in"),
    (["comes straight at", "straight at", "bears down", "opposite flow", "opposite direction", "oncoming"], "oncoming"),
    (["holding ego up", "holding us up", "blocking progress", "idling in front", "sitting in", "stopped", "stationary", "parked", "blocking"], "stopped_lead"),
]

RISK_RULES = [
    (["uncomfortably close", "very close", "hugging ego", "hugging", "close", "near", "risky", "dangerous", "corner case"], "risky"),
    (["collision", "crash"], "urgent"),
]


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower().replace("-", " "))


def _token_present(normalized_text: str, token: str) -> bool:
    token_pattern = r"\b" + re.escape(token).replace(r"\ ", r"\s+") + r"\b"
    return re.search(token_pattern, normalized_text) is not None


def _collect_matches(normalized_text: str, rules: List[tuple]) -> List[str]:
    hits: List[str] = []
    for tokens, tag in rules:
        if any(_token_present(normalized_text, token) for token in tokens):
            if isinstance(tag, list):
                for item in tag:
                    if item not in hits:
                        hits.append(item)
            elif tag not in hits:
                hits.append(tag)
    return hits


def _clean_actor_matches(normalized_text: str, category_groups: List[str]) -> List[str]:
    cleaned = list(category_groups)
    if "vehicle" in cleaned and any(item in cleaned for item in ["pedestrian", "bicycle", "motorcycle"]):
        ego_vehicle_markers = [
            "front of the car",
            "ahead of the car",
            "path in front of the car",
            "holding ego up",
            "ego's lane",
            "ego lane",
        ]
        if any(marker in normalized_text for marker in ego_vehicle_markers):
            cleaned = [item for item in cleaned if item != "vehicle"]
    return cleaned


def parse_query(text: str) -> ParsedQuery:
    normalized = _normalize_text(text)
    category_groups = _clean_actor_matches(normalized, _collect_matches(normalized, CATEGORY_RULES))
    positions = _collect_matches(normalized, POSITION_RULES)
    behaviors = _collect_matches(normalized, BEHAVIOR_RULES)
    risk_terms = _collect_matches(normalized, RISK_RULES)

    distance_match = re.search(r"(\d+(?:\.\d+)?)\s*(m|meter|meters)", normalized)
    if distance_match:
        near_distance_m = float(distance_match.group(1))
    elif risk_terms:
        near_distance_m = 15.0
    else:
        near_distance_m = 25.0

    if "crossing" in behaviors and not category_groups:
        category_groups = ["pedestrian", "bicycle", "motorcycle"]

    if "cut_in" in behaviors and not category_groups:
        category_groups = ["vehicle", "bus", "truck"]

    if "stopped_lead" in behaviors and "front" not in positions:
        positions.append("front")

    max_ttc_s = 4.0 if "urgent" in risk_terms else 6.0

    return ParsedQuery(
        original_text=text,
        normalized_text=normalized,
        category_groups=category_groups,
        positions=positions,
        behaviors=behaviors,
        near_distance_m=near_distance_m,
        max_ttc_s=max_ttc_s,
        risk_terms=risk_terms,
        specific_keywords=[],
    )

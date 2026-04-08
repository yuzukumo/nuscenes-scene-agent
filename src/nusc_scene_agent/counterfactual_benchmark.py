from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import yaml

from nusc_scene_agent.query_parser import parse_query


ALTERNATE_ACTORS = {
    "pedestrian": "bicycle",
    "bicycle": "pedestrian",
    "motorcycle": "pedestrian",
    "vehicle": "truck",
    "truck": "bus",
    "bus": "truck",
}

ALTERNATE_BEHAVIORS = {
    "crossing": "oncoming",
    "cut_in": "stopped_lead",
    "oncoming": "stopped_lead",
    "stopped_lead": "cut_in",
}


def _case_semantics(entry: Dict[str, object]) -> Dict[str, object]:
    parsed_queries = [parse_query(text) for text in entry.get("source_queries") or [] if str(text).strip()]
    category_group = str(entry.get("category_group") or "vehicle")
    category_name = str(entry.get("category_name") or category_group)

    behaviors: List[str] = []
    positions: List[str] = []
    risk_terms: List[str] = []
    actors: List[str] = []
    for query in parsed_queries:
        for item in query.behaviors:
            if item not in behaviors:
                behaviors.append(item)
        for item in query.positions:
            if item not in positions:
                positions.append(item)
        for item in query.risk_terms:
            if item not in risk_terms:
                risk_terms.append(item)
        for item in query.category_groups:
            if item not in actors:
                actors.append(item)

    if not actors:
        actors = [category_group]

    primary_behavior = behaviors[0] if behaviors else "proximity"
    return {
        "actor": category_group,
        "actor_name": category_name,
        "actors": actors,
        "behaviors": behaviors,
        "positions": positions,
        "risk_terms": risk_terms,
        "primary_behavior": primary_behavior,
    }


def _actor_phrase(actor: str) -> str:
    return {
        "pedestrian": "pedestrian",
        "bicycle": "bike rider",
        "motorcycle": "motorcyclist",
        "bus": "bus",
        "truck": "truck",
        "vehicle": "vehicle",
    }.get(actor, actor)


def _position_phrase(positions: Sequence[str]) -> str:
    pos = set(positions)
    if "left" in pos:
        return "on the left side of ego"
    if "right" in pos:
        return "on the right side of ego"
    if "rear" in pos:
        return "behind ego"
    return "ahead of ego"


def _canonical_positive_query(semantics: Dict[str, object]) -> str:
    actor = _actor_phrase(str(semantics["actor"]))
    behavior = str(semantics["primary_behavior"])
    positions = list(semantics["positions"])
    if behavior == "crossing":
        return "a {0} crosses ahead of ego at a crosswalk".format(actor)
    if behavior == "cut_in":
        side = "the right" if "right" in positions else "the left" if "left" in positions else "the adjacent lane"
        return "a {0} merges into ego lane from {1}".format(actor, side)
    if behavior == "oncoming":
        return "an oncoming {0} approaches ego in the same corridor".format(actor)
    if behavior == "stopped_lead":
        return "a stopped {0} is blocking ego in lane ahead".format(actor)
    return "a {0} is at close range {1}".format(actor, _position_phrase(positions))


def _paraphrase_positive_query(semantics: Dict[str, object]) -> str:
    actor = _actor_phrase(str(semantics["actor"]))
    behavior = str(semantics["primary_behavior"])
    positions = list(semantics["positions"])
    if behavior == "crossing":
        return "a {0} crosses into ego path ahead".format(actor)
    if behavior == "cut_in":
        side = "the passenger side" if "right" in positions else "the driver's side" if "left" in positions else "an adjacent lane"
        return "a {0} merges from {1} into ego lane".format(actor, side)
    if behavior == "oncoming":
        return "a {0} approaches ego in the opposite direction".format(actor)
    if behavior == "stopped_lead":
        return "the lead {0} is stationary in ego lane".format(actor)
    return "a {0} remains at close range {1}".format(actor, _position_phrase(positions))


def _hard_negative_actor_query(semantics: Dict[str, object]) -> Tuple[str, str]:
    actor = str(semantics["actor"])
    negative_actor = ALTERNATE_ACTORS.get(actor, "vehicle")
    negative = dict(semantics)
    negative["actor"] = negative_actor
    negative["actors"] = [negative_actor]
    return _canonical_positive_query(negative), negative_actor


def _hard_negative_behavior_query(semantics: Dict[str, object]) -> Tuple[str, str]:
    behavior = str(semantics["primary_behavior"])
    negative_behavior = ALTERNATE_BEHAVIORS.get(behavior, "stopped_lead")
    negative = dict(semantics)
    negative["primary_behavior"] = negative_behavior
    negative["behaviors"] = [negative_behavior]
    if negative_behavior == "stopped_lead":
        negative["positions"] = ["front"]
    return _canonical_positive_query(negative), negative_behavior


def _thresholds_for_entry(entry: Dict[str, object]) -> Dict[str, float]:
    min_distance = float(entry.get("min_distance_m") or 10.0)
    min_ttc = entry.get("min_ttc_s")
    near_distance_m = max(8.0, min(35.0, round(min_distance * 2.2, 1)))
    if min_ttc is None:
        max_ttc_s = 6.0
    else:
        max_ttc_s = max(3.0, min(8.0, round(float(min_ttc) * 2.5, 1)))
    return {"near_distance_m": near_distance_m, "max_ttc_s": max_ttc_s}


def _variant_specs(entry: Dict[str, object]) -> List[Dict[str, object]]:
    semantics = _case_semantics(entry)
    case_key = str(entry["case_key"])
    group = "anchor_" + case_key.replace(":", "_")
    thresholds = _thresholds_for_entry(entry)
    reference_scene_names = [str(entry["scene_name"])] if entry.get("scene_name") else []
    reference_instance_tokens = [str(entry["instance_token"])] if entry.get("instance_token") else []
    reference_event_sample_range = []
    if entry.get("event_start_sample_idx") is not None and entry.get("event_end_sample_idx") is not None:
        reference_event_sample_range = [
            int(entry["event_start_sample_idx"]),
            int(entry["event_end_sample_idx"]),
        ]
    reference_peak_sample_idx = (
        int(entry["event_peak_sample_idx"]) if entry.get("event_peak_sample_idx") is not None else None
    )
    actor = str(semantics["actor"])
    behaviors = list(semantics["behaviors"]) or ([str(semantics["primary_behavior"])] if semantics["primary_behavior"] != "proximity" else [])
    positions = list(semantics["positions"])
    risk_terms = list(semantics["risk_terms"])

    positive_canonical = _canonical_positive_query(semantics)
    positive_paraphrase = _paraphrase_positive_query(semantics)
    negative_actor_query, negative_actor = _hard_negative_actor_query(semantics)
    negative_behavior_query, negative_behavior = _hard_negative_behavior_query(semantics)

    variant_defs = [
        {
            "variant_type": "positive_canonical",
            "expect_match": True,
            "query_text": positive_canonical,
            "actors": [actor],
            "behaviors": behaviors,
            "positions": positions,
            "risk_terms": risk_terms,
            "tags": ["counterfactual", "positive", "canonical"],
        },
        {
            "variant_type": "positive_paraphrase",
            "expect_match": True,
            "query_text": positive_paraphrase,
            "actors": [actor],
            "behaviors": behaviors,
            "positions": positions,
            "risk_terms": list(dict.fromkeys(risk_terms + ["risky"])),
            "tags": ["counterfactual", "positive", "paraphrase"],
        },
        {
            "variant_type": "negative_actor_swap",
            "expect_match": False,
            "query_text": negative_actor_query,
            "actors": [negative_actor],
            "behaviors": behaviors,
            "positions": positions,
            "risk_terms": risk_terms,
            "tags": ["counterfactual", "negative", "actor_swap"],
        },
        {
            "variant_type": "negative_behavior_swap",
            "expect_match": False,
            "query_text": negative_behavior_query,
            "actors": [actor],
            "behaviors": [] if negative_behavior == "proximity" else [negative_behavior],
            "positions": positions,
            "risk_terms": risk_terms,
            "tags": ["counterfactual", "negative", "behavior_swap"],
        },
    ]

    specs: List[Dict[str, object]] = []
    for variant in variant_defs:
        specs.append(
            {
                "id": "{0}_{1}".format(group, variant["variant_type"]),
                "description": "{0} {1}".format(entry["scene_name"], variant["variant_type"]),
                "top_k": 3,
                "candidate_pool": 12,
                "apply_query_overrides": False,
                "tags": list(variant["tags"]) + list(behaviors or ["proximity"]),
                "reference_case_keys": [case_key],
                "reference_scene_names": list(reference_scene_names),
                "reference_instance_tokens": list(reference_instance_tokens),
                "reference_event_sample_range": list(reference_event_sample_range),
                "reference_peak_sample_idx": reference_peak_sample_idx,
                "expect_match": bool(variant["expect_match"]),
                "benchmark_group": group,
                "variant_type": str(variant["variant_type"]),
                "query": {
                    "natural_language": str(variant["query_text"]),
                    "actors": list(variant["actors"]),
                    "positions": list(variant["positions"]),
                    "behaviors": list(variant["behaviors"]),
                    "risk_terms": list(variant["risk_terms"]),
                    "thresholds": dict(thresholds),
                },
            }
        )
    return specs


def _diverse_anchor_entries(entries: Sequence[Dict[str, object]], max_cases: int) -> List[Dict[str, object]]:
    passed_entries = [entry for entry in entries if bool(entry.get("passed"))]
    if not passed_entries:
        passed_entries = list(entries)

    scored: List[Tuple[Tuple[float, int, float], Dict[str, object], Dict[str, object]]] = []
    for entry in passed_entries:
        semantics = _case_semantics(entry)
        score = (
            float(entry.get("validation_score") or 0.0),
            len(entry.get("source_query_ids") or []),
            -float(entry.get("min_distance_m") or 999.0),
        )
        scored.append((score, entry, semantics))
    scored.sort(key=lambda item: item[0], reverse=True)

    selected: List[Dict[str, object]] = []
    seen_case_keys = set()

    # First pass: guarantee broad behavior coverage.
    covered_behaviors = set()
    for _, entry, semantics in scored:
        if len(selected) >= max_cases:
            break
        case_key = str(entry["case_key"])
        primary_behavior = str(semantics["primary_behavior"])
        if case_key in seen_case_keys or primary_behavior in covered_behaviors:
            continue
        selected.append(entry)
        seen_case_keys.add(case_key)
        covered_behaviors.add(primary_behavior)

    # Second pass: improve actor diversity.
    covered_actors = {str(_case_semantics(entry)["actor"]) for entry in selected}
    for _, entry, semantics in scored:
        if len(selected) >= max_cases:
            break
        case_key = str(entry["case_key"])
        actor = str(semantics["actor"])
        if case_key in seen_case_keys or actor in covered_actors:
            continue
        selected.append(entry)
        seen_case_keys.add(case_key)
        covered_actors.add(actor)

    # Final pass: fill remaining slots by score.
    for _, entry, _ in scored:
        if len(selected) >= max_cases:
            break
        case_key = str(entry["case_key"])
        if case_key in seen_case_keys:
            continue
        selected.append(entry)
        seen_case_keys.add(case_key)
    return selected


def generate_counterfactual_benchmark_from_case_library(
    case_library_path: Path,
    output_path: Path,
    max_cases: int = 6,
) -> Dict[str, object]:
    source_case_library = str(case_library_path)
    case_library_path = case_library_path.resolve()
    output_path = output_path.resolve()
    entries = json.loads(case_library_path.read_text(encoding="utf-8"))
    anchors = _diverse_anchor_entries(entries, max_cases=max_cases)

    queries: List[Dict[str, object]] = []
    for entry in anchors:
        queries.extend(_variant_specs(entry))

    payload = {
        "metadata": {
            "generator": "counterfactual_benchmark_generator_v1",
            "source_case_library": source_case_library,
            "anchor_case_count": len(anchors),
            "query_count": len(queries),
        },
        "queries": queries,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return payload["metadata"]

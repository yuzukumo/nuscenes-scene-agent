from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import yaml


DEFAULT_SCENARIO_TAXONOMY = Path("configs/scenario_taxonomy.yaml")
SCENARIO_TAXONOMY_SCHEMA = "scenario_taxonomy_v1"


def load_scenario_taxonomy(path: Path = DEFAULT_SCENARIO_TAXONOMY) -> Dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if payload.get("schema") != SCENARIO_TAXONOMY_SCHEMA:
        raise ValueError(
            "Unsupported scenario taxonomy schema: {0!r}.".format(payload.get("schema"))
        )
    families = list(payload.get("families") or [])
    family_ids = [str(item.get("id") or "") for item in families]
    if not families or any(not item for item in family_ids) or len(set(family_ids)) != len(family_ids):
        raise ValueError("Scenario taxonomy family IDs must be non-empty and unique.")
    return payload


def taxonomy_label_index(
    taxonomy: Mapping[str, Any],
    backend: str,
) -> Dict[str, list[str]]:
    index: Dict[str, list[str]] = defaultdict(list)
    for family in taxonomy.get("families", []):
        family_id = str(family.get("id") or "")
        backend_spec = dict(dict(family.get("backends") or {}).get(backend) or {})
        match = dict(backend_spec.get("match") or {})
        values = list(match.get("values") or backend_spec.get("labels") or [])
        for value in values:
            label = str(value).strip()
            if label and family_id not in index[label]:
                index[label].append(family_id)
    return dict(index)


def map_scenario_labels(
    backend: str,
    labels: Iterable[str],
    taxonomy: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    taxonomy = dict(taxonomy or load_scenario_taxonomy())
    index = taxonomy_label_index(taxonomy, backend)
    normalized = list(dict.fromkeys(str(item).strip() for item in labels if str(item).strip()))
    matched: Dict[str, list[str]] = {label: list(index.get(label) or []) for label in normalized}
    family_ids = sorted({family_id for values in matched.values() for family_id in values})
    return {
        "backend": backend,
        "labels": normalized,
        "family_ids": family_ids,
        "matched_labels": {key: value for key, value in matched.items() if value},
        "unmapped_labels": sorted(key for key, value in matched.items() if not value),
        "ambiguous_labels": {key: value for key, value in matched.items() if len(value) > 1},
    }


def build_taxonomy_coverage(
    backend: str,
    label_counts: Mapping[str, int] | Sequence[str],
    taxonomy: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    counts = Counter(label_counts) if not isinstance(label_counts, Mapping) else Counter(
        {str(key): int(value) for key, value in label_counts.items()}
    )
    taxonomy = dict(taxonomy or load_scenario_taxonomy())
    index = taxonomy_label_index(taxonomy, backend)
    family_counts: Counter[str] = Counter()
    unmapped: Dict[str, int] = {}
    ambiguous: Dict[str, list[str]] = {}
    for label, count in counts.items():
        families = list(index.get(str(label)) or [])
        if not families:
            unmapped[str(label)] = int(count)
            continue
        if len(families) > 1:
            ambiguous[str(label)] = families
        for family_id in families:
            family_counts[family_id] += int(count)
    total = sum(counts.values())
    mapped = total - sum(unmapped.values())
    return {
        "schema": "scenario_taxonomy_coverage_v1",
        "backend": backend,
        "label_count": len(counts),
        "sample_count": total,
        "mapped_sample_count": mapped,
        "mapped_sample_rate": round(mapped / total, 6) if total else 0.0,
        "canonical_family_counts": dict(sorted(family_counts.items())),
        "unmapped_label_counts": dict(sorted(unmapped.items())),
        "ambiguous_labels": dict(sorted(ambiguous.items())),
    }

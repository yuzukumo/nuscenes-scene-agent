from __future__ import annotations

import csv
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


HUMAN_AUDIT_ITEM_SCHEMA = "human_audit_item_v1"
HUMAN_AUDIT_SET_SCHEMA = "human_audit_set_v1"
HUMAN_AUDIT_METRICS_SCHEMA = "human_audit_metrics_v1"


def _as_list(value: object) -> List[object]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str) and "|" in value:
        return [item for item in value.split("|") if item]
    return [value]


def _as_strings(value: object) -> List[str]:
    return [str(item) for item in _as_list(value) if str(item)]


def _optional_float(value: object) -> Optional[float]:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except Exception:  # noqa: BLE001
        return None


def _optional_int(value: object) -> Optional[int]:
    try:
        if value in (None, ""):
            return None
        return int(float(value))
    except Exception:  # noqa: BLE001
        return None


def _optional_bool(value: object) -> Optional[bool]:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "match", "correct", "pass", "positive"}:
        return True
    if text in {"0", "false", "no", "n", "mismatch", "incorrect", "fail", "negative"}:
        return False
    return None


def _behavior_key(entry: Mapping[str, object]) -> str:
    for key in ("all_behaviors", "matched_behaviors"):
        values = _as_strings(entry.get(key))
        if values:
            return values[0]
    value = str(entry.get("event_primary_behavior") or "").strip()
    return value or "unspecified"


def _difficulty_bucket(entry: Mapping[str, object]) -> str:
    if not _optional_bool(entry.get("passed")):
        return "not_validation_passed"
    score = _optional_float(
        entry.get("validation_quality_score", entry.get("validation_score"))
    ) or 0.0
    if score < 80.0:
        return "low_score"
    if score < 90.0:
        return "borderline_score"
    return "high_score"


def _relative_reference(value: object, base_dir: Path) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    path = Path(text)
    if not path.is_absolute():
        path = path.resolve()
    return Path(os.path.relpath(path, start=base_dir)).as_posix()


def _load_case_library(case_library_path: Path) -> List[Dict[str, object]]:
    payload = json.loads(case_library_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Case library must be a JSON list: {0}".format(case_library_path))
    return [dict(item) for item in payload]


def _stratified_sample(
    entries: Sequence[Mapping[str, object]],
    sample_size: int,
    seed: int,
    max_per_behavior: int = 0,
) -> List[Mapping[str, object]]:
    if sample_size <= 0 or sample_size >= len(entries):
        return list(entries)

    rng = random.Random(seed)
    buckets: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
    for entry in entries:
        buckets[_behavior_key(entry)].append(entry)

    for bucket in buckets.values():
        bucket.sort(key=lambda item: str(item.get("case_key") or ""))
        rng.shuffle(bucket)
        if max_per_behavior > 0:
            del bucket[max_per_behavior:]

    selected: List[Mapping[str, object]] = []
    bucket_names = sorted(buckets)
    while len(selected) < sample_size and bucket_names:
        made_progress = False
        for name in list(bucket_names):
            bucket = buckets[name]
            if not bucket:
                bucket_names.remove(name)
                continue
            selected.append(bucket.pop(0))
            made_progress = True
            if len(selected) >= sample_size:
                break
        if not made_progress:
            break
    return selected


def _build_audit_item(
    entry: Mapping[str, object],
    audit_index: int,
    output_dir: Path,
) -> Dict[str, object]:
    behavior = _behavior_key(entry)
    return {
        "schema": HUMAN_AUDIT_ITEM_SCHEMA,
        "audit_id": "audit_{0:04d}".format(audit_index),
        "case_key": str(entry.get("case_key") or ""),
        "behavior": behavior,
        "difficulty_bucket": _difficulty_bucket(entry),
        "source": {
            "query_ids": _as_strings(entry.get("source_query_ids")),
            "queries": _as_strings(entry.get("source_queries")),
            "query_tags": _as_strings(entry.get("source_query_tags")),
        },
        "candidate": {
            "scene_name": str(entry.get("scene_name") or ""),
            "scene_token": str(entry.get("scene_token") or ""),
            "sample_idx": _optional_int(entry.get("sample_idx")),
            "sample_token": str(entry.get("sample_token") or ""),
            "instance_token": str(entry.get("instance_token") or ""),
            "category_name": str(entry.get("category_name") or ""),
            "category_group": str(entry.get("category_group") or ""),
            "location": str(entry.get("location") or ""),
        },
        "system": {
            "passed": bool(_optional_bool(entry.get("passed"))),
            "validation_score": _optional_float(entry.get("validation_score")),
            "validation_quality_score": _optional_float(
                entry.get("validation_quality_score", entry.get("validation_score"))
            ),
            "retrieval_score": _optional_float(entry.get("retrieval_score")),
            "min_distance_m": _optional_float(entry.get("min_distance_m")),
            "min_ttc_s": _optional_float(entry.get("min_ttc_s")),
            "matched_behaviors": _as_strings(entry.get("matched_behaviors")),
            "all_behaviors": _as_strings(entry.get("all_behaviors")),
            "event_start_sample_idx": _optional_int(entry.get("event_start_sample_idx")),
            "event_end_sample_idx": _optional_int(entry.get("event_end_sample_idx")),
            "event_peak_sample_idx": _optional_int(entry.get("event_peak_sample_idx")),
        },
        "evidence": {
            "figure_path": _relative_reference(entry.get("figure_path"), output_dir),
            "report_dir": _relative_reference(entry.get("report_dir"), output_dir),
            "notes": _as_strings(entry.get("notes")),
        },
        "labels": {
            "semantic_match": None,
            "primary_actor_correct": None,
            "behavior_correct": None,
            "event_window_correct": None,
            "event_start_sample_idx": None,
            "event_end_sample_idx": None,
            "event_peak_sample_idx": None,
            "confidence": None,
            "notes": "",
        },
    }


CSV_FIELDS = [
    "audit_id",
    "case_key",
    "behavior",
    "difficulty_bucket",
    "query_ids",
    "queries",
    "scene_name",
    "sample_idx",
    "category_name",
    "validation_score",
    "validation_quality_score",
    "passed",
    "system_event_start_sample_idx",
    "system_event_end_sample_idx",
    "system_event_peak_sample_idx",
    "figure_path",
    "report_dir",
    "semantic_match",
    "primary_actor_correct",
    "behavior_correct",
    "event_window_correct",
    "event_start_sample_idx",
    "event_end_sample_idx",
    "event_peak_sample_idx",
    "confidence",
    "notes",
]


def _item_to_csv_row(item: Mapping[str, object]) -> Dict[str, object]:
    source = dict(item.get("source") or {})
    candidate = dict(item.get("candidate") or {})
    system = dict(item.get("system") or {})
    evidence = dict(item.get("evidence") or {})
    labels = dict(item.get("labels") or {})
    return {
        "audit_id": item.get("audit_id", ""),
        "case_key": item.get("case_key", ""),
        "behavior": item.get("behavior", ""),
        "difficulty_bucket": item.get("difficulty_bucket", ""),
        "query_ids": "|".join(_as_strings(source.get("query_ids"))),
        "queries": "|".join(_as_strings(source.get("queries"))),
        "scene_name": candidate.get("scene_name", ""),
        "sample_idx": candidate.get("sample_idx", ""),
        "category_name": candidate.get("category_name", ""),
        "validation_score": system.get("validation_score", ""),
        "validation_quality_score": system.get("validation_quality_score", ""),
        "passed": system.get("passed", ""),
        "system_event_start_sample_idx": system.get("event_start_sample_idx", ""),
        "system_event_end_sample_idx": system.get("event_end_sample_idx", ""),
        "system_event_peak_sample_idx": system.get("event_peak_sample_idx", ""),
        "figure_path": evidence.get("figure_path", ""),
        "report_dir": evidence.get("report_dir", ""),
        "semantic_match": labels.get("semantic_match", ""),
        "primary_actor_correct": labels.get("primary_actor_correct", ""),
        "behavior_correct": labels.get("behavior_correct", ""),
        "event_window_correct": labels.get("event_window_correct", ""),
        "event_start_sample_idx": labels.get("event_start_sample_idx", ""),
        "event_end_sample_idx": labels.get("event_end_sample_idx", ""),
        "event_peak_sample_idx": labels.get("event_peak_sample_idx", ""),
        "confidence": labels.get("confidence", ""),
        "notes": labels.get("notes", ""),
    }


def _write_jsonl(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def _write_csv(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        for item in rows:
            writer.writerow({field: _item_to_csv_row(item).get(field, "") for field in CSV_FIELDS})


def _write_audit_readme(items: Sequence[Mapping[str, object]], output_dir: Path) -> None:
    behavior_counts: Dict[str, int] = defaultdict(int)
    for item in items:
        behavior_counts[str(item.get("behavior") or "unspecified")] += 1

    lines = [
        "# Human Audit Set",
        "",
        "This directory contains an unlabeled template for manual author review of mined risk-scene anchors.",
        "",
        "## Files",
        "",
        "- `human_audit_items.jsonl`: canonical annotation file.",
        "- `human_audit_items.csv`: spreadsheet-friendly copy with the same label fields.",
        "- `human_audit_manifest.json`: sampling metadata.",
        "- `review_queue.md`: compact review table with evidence links.",
        "",
        "## Label Fields",
        "",
        "- `semantic_match`: whether the scene matches the source query semantics.",
        "- `primary_actor_correct`: whether the selected actor is the main risk actor.",
        "- `behavior_correct`: whether the system behavior label is correct.",
        "- `event_window_correct`: whether the system event window is acceptable.",
        "- `event_start_sample_idx`, `event_end_sample_idx`, `event_peak_sample_idx`: optional corrected event indices.",
        "- `confidence`: reviewer confidence in `[0, 1]`.",
        "- `notes`: short free-text justification for ambiguous cases.",
        "",
        "Use `true`, `false`, or blank for boolean fields. Blank fields are ignored by the evaluator.",
        "",
        "## Sample",
        "",
        "| Behavior | Cases |",
        "| --- | ---: |",
    ]
    for behavior, count in sorted(behavior_counts.items()):
        lines.append("| {0} | {1} |".format(behavior, count))
    output_dir.joinpath("README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_review_queue(items: Sequence[Mapping[str, object]], output_dir: Path) -> None:
    lines = [
        "# Author Audit Review Queue",
        "",
        "Review the evidence for each row and fill the label columns in `human_audit_items.jsonl` or `human_audit_items.csv`.",
        "",
        "| ID | Behavior | Query | Scene | Validation Quality | Evidence | Report |",
        "| --- | --- | --- | --- | ---: | --- | --- |",
    ]
    for item in items:
        source = dict(item.get("source") or {})
        candidate = dict(item.get("candidate") or {})
        system = dict(item.get("system") or {})
        evidence = dict(item.get("evidence") or {})
        queries = _as_strings(source.get("queries"))
        query_text = queries[0] if queries else ""
        figure_path = str(evidence.get("figure_path") or "")
        report_dir = str(evidence.get("report_dir") or "")
        evidence_link = "[evidence]({0})".format(figure_path) if figure_path else ""
        report_link = "[report]({0})".format(report_dir) if report_dir else ""
        lines.append(
            "| {audit_id} | {behavior} | {query} | {scene} #{sample} | {score} | {evidence} | {report} |".format(
                audit_id=item.get("audit_id", ""),
                behavior=item.get("behavior", ""),
                query=query_text.replace("|", "/"),
                scene=str(candidate.get("scene_name") or ""),
                sample=str(candidate.get("sample_idx") or ""),
                score=system.get("validation_quality_score", system.get("validation_score", "")),
                evidence=evidence_link,
                report=report_link,
            )
        )
    output_dir.joinpath("review_queue.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_human_audit_set(
    case_library_path: Path,
    output_dir: Path,
    sample_size: int = 100,
    seed: int = 7,
    max_per_behavior: int = 0,
) -> Dict[str, object]:
    case_library_path = case_library_path.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    entries = _load_case_library(case_library_path)
    sampled = _stratified_sample(entries, sample_size=sample_size, seed=seed, max_per_behavior=max_per_behavior)
    items = [
        _build_audit_item(entry, idx, output_dir)
        for idx, entry in enumerate(sampled, start=1)
    ]

    jsonl_path = output_dir / "human_audit_items.jsonl"
    csv_path = output_dir / "human_audit_items.csv"
    manifest_path = output_dir / "human_audit_manifest.json"
    _write_jsonl(items, jsonl_path)
    _write_csv(items, csv_path)
    _write_audit_readme(items, output_dir)
    _write_review_queue(items, output_dir)

    manifest = {
        "schema": HUMAN_AUDIT_SET_SCHEMA,
        "path_base": "audit_directory",
        "case_library": _relative_reference(case_library_path, output_dir),
        "output_dir": ".",
        "case_library_count": len(entries),
        "audit_item_count": len(items),
        "requested_sample_size": int(sample_size),
        "sample_size": len(items),
        "sample_size_limited_by_available_cases": bool(
            sample_size > 0 and len(items) < sample_size
        ),
        "seed": int(seed),
        "max_per_behavior": int(max_per_behavior),
        "jsonl": jsonl_path.name,
        "csv": csv_path.name,
        "review_queue": "review_queue.md",
        "behavior_counts": {
            behavior: sum(1 for item in items if item.get("behavior") == behavior)
            for behavior in sorted({str(item.get("behavior")) for item in items})
        },
        "label_status": "unlabeled_template",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    runtime_manifest = dict(manifest)
    runtime_manifest.update(
        {
            "case_library": str(case_library_path),
            "output_dir": str(output_dir),
            "jsonl": str(jsonl_path),
            "csv": str(csv_path),
            "review_queue": str(output_dir / "review_queue.md"),
            "manifest": str(manifest_path),
        }
    )
    return runtime_manifest


def _read_jsonl(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(dict(json.loads(line)))
    return rows


def _read_csv(path: Path) -> List[Dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _load_audit_annotations(path: Path) -> List[Dict[str, object]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return _read_jsonl(path)
    if suffix == ".csv":
        return _read_csv(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [dict(item) for item in payload]
    if isinstance(payload, dict) and isinstance(payload.get("items"), list):
        return [dict(item) for item in payload["items"]]
    raise ValueError("Unsupported annotation format: {0}".format(path))


def _labels_from_row(row: Mapping[str, object]) -> Dict[str, object]:
    if "labels" in row and isinstance(row["labels"], Mapping):
        return dict(row["labels"])
    return {
        "semantic_match": row.get("semantic_match"),
        "primary_actor_correct": row.get("primary_actor_correct"),
        "behavior_correct": row.get("behavior_correct"),
        "event_window_correct": row.get("event_window_correct"),
        "event_start_sample_idx": row.get("event_start_sample_idx"),
        "event_end_sample_idx": row.get("event_end_sample_idx"),
        "event_peak_sample_idx": row.get("event_peak_sample_idx"),
        "confidence": row.get("confidence"),
        "notes": row.get("notes"),
    }


def _system_from_row(row: Mapping[str, object]) -> Dict[str, object]:
    if "system" in row and isinstance(row["system"], Mapping):
        return dict(row["system"])
    return {
        "event_start_sample_idx": row.get("system_event_start_sample_idx") or row.get("event_start_sample_idx_system"),
        "event_end_sample_idx": row.get("system_event_end_sample_idx") or row.get("event_end_sample_idx_system"),
        "event_peak_sample_idx": row.get("system_event_peak_sample_idx") or row.get("event_peak_sample_idx_system"),
        "validation_score": row.get("validation_score"),
        "passed": row.get("passed"),
    }


def _event_iou(system: Mapping[str, object], labels: Mapping[str, object]) -> Optional[float]:
    system_start = _optional_int(system.get("event_start_sample_idx"))
    system_end = _optional_int(system.get("event_end_sample_idx"))
    label_start = _optional_int(labels.get("event_start_sample_idx"))
    label_end = _optional_int(labels.get("event_end_sample_idx"))
    if None in {system_start, system_end, label_start, label_end}:
        return None
    assert system_start is not None
    assert system_end is not None
    assert label_start is not None
    assert label_end is not None
    intersection = max(0, min(system_end, label_end) - max(system_start, label_start) + 1)
    union = max(system_end, label_end) - min(system_start, label_start) + 1
    if union <= 0:
        return None
    return float(intersection) / float(union)


def _rate(values: Iterable[Optional[bool]]) -> Optional[float]:
    usable = [value for value in values if value is not None]
    if not usable:
        return None
    return sum(1 for value in usable if value) / len(usable)


def _round_optional(value: Optional[float], digits: int = 4) -> Optional[float]:
    return round(float(value), digits) if value is not None else None


def _annotation_record(row: Mapping[str, object]) -> Dict[str, object]:
    labels = _labels_from_row(row)
    system = _system_from_row(row)
    semantic_match = _optional_bool(labels.get("semantic_match"))
    event_iou = _event_iou(system, labels)
    return {
        "audit_id": str(row.get("audit_id") or ""),
        "case_key": str(row.get("case_key") or ""),
        "behavior": str(row.get("behavior") or "unspecified"),
        "semantic_match": semantic_match,
        "primary_actor_correct": _optional_bool(labels.get("primary_actor_correct")),
        "behavior_correct": _optional_bool(labels.get("behavior_correct")),
        "event_window_correct": _optional_bool(labels.get("event_window_correct")),
        "event_iou": event_iou,
        "confidence": _optional_float(labels.get("confidence")),
        "notes": str(labels.get("notes") or ""),
    }


def _summarize_records(records: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    labeled = [row for row in records if row.get("semantic_match") is not None]
    positive = [row for row in labeled if row.get("semantic_match") is True]
    event_ious = [float(row["event_iou"]) for row in labeled if row.get("event_iou") is not None]
    confidences = [float(row["confidence"]) for row in labeled if row.get("confidence") is not None]
    return {
        "item_count": len(records),
        "labeled_count": len(labeled),
        "semantic_match_count": len(positive),
        "semantic_match_rate": _round_optional(_rate([row.get("semantic_match") for row in records])),
        "primary_actor_correct_rate": _round_optional(_rate([row.get("primary_actor_correct") for row in labeled])),
        "behavior_correct_rate": _round_optional(_rate([row.get("behavior_correct") for row in labeled])),
        "event_window_correct_rate": _round_optional(_rate([row.get("event_window_correct") for row in labeled])),
        "mean_event_iou": round(mean(event_ious), 4) if event_ious else None,
        "mean_confidence": round(mean(confidences), 4) if confidences else None,
    }


def _group_by_behavior(records: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
    for row in records:
        grouped[str(row.get("behavior") or "unspecified")].append(row)
    rows: List[Dict[str, object]] = []
    for behavior, behavior_records in sorted(grouped.items()):
        summary = _summarize_records(behavior_records)
        summary["behavior"] = behavior
        rows.append(summary)
    return rows


def _write_metrics_markdown(payload: Mapping[str, object], output_dir: Path) -> None:
    overview = dict(payload.get("overview") or {})
    lines = [
        "# Human Audit Metrics",
        "",
        "- Items: {0}".format(overview.get("item_count")),
        "- Labeled items: {0}".format(overview.get("labeled_count")),
        "- Semantic match rate: {0}".format(overview.get("semantic_match_rate")),
        "- Primary actor correct rate: {0}".format(overview.get("primary_actor_correct_rate")),
        "- Behavior correct rate: {0}".format(overview.get("behavior_correct_rate")),
        "- Event window correct rate: {0}".format(overview.get("event_window_correct_rate")),
        "- Mean event IoU: {0}".format(overview.get("mean_event_iou")),
        "",
        "## Behavior Breakdown",
        "",
        "| Behavior | Items | Labeled | Semantic Match Rate | Actor Correct Rate | Behavior Correct Rate | Event IoU |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload.get("by_behavior", []):
        lines.append(
            "| {behavior} | {item_count} | {labeled_count} | {semantic_match_rate} | "
            "{primary_actor_correct_rate} | {behavior_correct_rate} | {mean_event_iou} |".format(**row)
        )
    output_dir.joinpath("human_audit_metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_behavior_csv(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    fieldnames = [
        "behavior",
        "item_count",
        "labeled_count",
        "semantic_match_count",
        "semantic_match_rate",
        "primary_actor_correct_rate",
        "behavior_correct_rate",
        "event_window_correct_rate",
        "mean_event_iou",
        "mean_confidence",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def evaluate_human_audit_set(annotation_path: Path, output_dir: Path) -> Dict[str, object]:
    annotation_path = annotation_path.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_audit_annotations(annotation_path)
    records = [_annotation_record(row) for row in rows]
    by_behavior = _group_by_behavior(records)
    false_positive_cases = [
        {
            "audit_id": row["audit_id"],
            "case_key": row["case_key"],
            "behavior": row["behavior"],
            "notes": row["notes"],
        }
        for row in records
        if row.get("semantic_match") is False
    ]
    payload = {
        "schema": HUMAN_AUDIT_METRICS_SCHEMA,
        "annotation_path": str(annotation_path),
        "overview": _summarize_records(records),
        "by_behavior": by_behavior,
        "false_positive_cases": false_positive_cases,
        "metric_note": (
            "These metrics use human audit labels. Unlabeled rows are excluded from rates except item counts."
        ),
    }

    output_dir.joinpath("human_audit_metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_behavior_csv(by_behavior, output_dir / "human_audit_by_behavior.csv")
    _write_metrics_markdown(payload, output_dir)
    return payload

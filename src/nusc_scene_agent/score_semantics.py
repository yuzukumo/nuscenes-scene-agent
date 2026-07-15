"""Shared field semantics for validation-quality artifacts.

The project used ``validation_score`` in early artifacts.  New artifacts use
``validation_quality_score`` to distinguish a continuous evidence score from
the binary validation-acceptance gate.  These helpers keep old artifacts
readable while making the canonical field explicit at every reporting layer.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


VALIDATION_QUALITY_SCORE_FIELD = "validation_quality_score"
LEGACY_VALIDATION_SCORE_FIELD = "validation_score"


def get_validation_quality_score(record: Mapping[str, Any], default: float = 0.0) -> float:
    """Read the canonical quality score with a legacy-artifact fallback."""
    value = record.get(VALIDATION_QUALITY_SCORE_FIELD)
    if value is None:
        value = record.get(LEGACY_VALIDATION_SCORE_FIELD, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def get_best_validation_quality_score(record: Mapping[str, Any], default: float = 0.0) -> float:
    """Read the quality score attached to the selected best case.

    New benchmark artifacts define this as the highest-quality accepted case,
    or the highest-quality rejected case when no candidate passes. Older
    artifacts used ``best_validation_score`` for the same selection policy.
    """
    value = record.get("best_validation_quality_score")
    if value is None:
        value = record.get("best_validation_score", default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def get_max_validation_quality_score(record: Mapping[str, Any], default: float = 0.0) -> float:
    """Read the maximum quality score without applying the acceptance gate."""
    value = record.get("max_validation_quality_score")
    if value is None:
        value = record.get(VALIDATION_QUALITY_SCORE_FIELD)
    if value is None:
        value = record.get(LEGACY_VALIDATION_SCORE_FIELD, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def get_mean_best_validation_quality_score(record: Mapping[str, Any], default: float = 0.0) -> float:
    """Read a profile-level mean quality score with compatibility fallback."""
    value = record.get("mean_best_validation_quality_score")
    if value is None:
        value = record.get("mean_best_validation_score", default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def get_average_best_validation_quality_score(record: Mapping[str, Any], default: float = 0.0) -> float:
    """Read a behavior/actor aggregate quality score with compatibility fallback."""
    value = record.get("avg_best_validation_quality_score")
    if value is None:
        value = record.get("avg_best_validation_score", default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)

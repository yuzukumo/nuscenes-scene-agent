from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from nusc_scene_agent.validation import ValidationConfig


def default_ablation_profiles(
    base_query_mode: str = "hybrid",
    base_rerank_mode: str = "llm",
) -> List[Dict[str, object]]:
    profiles: List[Dict[str, object]] = [
        {
            "name": "full_system",
            "label": "Full-System",
            "description": "Base configuration with all enabled modules.",
            "query_mode": base_query_mode,
            "rerank_mode": base_rerank_mode,
            "validation_config": ValidationConfig(name="full_system"),
        }
    ]

    if base_rerank_mode != "none":
        profiles.append(
            {
                "name": "no_rerank",
                "label": "No-Rerank",
                "description": "Disable the LLM reranking stage while keeping the planner unchanged.",
                "query_mode": base_query_mode,
                "rerank_mode": "none",
                "validation_config": ValidationConfig(name="no_rerank"),
            }
        )

    profiles.extend(
        [
            {
                "name": "no_map_context",
                "label": "No-Map-Context",
                "description": "Remove map-aware validation support from the scoring pipeline.",
                "query_mode": base_query_mode,
                "rerank_mode": base_rerank_mode,
                "validation_config": ValidationConfig(
                    name="no_map_context",
                    enable_map_context=False,
                ),
            },
            {
                "name": "no_event_localization",
                "label": "No-Event-Localization",
                "description": "Disable event-window localization while keeping retrieval and validation active.",
                "query_mode": base_query_mode,
                "rerank_mode": base_rerank_mode,
                "validation_config": ValidationConfig(
                    name="no_event_localization",
                    enable_event_localization=False,
                ),
            },
        ]
    )
    return profiles


def write_ablation_manifest(profiles: List[Dict[str, object]], output_dir: Path) -> None:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = []
    for profile in profiles:
        validation_config = profile.get("validation_config")
        payload.append(
            {
                "name": str(profile["name"]),
                "label": str(profile.get("label") or profile["name"]),
                "description": str(profile.get("description") or ""),
                "query_mode": str(profile.get("query_mode") or ""),
                "rerank_mode": str(profile.get("rerank_mode") or ""),
                "validation_config": validation_config.to_dict() if isinstance(validation_config, ValidationConfig) else {},
                "output_dir": str(profile.get("output_dir") or ""),
            }
        )
    (output_dir / "ablation_manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

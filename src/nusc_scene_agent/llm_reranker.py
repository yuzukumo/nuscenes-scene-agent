from __future__ import annotations

import json
from typing import Dict, List, Sequence

from nusc_scene_agent.llm_client import LLMConfig, llm_json
from nusc_scene_agent.models import ParsedQuery, RetrievalCandidate


def rerank_candidates_with_llm(
    query: ParsedQuery,
    candidates: Sequence[RetrievalCandidate],
    config: LLMConfig,
) -> List[RetrievalCandidate]:
    if len(candidates) <= 1:
        return list(candidates)

    candidate_payload: List[Dict[str, object]] = []
    for idx, candidate in enumerate(candidates, start=1):
        candidate_payload.append(
            {
                "id": idx,
                "scene_name": candidate.scene_name,
                "sample_idx": candidate.sample_idx,
                "actor": candidate.category_name,
                "category_group": candidate.category_group,
                "location": candidate.location,
                "distance": round(candidate.distance, 3),
                "ttc": None if candidate.ttc == float("inf") else round(candidate.ttc, 3),
                "x_ego": round(candidate.x_ego, 3),
                "y_ego": round(candidate.y_ego, 3),
                "speed": round(candidate.speed, 3),
                "rel_vx": round(candidate.rel_vx, 3),
                "rel_vy": round(candidate.rel_vy, 3),
                "heading_delta": round(candidate.heading_delta, 3),
                "retrieval_score": round(candidate.retrieval_score, 3),
                "scene_description": candidate.scene_description[:160],
            }
        )

    system_prompt = (
        "You are ranking retrieved nuScenes scene candidates for a risky-scene query. "
        "Return JSON only with a key named ranking whose value is a list of candidate ids in the best-first order. "
        "Prefer candidates whose actor type, relative position, motion pattern, and geometry best match the query."
    )
    user_prompt = {
        "query": query.to_dict(),
        "candidates": candidate_payload,
    }
    payload = llm_json(
        config=config,
        system_prompt=system_prompt,
        user_prompt=json.dumps(user_prompt, ensure_ascii=False),
        temperature=0.0,
    )
    ranking = payload.get("ranking") or []
    order: List[int] = []
    for item in ranking:
        try:
            idx = int(item)
        except Exception:  # noqa: BLE001
            continue
        if 1 <= idx <= len(candidates) and idx not in order:
            order.append(idx)

    ordered: List[RetrievalCandidate] = [candidates[idx - 1] for idx in order]
    remaining = [candidate for idx, candidate in enumerate(candidates, start=1) if idx not in order]
    return ordered + remaining

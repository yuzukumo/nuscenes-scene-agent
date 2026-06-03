from __future__ import annotations

import os
import sqlite3
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from nusc_scene_agent.benchmark_schema import BenchmarkQuerySpec, apply_benchmark_spec
from nusc_scene_agent.learned_retrieval import DEFAULT_LEARNED_RETRIEVER_CHECKPOINT, rerank_candidates_with_learned_model
from nusc_scene_agent.llm_client import LLMConfig
from nusc_scene_agent.llm_query_planner import resolve_hybrid_queries, resolve_query
from nusc_scene_agent.llm_reranker import rerank_candidates_with_llm
from nusc_scene_agent.models import ParsedQuery, ValidatedCase
from nusc_scene_agent.multimodal_retrieval import rerank_candidates_with_multimodal_model
from nusc_scene_agent.reporting import slugify, write_query_report
from nusc_scene_agent.retrieval import retrieve_candidates
from nusc_scene_agent.validation import ValidationConfig, validate_candidate


def _select_diverse_cases(validated: Sequence[ValidatedCase], top_k: int) -> List[ValidatedCase]:
    selected: List[ValidatedCase] = []
    seen_samples = set()

    for case in validated:
        sample_token = case.candidate.sample_token
        if sample_token in seen_samples:
            continue
        seen_samples.add(sample_token)
        selected.append(case)
        if len(selected) >= top_k:
            return selected

    for case in validated:
        if case in selected:
            continue
        selected.append(case)
        if len(selected) >= top_k:
            break
    return selected


def _hypothesis_name(query: ParsedQuery) -> str:
    for marker, name in [
        ("planner:hybrid_merge", "hybrid_merge"),
        ("planner:llm_only", "llm"),
        ("planner:rule", "rule"),
    ]:
        if marker in query.specific_keywords:
            return name
    return "query"


def _query_trace_entry(
    query: ParsedQuery,
    evaluated: Dict[str, object],
    selected_name: str,
) -> Dict[str, object]:
    validated = list(evaluated["validated"])
    best_case = max(validated, key=lambda item: float(item.validation_score), default=None)
    return {
        "name": str(evaluated["name"]),
        "selected": bool(evaluated["name"] == selected_name),
        "query": query.to_dict(),
        "candidate_count": len(evaluated["candidates"]),
        "passed_count": sum(1 for case in validated if case.passed),
        "best_validation_score": float(best_case.validation_score) if best_case is not None else 0.0,
        "best_scene_name": str(best_case.candidate.scene_name) if best_case is not None else "",
        "best_sample_idx": int(best_case.candidate.sample_idx) if best_case is not None else -1,
    }


def _build_agent_trace(
    mode: str,
    evaluated_hypotheses: Sequence[Dict[str, object]],
    selected_query: ParsedQuery,
) -> Dict[str, object]:
    selected_name = _hypothesis_name(selected_query)
    return {
        "mode": mode,
        "selected_hypothesis": selected_name,
        "selection_policy": (
            "Prefer hypotheses with passing validated cases, then higher best passing score, "
            "then more passed cases, then higher best score, then richer structured signal."
        ),
        "hypotheses": [
            _query_trace_entry(item["query"], item, selected_name)
            for item in evaluated_hypotheses
        ],
    }


def _hypothesis_priority(query: ParsedQuery, validated: Sequence[ValidatedCase]) -> tuple:
    passed_scores = [float(case.validation_score) for case in validated if case.passed]
    all_scores = [float(case.validation_score) for case in validated]
    passed_count = sum(1 for case in validated if case.passed)
    signal_count = (
        2 * len(query.behaviors)
        + len(query.positions)
        + len(query.risk_terms)
        + len(query.category_groups)
    )
    return (
        bool(passed_scores),
        max(passed_scores) if passed_scores else -1.0,
        passed_count,
        max(all_scores) if all_scores else -1.0,
        signal_count,
    )


def _evaluate_query_hypothesis(
    conn: sqlite3.Connection,
    query: ParsedQuery,
    top_k: int,
    candidate_pool: int,
    rerank_mode: str,
    llm_config: Optional[LLMConfig],
    learned_reranker_checkpoint: Optional[Path] = None,
    validation_config: Optional[ValidationConfig] = None,
) -> Dict[str, object]:
    validation_config = validation_config or ValidationConfig()
    candidates = retrieve_candidates(conn, query=query, top_k=top_k, candidate_pool=max(candidate_pool, top_k))
    if rerank_mode == "learned" and candidates:
        checkpoint_path = learned_reranker_checkpoint or Path(
            os.environ.get("NUSC_SCENE_AGENT_LEARNED_RERANKER", str(DEFAULT_LEARNED_RETRIEVER_CHECKPOINT))
        )
        candidates = rerank_candidates_with_learned_model(query, candidates, checkpoint_path=checkpoint_path)
    elif rerank_mode == "multimodal" and candidates:
        candidates = rerank_candidates_with_multimodal_model(query, candidates)
    elif rerank_mode == "llm" and llm_config is not None and candidates:
        try:
            candidates = rerank_candidates_with_llm(query, candidates, llm_config)
        except Exception:  # noqa: BLE001
            pass
    validated: List[ValidatedCase] = [
        validate_candidate(
            conn,
            query,
            candidate,
            include_map_geometries=False,
            validation_config=validation_config,
        )
        for candidate in candidates
    ]
    validated.sort(key=lambda item: (item.passed, item.validation_score), reverse=True)
    return {
        "query": query,
        "candidates": candidates,
        "validated": validated,
        "priority": _hypothesis_priority(query, validated),
        "name": _hypothesis_name(query),
    }


def _select_best_hypothesis(evaluated_hypotheses: Sequence[Dict[str, object]]) -> Dict[str, object]:
    return max(
        evaluated_hypotheses,
        key=lambda item: tuple(item["priority"]),
    )


def run_query_pipeline(
    db_path: Path,
    query_text: str,
    output_root: Path,
    top_k: int = 5,
    candidate_pool: int = 30,
    benchmark_spec: Optional[BenchmarkQuerySpec] = None,
    query_mode: str = "rule",
    rerank_mode: str = "none",
    llm_config: Optional[LLMConfig] = None,
    learned_reranker_checkpoint: Optional[Path] = None,
    validation_config: Optional[ValidationConfig] = None,
) -> Dict[str, object]:
    db_path = db_path.resolve()
    if not db_path.exists():
        raise FileNotFoundError("Missing SQLite index: {0}".format(db_path))

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    hypothesis_results: List[Dict[str, object]] = []
    validation_config = validation_config or ValidationConfig()

    if query_mode == "hybrid" and llm_config is not None:
        hypotheses = resolve_hybrid_queries(query_text, llm_config)
        if benchmark_spec is not None:
            hypotheses = [apply_benchmark_spec(query, benchmark_spec) for query in hypotheses]
        hypothesis_results = [
            _evaluate_query_hypothesis(
                conn=conn,
                query=query,
                top_k=top_k,
                candidate_pool=candidate_pool,
                rerank_mode=rerank_mode,
                llm_config=llm_config,
                learned_reranker_checkpoint=learned_reranker_checkpoint,
                validation_config=validation_config,
            )
            for query in hypotheses
        ]
        best_hypothesis = _select_best_hypothesis(hypothesis_results)
        query = replace(
            best_hypothesis["query"],
            specific_keywords=list(best_hypothesis["query"].specific_keywords)
            + ["planner:hybrid_selected:{0}".format(best_hypothesis["name"])],
        )
        candidates = list(best_hypothesis["candidates"])
        validated = list(best_hypothesis["validated"])
        agent_trace = _build_agent_trace(query_mode, hypothesis_results, query)
    else:
        query = resolve_query(query_text, mode=query_mode, config=llm_config)
        if benchmark_spec is not None:
            query = apply_benchmark_spec(query, benchmark_spec)
        evaluated = _evaluate_query_hypothesis(
            conn=conn,
            query=query,
            top_k=top_k,
            candidate_pool=candidate_pool,
            rerank_mode=rerank_mode,
            llm_config=llm_config,
            learned_reranker_checkpoint=learned_reranker_checkpoint,
            validation_config=validation_config,
        )
        candidates = list(evaluated["candidates"])
        validated = list(evaluated["validated"])
        agent_trace = _build_agent_trace(query_mode, [evaluated], query)

    selected_candidates = [case.candidate for case in _select_diverse_cases(validated, top_k=top_k)]
    selected = [
        validate_candidate(
            conn,
            query,
            candidate,
            include_map_geometries=True,
            validation_config=validation_config,
        )
        for candidate in selected_candidates
    ]
    conn.close()

    query_dir = output_root.resolve() / slugify(query_text)
    rows = write_query_report(query, selected, query_dir, agent_trace=agent_trace)
    return {
        "query": query,
        "query_spec": benchmark_spec,
        "query_dir": query_dir,
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "summary_rows": rows,
        "selected_cases": selected,
        "hypothesis_results": hypothesis_results,
        "agent_trace": agent_trace,
    }

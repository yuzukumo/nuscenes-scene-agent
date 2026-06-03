from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, TypedDict

from nusc_scene_agent.benchmark_schema import BenchmarkQuerySpec, apply_benchmark_spec
from nusc_scene_agent.llm_client import LLMConfig
from nusc_scene_agent.llm_query_planner import resolve_hybrid_queries, resolve_query
from nusc_scene_agent.models import ParsedQuery, RetrievalCandidate, ValidatedCase
from nusc_scene_agent.pipeline import (
    _build_agent_trace,
    _evaluate_query_hypothesis,
    _hypothesis_name,
    _select_best_hypothesis,
    _select_diverse_cases,
)
from nusc_scene_agent.reporting import slugify, write_query_report
from nusc_scene_agent.validation import validate_candidate


class LangGraphState(TypedDict, total=False):
    db_path: Path
    query_text: str
    output_root: Path
    top_k: int
    candidate_pool: int
    benchmark_spec: Optional[BenchmarkQuerySpec]
    query_mode: str
    rerank_mode: str
    llm_config: Optional[LLMConfig]
    learned_reranker_checkpoint: Optional[Path]
    hypotheses: List[ParsedQuery]
    evaluated_hypotheses: List[Dict[str, object]]
    selected_query: ParsedQuery
    selected_candidates: List[RetrievalCandidate]
    selected_cases: List[ValidatedCase]
    query_dir: Path
    summary_rows: List[Dict[str, object]]
    candidate_count: int
    selected_count: int
    agent_trace: Dict[str, object]
    framework_trace: Dict[str, object]


def _load_langgraph_runtime():
    try:
        from langgraph.graph import END, START, StateGraph
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            'LangGraph is not installed. Install it with `pip install -e ".[agent]"` '
            "or add `langgraph` to the active environment."
        ) from exc
    return StateGraph, START, END


def _resolve_hypotheses_node(state: LangGraphState) -> LangGraphState:
    query_text = str(state["query_text"])
    query_mode = str(state.get("query_mode") or "rule")
    llm_config = state.get("llm_config")
    benchmark_spec = state.get("benchmark_spec")

    if query_mode == "hybrid" and llm_config is not None:
        hypotheses = resolve_hybrid_queries(query_text, llm_config)
    else:
        hypotheses = [resolve_query(query_text, mode=query_mode, config=llm_config)]

    if benchmark_spec is not None:
        hypotheses = [apply_benchmark_spec(query, benchmark_spec) for query in hypotheses]

    return {"hypotheses": hypotheses}


def _evaluate_hypotheses_node(state: LangGraphState) -> LangGraphState:
    db_path = Path(state["db_path"]).resolve()
    if not db_path.exists():
        raise FileNotFoundError("Missing SQLite index: {0}".format(db_path))

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        evaluated_hypotheses = [
            _evaluate_query_hypothesis(
                conn=conn,
                query=query,
                top_k=int(state.get("top_k") or 5),
                candidate_pool=int(state.get("candidate_pool") or 30),
                rerank_mode=str(state.get("rerank_mode") or "none"),
                llm_config=state.get("llm_config"),
                learned_reranker_checkpoint=state.get("learned_reranker_checkpoint"),
            )
            for query in state.get("hypotheses") or []
        ]
    finally:
        conn.close()
    return {"evaluated_hypotheses": evaluated_hypotheses}


def _best_validation_score(cases: List[ValidatedCase]) -> float:
    if not cases:
        return 0.0
    return max(float(case.validation_score) for case in cases)


def _build_framework_trace(
    mode: str,
    evaluated_hypotheses: List[Dict[str, object]],
    selected_query: ParsedQuery,
) -> Dict[str, object]:
    selected_name = _hypothesis_name(selected_query)
    return {
        "framework": "langgraph",
        "mode": mode,
        "selected_hypothesis": selected_name,
        "nodes": [
            {
                "name": "resolve_hypotheses",
                "hypothesis_count": len(evaluated_hypotheses),
                "hypothesis_names": [str(item["name"]) for item in evaluated_hypotheses],
            },
            {
                "name": "evaluate_hypotheses",
                "candidate_counts": {
                    str(item["name"]): len(item["candidates"]) for item in evaluated_hypotheses
                },
                "passed_counts": {
                    str(item["name"]): sum(1 for case in item["validated"] if case.passed)
                    for item in evaluated_hypotheses
                },
                "best_validation_scores": {
                    str(item["name"]): _best_validation_score(list(item["validated"]))
                    for item in evaluated_hypotheses
                },
            },
            {
                "name": "select_hypothesis",
                "selected_hypothesis": selected_name,
            },
        ],
    }


def _select_hypothesis_node(state: LangGraphState) -> LangGraphState:
    evaluated_hypotheses = list(state.get("evaluated_hypotheses") or [])
    if not evaluated_hypotheses:
        raise ValueError("No query hypotheses were evaluated.")

    best_hypothesis = _select_best_hypothesis(evaluated_hypotheses)
    query_mode = str(state.get("query_mode") or "rule")
    llm_config = state.get("llm_config")

    selected_query = best_hypothesis["query"]
    if query_mode == "hybrid" and llm_config is not None and len(evaluated_hypotheses) > 1:
        selected_query = replace(
            selected_query,
            specific_keywords=list(selected_query.specific_keywords)
            + ["planner:hybrid_selected:{0}".format(best_hypothesis["name"])],
        )

    selected_candidates = [
        case.candidate
        for case in _select_diverse_cases(list(best_hypothesis["validated"]), top_k=int(state.get("top_k") or 5))
    ]
    agent_trace = _build_agent_trace(query_mode, evaluated_hypotheses, selected_query)
    framework_trace = _build_framework_trace(query_mode, evaluated_hypotheses, selected_query)

    return {
        "selected_query": selected_query,
        "selected_candidates": selected_candidates,
        "candidate_count": len(best_hypothesis["candidates"]),
        "agent_trace": agent_trace,
        "framework_trace": framework_trace,
    }


def _report_results_node(state: LangGraphState) -> LangGraphState:
    db_path = Path(state["db_path"]).resolve()
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        selected_cases = [
            validate_candidate(conn, state["selected_query"], candidate, include_map_geometries=True)
            for candidate in state.get("selected_candidates") or []
        ]
    finally:
        conn.close()

    query_dir = Path(state["output_root"]).resolve() / slugify(str(state["query_text"]))
    summary_rows = write_query_report(
        state["selected_query"],
        selected_cases,
        query_dir,
        agent_trace=state.get("agent_trace"),
    )

    framework_trace = dict(state.get("framework_trace") or {})
    framework_trace["nodes"] = list(framework_trace.get("nodes") or []) + [
        {
            "name": "report_results",
            "query_dir": str(query_dir),
            "selected_count": len(selected_cases),
        }
    ]
    (query_dir / "langgraph_trace.json").write_text(
        json.dumps(framework_trace, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return {
        "selected_cases": selected_cases,
        "selected_count": len(selected_cases),
        "query_dir": query_dir,
        "summary_rows": summary_rows,
        "framework_trace": framework_trace,
    }


def _build_langgraph_app():
    StateGraph, START, END = _load_langgraph_runtime()
    graph = StateGraph(LangGraphState)
    graph.add_node("resolve_hypotheses", _resolve_hypotheses_node)
    graph.add_node("evaluate_hypotheses", _evaluate_hypotheses_node)
    graph.add_node("select_hypothesis", _select_hypothesis_node)
    graph.add_node("report_results", _report_results_node)
    graph.add_edge(START, "resolve_hypotheses")
    graph.add_edge("resolve_hypotheses", "evaluate_hypotheses")
    graph.add_edge("evaluate_hypotheses", "select_hypothesis")
    graph.add_edge("select_hypothesis", "report_results")
    graph.add_edge("report_results", END)
    return graph.compile()


def run_langgraph_query_pipeline(
    db_path: Path,
    query_text: str,
    output_root: Path,
    top_k: int = 5,
    candidate_pool: int = 30,
    benchmark_spec: Optional[BenchmarkQuerySpec] = None,
    query_mode: str = "hybrid",
    rerank_mode: str = "none",
    llm_config: Optional[LLMConfig] = None,
    learned_reranker_checkpoint: Optional[Path] = None,
) -> Dict[str, object]:
    app = _build_langgraph_app()
    final_state = app.invoke(
        {
            "db_path": Path(db_path).resolve(),
            "query_text": query_text,
            "output_root": Path(output_root).resolve(),
            "top_k": int(top_k),
            "candidate_pool": int(candidate_pool),
            "benchmark_spec": benchmark_spec,
            "query_mode": query_mode,
            "rerank_mode": rerank_mode,
            "llm_config": llm_config,
            "learned_reranker_checkpoint": learned_reranker_checkpoint,
        }
    )
    return {
        "query": final_state["selected_query"],
        "query_spec": benchmark_spec,
        "query_dir": final_state["query_dir"],
        "candidate_count": int(final_state["candidate_count"]),
        "selected_count": int(final_state["selected_count"]),
        "summary_rows": list(final_state.get("summary_rows") or []),
        "selected_cases": list(final_state.get("selected_cases") or []),
        "hypothesis_results": list(final_state.get("evaluated_hypotheses") or []),
        "agent_trace": dict(final_state.get("agent_trace") or {}),
        "framework_trace": dict(final_state.get("framework_trace") or {}),
    }

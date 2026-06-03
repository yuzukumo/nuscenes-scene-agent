from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, TypedDict

from nusc_scene_agent.benchmark_registry import build_default_benchmark_registry
from nusc_scene_agent.llm_client import LLMConfig, llm_json


DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "gemma4:latest"
DEFAULT_RESEARCH_AGENT_OUTPUT = Path("outputs/research_agent_review_v1")
DEFAULT_RESEARCH_AGENT_ARTIFACTS = [
    Path("outputs/trainval_scenario_mining_v1_hybrid/benchmark_metrics_summary.md"),
    Path("outputs/trainval_hybrid_ablation_v1/benchmark_profile_comparison_summary.md"),
    Path("outputs/external_official_tracking_comparison/perception_comparison_summary.md"),
    Path("outputs/trainval_bev_occupancy_proxy_study_v1/bev_occupancy_comparison_summary.md"),
    Path("outputs/contextvae_world_model_study_v1/comparison/world_model_comparison_summary.md"),
    Path("outputs/nuplan_replay_sweep_v1/nuplan_replay_sweep_summary.md"),
]


class ResearchAgentState(TypedDict, total=False):
    output_dir: Path
    artifact_paths: List[Path]
    focus: str
    llm_config: LLMConfig
    context: Dict[str, Any]
    analysis: Dict[str, Any]
    report_path: Path


def _load_langgraph_runtime():
    try:
        from langgraph.graph import END, START, StateGraph
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            'LangGraph is not installed. Install it with `pip install -e ".[agent]"` '
            "or add `langgraph` to the active environment."
        ) from exc
    return StateGraph, START, END


def run_research_agent(
    output_dir: Path = DEFAULT_RESEARCH_AGENT_OUTPUT,
    llm_config: Optional[LLMConfig] = None,
    artifact_paths: Optional[Sequence[Path]] = None,
    focus: str = "Assess benchmark gaps and propose evidence-based next actions.",
) -> Dict[str, Any]:
    if llm_config is None:
        raise ValueError("run_research_agent requires an LLMConfig. Configure a local Ollama endpoint.")

    app = _build_research_agent_app()
    final_state = app.invoke(
        {
            "output_dir": Path(output_dir).resolve(),
            "artifact_paths": [Path(path) for path in (artifact_paths or DEFAULT_RESEARCH_AGENT_ARTIFACTS)],
            "focus": focus,
            "llm_config": llm_config,
        }
    )
    return {
        "schema": "research_agent_result_v1",
        "output_dir": str(Path(final_state["output_dir"]).resolve()),
        "report_path": str(Path(final_state["report_path"]).resolve()),
        "analysis": dict(final_state["analysis"]),
        "context_path": str(Path(final_state["output_dir"]).resolve() / "research_agent_context.json"),
    }


def _build_research_agent_app():
    StateGraph, START, END = _load_langgraph_runtime()
    graph = StateGraph(ResearchAgentState)
    graph.add_node("collect_artifacts", _collect_artifacts_node)
    graph.add_node("analyze_with_llm", _analyze_with_llm_node)
    graph.add_node("write_report", _write_report_node)
    graph.add_edge(START, "collect_artifacts")
    graph.add_edge("collect_artifacts", "analyze_with_llm")
    graph.add_edge("analyze_with_llm", "write_report")
    graph.add_edge("write_report", END)
    return graph.compile()


def _collect_artifacts_node(state: ResearchAgentState) -> ResearchAgentState:
    output_dir = Path(state["output_dir"]).resolve()
    artifacts = [_artifact_digest(Path(path)) for path in state.get("artifact_paths") or []]
    context = {
        "schema": "research_agent_context_v1",
        "focus": str(state.get("focus") or ""),
        "benchmark_registry": build_default_benchmark_registry(),
        "artifacts": artifacts,
        "agent_nodes": [
            "collect_artifacts",
            "analyze_with_llm",
            "write_report",
        ],
        "constraints": [
            "Use only provided artifacts.",
            "Do not invent benchmark numbers.",
            "Separate completed capabilities from remaining gaps.",
            "Prefer concrete next actions with expected artifacts.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "research_agent_context.json").write_text(
        json.dumps(context, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return {"context": context}


def _analyze_with_llm_node(state: ResearchAgentState) -> ResearchAgentState:
    context = dict(state["context"])
    llm_config = state["llm_config"]
    system_prompt = (
        "You are a rigorous autonomous-driving benchmark research agent. "
        "Analyze the supplied local benchmark artifacts and return JSON only. "
        "Do not use promotional language. Do not invent metrics or cite missing artifacts. "
        "Required JSON schema: {"
        "'project_positioning': string, "
        "'completed_capabilities': [{'capability': string, 'evidence': string}], "
        "'industry_gaps': [{'gap': string, 'evidence': string, 'priority': 'high'|'medium'|'low'}], "
        "'next_actions': [{'action': string, 'rationale': string, 'expected_artifact': string, 'risk': string}], "
        "'agent_workflow': [{'node': string, 'purpose': string}], "
        "'benchmark_update_queries': [string], "
        "'claims_to_avoid': [string]"
        "}."
    )
    user_prompt = json.dumps(context, ensure_ascii=False)
    analysis = llm_json(
        llm_config,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.0,
    )
    normalized = _normalize_analysis(analysis)
    return {"analysis": normalized}


def _write_report_node(state: ResearchAgentState) -> ResearchAgentState:
    output_dir = Path(state["output_dir"]).resolve()
    analysis = dict(state["analysis"])
    json_path = output_dir / "research_agent_report.json"
    md_path = output_dir / "research_agent_report.md"
    json_path.write_text(json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(_render_markdown_report(analysis), encoding="utf-8")
    return {"report_path": md_path}


def _artifact_digest(path: Path, max_chars: int = 12000) -> Dict[str, Any]:
    resolved = path.resolve()
    digest: Dict[str, Any] = {
        "path": str(path),
        "exists": resolved.exists(),
    }
    if not resolved.exists():
        digest["summary"] = "missing"
        return digest

    digest["size_bytes"] = resolved.stat().st_size
    suffix = resolved.suffix.lower()
    try:
        if suffix == ".json":
            payload = json.loads(resolved.read_text(encoding="utf-8"))
            digest["summary"] = _summarize_json(payload)
            digest["content_excerpt"] = json.dumps(payload, indent=2, ensure_ascii=False)[:max_chars]
        elif suffix == ".csv":
            rows = _read_csv_rows(resolved, limit=12)
            digest["summary"] = {"row_count_sampled": len(rows), "columns": list(rows[0].keys()) if rows else []}
            digest["rows"] = rows
        else:
            text = resolved.read_text(encoding="utf-8", errors="replace")
            digest["summary"] = {"line_count": len(text.splitlines())}
            digest["content_excerpt"] = text[:max_chars]
    except Exception as exc:  # noqa: BLE001
        digest["summary"] = "read_error"
        digest["error"] = str(exc)
    return digest


def _summarize_json(payload: Any) -> Dict[str, Any]:
    if isinstance(payload, Mapping):
        summary: Dict[str, Any] = {"type": "object", "keys": sorted(str(key) for key in payload.keys())[:30]}
        for key in ("schema", "case_count", "profile_count", "experiment_type"):
            if key in payload:
                summary[key] = payload[key]
        if "overview" in payload:
            summary["overview"] = payload["overview"]
        if "result" in payload and isinstance(payload["result"], Mapping):
            summary["result_keys"] = sorted(str(key) for key in payload["result"].keys())[:30]
        return summary
    if isinstance(payload, list):
        return {"type": "list", "length": len(payload)}
    return {"type": type(payload).__name__}


def _read_csv_rows(path: Path, limit: int) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows: List[Dict[str, str]] = []
        for row in reader:
            rows.append(dict(row))
            if len(rows) >= limit:
                break
    return rows


def _normalize_analysis(payload: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "schema": "research_agent_report_v1",
        "project_positioning": str(payload.get("project_positioning") or ""),
        "completed_capabilities": _normalize_list_of_dicts(
            payload.get("completed_capabilities"),
            required_keys=["capability", "evidence"],
        ),
        "industry_gaps": _normalize_list_of_dicts(
            payload.get("industry_gaps"),
            required_keys=["gap", "evidence", "priority"],
        ),
        "next_actions": _normalize_list_of_dicts(
            payload.get("next_actions"),
            required_keys=["action", "rationale", "expected_artifact", "risk"],
        ),
        "agent_workflow": _normalize_list_of_dicts(
            payload.get("agent_workflow"),
            required_keys=["node", "purpose"],
        ),
        "benchmark_update_queries": [
            str(item).strip() for item in list(payload.get("benchmark_update_queries") or []) if str(item).strip()
        ],
        "claims_to_avoid": [
            str(item).strip() for item in list(payload.get("claims_to_avoid") or []) if str(item).strip()
        ],
    }


def _normalize_list_of_dicts(value: Any, required_keys: Sequence[str]) -> List[Dict[str, str]]:
    rows = []
    if not isinstance(value, list):
        return rows
    for item in value:
        if not isinstance(item, Mapping):
            continue
        row = {key: str(item.get(key) or "").strip() for key in required_keys}
        if any(row.values()):
            rows.append(row)
    return rows


def _render_markdown_report(analysis: Mapping[str, Any]) -> str:
    lines = [
        "# Research Agent Report",
        "",
        "## Project Positioning",
        "",
        str(analysis.get("project_positioning") or "").strip(),
        "",
        "## Completed Capabilities",
        "",
    ]
    lines.extend(_render_dict_rows(analysis.get("completed_capabilities"), ["capability", "evidence"]))
    lines.extend(["", "## Industry Gaps", ""])
    lines.extend(_render_dict_rows(analysis.get("industry_gaps"), ["gap", "priority", "evidence"]))
    lines.extend(["", "## Next Actions", ""])
    lines.extend(_render_dict_rows(analysis.get("next_actions"), ["action", "rationale", "expected_artifact", "risk"]))
    lines.extend(["", "## Agent Workflow", ""])
    lines.extend(_render_dict_rows(analysis.get("agent_workflow"), ["node", "purpose"]))
    lines.extend(["", "## Benchmark Update Queries", ""])
    queries = list(analysis.get("benchmark_update_queries") or [])
    lines.extend(["- {0}".format(str(item)) for item in queries] or ["- None"])
    lines.extend(["", "## Claims To Avoid", ""])
    claims = list(analysis.get("claims_to_avoid") or [])
    lines.extend(["- {0}".format(str(item)) for item in claims] or ["- None"])
    lines.append("")
    return "\n".join(lines)


def _render_dict_rows(value: Any, keys: Sequence[str]) -> List[str]:
    rows = []
    for item in list(value or []):
        if not isinstance(item, Mapping):
            continue
        parts = []
        for key in keys:
            text = str(item.get(key) or "").strip()
            if text:
                parts.append("{0}: {1}".format(key.replace("_", " ").title(), text))
        if parts:
            rows.append("- " + " | ".join(parts))
    return rows or ["- None"]

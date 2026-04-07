from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

from nusc_scene_agent.benchmark_comparison import (
    build_benchmark_comparison,
    default_benchmark_profiles,
    write_benchmark_comparison,
)
from nusc_scene_agent.ablation_study import default_ablation_profiles, write_ablation_manifest
from nusc_scene_agent.case_library_enrichment import enrich_case_library
from nusc_scene_agent.counterfactual_benchmark import generate_counterfactual_benchmark_from_case_library
from nusc_scene_agent.benchmark_exports import write_benchmark_exports
from nusc_scene_agent.benchmark_metrics import build_benchmark_metrics, write_benchmark_metrics
from nusc_scene_agent.case_library import build_case_library, write_case_library
from nusc_scene_agent.benchmark_schema import load_benchmark_config
from nusc_scene_agent.data_utils import DEFAULT_DATAROOT, PREPARE_PROFILES, discover_archive_inventory, prepare_data
from nusc_scene_agent.gallery import build_benchmark_gallery, build_comparison_browser
from nusc_scene_agent.indexing import build_index
from nusc_scene_agent.langgraph_agent import run_langgraph_query_pipeline
from nusc_scene_agent.llm_client import DEFAULT_TIMEOUT_S, LLMConfig
from nusc_scene_agent.pipeline import run_query_pipeline
from nusc_scene_agent.scenario_mining_benchmark import generate_scenario_mining_benchmark_from_case_library
from nusc_scene_agent.validation import ValidationConfig


DEFAULT_DB = Path("artifacts/index/v1.0-mini.sqlite")
DEFAULT_BENCHMARK = Path("benchmarks/mvp_queries.yaml")
DEFAULT_TRAINVAL_BENCHMARK = Path("benchmarks/trainval_suite_v1.yaml")
DEFAULT_COMPARE_BENCHMARK = Path("benchmarks/trainval_language_stress_v1.yaml")


def _add_llm_connection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--llm-base-url", default="")
    parser.add_argument("--llm-api-key", default="")
    parser.add_argument("--llm-model", default="")
    parser.add_argument("--llm-timeout", type=float, default=DEFAULT_TIMEOUT_S)


def _add_llm_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--query-mode", choices=["rule", "llm", "hybrid"], default="rule")
    parser.add_argument("--rerank-mode", choices=["none", "llm"], default="none")
    _add_llm_connection_args(parser)


def _resolve_llm_config(args: argparse.Namespace) -> Optional[LLMConfig]:
    env_config = LLMConfig.from_env()
    base_url = str(getattr(args, "llm_base_url", "") or (env_config.base_url if env_config else "")).strip()
    api_key = str(getattr(args, "llm_api_key", "") or (env_config.api_key if env_config else "")).strip()
    model = str(getattr(args, "llm_model", "") or (env_config.model if env_config else "")).strip()
    timeout_s = float(getattr(args, "llm_timeout", DEFAULT_TIMEOUT_S))
    if not (base_url and api_key and model):
        return None
    return LLMConfig(base_url=base_url, api_key=api_key, model=model, timeout_s=timeout_s)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="nuScenes scene mining and benchmark generation toolkit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect-archives", help="Inspect local nuScenes archives and readiness.")
    inspect_parser.add_argument("--workspace", default=".", help="Workspace root that contains the archives.")

    prepare_parser = subparsers.add_parser("prepare-data", help="Unpack nuScenes archives into a dataroot.")
    prepare_parser.add_argument("--archive", action="append", default=[], help="Archive path, repeatable.")
    prepare_parser.add_argument("--workspace", default=".", help="Workspace root that contains the archives.")
    prepare_parser.add_argument("--dataroot", default=str(DEFAULT_DATAROOT), help="Extraction target.")
    prepare_parser.add_argument("--profile", choices=PREPARE_PROFILES, default="mini")

    build_parser = subparsers.add_parser("build-index", help="Build a SQLite index from nuScenes annotations.")
    build_parser.add_argument("--version", default="v1.0-mini")
    build_parser.add_argument("--dataroot", default=str(DEFAULT_DATAROOT))
    build_parser.add_argument("--db", default=str(DEFAULT_DB))
    build_parser.add_argument("--scene-limit", type=int, default=0)

    query_parser = subparsers.add_parser("query", help="Run one natural-language risk query.")
    query_parser.add_argument("text", help="Natural-language risk description.")
    query_parser.add_argument("--db", default=str(DEFAULT_DB))
    query_parser.add_argument("--output", default="outputs/query")
    query_parser.add_argument("--top-k", type=int, default=5)
    query_parser.add_argument("--candidate-pool", type=int, default=12)
    _add_llm_args(query_parser)

    langgraph_query_parser = subparsers.add_parser(
        "langgraph-query",
        help="Run one natural-language risk query through a LangGraph orchestration layer.",
    )
    langgraph_query_parser.add_argument("text", help="Natural-language risk description.")
    langgraph_query_parser.add_argument("--db", default=str(DEFAULT_DB))
    langgraph_query_parser.add_argument("--output", default="outputs/langgraph_query")
    langgraph_query_parser.add_argument("--top-k", type=int, default=5)
    langgraph_query_parser.add_argument("--candidate-pool", type=int, default=12)
    _add_llm_args(langgraph_query_parser)

    benchmark_parser = subparsers.add_parser("benchmark", help="Run a YAML benchmark query suite.")
    benchmark_parser.add_argument("--config", default=str(DEFAULT_BENCHMARK))
    benchmark_parser.add_argument("--db", default=str(DEFAULT_DB))
    benchmark_parser.add_argument("--output", default="outputs/benchmark")
    benchmark_parser.add_argument("--candidate-pool", type=int, default=12)
    _add_llm_args(benchmark_parser)

    generate_counterfactual_parser = subparsers.add_parser(
        "generate-counterfactual-benchmark",
        help="Generate a contrastive counterfactual benchmark from a case library.",
    )
    generate_counterfactual_parser.add_argument(
        "--case-library",
        default="outputs/trainval_suite_llm_hybrid_en_v1/case_library.json",
        help="Path to case_library.json used as benchmark anchors.",
    )
    generate_counterfactual_parser.add_argument(
        "--output",
        default="benchmarks/trainval_counterfactual_reference_v1.yaml",
        help="Output YAML path for the generated benchmark.",
    )
    generate_counterfactual_parser.add_argument("--max-cases", type=int, default=6)

    generate_scenario_parser = subparsers.add_parser(
        "generate-scenario-mining-benchmark",
        help="Generate a planning-centric scenario mining benchmark from a case library.",
    )
    generate_scenario_parser.add_argument(
        "--case-library",
        default="outputs/trainval_suite_llm_hybrid_en_v1/case_library.json",
        help="Path to case_library.json used as scenario anchors.",
    )
    generate_scenario_parser.add_argument(
        "--output",
        default="benchmarks/trainval_scenario_mining_v1.yaml",
        help="Output YAML path for the generated benchmark.",
    )
    generate_scenario_parser.add_argument("--max-cases", type=int, default=8)

    enrich_case_library_parser = subparsers.add_parser(
        "enrich-case-library",
        help="Re-validate a case library to populate actor grounding and event localization fields.",
    )
    enrich_case_library_parser.add_argument(
        "--case-library",
        default="outputs/trainval_suite_llm_hybrid_en_v1/case_library.json",
        help="Input case library JSON path.",
    )
    enrich_case_library_parser.add_argument(
        "--db",
        default="artifacts/index/v1.0-trainval.sqlite",
        help="SQLite index used for re-validation.",
    )
    enrich_case_library_parser.add_argument(
        "--output",
        default="outputs/trainval_suite_llm_hybrid_en_v1/case_library_enriched.json",
        help="Output JSON path for the enriched case library.",
    )

    compare_parser = subparsers.add_parser(
        "benchmark-compare",
        help="Run the benchmark across rule, llm, and hybrid profiles and export a comparison summary.",
    )
    compare_parser.add_argument("--config", default=str(DEFAULT_COMPARE_BENCHMARK))
    compare_parser.add_argument("--db", default="artifacts/index/v1.0-trainval.sqlite")
    compare_parser.add_argument("--output", default="outputs/trainval_profile_comparison_v1")
    compare_parser.add_argument("--candidate-pool", type=int, default=12)
    _add_llm_connection_args(compare_parser)

    ablate_parser = subparsers.add_parser(
        "benchmark-ablate",
        help="Run an ablation study over the benchmark and export comparison artifacts.",
    )
    ablate_parser.add_argument("--config", default="benchmarks/trainval_scenario_mining_v1.yaml")
    ablate_parser.add_argument("--db", default="artifacts/index/v1.0-trainval.sqlite")
    ablate_parser.add_argument("--output", default="outputs/trainval_hybrid_ablation_v1")
    ablate_parser.add_argument("--candidate-pool", type=int, default=12)
    ablate_parser.add_argument("--base-query-mode", choices=["rule", "llm", "hybrid"], default="hybrid")
    ablate_parser.add_argument("--base-rerank-mode", choices=["none", "llm"], default="llm")
    ablate_parser.add_argument("--reuse-existing", action="store_true")
    _add_llm_connection_args(ablate_parser)

    gallery_parser = subparsers.add_parser(
        "build-gallery",
        help="Build a static query gallery for one benchmark output or a side-by-side browser for a comparison output.",
    )
    gallery_group = gallery_parser.add_mutually_exclusive_group(required=True)
    gallery_group.add_argument(
        "--benchmark-output",
        default="",
        help="Benchmark output directory that contains benchmark_summary.json.",
    )
    gallery_group.add_argument(
        "--comparison-output",
        default="",
        help="Comparison output directory that contains benchmark_profile_comparison.json.",
    )
    gallery_parser.add_argument("--title", default="")

    demo_parser = subparsers.add_parser("demo", help="Run the end-to-end MVP demo on v1.0-mini.")
    demo_parser.add_argument("--workspace", default=".")
    demo_parser.add_argument("--dataroot", default=str(DEFAULT_DATAROOT))
    demo_parser.add_argument("--db", default=str(DEFAULT_DB))
    demo_parser.add_argument("--config", default=str(DEFAULT_BENCHMARK))
    demo_parser.add_argument("--output", default="outputs/demo")
    demo_parser.add_argument("--candidate-pool", type=int, default=12)
    _add_llm_args(demo_parser)
    return parser


def _run_benchmark(
    config_path: Path,
    db_path: Path,
    output_dir: Path,
    candidate_pool: int,
    query_mode: str = "rule",
    rerank_mode: str = "none",
    llm_config: Optional[LLMConfig] = None,
    validation_config: Optional[ValidationConfig] = None,
) -> List[dict]:
    queries = load_benchmark_config(config_path)
    summaries: List[dict] = []
    benchmark_results: List[dict] = []
    for spec in queries:
        result = run_query_pipeline(
            db_path=db_path,
            query_text=spec.natural_language,
            output_root=output_dir,
            top_k=spec.top_k,
            candidate_pool=spec.candidate_pool or candidate_pool,
            benchmark_spec=spec,
            query_mode=query_mode,
            rerank_mode=rerank_mode,
            llm_config=llm_config,
            validation_config=validation_config,
        )
        result["id"] = spec.id
        benchmark_results.append(result)
        summaries.append(
            {
                "id": spec.id,
                "description": spec.description,
                "query_dir": str(result["query_dir"]),
                "candidate_count": result["candidate_count"],
                "selected_count": result["selected_count"],
                "tags": spec.tags,
                "behaviors": spec.behaviors,
                "actors": spec.actors,
            }
        )

    (output_dir / "benchmark_summary.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    case_library_entries = build_case_library(benchmark_results)
    write_case_library(case_library_entries, output_dir)
    write_benchmark_metrics(build_benchmark_metrics(benchmark_results, case_library_entries), output_dir)
    write_benchmark_exports(benchmark_results, case_library_entries, output_dir)
    return summaries


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "inspect-archives":
        inventory = discover_archive_inventory(Path(args.workspace))
        print(json.dumps(inventory.to_dict(), indent=2, ensure_ascii=False))
        return

    if args.command == "prepare-data":
        extracted = prepare_data(
            workspace=Path(args.workspace),
            dataroot=Path(args.dataroot),
            archives=args.archive or None,
            profile=args.profile,
        )
        print("Prepared dataroot:", Path(args.dataroot).resolve())
        for item in extracted:
            print("  extracted:", item)
        return

    if args.command == "build-index":
        stats = build_index(
            version=args.version,
            dataroot=Path(args.dataroot),
            db_path=Path(args.db),
            scene_limit=args.scene_limit,
            verbose=True,
        )
        print("Built index:", Path(args.db).resolve())
        print(json.dumps(stats, indent=2))
        return

    if args.command == "query":
        llm_config = _resolve_llm_config(args)
        result = run_query_pipeline(
            db_path=Path(args.db),
            query_text=args.text,
            output_root=Path(args.output),
            top_k=args.top_k,
            candidate_pool=args.candidate_pool,
            query_mode=args.query_mode,
            rerank_mode=args.rerank_mode,
            llm_config=llm_config,
        )
        print("Query output:", result["query_dir"])
        print("Candidates:", result["candidate_count"], "Selected:", result["selected_count"])
        return

    if args.command == "langgraph-query":
        llm_config = _resolve_llm_config(args)
        result = run_langgraph_query_pipeline(
            db_path=Path(args.db),
            query_text=args.text,
            output_root=Path(args.output),
            top_k=args.top_k,
            candidate_pool=args.candidate_pool,
            query_mode=args.query_mode,
            rerank_mode=args.rerank_mode,
            llm_config=llm_config,
        )
        print("LangGraph query output:", result["query_dir"])
        print("Candidates:", result["candidate_count"], "Selected:", result["selected_count"])
        return

    if args.command == "benchmark":
        output_dir = Path(args.output).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        llm_config = _resolve_llm_config(args)
        summaries = _run_benchmark(
            config_path=Path(args.config),
            db_path=Path(args.db),
            output_dir=output_dir,
            candidate_pool=args.candidate_pool,
            query_mode=args.query_mode,
            rerank_mode=args.rerank_mode,
            llm_config=llm_config,
        )
        print("Benchmark output:", output_dir)
        print("Queries:", len(summaries))
        return

    if args.command == "generate-counterfactual-benchmark":
        metadata = generate_counterfactual_benchmark_from_case_library(
            case_library_path=Path(args.case_library),
            output_path=Path(args.output),
            max_cases=args.max_cases,
        )
        print("Counterfactual benchmark:", Path(args.output).resolve())
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        return

    if args.command == "generate-scenario-mining-benchmark":
        metadata = generate_scenario_mining_benchmark_from_case_library(
            case_library_path=Path(args.case_library),
            output_path=Path(args.output),
            max_cases=args.max_cases,
        )
        print("Scenario mining benchmark:", Path(args.output).resolve())
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        return

    if args.command == "enrich-case-library":
        metadata = enrich_case_library(
            case_library_path=Path(args.case_library),
            db_path=Path(args.db),
            output_path=Path(args.output),
        )
        print("Enriched case library:", Path(args.output).resolve())
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        return

    if args.command == "benchmark-compare":
        output_dir = Path(args.output).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        llm_config = _resolve_llm_config(args)
        profile_runs = default_benchmark_profiles(include_llm=llm_config is not None)

        for profile in profile_runs:
            profile_output = output_dir / profile["name"]
            profile_output.mkdir(parents=True, exist_ok=True)
            _run_benchmark(
                config_path=Path(args.config),
                db_path=Path(args.db),
                output_dir=profile_output,
                candidate_pool=args.candidate_pool,
                query_mode=str(profile["query_mode"]),
                rerank_mode=str(profile["rerank_mode"]),
                llm_config=llm_config,
            )
            profile["output_dir"] = str(profile_output)

        comparison = build_benchmark_comparison(profile_runs)
        write_benchmark_comparison(comparison, output_dir)
        print("Benchmark comparison output:", output_dir)
        print("Profiles:", ", ".join(str(profile["name"]) for profile in profile_runs))
        return

    if args.command == "benchmark-ablate":
        output_dir = Path(args.output).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        llm_config = _resolve_llm_config(args)
        ablation_runs = default_ablation_profiles(
            base_query_mode=args.base_query_mode,
            base_rerank_mode=args.base_rerank_mode,
        )
        requires_llm = any(
            str(profile["query_mode"]) != "rule" or str(profile["rerank_mode"]) == "llm"
            for profile in ablation_runs
        )
        if requires_llm and llm_config is None:
            raise ValueError("The selected ablation profiles require LLM configuration.")

        for profile in ablation_runs:
            profile_output = output_dir / str(profile["name"])
            profile_output.mkdir(parents=True, exist_ok=True)
            benchmark_metrics_path = profile_output / "benchmark_metrics.json"
            if not (args.reuse_existing and benchmark_metrics_path.exists()):
                _run_benchmark(
                    config_path=Path(args.config),
                    db_path=Path(args.db),
                    output_dir=profile_output,
                    candidate_pool=args.candidate_pool,
                    query_mode=str(profile["query_mode"]),
                    rerank_mode=str(profile["rerank_mode"]),
                    llm_config=llm_config,
                    validation_config=profile.get("validation_config"),
                )
            profile["output_dir"] = str(profile_output)

        comparison = build_benchmark_comparison(ablation_runs)
        write_benchmark_comparison(comparison, output_dir)
        write_ablation_manifest(ablation_runs, output_dir)
        build_comparison_browser(
            comparison_output_dir=output_dir,
            title="nuScenes Ablation Browser",
        )
        print("Benchmark ablation output:", output_dir)
        print("Profiles:", ", ".join(str(profile["name"]) for profile in ablation_runs))
        return

    if args.command == "build-gallery":
        if args.benchmark_output:
            metadata = build_benchmark_gallery(
                benchmark_output_dir=Path(args.benchmark_output),
                title=args.title or "nuScenes Benchmark Query Gallery",
            )
        else:
            metadata = build_comparison_browser(
                comparison_output_dir=Path(args.comparison_output),
                title=args.title or "nuScenes Benchmark Comparison Browser",
            )
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        return

    if args.command == "demo":
        workspace = Path(args.workspace).resolve()
        dataroot = Path(args.dataroot).resolve()
        db_path = Path(args.db).resolve()
        output_dir = Path(args.output).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        llm_config = _resolve_llm_config(args)

        if not (dataroot / "v1.0-mini").exists():
            extracted = prepare_data(workspace=workspace, dataroot=dataroot, archives=None)
            print("Prepared data from", len(extracted), "archives")

        if not db_path.exists():
            stats = build_index(version="v1.0-mini", dataroot=dataroot, db_path=db_path, verbose=True)
            print("Built index:", stats)

        summaries = _run_benchmark(
            config_path=Path(args.config),
            db_path=db_path,
            output_dir=output_dir,
            candidate_pool=args.candidate_pool,
            query_mode=args.query_mode,
            rerank_mode=args.rerank_mode,
            llm_config=llm_config,
        )
        print("Demo complete:", output_dir)
        print(json.dumps(summaries, indent=2, ensure_ascii=False))
        return


if __name__ == "__main__":
    main()

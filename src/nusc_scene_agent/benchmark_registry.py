from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping


BENCHMARK_REGISTRY_SCHEMA = "benchmark_registry_v1"


@dataclass
class BenchmarkLayerSpec:
    layer_id: str
    dataset: str
    task: str
    description: str
    default_command: str
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    metrics: List[str] = field(default_factory=list)
    case_schema: str = "unified_risk_case_v1"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_default_benchmark_registry() -> Dict[str, Any]:
    layers = [
        BenchmarkLayerSpec(
            layer_id="scenario_mining",
            dataset="nuScenes",
            task="natural-language risk retrieval",
            description="Retrieve and validate risky scene anchors from a SQLite scene index.",
            default_command="python -m nusc_scene_agent run-experiment-config --config configs/risk_benchmark_suite.yaml",
            inputs=["nuScenes SQLite index", "YAML query benchmark"],
            outputs=["scenario-mining YAML", "case_library.json", "benchmark metrics", "derived benchmark slices"],
            metrics=["Pass@1", "Pass@K", "reference precision", "event localization"],
        ),
        BenchmarkLayerSpec(
            layer_id="perception_slices",
            dataset="nuScenes",
            task="scenario-conditioned perception evaluation",
            description="Evaluate external tracking or detection outputs on mined risk slices.",
            default_command="python -m nusc_scene_agent run-experiment-config --config configs/risk_benchmark_suite.yaml",
            inputs=["scenario-mining benchmark", "prediction tracks"],
            outputs=["perception metrics", "leaderboard CSV"],
            metrics=["coverage", "track recall", "localization error", "risk actor recall"],
        ),
        BenchmarkLayerSpec(
            layer_id="bev_occupancy_slices",
            dataset="nuScenes",
            task="risk-conditioned sparse BEV occupancy evaluation",
            description="Evaluate sparse BEV occupancy coverage on mined risk slices using primary and context actor cells.",
            default_command="python -m nusc_scene_agent run-experiment-config --config configs/risk_benchmark_suite.yaml",
            inputs=["perception-slice benchmark", "nuScenes SQLite index", "BEV occupancy predictions"],
            outputs=["BEV occupancy benchmark", "occupancy metrics", "leaderboard CSV"],
            metrics=["occupancy IoU", "primary actor recall", "context recall", "risk fidelity"],
        ),
        BenchmarkLayerSpec(
            layer_id="world_model_slices",
            dataset="nuScenes",
            task="scenario-conditioned future prediction",
            description="Evaluate future trajectory and occupancy forecasts on compact risk slices.",
            default_command="python -m nusc_scene_agent run-experiment-config --config configs/risk_benchmark_suite.yaml",
            inputs=["world-model benchmark", "future rollout predictions"],
            outputs=["world-model metrics", "case studies", "JSONL/MCAP replay export"],
            metrics=["ADE", "FDE", "MinADE@K", "occupancy IoU", "risk fidelity"],
        ),
        BenchmarkLayerSpec(
            layer_id="nuplan_replay_regression",
            dataset="nuPlan",
            task="simulation-style replay regression",
            description="Generate compact replay cases from nuPlan SQLite scenario tags and evaluate ego rollouts.",
            default_command="python -m nusc_scene_agent run-experiment-config --config configs/nuplan_replay_sweep_medium.yaml",
            inputs=["nuPlan SQLite split", "ego rollout predictions"],
            outputs=["replay benchmark", "rollout metrics", "comparison leaderboard", "sweep summary", "case-study figures"],
            metrics=["ego ADE", "ego FDE", "min-distance error", "min-TTC error", "risk fidelity"],
        ),
        BenchmarkLayerSpec(
            layer_id="nuplan_closed_loop_replay",
            dataset="nuPlan",
            task="closed-loop replay simulation",
            description="Roll ego state forward with planner profiles while replaying logged actors and traffic-light context.",
            default_command="python -m nusc_scene_agent run-experiment-config --config configs/nuplan_closed_loop_sweep_medium.yaml",
            inputs=["nuPlan SQLite splits", "closed-loop planner profiles"],
            outputs=["closed-loop benchmarks", "closed-loop metrics", "cross-split leaderboard CSV", "case-study figures"],
            metrics=["ego ADE", "progress ratio", "collision proxy mismatch", "comfort violation", "closed-loop score"],
        ),
    ]
    return {
        "schema": BENCHMARK_REGISTRY_SCHEMA,
        "layers": {layer.layer_id: layer.to_dict() for layer in layers},
    }


def write_benchmark_registry(output_path: Path, registry: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    payload = dict(registry or build_default_benchmark_registry())
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload

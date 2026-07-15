from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping


BENCHMARK_CATALOG_SCHEMA = "benchmark_catalog_v1"


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


def build_default_benchmark_catalog() -> Dict[str, Any]:
    layers = [
        BenchmarkLayerSpec(
            layer_id="scenario_mining",
            dataset="nuScenes",
            task="natural-language risk retrieval",
            description="Retrieve and validate risky scene anchors from a SQLite scene index.",
            default_command="python -m nusc_scene_agent run-experiment-config --config configs/risk_benchmark_suite.yaml",
            inputs=["nuScenes SQLite index", "YAML query benchmark"],
            outputs=["scenario-mining YAML", "case_library.json", "benchmark metrics", "derived benchmark slices"],
            metrics=[
                "validation acceptance@1",
                "validation acceptance@K",
                "reference consistency",
                "event localization",
            ],
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
            task="ego-only replay simulation",
            description="Roll ego state forward with planner profiles while replaying logged actor states and traffic-light context; other agents do not react to ego actions.",
            default_command="python -m nusc_scene_agent run-experiment-config --config configs/nuplan_closed_loop_sweep_medium.yaml",
            inputs=["nuPlan SQLite splits", "closed-loop planner profiles"],
            outputs=["closed-loop benchmarks", "closed-loop metrics", "cross-split leaderboard CSV", "case-study figures"],
            metrics=["ego ADE", "progress ratio", "collision proxy mismatch", "comfort violation", "replay score"],
        ),
        BenchmarkLayerSpec(
            layer_id="bench2drive_vision_planner",
            dataset="Bench2Drive",
            task="multi-camera vision planning",
            description="Train and evaluate a supervised ego-trajectory planner from six camera views and route features.",
            default_command="torchrun --standalone --nproc_per_node=8 -m nusc_scene_agent train-bench2drive-vision-planner --manifest artifacts/bench2drive/vision_e2e_manifest_tensor_160.jsonl --output outputs/bench2drive_vision_e2e_final --epochs 24 --batch-size 64 --image-size 160 --model-size research --architecture trajectory_transformer --camera-pooling transformer --trajectory-modes 4 --trajectory-selection expected --trajectory-temperature 0.5 --dropout 0.05 --num-workers 4 --prefetch-factor 4 --precision fp16 --selection-metric risk_aware --brake-loss-weight 0.25 --brake-positive-weight 1.25 --risk-sample-weight 1.25 --lateral-loss-weight 2.0 --turn-sample-weight 2.0 --selected-waypoint-loss-weight 0.5 --displacement-loss-weight 0.1 --endpoint-loss-weight 0.025 --path-length-loss-weight 0.025 --mode-classification-weight 0.05",
            inputs=["Bench2Drive Base1000 archives", "predecoded multi-camera tensor cache"],
            outputs=["training report", "planner checkpoint", "evaluation report", "prediction examples"],
            metrics=["ADE", "FDE", "control loss", "brake accuracy", "training throughput"],
        ),
        BenchmarkLayerSpec(
            layer_id="bench2drive_vision_closed_loop",
            dataset="Bench2Drive",
            task="model-in-the-loop vision closed-loop evaluation",
            description="Roll a trained multi-camera vision planner forward with predicted-waypoint control and vehicle dynamics on Bench2Drive clips.",
            default_command="python -m nusc_scene_agent run-experiment-config --config configs/bench2drive_vision_closed_loop.yaml",
            inputs=["Bench2Drive tensor manifest", "vision planner checkpoint"],
            outputs=["closed-loop report", "case metrics", "rollout figures", "optional per-case media"],
            metrics=["closed-loop ADE", "closed-loop FDE", "route completion", "lateral error", "closed-loop score"],
        ),
        BenchmarkLayerSpec(
            layer_id="carla_semantic_demo",
            dataset="CARLA",
            task="semantic-gated visual closed-loop demonstration",
            description="Search CARLA route/traffic configurations and keep only vision closed-loop rollouts that pass video, control, and semantic evidence checks.",
            default_command="python -m nusc_scene_agent run-experiment-config --config configs/carla_semantic_demo.yaml",
            inputs=["CARLA runtime", "Bench2Drive vision planner checkpoint"],
            outputs=["mining report", "semantic audit", "states CSV", "route trace", "rollout figure", "1080p HEVC MP4"],
            metrics=[
                "passed target count",
                "attempt count",
                "Traffic Manager vehicle count",
                "scripted vehicle count",
                "semantic audit status",
            ],
        ),
    ]
    return {
        "schema": BENCHMARK_CATALOG_SCHEMA,
        "layers": {layer.layer_id: layer.to_dict() for layer in layers},
    }


def write_benchmark_catalog(output_path: Path, catalog: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    payload = dict(catalog or build_default_benchmark_catalog())
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload

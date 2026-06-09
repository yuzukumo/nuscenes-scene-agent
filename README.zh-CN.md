<div align="center">

# Autonomous Driving Risk Scenario Benchmark Agent

面向 `nuScenes`、`nuPlan`、`Bench2Drive` 和 `CARLA` 的风险场景挖掘、基准生成、回放评估与视觉 E2E 规划器验证系统。

`Python 3.10+` `Conda` `nuScenes` `nuPlan` `CARLA` `Bench2Drive` `Ollama` `Benchmarking`

[English](README.md) | [简体中文](README.zh-CN.md)

</div>

<p align="center">
  <video src="https://github.com/user-attachments/assets/73f1467c-1bce-4f4a-b5b6-22fafa78e848" controls muted playsinline width="100%"></video>
  <br>
</p>

`Autonomous Driving Risk Scenario Benchmark Agent` 以统一的风险场景分类体系为核心。`nuScenes` 用于真实道路日志中的场景挖掘与验证，`nuPlan` 用于日志回放和回放式闭环评估，`Bench2Drive` 用于训练和诊断多相机视觉 E2E 轨迹规划器，`CARLA` 用于生成经过语义审查的闭环可视化证据。各数据源和仿真后端通过同一组场景定义连接。

## 场景体系

[configs/scenario_taxonomy.yaml](configs/scenario_taxonomy.yaml) 定义了数据挖掘、模型评估和仿真演示之间共享的风险场景词表。

| 后端 | 作用 |
| --- | --- |
| `nuScenes` | 从真实道路日志中挖掘风险场景锚点，并导出检索、感知、BEV 占用和世界模型切片。 |
| `nuPlan` | 在相同场景族下执行日志回放和回放式闭环误差评估。 |
| `Bench2Drive` | 基于仿真驾驶数据训练和评估多相机视觉 E2E 轨迹规划器。 |
| `CARLA` | 对选定场景目标执行语义审查后的视觉闭环 rollout。 |

<p align="center">
  <img src="./assets/pipeline_overview.png" alt="Pipeline overview" width="100%">
</p>

## 视觉 E2E 规划器训练

Bench2Drive 组件从六路 RGB 相机和路线特征训练视觉 E2E 轨迹规划器。模型预测多模态未来 ego 路径点，并同时输出控制和刹车头；结构上使用 transformer 对相机、路线和轨迹模态 token 进行融合。训练后的 checkpoint 通过监督验证、简化模型在环 rollout 和选定 CARLA 语义 rollout 进行评估。

| 项目 | 数值 |
| --- | --- |
| 输入 | 六路 RGB 相机视图和路线特征 |
| 模型 | `research` trajectory transformer，`4` 个轨迹模态 |
| 训练集 | `40,223` 个训练样本和 `4,717` 个验证样本 |
| 训练运行 | `8` 卡 DDP，`24` epochs，`315.983s` |
| 监督验证 | ADE `1.599`，FDE `2.625`，brake F1 `0.884` |
| 闭环诊断 | `64` 个 proxy rollout 案例；route completion `0.754`；closed-loop score `0.105` |
| CARLA 证据 | 保留 1 个闭环 demo；`0` 次碰撞；safety override ratio `0.105` |

## 结果概览

trainval 评估套件导出 `24` 个场景锚点、`48` 个成对场景挖掘查询，以及对齐的感知、BEV 占用和世界模型切片。表中的导出数量是对已验证挖掘样本的采样上限。

| 层级 | 结果 |
| --- | --- |
| 场景挖掘 | `24` 个锚点和 `48` 个 reference-aware 查询 |
| 学习式重排序 | `4,000` 个弱监督 trainval 组；scene-held-out Recall@1 `1.000` |
| 感知切片 | `24` 个带事件窗口 actor 监督的风险切片 |
| BEV 占用切片 | `oracle_occupancy` IoU `1.000`；`context_drop_occupancy` IoU `0.553`；`risk_actor_only` IoU `0.105` |
| 世界模型基准 | `24` 个场景条件切片；`kinematic_rollout` risk fidelity `0.869` |
| `ContextVAE` 基线 | `7` 个 forecast-compatible 切片；`ADE 0.280`；`MinADE@5 0.207`；risk fidelity `0.841` |
| `nuPlan` 回放回归 | 扫描 `576` 个 SQLite logs；`1556` 个候选；`112` 个回放案例；`history_kinematic` ADE `0.916` |
| `nuPlan` 闭环回放 | `112` 个 replay-simulation 案例；`history_kinematic` ADE `1.027`；closed-loop score `0.950` |
| Bench2Drive vision E2E trajectory transformer | `44,940` 个缓存多相机样本；`8` 卡 DDP 运行时间 `316.0s`；temperature-calibrated ADE `1.599`；FDE `2.625`；brake F1 `0.884` |
| Bench2Drive model-in-the-loop proxy | `64` 个案例；route completion `0.754`；mean lateral error `1.332 m`；closed-loop score `0.105` |
| CARLA semantic demo mining | `1/1` 个 demo 目标通过审查；`267` 帧；`13` 辆 Traffic Manager 车辆；`9` 个斑马线行人；direct model-control ratio `1.000`；safety override ratio `0.105`；`0` 辆 scripted vehicle；`0` 次碰撞 |
| 失败挖掘 | `401` 条失败记录、`83` 个簇和 `24` 个 benchmark update queries |
| failure-aware ML retrieval | Pass@K 从 `20/24` 提升到 `24/24`，候选生成经过验证约束 |

<p align="center">
  <img src="./assets/readme_overview.png" alt="Representative scene-mining outputs" width="100%">
</p>

<p align="center">
  <img src="./assets/world_model_results_overview.png" alt="World-model evaluation overview" width="100%">
</p>

<p align="center">
  <img src="./assets/nuplan_replay_case_studies.png" alt="nuPlan replay-regression case studies" width="100%">
</p>

<p align="center">
  <img src="./assets/nuplan_closed_loop_case_studies.png" alt="nuPlan closed-loop replay case studies" width="100%">
</p>

详细结果表见 [docs/benchmark_snapshot.md](docs/benchmark_snapshot.md)。

## 系统组件

- 基于本地 Ollama 的自然语言查询规划，并结合确定性检索和验证。
- Actor grounding、事件窗口定位、TTC、车道关系、斑马线语义和 BEV 证据渲染。
- 带场景、actor 和事件窗口监督的 reference-aware 场景挖掘基准。
- 场景条件感知切片、稀疏 BEV 占用切片和世界模型基准切片。
- 弱监督 query-scene 重排序和 failure-aware 候选生成。
- 覆盖感知、占用、世界模型、回放回归和闭环指标的模型在环失败挖掘。
- 对齐 `nuScenes` 场景挖掘、`nuPlan` 回放、Bench2Drive 视觉 E2E 规划器训练和 CARLA 语义演示的场景分类体系。
- 结果注册表、artifact manifest 和数据后端检查。

## 快速开始

```bash
conda env create -f environment.yml
conda activate nuscenes
```

数据下载链接和本地 archive 布局见 [docs/dataset_downloads.md](docs/dataset_downloads.md)。

准备数据并构建 `nuScenes` trainval 索引：

```bash
python -m nusc_scene_agent inspect-archives --workspace .
python -m nusc_scene_agent prepare-data --workspace . --profile trainval-full

python -m nusc_scene_agent build-index \
  --version v1.0-trainval \
  --dataroot data/sets/nuscenes \
  --db artifacts/index/v1.0-trainval.sqlite
```

启动本地模型服务。如果服务尚未运行，在单独 shell 中执行 `ollama serve`：

```bash
ollama pull gemma4:latest
ollama serve
```

运行完整基准套件：

```bash
python -m nusc_scene_agent run-full-benchmark-suite
```

完整套件配置见 [configs/full_benchmark_suite.yaml](configs/full_benchmark_suite.yaml)。分阶段命令见 [docs/usage.md](docs/usage.md)。

## 数据策略

数据集压缩包、解压后的数据、地图文件、SQLite 索引、生成结果、外部代码库和外部预测文件不纳入版本控制。相关目录包括 `archives/`、`data/`、`artifacts/`、`outputs/`、`external/` 和 `external_predictions/`。

## 项目结构

```text
src/nusc_scene_agent/    核心库和 CLI
benchmarks/              基准配置和导出的 benchmark JSON
configs/                 结构化实验配置
assets/                  README 和文档引用的静态图像与演示视频
docs/                    架构、结果快照、使用说明和数据下载说明
tests/                   检索、验证、报告和基准相关单元测试
environment.yml          Conda-first 环境配置
```

## 文档

- [Usage](docs/usage.md)
- [Architecture Notes](docs/architecture.md)
- [Benchmark Snapshot Notes](docs/benchmark_snapshot.md)
- [Dataset Downloads](docs/dataset_downloads.md)

## License

Released under the [MIT License](LICENSE).

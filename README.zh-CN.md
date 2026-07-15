<div align="center">

# Autonomous Driving Risk Scenario Benchmark Agent

面向 `nuScenes`、`nuPlan`、`Bench2Drive` 和 `CARLA` 的风险场景挖掘、基准生成、回放评估与视觉 E2E 规划器验证系统。

`Python 3.10+` `Conda` `nuScenes` `nuPlan` `CARLA` `Bench2Drive` `Ollama` `Benchmarking`

[English](README.md) | [简体中文](README.zh-CN.md)

</div>

<p align="center">
  <video src="https://github.com/user-attachments/assets/73f1467c-1bce-4f4a-b5b6-22fafa78e848" controls muted playsinline width="100%"></video>
</p>

## 配对模型比较

主要模型比较在同一组 `64` 个留出测试案例上，对 `dynamics_regularized_half` 与 `trajectory_baseline` 进行模型在环评估。表中正值表示改善；置信区间采用按案例配对的 percentile bootstrap 计算（`10,000` 次重复，随机种子 `7`）。

| 指标 | 基线 | 候选模型 | 改善量 | 95% CI |
| --- | ---: | ---: | ---: | ---: |
| Closed-loop ADE | 11.260 m | 9.721 m | `+1.539 m` | `[+0.739, +2.382]` |
| Closed-loop FDE | 23.690 m | 19.471 m | `+4.218 m` | `[+2.051, +6.541]` |
| Route completion | 0.746 | 0.799 | `+0.053` | `[+0.009, +0.105]` |
| Closed-loop score | 0.087 | 0.157 | `+0.070` | `[+0.023, +0.120]` |
| 平均横向误差 | 1.436 m | 1.568 m | `-0.132 m` | `[-0.406, +0.140]` |

候选模型改善了闭环行驶进度和累积轨迹误差，横向误差变化没有统计上的确定性。在独立开环测试集上（`4,334` 个样本、`97` 个 clip），路径长度误差改善 `0.069 m`（`95% CI [+0.036, +0.101]`），横向 MAE 增加 `0.028 m`（`95% CI [-0.045, -0.013]`）；ADE、FDE 和 brake F1 的变化均不确定。该权衡构成模型实验的主要结论。

`Autonomous Driving Risk Scenario Benchmark Agent` 使用统一的风险场景分类体系。`nuScenes` 用于真实道路日志中的场景挖掘与验证，`nuPlan` 用于日志回放和回放式闭环评估，`Bench2Drive` 用于训练和诊断多相机视觉 E2E 轨迹规划器，`CARLA` 用于生成通过预设审查条件的闭环可视化证据。

## 场景体系

[configs/scenario_taxonomy.yaml](configs/scenario_taxonomy.yaml) 定义了数据挖掘、模型评估和仿真演示之间共享的风险场景词表。

| 后端 | 作用 |
| --- | --- |
| `nuScenes` | 从真实道路日志中挖掘风险场景锚点，并导出检索、感知、BEV 占用和世界模型切片。 |
| `nuPlan` | 在相同场景族下执行日志回放和回放式闭环误差评估。 |
| `Bench2Drive` | 基于仿真驾驶数据训练和评估多相机视觉 E2E 轨迹规划器。 |
| `CARLA` | 对选定场景执行视觉闭环测试，并应用语义审查条件。 |

<p align="center">
  <img src="./assets/pipeline_overview.png" alt="Pipeline overview" width="100%">
</p>

## 视觉 E2E 规划器训练

Bench2Drive 组件以六路 RGB 图像和路线特征为输入，训练视觉 E2E 轨迹规划器。模型使用 Transformer 融合相机、路线和轨迹模态 token，预测多模态未来自车路径点，并输出控制量和制动概率。模型通过监督评估、简化的模型在环测试和 CARLA 语义场景测试进行验证。

| 项目 | 数值 |
| --- | --- |
| 输入 | 六路 RGB 相机视图和路线特征 |
| 模型 | `research` 配置的 trajectory transformer，`4` 个轨迹模态 |
| 训练集 | `35,629` 个训练样本、`4,977` 个验证样本和 `4,334` 个独立测试样本 |
| 训练配置 | `8` 卡 DDP，`24` epochs，`289.538s` |
| 候选模型（留出测试集） | ADE `1.653`，FDE `2.697`，横向 MAE `0.552 m`，brake F1 `0.828` |
| 闭环诊断 | `64` 个独立测试案例；route completion `0.799`；closed-loop score `0.157` |
| CARLA 证据 | `1` 个通过审查的闭环演示；`0` 次碰撞；安全层介入比例 `0.096` |

## 结果概览

`trainval` 评估套件导出 `24` 个场景锚点、`48` 个成对场景挖掘查询，以及对齐的感知、BEV 占用和世界模型切片。表中的导出数量是对已验证挖掘样本的采样上限。

| 层级 | 结果 |
| --- | --- |
| 场景挖掘 | `24` 个锚点和 `48` 个 reference-aware 查询 |
| 启发式敏感性 | 在不同检索权重、验证质量权重和几何阈值配置下，validation acceptance@1 均为 `16/16` |
| 学习式重排序 | `4,000` 个弱监督 trainval 组；scene-held-out weak-anchor consistency@1 `1.000` |
| 感知切片 | `24` 个带事件窗口 actor 监督的风险切片 |
| BEV 占用切片 | `oracle_occupancy` IoU `1.000`；`context_drop_occupancy` IoU `0.553`；`risk_actor_only` IoU `0.105` |
| 世界模型基准 | `24` 个场景条件切片；`kinematic_rollout` risk fidelity `0.869` |
| `ContextVAE` 基线 | `7` 个 forecast-compatible 切片；`ADE 0.280`；`MinADE@5 0.207`；risk fidelity `0.841` |
| `nuPlan` 回放回归 | 扫描 `576` 个 SQLite 日志；`1556` 个候选；`112` 个回放案例；`history_kinematic` ADE `0.916` |
| `nuPlan` 闭环回放 | `112` 个 replay-simulation 案例；`history_kinematic` ADE `1.027`；closed-loop score `0.950` |
| Bench2Drive vision E2E trajectory transformer | `44,940` 个缓存多相机样本；`8` 卡 DDP 运行时间 `289.5s`；独立测试候选模型 ADE `1.653`；FDE `2.697`；brake F1 `0.828` |
| Bench2Drive model-in-the-loop proxy | `64` 个独立测试案例；route completion `0.799`；平均横向误差 `1.568 m`；closed-loop score `0.157` |
| CARLA 语义演示挖掘 | `1/1` 个场景目标通过审查；`272` 帧；`13` 辆 Traffic Manager 车辆；`9` 名斑马线行人；模型路径点控制器占比 `1.000`；安全层介入比例 `0.096`；`0` 辆脚本控制车辆；`0` 次碰撞 |
| 失败挖掘 | `401` 条失败记录、`83` 个簇和 `24` 个 benchmark update queries |
| failure-aware ML retrieval | validation-gated acceptance@K 从 `20/24` 提升到 `24/24` |

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

<p align="center">
  <img src="./assets/bench2drive_prediction_comparison.png" alt="Bench2Drive 开环配对比较" width="100%">
</p>

<p align="center">
  <img src="./assets/bench2drive_closed_loop_comparison.png" alt="Bench2Drive 闭环配对比较" width="100%">
</p>

详细结果表见 [docs/benchmark_snapshot.md](docs/benchmark_snapshot.md)。

## 评估边界

`nuScenes` 参考锚点来源于经验证的案例库，属于弱监督标签。相关一致性指标衡量系统与确定性锚点的匹配程度，不等同于基于独立人工标注的语义召回率。世界模型比较在共同案例交集上进行并报告 bootstrap 区间，但可用于预测评估的子集仍然较小。Bench2Drive 使用按数据归档文件隔离的留出测试集，并按 clip 或 case 进行配对比较。Bench2Drive 闭环层仅作为模型在环诊断；CARLA 闭环测试是通过预设审查条件的定性证据，不构成具有统计效力的闭环驾驶基准。

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

数据下载链接和本地压缩包目录结构见 [docs/dataset_downloads.md](docs/dataset_downloads.md)。

准备数据并构建 `nuScenes` trainval 索引：

```bash
python -m nusc_scene_agent inspect-archives --workspace .
python -m nusc_scene_agent prepare-data --workspace . --profile trainval-full

python -m nusc_scene_agent build-index \
  --version v1.0-trainval \
  --dataroot data/sets/nuscenes \
  --db artifacts/index/v1.0-trainval.sqlite
```

启动本地模型服务。如果服务尚未运行，在另一终端中执行 `ollama serve`：

```bash
ollama pull gemma4:latest
ollama serve

python -m nusc_scene_agent inspect-ollama-model \
  --output outputs/ollama_model_metadata.json
export NUSC_SCENE_AGENT_OLLAMA_DIGEST="$(python -c 'import json; print(json.load(open("outputs/ollama_model_metadata.json"))["digest"])')"
```

运行完整基准套件：

```bash
python -m nusc_scene_agent run-full-benchmark-suite
```

完整套件配置见 [configs/full_benchmark_suite.yaml](configs/full_benchmark_suite.yaml)。分阶段命令见 [docs/usage.md](docs/usage.md)。
完整套件要求校验已记录的 Ollama digest；临时查询命令可以使用可变的 `gemma4:latest` 标签。

## 数据策略

数据集压缩包、解压后的数据、地图文件、SQLite 索引、生成结果、外部代码库和外部预测文件不纳入版本控制。相关目录包括 `archives/`、`data/`、`artifacts/`、`outputs/`、`external/` 和 `external_predictions/`。

## 项目结构

```text
src/nusc_scene_agent/    核心库和 CLI
benchmarks/              基准配置和导出的基准 JSON
configs/                 结构化实验配置
assets/                  README 和文档引用的静态图像与演示视频
docs/                    架构、结果快照、使用说明和数据下载说明
tests/                   检索、验证、报告和基准相关单元测试
environment.yml          以 Conda 为主的环境配置
```

## 文档

- [Usage](docs/usage.md)
- [Architecture Notes](docs/architecture.md)
- [Benchmark Snapshot Notes](docs/benchmark_snapshot.md)
- [Dataset Downloads](docs/dataset_downloads.md)

## 许可证

本项目采用 [MIT License](LICENSE)。

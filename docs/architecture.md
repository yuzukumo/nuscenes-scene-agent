# Architecture Notes

## System Goal

This project turns free-form risky-scene descriptions into validated `nuScenes` cases and benchmark-ready artifacts. It also includes a lightweight `nuPlan` replay-regression extension for simulation-style evaluation.

The target workflow is:

1. describe a risk scenario in natural language
2. retrieve candidate scenes from a prebuilt `SQLite` index
3. validate them with deterministic checks
4. export evidence figures, reports, and benchmark summaries

## Pipeline

### 1. Offline indexing

- parse `nuScenes` samples, annotations, ego poses, and map metadata
- build a compact `SQLite` index for fast candidate retrieval
- keep the runtime compact for repeated experiments

### 2. Query planning

- `rule` mode uses a deterministic parser over scene-language patterns
- `llm` mode asks a local Ollama model to produce structured constraints
- `hybrid` mode evaluates multiple query hypotheses instead of unconditionally merging planner outputs

## 3. Retrieval

- filter and score candidates using actor type, relative position, behavior labels, and temporal context
- support scenario families such as `crossing`, `cut_in`, `oncoming`, and `stopped_lead`
- batch temporal feature loading where needed to avoid large-query `SQLite` failures

## 4. Validation

- compute geometry and relative motion features
- estimate risk-related signals such as proximity and TTC
- use map context for lane relation, road direction, crosswalk preference, and walkway support
- keep validation deterministic so benchmark behavior is inspectable and reproducible
- localize the critical event window with start, end, and peak sample indices
- export grounded primary-actor metadata for scenario-level reporting

## 5. Reporting

- export `Markdown` and `HTML` summaries
- generate BEV evidence images
- write case-library artifacts and benchmark metrics
- surface failure-pattern groupings and benchmark-profile comparisons
- export reference-aware scenario-mining artifacts such as event windows and grounded actors
- export scenario-group summaries that score anchor consistency across paraphrases
- export leaderboard-style HTML and CSV artifacts for profile comparison and behavior-level failure analysis
- export static benchmark browsers that link query summaries, case reports, and evidence figures across profiles
- support explicit ablation studies over reranking, map context, and event localization
- export scenario-conditioned perception slices and model-agnostic evaluation summaries for external BEV tracking outputs
- export sparse BEV occupancy slices for primary and context actor coverage analysis
- export scenario-conditioned world-model slices with future trajectory and occupancy supervision
- export replay-ready JSONL or optional MCAP assets for offline system testing
- export compact `nuPlan` replay-regression cases from SQLite scenario tags
- export Ollama-generated benchmark review reports from local artifacts
- export unified risk-case collections, benchmark-layer registries, dataset-backend inventories, and artifact manifests

## 6. Counterfactual Benchmark Generation

The repository supports counterfactual benchmark generation from validated case libraries.

The generator:

- selects diverse anchor cases from an existing case library
- creates positive canonical and paraphrase variants
- creates actor-swap and behavior-swap negatives
- keeps explicit `reference_case_keys` for objective evaluation
- optionally carries reference event windows for localization-aware scoring

This extends the repository from case retrieval to benchmark construction and contrastive evaluation.

## 7. Scenario Mining Benchmark Layer

The repository exposes a reference-aware benchmark layer derived from validated case libraries.

This layer keeps explicit:

- reference scene names
- reference actor instance tokens
- reference event start and end sample indices
- reference event peak sample indices

That allows the metrics layer to move beyond generic retrieval scores and report scenario-mining-style outputs such as:

- scene-level objective@1 and objective@K
- actor-level objective@1 and objective@K
- event-window overlap
- peak-sample localization error

## 8. Scenario-Conditioned Perception Benchmark Layer

The repository also derives a compact perception benchmark from scenario-mining anchors.

This layer exports:

- short event-window actor trajectories in ego coordinates
- anchor sample indices and event-window bounds
- behavior and actor labels inherited from validated scenario-mining cases
- benchmark-side risk facets such as distance band, TTC band, visibility band, map relation, and occlusion proxy

That makes it possible to score external detector or tracker outputs on mined risk slices with metrics such as:

- anchor recall
- full-track success
- event recall
- contiguous temporal coverage
- first-match lag
- center localization error

The layer also includes:

- an adapter from official `nuScenes` detection and tracking JSON into the local slice-evaluation schema
- greedy temporal linking for detection-only outputs so short event windows can still be evaluated without native track IDs
- coverage-aware benchmark filtering so split-specific prediction files can be aligned to the subset they actually cover
- per-case `CSV`, `JSON`, `Markdown`, and `HTML` exports for downstream analysis

## 9. Risk-Conditioned BEV Occupancy Benchmark Layer

The repository derives sparse BEV occupancy slices from the perception benchmark and the indexed `nuScenes` actor table.

This layer keeps:

- event-window sample tokens and sample indices
- primary risk actor occupancy cells
- context actor occupancy cells
- union occupancy cells over a fixed ego-frame BEV grid
- inherited behavior labels and risk facets

The evaluation layer reports:

- occupancy IoU
- primary actor recall
- context recall
- anchor-frame occupancy IoU
- composite risk-fidelity score

This layer uses sparse center-cell occupancy labels to test whether perception outputs cover the risk actor and nearby dynamic context on mined event windows.

## 10. Scenario-Conditioned World-Model Benchmark Layer

The repository also derives a compact world-model benchmark from the perception-slice layer.

This layer keeps:

- a short observed history up to the rollout anchor
- a short future horizon for the primary risk actor in ego coordinates
- sparse future occupancy targets for the primary actor and surrounding context actors
- benchmark-side risk facets inherited from the validated case anchor

That makes it possible to score future rollouts with metrics such as:

- full-horizon success
- horizon recall
- average displacement error
- final displacement error
- occupancy IoU
- closest-approach distance error
- closest-approach timing error
- composite risk-fidelity score

The layer also includes:

- proxy rollout generators for controlled studies
- an adapter for compact external rollout formats keyed by benchmark group or reference case
- an adapter for `nuScenes prediction challenge` style multi-modal forecast outputs keyed by `(instance, sample)`
- direct execution of the official `nuScenes` physics forecast baselines on benchmark anchors
- per-case `CSV`, `JSON`, `Markdown`, and `HTML` evaluation exports
- multi-profile comparison exports for behavior-wise and risk-wise analysis
- challenge-track breakdowns for benchmark-style reporting
- qualitative case-study rendering for benchmark-aligned trajectory comparisons
- replay export as newline-delimited JSON and optional `MCAP`

## 11. nuPlan Replay-Regression Extension

The `nuPlan` extension reads local SQLite logs directly and constructs compact replay-regression cases from scenario tags.

Each case stores:

- an anchor lidar timestamp and scenario tag
- a scenario-family taxonomy, difficulty label, and risk-facet summary
- ego history and future ego states
- a primary risk actor when the tag is agent-specific
- traffic-light status counts for the replay window
- risk targets such as minimum actor distance, minimum TTC, and collision-proxy consistency
- comfort targets such as maximum acceleration, jerk, and yaw rate

The evaluation layer compares predicted ego rollouts against logged replay windows with:

- full-horizon coverage
- ego ADE and FDE
- minimum-distance error to the primary actor
- minimum-TTC error
- red-light context recall
- acceleration, jerk, and yaw-rate errors
- collision-proxy mismatch
- composite risk-fidelity score
- profile comparison summaries and leaderboard CSV exports
- cross-split and city-level sweep summaries
- failure-taxonomy CSV exports by profile, scenario family, tag, and difficulty
- qualitative replay case-study figures with per-profile trajectory overlays

This extension is a compact regression harness that connects the repository to simulation-style testing while preserving the main scope of dataset indexing, scenario mining, and benchmark generation.

## 12. Unified Benchmark Interface

The project exposes a structural layer above individual benchmark implementations.

The interface includes:

- `unified_risk_case_v1` for common case metadata across `nuScenes` and `nuPlan`
- `benchmark_registry_v1` for declaring benchmark layers, inputs, outputs, and metrics
- `dataset_backend_inventory_v1` for checking local `nuScenes` and `nuPlan` readiness
- `benchmark_artifact_manifest_v1` for recording benchmark outputs and evidence artifacts
- YAML experiment configs for reproducible study entry points

Benchmark-specific schemas remain the source of task-level detail. The interface adds a common indexing and reporting layer so scenario mining, perception slices, world-model slices, and replay regression can be compared as parts of one evaluation framework.

The default `nuScenes` benchmark suite is generated through `configs/risk_benchmark_suite.yaml`. This config derives scenario-mining queries, perception slices, world-model slices, sparse BEV occupancy slices, and proxy-study outputs from the same validated case library and SQLite index. The default `max_cases` controls the exported benchmark size; it is a sampling cap rather than a dataset-size estimate.

## 13. Structural Multimodal Retrieval

The retrieval pipeline includes a deterministic multimodal reranking layer. It scores each candidate with four structural signals:

- language-intent consistency
- BEV geometry consistency
- motion consistency
- sensor visibility

The module writes per-candidate modality scores and fused scores to JSON, CSV, and Markdown. It is an auditable retrieval model for scene mining; learned perception backbones remain outside this layer.

## 14. Learned Scene Reranking

The learned retrieval layer trains a compact PyTorch query-scene scorer from weakly labeled `v1.0-trainval` scenario families.

Training data is constructed from:

- rule-mined positives for crossing, stopped-lead, lateral cut-in, and oncoming scenario families
- near-miss negatives mined from the same trainval SQLite index
- scene-level validation splits for weak-supervised training
- query features from language, actor type, position, behavior, and risk thresholds
- candidate features from BEV geometry, motion, TTC, and sensor visibility
- explicit query-candidate compatibility features

The model is a small pairwise MLP rather than an end-to-end perception or driving model. It is used only as a reranker after candidate retrieval. The trained checkpoint is evaluated both on weak-supervised held-out scenes and on the reference-aware scenario-mining benchmark.

Outputs include a checkpoint, JSON training report, Markdown training summary, and per-query learned retrieval reports. The trained checkpoint can be used through `--rerank-mode learned`.

## 15. Model-in-the-loop Failure Mining

The failure-mining layer reads metric artifacts from perception, BEV occupancy, world-model, external forecast, and `nuPlan` replay-regression studies.

It exports:

- failure records with source, profile, case key, failure tag, severity, behavior, actor, and scenario context
- clustered failure summaries by source, failure mode, behavior or `nuPlan` scenario family, and actor or scenario tag
- benchmark update queries that can be fed back into the scenario-mining pipeline

This creates a closed evaluation loop: model outputs are evaluated, failure patterns are mined, and new retrieval queries are generated from the observed failures.

## 16. Research Agent

The research-agent layer is a LangGraph workflow that reads local benchmark artifacts and calls a local Ollama model.

The graph uses three nodes:

- `collect_artifacts` records benchmark summaries, leaderboards, registry metadata, and the analysis constraints
- `analyze_with_llm` asks the local model for a structured gap analysis and next-action plan
- `write_report` exports a JSON and Markdown report

Deterministic metrics remain the evidence source. The agent operates after evaluation and is used to inspect completed capabilities, identify remaining gaps, propose benchmark update queries, and list unsupported claims.

## Agent Formulation

The agent component is implemented as an orchestration loop:

- interpret user intent from open-ended risk language
- produce retrieval hypotheses
- search a structured memory of scenes
- validate the results with explicit evidence
- package the result into usable evaluation artifacts

This is a hybrid orchestration pattern: a local Ollama model handles intent interpretation, while deterministic code handles evidence generation and reproducibility.

## Hybrid Formulation

The `hybrid_agent` evaluates rule and local-model outputs as separate hypotheses before selecting an evidence-supported result:

- evaluates rule-based and Ollama-based query hypotheses separately
- builds conservative merged hypotheses when structured overlap supports that choice
- retrieves and validates cases for each hypothesis
- chooses the most evidence-supported interpretation at runtime

This design is intended to reduce instability relative to direct union of planner outputs on the language-robustness benchmark.

## Scope Boundaries

The maintained scope is indexing, retrieval, validation, reranking, benchmark generation, external-baseline adapters, and compact replay regression.

Large-stack components remain outside the repository scope:

- end-to-end driving policy training
- full autonomy-stack simulation
- production UI serving
- large-scale platform maintenance

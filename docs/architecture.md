# Architecture Notes

## System Goal

This project turns free-form risky-scene descriptions into validated `nuScenes` cases and benchmark-ready artifacts.

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
- `llm` mode asks an OpenAI-compatible model to produce structured constraints
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
- export scenario-conditioned world-model slices with future trajectory and occupancy supervision
- export replay-ready JSONL or optional MCAP assets for offline system testing

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

## 9. Scenario-Conditioned World-Model Benchmark Layer

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

## Agent Formulation

The repository does not implement a driving policy. The agent component is the orchestration loop:

- interpret user intent from open-ended risk language
- produce retrieval hypotheses
- search a structured memory of scenes
- validate the results with explicit evidence
- package the result into usable evaluation artifacts

This is a hybrid orchestration pattern: `LLM` for intent interpretation and deterministic code for evidence generation and reproducibility.

## Hybrid Formulation

The current `hybrid_agent` does not simply union rule and LLM outputs.

Instead, it:

- evaluates rule-based and LLM-based query hypotheses separately
- builds conservative merged hypotheses when structured overlap supports that choice
- retrieves and validates cases for each hypothesis
- chooses the most evidence-supported interpretation at runtime

This design is intended to reduce instability relative to direct union of planner outputs on the language-robustness benchmark.

## Solo-Developer Design Constraints

The repository is intentionally designed to remain compact:

- no heavy training pipeline
- no end-to-end driving stack
- no simulator-first workflow
- no requirement to maintain a large web application

The scope remains centered on indexing, retrieval, validation, and benchmark generation, which preserves research depth without requiring a large training or platform stack.

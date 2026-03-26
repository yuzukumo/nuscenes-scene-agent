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
- keep the runtime lightweight for repeated experiments

### 2. Query planning

- `rule` mode uses a deterministic parser over scene-language patterns
- `llm` mode asks an OpenAI-compatible model to produce structured constraints
- `hybrid` mode evaluates multiple query hypotheses instead of blindly merging planner outputs

## 3. Retrieval

- filter and score candidates using actor type, relative position, behavior labels, and temporal context
- support scenario families such as `crossing`, `cut_in`, `oncoming`, and `stopped_lead`
- batch temporal feature loading where needed to avoid large-query `SQLite` failures

## 4. Validation

- compute geometry and relative motion features
- estimate risk-related signals such as proximity and TTC
- use map context for lane relation, road direction, crosswalk preference, and walkway support
- keep validation deterministic so benchmark behavior is inspectable and reproducible

## 5. Reporting

- export `Markdown` and `HTML` summaries
- generate BEV evidence images
- write case-library artifacts and benchmark metrics
- surface hard-case groupings and benchmark-profile comparisons

## Why This Counts As An Agent

The repo does not implement a driving policy. The agentic part is the orchestration loop:

- interpret user intent from open-ended risk language
- produce retrieval hypotheses
- search a structured memory of scenes
- validate the results with explicit evidence
- package the result into usable evaluation artifacts

This is a practical hybrid agent pattern: `LLM for intent understanding`, `code for evidence and stability`.

## Why The Hybrid Path Matters

The current `hybrid_agent` does not simply union rule and LLM outputs.

Instead, it:

- evaluates rule-based and LLM-based query hypotheses separately
- builds conservative merged hypotheses when useful
- retrieves and validates cases for each hypothesis
- chooses the most evidence-supported interpretation at runtime

That makes the system more stable on language-stress benchmarks than naive fusion.

## Solo-Developer Design Constraints

The repo is intentionally designed to stay lightweight:

- no heavy training pipeline
- no end-to-end driving stack
- no simulator-first workflow
- no requirement to maintain a large web application

The focus stays on index, retrieval, validation, and benchmark generation, which gives good research and portfolio value without turning into a multi-month platform project.

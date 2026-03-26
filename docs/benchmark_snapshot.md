# Benchmark Snapshot Notes

The repository keeps large generated outputs local by default, but the current local benchmark runs provide a useful picture of the system's behavior.

## Core Trainval Suite

Dataset:

- `nuScenes v1.0-trainval`
- `map expansion`

Current snapshot:

| Metric | Value |
| --- | --- |
| Queries | 16 |
| Pass@1 | 16/16 |
| Pass@K | 16/16 |
| Selected cases | 36 |
| Unique cases | 33 |
| Unique passed cases | 29 |
| Mean best validation score | 89.26 |

Behavior coverage:

- `crossing`
- `stopped_lead`
- `oncoming`
- `cut_in`
- close-proximity queries without explicit behavior labels

## Language-Stress Comparison

The language-stress suite is designed to test phrasing robustness rather than simply replay structured benchmark labels.

Current snapshot:

| Profile | Pass@1 | Mean Best Score |
| --- | --- | --- |
| `rule_only` | 16/16 | 90.64 |
| `llm_planner` | 15/16 | 86.29 |
| `hybrid_agent` | 16/16 | 91.34 |

Additional signal:

- queries: `16`
- planner signal divergence: `12`

## Why These Results Matter

The interesting result is not only that the benchmark passes.

It also shows:

- language phrasing still changes the parsed structured signal
- pure LLM planning can degrade on some scenario phrasings
- evidence-guided hybrid arbitration recovers stability without giving up flexibility

That makes the repo useful both as a retrieval toolkit and as a benchmark-design sandbox for studying planner robustness.

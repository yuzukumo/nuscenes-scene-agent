# Human Audit Set

This directory contains an unlabeled template for manual author review of mined risk-scene anchors.

## Files

- `human_audit_items.jsonl`: canonical annotation file.
- `human_audit_items.csv`: spreadsheet-friendly copy with the same label fields.
- `human_audit_manifest.json`: sampling metadata.
- `review_queue.md`: compact review table with evidence links.

## Label Fields

- `semantic_match`: whether the scene matches the source query semantics.
- `primary_actor_correct`: whether the selected actor is the main risk actor.
- `behavior_correct`: whether the system behavior label is correct.
- `event_window_correct`: whether the system event window is acceptable.
- `event_start_sample_idx`, `event_end_sample_idx`, `event_peak_sample_idx`: optional corrected event indices.
- `confidence`: reviewer confidence in `[0, 1]`.
- `notes`: short free-text justification for ambiguous cases.

Use `true`, `false`, or blank for boolean fields. Blank fields are ignored by the evaluator.

## Sample

| Behavior | Cases |
| --- | ---: |
| crossing | 7 |
| cut_in | 7 |
| oncoming | 7 |
| proximity | 5 |
| stopped_lead | 7 |

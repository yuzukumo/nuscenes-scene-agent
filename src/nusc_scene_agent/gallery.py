from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from jinja2 import Template


BENCHMARK_GALLERY_TEMPLATE = Template(
    """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{{ title }}</title>
  <style>
    :root {
      --bg: #f4efe7;
      --panel: #fffdf9;
      --line: #ded4c5;
      --ink: #1f1f1f;
      --muted: #6e665d;
      --head: #ebe2d5;
      --accent: #264653;
      --good: #2a9d8f;
      --warn: #c46a00;
      --bad: #b00020;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      padding: 32px;
      font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
      background: linear-gradient(180deg, #faf6f0 0%, var(--bg) 100%);
      color: var(--ink);
    }
    h1, h2, h3, p { margin-top: 0; }
    .hero, .controls, .card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      box-shadow: 0 10px 30px rgba(31, 31, 31, 0.05);
    }
    .hero, .controls {
      padding: 20px 22px;
      margin-bottom: 18px;
    }
    .meta {
      color: var(--muted);
      margin-bottom: 12px;
    }
    .pill {
      display: inline-block;
      margin-right: 8px;
      margin-bottom: 8px;
      padding: 7px 12px;
      border-radius: 999px;
      background: #f2ece3;
      color: var(--accent);
      font-size: 13px;
      font-weight: 700;
    }
    .controls-grid {
      display: grid;
      grid-template-columns: 2fr 1fr 1fr 1fr;
      gap: 12px;
    }
    label {
      display: block;
      font-size: 13px;
      color: var(--muted);
      margin-bottom: 6px;
      font-weight: 700;
    }
    input, select {
      width: 100%;
      padding: 10px 12px;
      border: 1px solid var(--line);
      border-radius: 12px;
      background: white;
      color: var(--ink);
      font: inherit;
    }
    .gallery {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
      gap: 16px;
    }
    .card {
      overflow: hidden;
    }
    .card img {
      width: 100%;
      aspect-ratio: 1.3 / 1;
      object-fit: cover;
      display: block;
      background: #eee7dd;
    }
    .card-body {
      padding: 18px 18px 16px 18px;
    }
    .title-row {
      display: flex;
      gap: 10px;
      align-items: flex-start;
      justify-content: space-between;
      margin-bottom: 10px;
    }
    .title-row h3 {
      font-size: 18px;
      line-height: 1.3;
      margin-bottom: 0;
    }
    .badge {
      display: inline-block;
      padding: 5px 9px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 700;
      white-space: nowrap;
    }
    .badge-pass { background: rgba(42, 157, 143, 0.12); color: var(--good); }
    .badge-fail { background: rgba(176, 0, 32, 0.1); color: var(--bad); }
    .badge-ref { background: rgba(38, 70, 83, 0.1); color: var(--accent); }
    .stats {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px 14px;
      margin-bottom: 12px;
      font-size: 14px;
    }
    .stats div { color: var(--muted); }
    .stats strong { color: var(--ink); }
    .tag-list {
      margin-bottom: 12px;
    }
    .tag {
      display: inline-block;
      margin-right: 6px;
      margin-bottom: 6px;
      padding: 5px 8px;
      border-radius: 10px;
      background: #f4efe8;
      font-size: 12px;
      color: var(--accent);
    }
    .links a {
      color: #b14f00;
      text-decoration: none;
      margin-right: 14px;
      font-weight: 700;
    }
    .hidden { display: none !important; }
    .result-count { color: var(--muted); font-size: 14px; margin-top: 10px; }
    @media (max-width: 1000px) {
      .controls-grid { grid-template-columns: 1fr 1fr; }
    }
    @media (max-width: 640px) {
      body { padding: 18px; }
      .controls-grid { grid-template-columns: 1fr; }
      .gallery { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <section class="hero">
    <h1>{{ title }}</h1>
    <p class="meta">{{ subtitle }}</p>
    <div>
      <span class="pill">Queries: {{ cards|length }}</span>
      <span class="pill">Source: {{ source_name }}</span>
      <span class="pill">Top-1 previews with direct links to case reports</span>
    </div>
  </section>

  <section class="controls">
    <div class="controls-grid">
      <div>
        <label for="search">Search</label>
        <input id="search" type="search" placeholder="query text, scene, actor, behavior" />
      </div>
      <div>
        <label for="behavior">Behavior</label>
        <select id="behavior">
          <option value="">All behaviors</option>
          {% for item in behaviors %}
          <option value="{{ item }}">{{ item }}</option>
          {% endfor %}
        </select>
      </div>
      <div>
        <label for="actor">Actor</label>
        <select id="actor">
          <option value="">All actors</option>
          {% for item in actors %}
          <option value="{{ item }}">{{ item }}</option>
          {% endfor %}
        </select>
      </div>
      <div>
        <label for="status">Status</label>
        <select id="status">
          <option value="">All queries</option>
          <option value="pass">Top-1 passing</option>
          <option value="mismatch">Reference mismatch</option>
        </select>
      </div>
    </div>
    <div class="result-count"><span id="result-count">{{ cards|length }}</span> cards shown</div>
  </section>

  <section class="gallery" id="gallery">
    {% for card in cards %}
    <article
      class="card"
      data-search="{{ card.search_text }}"
      data-behaviors="{{ card.behavior_text }}"
      data-actors="{{ card.actor_text }}"
      data-status="{{ card.status_text }}"
    >
      {% if card.image_path %}
      <img src="{{ card.image_path }}" alt="{{ card.query_text }}" />
      {% endif %}
      <div class="card-body">
        <div class="title-row">
          <div>
            <h3>{{ card.query_text }}</h3>
            <div class="meta">{{ card.query_id }} | {{ card.description }}</div>
          </div>
          <div>
            <span class="badge {{ 'badge-pass' if card.pass_at_1 else 'badge-fail' }}">{{ "pass@1" if card.pass_at_1 else "top-1 fail" }}</span>
            {% if card.reference_mismatch %}
            <span class="badge badge-ref">ref mismatch</span>
            {% endif %}
          </div>
        </div>

        <div class="tag-list">
          {% for tag in card.tags %}
          <span class="tag">{{ tag }}</span>
          {% endfor %}
        </div>

        <div class="stats">
          <div><strong>Scene</strong><br />{{ card.scene_name }} / sample {{ card.sample_idx }}</div>
          <div><strong>Actor</strong><br />{{ card.actor_name }}</div>
          <div><strong>Score</strong><br />{{ "%.2f"|format(card.validation_score) }}</div>
          <div><strong>Distance / TTC</strong><br />{{ "%.2f"|format(card.min_distance_m) }} m / {{ "%.2f"|format(card.min_ttc_s) }} s</div>
          <div><strong>Scene@1 / Actor@1</strong><br />{{ card.scene_objective_at_1 }} / {{ card.actor_objective_at_1 }}</div>
          <div><strong>Event IoU / Peak Error</strong><br />{{ card.event_iou_text }} / {{ card.peak_error_text }}</div>
        </div>

        <div class="links">
          <a href="{{ card.summary_path }}">summary</a>
          <a href="{{ card.case_path }}">case</a>
          <a href="{{ card.summary_html_path }}">html</a>
        </div>
      </div>
    </article>
    {% endfor %}
  </section>

  <script>
    const searchInput = document.getElementById("search");
    const behaviorSelect = document.getElementById("behavior");
    const actorSelect = document.getElementById("actor");
    const statusSelect = document.getElementById("status");
    const cards = Array.from(document.querySelectorAll(".card"));
    const resultCount = document.getElementById("result-count");

    function applyFilters() {
      const search = searchInput.value.trim().toLowerCase();
      const behavior = behaviorSelect.value;
      const actor = actorSelect.value;
      const status = statusSelect.value;
      let shown = 0;

      cards.forEach((card) => {
        const matchesSearch = !search || card.dataset.search.includes(search);
        const matchesBehavior = !behavior || card.dataset.behaviors.split("|").includes(behavior);
        const matchesActor = !actor || card.dataset.actors.split("|").includes(actor);
        const matchesStatus = !status || card.dataset.status.split("|").includes(status);
        const visible = matchesSearch && matchesBehavior && matchesActor && matchesStatus;
        card.classList.toggle("hidden", !visible);
        if (visible) shown += 1;
      });

      resultCount.textContent = String(shown);
    }

    [searchInput, behaviorSelect, actorSelect, statusSelect].forEach((node) => {
      node.addEventListener("input", applyFilters);
      node.addEventListener("change", applyFilters);
    });
  </script>
</body>
</html>
"""
)


COMPARISON_BROWSER_TEMPLATE = Template(
    """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{{ title }}</title>
  <style>
    :root {
      --bg: #f5f0e8;
      --panel: #fffdfa;
      --line: #ddd3c4;
      --ink: #1f1f1f;
      --muted: #6f665c;
      --head: #e8decf;
      --accent: #264653;
      --good: #2a9d8f;
      --warn: #c46a00;
      --bad: #b00020;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      padding: 32px;
      font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
      background: linear-gradient(180deg, #faf6f0 0%, var(--bg) 100%);
      color: var(--ink);
    }
    .hero, .controls, .query-card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 20px;
      box-shadow: 0 10px 30px rgba(31, 31, 31, 0.05);
    }
    .hero, .controls {
      padding: 22px 24px;
      margin-bottom: 18px;
    }
    .hero p, .controls p {
      color: var(--muted);
      margin-bottom: 0;
    }
    .pill {
      display: inline-block;
      margin-right: 8px;
      margin-bottom: 8px;
      padding: 7px 12px;
      border-radius: 999px;
      background: #f2ece3;
      color: var(--accent);
      font-size: 13px;
      font-weight: 700;
    }
    .controls-grid {
      display: grid;
      grid-template-columns: 2fr 1fr 1fr 1fr 1fr;
      gap: 12px;
    }
    label {
      display: block;
      font-size: 13px;
      color: var(--muted);
      margin-bottom: 6px;
      font-weight: 700;
    }
    input, select {
      width: 100%;
      padding: 10px 12px;
      border: 1px solid var(--line);
      border-radius: 12px;
      background: white;
      color: var(--ink);
      font: inherit;
    }
    .query-list {
      display: grid;
      gap: 18px;
    }
    .query-card {
      padding: 18px;
    }
    .query-head {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 14px;
      margin-bottom: 12px;
    }
    .meta {
      color: var(--muted);
      font-size: 14px;
    }
    .badge {
      display: inline-block;
      padding: 5px 9px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 700;
      margin-left: 6px;
      margin-bottom: 6px;
    }
    .badge-div { background: rgba(196, 106, 0, 0.12); color: var(--warn); }
    .badge-best { background: rgba(38, 70, 83, 0.12); color: var(--accent); }
    .tag {
      display: inline-block;
      margin-right: 6px;
      margin-bottom: 6px;
      padding: 5px 8px;
      border-radius: 10px;
      background: #f4efe8;
      font-size: 12px;
      color: var(--accent);
    }
    .profile-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 14px;
      margin-top: 12px;
    }
    .profile-card {
      border: 1px solid var(--line);
      border-radius: 16px;
      overflow: hidden;
      background: white;
    }
    .profile-card img {
      width: 100%;
      aspect-ratio: 1.28 / 1;
      object-fit: cover;
      display: block;
      background: #eee7dd;
    }
    .profile-body {
      padding: 14px;
    }
    .profile-title {
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 8px;
      margin-bottom: 8px;
    }
    .profile-title h3 {
      margin: 0;
      font-size: 16px;
    }
    .status {
      font-size: 12px;
      font-weight: 700;
    }
    .status-good { color: var(--good); }
    .status-bad { color: var(--bad); }
    .stats {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 7px 10px;
      font-size: 13px;
      color: var(--muted);
      margin-bottom: 10px;
    }
    .stats strong { color: var(--ink); }
    .failure {
      font-size: 13px;
      color: var(--bad);
      min-height: 18px;
      margin-bottom: 10px;
    }
    .links a {
      color: #b14f00;
      text-decoration: none;
      font-weight: 700;
      margin-right: 12px;
    }
    .hidden { display: none !important; }
    .result-count { color: var(--muted); font-size: 14px; margin-top: 10px; }
    @media (max-width: 1100px) {
      .controls-grid { grid-template-columns: 1fr 1fr 1fr; }
      .profile-grid { grid-template-columns: 1fr; }
    }
    @media (max-width: 700px) {
      body { padding: 18px; }
      .controls-grid { grid-template-columns: 1fr; }
      .query-head { display: block; }
    }
  </style>
</head>
<body>
  <section class="hero">
    <h1>{{ title }}</h1>
    <p>{{ subtitle }}</p>
    <div style="margin-top: 12px;">
      <span class="pill">Queries: {{ cards|length }}</span>
      <span class="pill">Profiles: {{ profile_labels|join(", ") }}</span>
      <span class="pill">Divergent queries: {{ divergent_query_count }}</span>
      <span class="pill">Top profile: {{ top_profile }}</span>
    </div>
  </section>

  <section class="controls">
    <div class="controls-grid">
      <div>
        <label for="search">Search</label>
        <input id="search" type="search" placeholder="query, scene, behavior, actor" />
      </div>
      <div>
        <label for="behavior">Behavior</label>
        <select id="behavior">
          <option value="">All behaviors</option>
          {% for item in behaviors %}
          <option value="{{ item }}">{{ item }}</option>
          {% endfor %}
        </select>
      </div>
      <div>
        <label for="actor">Actor</label>
        <select id="actor">
          <option value="">All actors</option>
          {% for item in actors %}
          <option value="{{ item }}">{{ item }}</option>
          {% endfor %}
        </select>
      </div>
      <div>
        <label for="best-profile">Best Profile</label>
        <select id="best-profile">
          <option value="">All winners</option>
          {% for item in profile_options %}
          <option value="{{ item.name }}">{{ item.label }}</option>
          {% endfor %}
        </select>
      </div>
      <div>
        <label for="status">Status</label>
        <select id="status">
          <option value="">All queries</option>
          <option value="divergent">Divergent only</option>
          <option value="mismatch">Any mismatch</option>
        </select>
      </div>
    </div>
    <div class="result-count"><span id="result-count">{{ cards|length }}</span> query cards shown</div>
  </section>

  <section class="query-list" id="query-list">
    {% for card in cards %}
    <article
      class="query-card"
      data-search="{{ card.search_text }}"
      data-behaviors="{{ card.behavior_text }}"
      data-actors="{{ card.actor_text }}"
      data-best-profile="{{ card.best_profile }}"
      data-status="{{ card.status_text }}"
    >
      <div class="query-head">
        <div>
          <h2 style="margin-bottom: 6px;">{{ card.query_text }}</h2>
          <div class="meta">{{ card.query_id }} | {{ card.description }}</div>
          <div style="margin-top: 8px;">
            {% for tag in card.tags %}
            <span class="tag">{{ tag }}</span>
            {% endfor %}
          </div>
        </div>
        <div style="text-align: right;">
          <span class="badge badge-best">best: {{ card.best_profile_label }}</span>
          {% if card.signal_divergence %}
          <span class="badge badge-div">planner disagreement</span>
          {% endif %}
          {% if card.has_mismatch %}
          <span class="badge badge-div">reference mismatch</span>
          {% endif %}
          <div class="meta" style="margin-top: 8px;">score span: {{ "%.2f"|format(card.score_span) }}</div>
        </div>
      </div>

      <div class="profile-grid">
        {% for profile in card.profiles %}
        <section class="profile-card">
          {% if profile.image_path %}
          <img src="{{ profile.image_path }}" alt="{{ profile.label }} preview" />
          {% endif %}
          <div class="profile-body">
            <div class="profile-title">
              <h3>{{ profile.label }}</h3>
              <span class="status {{ 'status-good' if profile.reference_objective_at_1 else 'status-bad' }}">
                {{ "anchor ok" if profile.reference_objective_at_1 else "anchor miss" }}
              </span>
            </div>
            <div class="stats">
              <div><strong>Scene</strong><br />{{ profile.scene_name }} / sample {{ profile.sample_idx }}</div>
              <div><strong>Actor</strong><br />{{ profile.actor_name }}</div>
              <div><strong>Score</strong><br />{{ "%.2f"|format(profile.validation_score) }}</div>
              <div><strong>Distance / TTC</strong><br />{{ "%.2f"|format(profile.min_distance_m) }} m / {{ "%.2f"|format(profile.min_ttc_s) }} s</div>
              <div><strong>Scene@1 / Actor@1</strong><br />{{ profile.scene_objective_at_1 }} / {{ profile.actor_objective_at_1 }}</div>
              <div><strong>Event IoU / Peak Error</strong><br />{{ profile.event_iou_text }} / {{ profile.peak_error_text }}</div>
            </div>
            <div class="failure">{{ profile.failure_reason }}</div>
            <div class="links">
              <a href="{{ profile.summary_path }}">summary</a>
              <a href="{{ profile.case_path }}">case</a>
              <a href="{{ profile.gallery_path }}">gallery</a>
            </div>
          </div>
        </section>
        {% endfor %}
      </div>
    </article>
    {% endfor %}
  </section>

  <script>
    const searchInput = document.getElementById("search");
    const behaviorSelect = document.getElementById("behavior");
    const actorSelect = document.getElementById("actor");
    const bestProfileSelect = document.getElementById("best-profile");
    const statusSelect = document.getElementById("status");
    const cards = Array.from(document.querySelectorAll(".query-card"));
    const resultCount = document.getElementById("result-count");

    function applyFilters() {
      const search = searchInput.value.trim().toLowerCase();
      const behavior = behaviorSelect.value;
      const actor = actorSelect.value;
      const bestProfile = bestProfileSelect.value;
      const status = statusSelect.value;
      let shown = 0;

      cards.forEach((card) => {
        const matchesSearch = !search || card.dataset.search.includes(search);
        const matchesBehavior = !behavior || card.dataset.behaviors.split("|").includes(behavior);
        const matchesActor = !actor || card.dataset.actors.split("|").includes(actor);
        const matchesBestProfile = !bestProfile || card.dataset.bestProfile === bestProfile;
        const matchesStatus = !status || card.dataset.status.split("|").includes(status);
        const visible = matchesSearch && matchesBehavior && matchesActor && matchesBestProfile && matchesStatus;
        card.classList.toggle("hidden", !visible);
        if (visible) shown += 1;
      });

      resultCount.textContent = String(shown);
    }

    [searchInput, behaviorSelect, actorSelect, bestProfileSelect, statusSelect].forEach((node) => {
      node.addEventListener("input", applyFilters);
      node.addEventListener("change", applyFilters);
    });
  </script>
</body>
</html>
"""
)


def _load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _relative_path(base_dir: Path, target_path: Path) -> str:
    return Path(os.path.relpath(target_path.resolve(), start=base_dir.resolve())).as_posix()


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:  # noqa: BLE001
        return default


def _safe_int(value: object, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:  # noqa: BLE001
        return default


def _top_case_dir(query_dir: Path) -> Optional[Path]:
    rank_dirs = sorted(
        [path for path in query_dir.iterdir() if path.is_dir() and path.name.startswith("rank_")],
        key=lambda item: item.name,
    )
    return rank_dirs[0] if rank_dirs else None


def _case_preview(query_dir: Path, base_dir: Path) -> Dict[str, object]:
    top_case_dir = _top_case_dir(query_dir)
    preview: Dict[str, object] = {
        "query_text": query_dir.name.replace("_", " "),
        "scene_name": "unknown",
        "sample_idx": -1,
        "actor_name": "unknown",
        "validation_score": 0.0,
        "passed": False,
        "min_distance_m": 0.0,
        "min_ttc_s": 0.0,
        "event_iou_text": "-",
        "peak_error_text": "-",
        "case_path": "",
        "summary_path": "",
        "summary_html_path": "",
        "image_path": "",
        "tags": [],
        "notes": [],
    }
    summary_md = query_dir / "summary.md"
    summary_html = query_dir / "summary.html"
    if summary_md.exists():
        preview["summary_path"] = _relative_path(base_dir, summary_md)
    if summary_html.exists():
        preview["summary_html_path"] = _relative_path(base_dir, summary_html)
    if top_case_dir is None:
        return preview

    case_json = top_case_dir / "case.json"
    case_md = top_case_dir / "case.md"
    evidence_png = top_case_dir / "evidence.png"
    if case_md.exists():
        preview["case_path"] = _relative_path(base_dir, case_md)
    if evidence_png.exists():
        preview["image_path"] = _relative_path(base_dir, evidence_png)
    if not case_json.exists():
        return preview

    data = _load_json(case_json)
    candidate = dict(data.get("candidate") or {})
    evidence = dict(data.get("evidence") or {})
    query = dict(data.get("query") or {})
    event_localization = dict(data.get("event_localization") or {})

    preview.update(
        {
            "query_text": str(query.get("original_text") or preview["query_text"]),
            "scene_name": str(candidate.get("scene_name") or "unknown"),
            "sample_idx": _safe_int(candidate.get("sample_idx"), -1),
            "actor_name": str(candidate.get("category_name") or "unknown"),
            "validation_score": _safe_float(data.get("validation_score")),
            "passed": bool(data.get("passed")),
            "min_distance_m": _safe_float(evidence.get("min_distance_m")),
            "min_ttc_s": _safe_float(evidence.get("min_ttc_s")),
            "notes": list(data.get("notes") or []),
            "query_payload": query,
            "event_localization": event_localization,
        }
    )
    return preview


def _build_status_text(pass_at_1: bool, reference_mismatch: bool) -> str:
    items = ["pass" if pass_at_1 else "fail"]
    if reference_mismatch:
        items.append("mismatch")
    return "|".join(items)


def build_benchmark_gallery(
    benchmark_output_dir: Path,
    title: str = "nuScenes Benchmark Query Gallery",
) -> Dict[str, object]:
    benchmark_output_dir = benchmark_output_dir.resolve()
    summary_path = benchmark_output_dir / "benchmark_summary.json"
    metrics_path = benchmark_output_dir / "benchmark_metrics.json"
    summaries = list(_load_json(summary_path)) if summary_path.exists() else []
    metrics = _load_json(metrics_path) if metrics_path.exists() else {}
    query_metrics = {
        str(item.get("id") or ""): dict(item)
        for item in list(metrics.get("query_metrics") or [])
    }

    cards: List[Dict[str, object]] = []
    behaviors = set()
    actors = set()
    for item in summaries:
        query_id = str(item.get("id") or "")
        query_dir = Path(str(item.get("query_dir") or ""))
        preview = _case_preview(query_dir, benchmark_output_dir)
        query_metric = dict(query_metrics.get(query_id) or {})
        behavior_items = [str(value) for value in list(item.get("behaviors") or [])] or ["none"]
        actor_items = [str(value) for value in list(item.get("actors") or [])] or ["any"]
        tag_items = [str(value) for value in list(item.get("tags") or [])]
        behaviors.update(behavior_items)
        actors.update(actor_items)
        reference_mismatch = any(query_metric.get(name) is False for name in ["scene_objective_at_1", "actor_objective_at_1", "reference_objective_at_1"])

        cards.append(
            {
                "query_id": query_id,
                "description": str(item.get("description") or ""),
                "query_text": str(preview["query_text"]),
                "scene_name": str(preview["scene_name"]),
                "sample_idx": int(preview["sample_idx"]),
                "actor_name": str(preview["actor_name"]),
                "validation_score": float(preview["validation_score"]),
                "min_distance_m": float(preview["min_distance_m"]),
                "min_ttc_s": float(preview["min_ttc_s"]),
                "pass_at_1": bool(query_metric.get("pass_at_1", preview["passed"])),
                "reference_mismatch": reference_mismatch,
                "scene_objective_at_1": query_metric.get("scene_objective_at_1", "-"),
                "actor_objective_at_1": query_metric.get("actor_objective_at_1", "-"),
                "event_iou_text": "{0:.3f}".format(float(query_metric["event_iou"])) if query_metric.get("event_iou") is not None else "-",
                "peak_error_text": str(query_metric.get("peak_error")) if query_metric.get("peak_error") is not None else "-",
                "tags": tag_items,
                "image_path": str(preview["image_path"]),
                "case_path": str(preview["case_path"]),
                "summary_path": str(preview["summary_path"]),
                "summary_html_path": str(preview["summary_html_path"]),
                "search_text": " ".join(
                    [
                        query_id,
                        str(preview["query_text"]),
                        str(item.get("description") or ""),
                        str(preview["scene_name"]),
                        str(preview["actor_name"]),
                        " ".join(behavior_items),
                        " ".join(actor_items),
                        " ".join(tag_items),
                    ]
                ).lower(),
                "behavior_text": "|".join(behavior_items),
                "actor_text": "|".join(actor_items),
                "status_text": _build_status_text(bool(query_metric.get("pass_at_1", preview["passed"])), reference_mismatch),
            }
        )

    json_path = benchmark_output_dir / "query_gallery.json"
    html_path = benchmark_output_dir / "query_gallery.html"
    payload = {
        "title": title,
        "source_name": benchmark_output_dir.name,
        "query_count": len(cards),
        "cards": cards,
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    html_path.write_text(
        BENCHMARK_GALLERY_TEMPLATE.render(
            title=title,
            subtitle="Static query browser for {0}".format(benchmark_output_dir.name),
            source_name=benchmark_output_dir.name,
            cards=cards,
            behaviors=sorted(behaviors),
            actors=sorted(actors),
        ),
        encoding="utf-8",
    )
    return {
        "mode": "benchmark",
        "output_dir": str(benchmark_output_dir),
        "html_path": str(html_path),
        "json_path": str(json_path),
        "query_count": len(cards),
    }


def _profile_failure_reason(metrics: Dict[str, object]) -> str:
    reasons: List[str] = []
    if metrics.get("scene_objective_at_1") is False:
        reasons.append("scene mismatch")
    if metrics.get("actor_objective_at_1") is False:
        reasons.append("actor mismatch")
    if metrics.get("reference_objective_at_1") is False:
        reasons.append("anchor mismatch")
    if metrics.get("event_iou") is not None and float(metrics["event_iou"]) < 0.5:
        reasons.append("event drift")
    if metrics.get("peak_error") is not None and float(metrics["peak_error"]) > 2.0:
        reasons.append("peak shift")
    return ", ".join(reasons) if reasons else "reference-aligned"


def build_comparison_browser(
    comparison_output_dir: Path,
    title: str = "nuScenes Benchmark Comparison Browser",
) -> Dict[str, object]:
    comparison_output_dir = comparison_output_dir.resolve()
    comparison = _load_json(comparison_output_dir / "benchmark_profile_comparison.json")
    profile_rows = list(comparison.get("profiles") or [])
    profile_options = [{"name": str(row["name"]), "label": str(row["label"])} for row in profile_rows]
    profile_summary_maps: Dict[str, Dict[str, Dict[str, object]]] = {}

    for row in profile_rows:
        profile_output_dir = Path(str(row["output_dir"]))
        build_benchmark_gallery(
            benchmark_output_dir=profile_output_dir,
            title="nuScenes Query Gallery - {0}".format(str(row["label"])),
        )
        summary_entries = list(_load_json(profile_output_dir / "benchmark_summary.json"))
        profile_summary_maps[str(row["name"])] = {
            str(item.get("id") or ""): dict(item)
            for item in summary_entries
        }

    cards: List[Dict[str, object]] = []
    behaviors = set()
    actors = set()
    divergent_query_count = 0
    for row in list(comparison.get("query_comparison") or []):
        behavior_items = [str(value) for value in list(row.get("behaviors") or [])] or ["none"]
        actor_items = [str(value) for value in list(row.get("actors") or [])] or ["any"]
        behaviors.update(behavior_items)
        actors.update(actor_items)
        if row.get("signal_divergence"):
            divergent_query_count += 1

        profile_cards: List[Dict[str, object]] = []
        first_query_text = ""
        has_mismatch = False
        for profile in profile_rows:
            profile_name = str(profile["name"])
            profile_label = str(profile["label"])
            summary_entry = dict(profile_summary_maps.get(profile_name, {}).get(str(row["id"])) or {})
            query_dir = Path(str(summary_entry.get("query_dir") or ""))
            preview = _case_preview(query_dir, comparison_output_dir)
            metrics = dict(row["profiles"].get(profile_name) or {})
            if not first_query_text and preview.get("query_text"):
                first_query_text = str(preview["query_text"])
            reference_objective_at_1 = metrics.get("reference_objective_at_1")
            if reference_objective_at_1 is False:
                has_mismatch = True
            profile_cards.append(
                {
                    "name": profile_name,
                    "label": profile_label,
                    "image_path": str(preview["image_path"]),
                    "summary_path": str(preview["summary_html_path"] or preview["summary_path"]),
                    "case_path": str(preview["case_path"]),
                    "gallery_path": _relative_path(comparison_output_dir, Path(str(profile["output_dir"])) / "query_gallery.html"),
                    "scene_name": str(preview["scene_name"]),
                    "sample_idx": int(preview["sample_idx"]),
                    "actor_name": str(preview["actor_name"]),
                    "validation_score": float(metrics.get("best_validation_score") or preview["validation_score"]),
                    "min_distance_m": float(preview["min_distance_m"]),
                    "min_ttc_s": float(preview["min_ttc_s"]),
                    "scene_objective_at_1": metrics.get("scene_objective_at_1"),
                    "actor_objective_at_1": metrics.get("actor_objective_at_1"),
                    "reference_objective_at_1": reference_objective_at_1,
                    "event_iou_text": "{0:.3f}".format(float(metrics["event_iou"])) if metrics.get("event_iou") is not None else "-",
                    "peak_error_text": str(metrics.get("peak_error")) if metrics.get("peak_error") is not None else "-",
                    "failure_reason": _profile_failure_reason(metrics),
                }
            )

        cards.append(
            {
                "query_id": str(row["id"]),
                "query_text": first_query_text or str(row.get("description") or row.get("id") or "query"),
                "description": str(row.get("description") or ""),
                "tags": behavior_items + actor_items,
                "behaviors": behavior_items,
                "actors": actor_items,
                "best_profile": str(row.get("best_profile") or ""),
                "best_profile_label": next((item["label"] for item in profile_options if item["name"] == str(row.get("best_profile") or "")), str(row.get("best_profile") or "")),
                "signal_divergence": bool(row.get("signal_divergence")),
                "has_mismatch": has_mismatch,
                "score_span": float(row.get("score_span") or 0.0),
                "profiles": profile_cards,
                "search_text": " ".join(
                    [
                        str(row["id"]),
                        first_query_text,
                        str(row.get("description") or ""),
                        " ".join(behavior_items),
                        " ".join(actor_items),
                        " ".join(str(card["scene_name"]) for card in profile_cards),
                        " ".join(str(card["actor_name"]) for card in profile_cards),
                    ]
                ).lower(),
                "behavior_text": "|".join(behavior_items),
                "actor_text": "|".join(actor_items),
                "status_text": "|".join(
                    item
                    for item in [
                        "divergent" if bool(row.get("signal_divergence")) else "",
                        "mismatch" if has_mismatch else "",
                    ]
                    if item
                ),
            }
        )

    browser_payload = {
        "title": title,
        "query_count": len(cards),
        "profiles": profile_options,
        "cards": cards,
    }
    json_path = comparison_output_dir / "comparison_browser.json"
    html_path = comparison_output_dir / "comparison_browser.html"
    json_path.write_text(json.dumps(browser_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    top_profile = str((list(comparison.get("leaderboard") or [{}])[0]).get("label") or "")
    html_path.write_text(
        COMPARISON_BROWSER_TEMPLATE.render(
            title=title,
            subtitle="Side-by-side browser over profile outputs with direct links to per-query reports.",
            cards=cards,
            profile_labels=[item["label"] for item in profile_options],
            profile_options=profile_options,
            behaviors=sorted(behaviors),
            actors=sorted(actors),
            divergent_query_count=divergent_query_count,
            top_profile=top_profile,
        ),
        encoding="utf-8",
    )
    return {
        "mode": "comparison",
        "output_dir": str(comparison_output_dir),
        "html_path": str(html_path),
        "json_path": str(json_path),
        "query_count": len(cards),
        "profile_count": len(profile_options),
    }

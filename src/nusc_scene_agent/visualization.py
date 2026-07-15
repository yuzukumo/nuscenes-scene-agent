from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

from nusc_scene_agent.geometry import global_xy_to_anchor_ego, oriented_box_corners
from nusc_scene_agent.models import ValidatedCase


BACKGROUND = "#f5f1ea"
TARGET_COLOR = "#d95f02"
TARGET_LIGHT = "#f7a65a"
CONTEXT_COLOR = "#b8b5af"
EGO_COLOR = "#202124"
GRID_COLOR = "#d8d2ca"
DRIVABLE_COLOR = "#b9cde5"
LANE_COLOR = "#7f9fc4"
CONNECTOR_COLOR = "#93a8bf"
CROSSWALK_COLOR = "#5fb49c"
WALKWAY_COLOR = "#bed9a6"
STOP_LINE_COLOR = "#c94c4c"


def _draw_box(ax: plt.Axes, x: float, y: float, width: float, length: float, yaw: float, color: str, alpha: float, lw: float) -> None:
    corners = oriented_box_corners(x, y, width, length, yaw)
    patch = Polygon(corners, closed=True, facecolor=color, edgecolor=color, linewidth=lw, alpha=alpha)
    ax.add_patch(patch)


def _draw_map_layers(ax: plt.Axes, case: ValidatedCase) -> None:
    if not case.map_geometries:
        return

    for polygon in case.map_geometries.get("drivable_area", []):
        ax.fill(polygon[:, 0], polygon[:, 1], facecolor=DRIVABLE_COLOR, edgecolor="none", alpha=0.12, zorder=0)

    for polygon in case.map_geometries.get("lane", []):
        patch = Polygon(polygon, closed=True, facecolor=LANE_COLOR, edgecolor=LANE_COLOR, linewidth=0.8, alpha=0.10)
        patch.set_zorder(0.2)
        ax.add_patch(patch)

    for polygon in case.map_geometries.get("lane_connector", []):
        patch = Polygon(
            polygon,
            closed=True,
            facecolor=CONNECTOR_COLOR,
            edgecolor=CONNECTOR_COLOR,
            linewidth=0.8,
            alpha=0.10,
        )
        patch.set_zorder(0.22)
        ax.add_patch(patch)

    for polygon in case.map_geometries.get("walkway", []):
        patch = Polygon(polygon, closed=True, facecolor=WALKWAY_COLOR, edgecolor="none", linewidth=0.0, alpha=0.16)
        patch.set_zorder(0.24)
        ax.add_patch(patch)

    for polygon in case.map_geometries.get("ped_crossing", []):
        patch = Polygon(
            polygon,
            closed=True,
            facecolor=CROSSWALK_COLOR,
            edgecolor=CROSSWALK_COLOR,
            linewidth=0.8,
            alpha=0.24,
        )
        patch.set_zorder(0.26)
        ax.add_patch(patch)

    for polygon in case.map_geometries.get("stop_line", []):
        ax.plot(polygon[:, 0], polygon[:, 1], color=STOP_LINE_COLOR, linewidth=1.2, alpha=0.75, zorder=0.28)


def _draw_bev(ax: plt.Axes, case: ValidatedCase) -> None:
    ax.set_facecolor(BACKGROUND)
    ax.grid(True, color=GRID_COLOR, linewidth=0.6, alpha=0.7)
    ax.axhline(0.0, color="#99948c", linewidth=0.8, alpha=0.8)
    ax.axvline(0.0, color="#99948c", linewidth=0.8, alpha=0.8)
    _draw_map_layers(ax, case)

    anchor_row = case.timeline.loc[case.timeline["sample_token"] == case.candidate.sample_token].iloc[0]
    anchor_xy = np.asarray([anchor_row["ego_x"], anchor_row["ego_y"]], dtype=float)
    anchor_yaw = float(anchor_row["ego_yaw"])
    ego_points_global = case.ego_window[["ego_x", "ego_y"]].to_numpy(dtype=float)
    ego_points = global_xy_to_anchor_ego(ego_points_global, anchor_xy, anchor_yaw)
    ax.plot(ego_points[:, 0], ego_points[:, 1], color=EGO_COLOR, linewidth=2.2, label="ego trajectory")

    for _, row in case.context_agents.head(32).iterrows():
        if row["instance_token"] == case.candidate.instance_token:
            continue
        _draw_box(
            ax,
            float(row["x_ego"]),
            float(row["y_ego"]),
            float(row["width"]),
            float(row["length"]),
            float(row["yaw_ego"]),
            CONTEXT_COLOR,
            0.28,
            0.8,
        )

    ordered_track = case.timeline.sort_values("sample_idx")
    color_values = np.linspace(0.35, 1.0, len(ordered_track))
    target_points = ordered_track[["x_ego", "y_ego"]].to_numpy(dtype=float)
    ax.plot(target_points[:, 0], target_points[:, 1], color=TARGET_COLOR, linewidth=2.0, alpha=0.9, label="target actor")

    for shade, (_, row) in zip(color_values, ordered_track.iterrows()):
        highlight = 2.0 if row["sample_token"] == case.candidate.sample_token else 1.0
        color = TARGET_COLOR if row["sample_token"] == case.candidate.sample_token else TARGET_LIGHT
        _draw_box(
            ax,
            float(row["x_ego"]),
            float(row["y_ego"]),
            float(row["width"]),
            float(row["length"]),
            float(row["yaw_ego"]),
            color,
            float(shade) * 0.8,
            highlight,
        )

    _draw_box(ax, 0.0, 0.0, 2.0, 4.8, 0.0, EGO_COLOR, 0.85, 1.6)
    ax.scatter([0.0], [0.0], color=EGO_COLOR, s=42)
    ax.scatter(target_points[:, 0], target_points[:, 1], color=TARGET_COLOR, s=18, zorder=5)

    ax.set_xlim(-40.0, 40.0)
    ax.set_ylim(-30.0, 30.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Forward / x_ego (m)")
    ax.set_ylabel("Left / y_ego (m)")
    ax.set_title("{0} | sample {1}".format(case.candidate.scene_name, case.candidate.sample_idx))
    ax.legend(loc="upper right", frameon=False)


def _draw_distance_panel(ax: plt.Axes, case: ValidatedCase) -> None:
    ordered_track = case.timeline.sort_values("sample_idx")
    ax.plot(ordered_track["t_sec"], ordered_track["distance"], color=TARGET_COLOR, linewidth=2.2)
    ax.axhline(case.query.near_distance_m, color="#7a6f65", linestyle="--", linewidth=1.2)
    ax.fill_between(
        ordered_track["t_sec"].to_numpy(dtype=float),
        ordered_track["distance"].to_numpy(dtype=float),
        case.query.near_distance_m,
        color=TARGET_LIGHT,
        alpha=0.18,
    )
    ax.set_title("Distance Timeline")
    ax.set_xlabel("Relative time (s)")
    ax.set_ylabel("Distance (m)")
    ax.grid(True, alpha=0.25)

    ttc_values = ordered_track["ttc"].replace({None: np.nan}).dropna()
    if not ttc_values.empty:
        ax2 = ax.twinx()
        ttc_series = ordered_track["ttc"].astype(float).clip(upper=case.query.max_ttc_s * 2.0)
        ax2.plot(ordered_track["t_sec"], ttc_series, color="#4e79a7", linewidth=1.8, alpha=0.9)
        ax2.set_ylabel("TTC (s)")


def _draw_text_panel(ax: plt.Axes, case: ValidatedCase) -> None:
    ax.axis("off")
    query_tags = ", ".join(case.query.category_groups + case.query.positions + case.query.behaviors) or "generic risk"
    header = "Validation quality: {0:.1f} | {1}".format(
        case.validation_quality_score, "passed" if case.passed else "candidate"
    )
    lines = [
        header,
        "Query tags: {0}".format(query_tags),
        "Actor: {0}".format(case.candidate.category_name),
        "Location: {0}".format(case.candidate.location),
        "Min distance: {0:.2f} m".format(case.evidence.get("min_distance_m", float("nan"))),
    ]
    min_ttc = case.evidence.get("min_ttc_s")
    if min_ttc is not None:
        lines.append("Min TTC: {0:.2f} s".format(min_ttc))
    if case.behavior_matches:
        lines.extend(
            ["{0}: {1}".format(name, "yes" if matched else "no") for name, matched in sorted(case.behavior_matches.items())]
        )
    if case.map_context.get("available"):
        lines.append("Map lane share: {0}".format("yes" if case.map_context.get("shares_lane_at_anchor") else "no"))
        lines.append("Map crosswalk: {0}".format("yes" if case.map_context.get("actor_on_crosswalk_any") else "no"))
    lines.extend(case.notes[:3])
    ax.text(
        0.0,
        1.0,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=11,
        family="monospace",
    )


def render_case_figure(case: ValidatedCase, output_path: Path) -> Path:
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(14, 8))
    grid = fig.add_gridspec(2, 2, width_ratios=[2.2, 1.0], height_ratios=[1.0, 1.0], wspace=0.24, hspace=0.25)
    ax_bev = fig.add_subplot(grid[:, 0])
    ax_distance = fig.add_subplot(grid[0, 1])
    ax_text = fig.add_subplot(grid[1, 1])

    _draw_bev(ax_bev, case)
    _draw_distance_panel(ax_distance, case)
    _draw_text_panel(ax_text, case)

    fig.suptitle("nuScenes Scene Mining Evidence", fontsize=16, y=0.98)
    fig.savefig(str(output_path), dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path

from __future__ import annotations

import math
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from nuscenes.nuscenes import NuScenes

from nusc_scene_agent.geometry import build_world_to_ego_matrix, global_to_ego_point, normalize_angle
from nusc_scene_agent.geometry import velocity_world_to_ego, yaw_from_rotation


INDEX_SCHEMA_NAME = "nusc_scene_agent_index"
INDEX_SCHEMA_VERSION = "1"
INDEX_SCHEMA_COMPATIBLE_VERSIONS = {INDEX_SCHEMA_VERSION}

INDEX_REQUIRED_COLUMNS = {
    "samples": {
        "sample_token",
        "scene_token",
        "scene_name",
        "scene_description",
        "sample_idx",
        "timestamp_us",
        "location",
        "ego_x",
        "ego_y",
        "ego_yaw",
    },
    "agents": {
        "ann_token",
        "sample_token",
        "scene_token",
        "scene_name",
        "sample_idx",
        "instance_token",
        "category_name",
        "category_group",
        "x_ego",
        "y_ego",
        "z_ego",
        "distance",
        "visibility",
        "ttc",
        "speed",
        "rel_vx",
        "rel_vy",
        "heading_delta",
        "num_lidar_pts",
        "num_radar_pts",
        "width",
        "length",
        "height",
        "is_front",
        "is_rear",
        "is_left",
        "is_right",
        "is_stationary",
    },
}


def read_index_metadata(conn: sqlite3.Connection) -> Dict[str, str]:
    try:
        rows = conn.execute("SELECT key, value FROM metadata").fetchall()
    except sqlite3.Error as exc:
        raise RuntimeError("SQLite index does not contain readable metadata.") from exc
    return {str(key): str(value) for key, value in rows}


def validate_index_schema(conn: sqlite3.Connection) -> Dict[str, str]:
    metadata = read_index_metadata(conn)
    schema_name = metadata.get("schema", "")
    schema_version = metadata.get("schema_version", "")
    if schema_name != INDEX_SCHEMA_NAME:
        raise RuntimeError(
            "Unsupported SQLite index schema: {0!r}; expected {1!r}.".format(schema_name, INDEX_SCHEMA_NAME)
        )
    if schema_version not in INDEX_SCHEMA_COMPATIBLE_VERSIONS:
        raise RuntimeError(
            "Unsupported SQLite index schema version: {0!r}; compatible versions: {1}. Rebuild the index."
            .format(schema_version, sorted(INDEX_SCHEMA_COMPATIBLE_VERSIONS))
        )
    if metadata.get("build_complete") != "true":
        raise RuntimeError("SQLite index metadata does not mark the build as complete. Rebuild or migrate the index.")
    _validate_required_columns(conn)
    return metadata


def _validate_required_columns(conn: sqlite3.Connection) -> None:
    for table, required in INDEX_REQUIRED_COLUMNS.items():
        available = {str(row[1]) for row in conn.execute("PRAGMA table_info({0})".format(table)).fetchall()}
        missing = sorted(required - available)
        if missing:
            raise RuntimeError(
                "SQLite index table {0!r} is missing required columns: {1}. Rebuild the index.".format(
                    table,
                    ", ".join(missing),
                )
            )


def migrate_index_schema(db_path: Path) -> Dict[str, str]:
    """Stamp a structurally compatible pre-versioned index as schema version 1."""
    db_path = Path(db_path).resolve()
    if not db_path.exists():
        raise FileNotFoundError("Missing SQLite index: {0}".format(db_path))
    conn = sqlite3.connect(str(db_path))
    try:
        metadata = read_index_metadata(conn)
        if metadata.get("schema") == INDEX_SCHEMA_NAME:
            return validate_index_schema(conn)
        if metadata.get("schema"):
            raise RuntimeError("Cannot migrate unknown SQLite index schema: {0!r}.".format(metadata["schema"]))
        _validate_required_columns(conn)
        integrity = str(conn.execute("PRAGMA integrity_check;").fetchone()[0])
        if integrity.lower() != "ok":
            raise RuntimeError("SQLite integrity check failed: {0}".format(integrity))
        counts = {
            "sample_count": str(conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0]),
            "agent_count": str(conn.execute("SELECT COUNT(*) FROM agents").fetchone()[0]),
        }
        migrated_at = datetime.now(timezone.utc).isoformat()
        conn.execute("BEGIN IMMEDIATE")
        conn.executemany(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
            [
                ("schema", INDEX_SCHEMA_NAME),
                ("schema_version", INDEX_SCHEMA_VERSION),
                ("build_complete", "true"),
                ("migrated_at_utc", migrated_at),
                *counts.items(),
            ],
        )
        conn.commit()
        return validate_index_schema(conn)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def simplify_category(category_name: str) -> str:
    if category_name.startswith("human.pedestrian"):
        return "pedestrian"
    if category_name.startswith("vehicle.bicycle"):
        return "bicycle"
    if category_name.startswith("vehicle.motorcycle"):
        return "motorcycle"
    if category_name.startswith("vehicle.bus"):
        return "bus"
    if (
        category_name.startswith("vehicle.truck")
        or category_name.startswith("vehicle.trailer")
        or category_name.startswith("vehicle.construction")
    ):
        return "truck"
    if category_name.startswith("vehicle."):
        return "vehicle"
    return category_name.split(".")[0]


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _build_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        DROP TABLE IF EXISTS metadata;
        DROP TABLE IF EXISTS samples;
        DROP TABLE IF EXISTS agents;

        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE samples (
            sample_token TEXT PRIMARY KEY,
            scene_token TEXT NOT NULL,
            scene_name TEXT NOT NULL,
            scene_description TEXT,
            sample_idx INTEGER NOT NULL,
            timestamp_us INTEGER NOT NULL,
            ego_x REAL NOT NULL,
            ego_y REAL NOT NULL,
            ego_z REAL NOT NULL,
            ego_yaw REAL NOT NULL,
            ego_speed REAL NOT NULL,
            location TEXT NOT NULL
        );

        CREATE TABLE agents (
            ann_token TEXT PRIMARY KEY,
            sample_token TEXT NOT NULL,
            scene_token TEXT NOT NULL,
            scene_name TEXT NOT NULL,
            sample_idx INTEGER NOT NULL,
            instance_token TEXT NOT NULL,
            category_name TEXT NOT NULL,
            category_group TEXT NOT NULL,
            visibility INTEGER NOT NULL,
            num_lidar_pts INTEGER NOT NULL,
            num_radar_pts INTEGER NOT NULL,
            width REAL NOT NULL,
            length REAL NOT NULL,
            height REAL NOT NULL,
            x_ego REAL NOT NULL,
            y_ego REAL NOT NULL,
            z_ego REAL NOT NULL,
            distance REAL NOT NULL,
            yaw_ego REAL NOT NULL,
            heading_delta REAL NOT NULL,
            speed REAL NOT NULL,
            actor_vx REAL NOT NULL,
            actor_vy REAL NOT NULL,
            rel_vx REAL NOT NULL,
            rel_vy REAL NOT NULL,
            rel_speed REAL NOT NULL,
            ttc REAL,
            is_front INTEGER NOT NULL,
            is_rear INTEGER NOT NULL,
            is_left INTEGER NOT NULL,
            is_right INTEGER NOT NULL,
            is_stationary INTEGER NOT NULL,
            is_vru INTEGER NOT NULL,
            FOREIGN KEY(sample_token) REFERENCES samples(sample_token)
        );

        CREATE INDEX idx_samples_scene ON samples(scene_token, sample_idx);
        CREATE INDEX idx_agents_scene ON agents(scene_token, sample_idx);
        CREATE INDEX idx_agents_sample ON agents(sample_token);
        CREATE INDEX idx_agents_instance ON agents(instance_token);
        CREATE INDEX idx_agents_category ON agents(category_group);
        CREATE INDEX idx_agents_distance ON agents(distance);
        """
    )


def _compute_ttc(x_ego: float, rel_vx: float) -> float:
    if x_ego <= 0.0 or rel_vx >= -0.2:
        return float("inf")
    return float(x_ego / abs(rel_vx))


def _sample_pose(
    nusc: NuScenes, sample_token: str, cache: Dict[str, Tuple[np.ndarray, float]]
) -> Tuple[np.ndarray, float]:
    cached = cache.get(sample_token)
    if cached is not None:
        return cached

    sample = nusc.get("sample", sample_token)
    lidar_token = sample["data"]["LIDAR_TOP"]
    sample_data = nusc.get("sample_data", lidar_token)
    ego_pose = nusc.get("ego_pose", sample_data["ego_pose_token"])
    translation = np.asarray(ego_pose["translation"], dtype=float)
    timestamp_s = float(sample["timestamp"]) / 1_000_000.0
    cache[sample_token] = (translation, timestamp_s)
    return cache[sample_token]


def _estimate_ego_velocity(
    nusc: NuScenes, sample: Dict[str, object], cache: Dict[str, Tuple[np.ndarray, float]]
) -> np.ndarray:
    prev_token = sample["prev"]
    next_token = sample["next"]
    current_translation, current_t = _sample_pose(nusc, sample["token"], cache)

    if prev_token and next_token:
        prev_translation, prev_t = _sample_pose(nusc, prev_token, cache)
        next_translation, next_t = _sample_pose(nusc, next_token, cache)
        delta_t = next_t - prev_t
        if delta_t > 0.0:
            return (next_translation - prev_translation) / delta_t

    if next_token:
        next_translation, next_t = _sample_pose(nusc, next_token, cache)
        delta_t = next_t - current_t
        if delta_t > 0.0:
            return (next_translation - current_translation) / delta_t

    if prev_token:
        prev_translation, prev_t = _sample_pose(nusc, prev_token, cache)
        delta_t = current_t - prev_t
        if delta_t > 0.0:
            return (current_translation - prev_translation) / delta_t

    return np.zeros(3, dtype=float)


def _sample_location(nusc: NuScenes, scene: Dict[str, object], cache: Dict[str, str]) -> str:
    scene_token = str(scene["token"])
    if scene_token in cache:
        return cache[scene_token]
    log = nusc.get("log", scene["log_token"])
    cache[scene_token] = str(log["location"])
    return cache[scene_token]


def _flush_rows(
    conn: sqlite3.Connection, sample_rows: List[Tuple[object, ...]], agent_rows: List[Tuple[object, ...]]
) -> None:
    if sample_rows:
        conn.executemany(
            """
            INSERT INTO samples (
                sample_token, scene_token, scene_name, scene_description, sample_idx, timestamp_us,
                ego_x, ego_y, ego_z, ego_yaw, ego_speed, location
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            sample_rows,
        )
        sample_rows.clear()

    if agent_rows:
        conn.executemany(
            """
            INSERT INTO agents (
                ann_token, sample_token, scene_token, scene_name, sample_idx, instance_token,
                category_name, category_group, visibility, num_lidar_pts, num_radar_pts,
                width, length, height, x_ego, y_ego, z_ego, distance, yaw_ego, heading_delta,
                speed, actor_vx, actor_vy, rel_vx, rel_vy, rel_speed, ttc,
                is_front, is_rear, is_left, is_right, is_stationary, is_vru
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            agent_rows,
        )
        agent_rows.clear()


def build_index(
    version: str,
    dataroot: Path,
    db_path: Path,
    scene_limit: int = 0,
    verbose: bool = True,
) -> Dict[str, int]:
    dataroot = dataroot.resolve()
    db_path = db_path.resolve()
    if not (dataroot / version).exists():
        raise FileNotFoundError("Missing nuScenes tables under {0}".format(dataroot / version))

    _ensure_parent(db_path)
    nusc = NuScenes(version=version, dataroot=str(dataroot), verbose=verbose)

    build_path = db_path.with_name(db_path.name + ".building")
    for path in [build_path, Path(str(build_path) + "-wal"), Path(str(build_path) + "-shm")]:
        path.unlink(missing_ok=True)

    conn = sqlite3.connect(str(build_path))
    conn.execute("PRAGMA journal_mode=DELETE;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    _build_schema(conn)
    conn.executemany(
        "INSERT INTO metadata(key, value) VALUES (?, ?)",
        [
            ("schema", INDEX_SCHEMA_NAME),
            ("schema_version", INDEX_SCHEMA_VERSION),
            ("version", version),
            ("dataroot", str(dataroot)),
            ("build_started_at_utc", datetime.now(timezone.utc).isoformat()),
        ],
    )

    pose_cache: Dict[str, Tuple[np.ndarray, float]] = {}
    location_cache: Dict[str, str] = {}
    sample_rows: List[Tuple[object, ...]] = []
    agent_rows: List[Tuple[object, ...]] = []

    scenes = nusc.scene[:scene_limit] if scene_limit else nusc.scene
    stats = {"scene_count": len(scenes), "sample_count": 0, "agent_count": 0}

    for scene in scenes:
        scene_name = str(scene["name"])
        scene_description = str(scene.get("description") or "")
        scene_token = str(scene["token"])
        location = _sample_location(nusc, scene, location_cache)

        sample_token = scene["first_sample_token"]
        sample_idx = 0

        while sample_token:
            sample = nusc.get("sample", sample_token)
            lidar_token = sample["data"]["LIDAR_TOP"]
            sample_data = nusc.get("sample_data", lidar_token)
            ego_pose = nusc.get("ego_pose", sample_data["ego_pose_token"])
            ego_speed_global = _estimate_ego_velocity(nusc, sample, pose_cache)
            ego_speed = float(np.linalg.norm(ego_speed_global[:2]))
            ego_yaw = yaw_from_rotation(ego_pose["rotation"])
            world_to_ego = build_world_to_ego_matrix(ego_pose["translation"], ego_pose["rotation"])

            sample_rows.append(
                (
                    str(sample["token"]),
                    scene_token,
                    scene_name,
                    scene_description,
                    sample_idx,
                    int(sample["timestamp"]),
                    float(ego_pose["translation"][0]),
                    float(ego_pose["translation"][1]),
                    float(ego_pose["translation"][2]),
                    float(ego_yaw),
                    ego_speed,
                    location,
                )
            )
            stats["sample_count"] += 1

            for ann_token in sample["anns"]:
                ann = nusc.get("sample_annotation", ann_token)
                center_ego = global_to_ego_point(ann["translation"], world_to_ego)
                actor_velocity = np.asarray(nusc.box_velocity(ann_token)[:2], dtype=float)
                if np.isnan(actor_velocity).any():
                    actor_velocity = np.zeros(2, dtype=float)

                actor_velocity_ego = velocity_world_to_ego(actor_velocity, ego_yaw)
                rel_velocity_ego = velocity_world_to_ego(actor_velocity - ego_speed_global[:2], ego_yaw)
                actor_yaw = yaw_from_rotation(ann["rotation"])
                yaw_ego = normalize_angle(actor_yaw - ego_yaw)
                x_ego = float(center_ego[0])
                y_ego = float(center_ego[1])
                distance = float(math.hypot(x_ego, y_ego))
                speed = float(np.linalg.norm(actor_velocity))
                rel_speed = float(np.linalg.norm(rel_velocity_ego))
                ttc = _compute_ttc(x_ego, float(rel_velocity_ego[0]))
                category_name = str(ann["category_name"])
                category_group = simplify_category(category_name)
                is_vru = 1 if category_group in {"pedestrian", "bicycle", "motorcycle"} else 0
                width, length, height = [float(value) for value in ann["size"]]

                agent_rows.append(
                    (
                        str(ann["token"]),
                        str(sample["token"]),
                        scene_token,
                        scene_name,
                        sample_idx,
                        str(ann["instance_token"]),
                        category_name,
                        category_group,
                        int(ann["visibility_token"]),
                        int(ann["num_lidar_pts"]),
                        int(ann["num_radar_pts"]),
                        width,
                        length,
                        height,
                        x_ego,
                        y_ego,
                        float(center_ego[2]),
                        distance,
                        yaw_ego,
                        yaw_ego,
                        speed,
                        float(actor_velocity_ego[0]),
                        float(actor_velocity_ego[1]),
                        float(rel_velocity_ego[0]),
                        float(rel_velocity_ego[1]),
                        rel_speed,
                        None if math.isinf(ttc) else ttc,
                        1 if x_ego >= 0.0 else 0,
                        1 if x_ego < 0.0 else 0,
                        1 if y_ego >= 0.5 else 0,
                        1 if y_ego <= -0.5 else 0,
                        1 if speed <= 1.0 else 0,
                        is_vru,
                    )
                )
                stats["agent_count"] += 1

            if len(sample_rows) >= 256:
                _flush_rows(conn, sample_rows, agent_rows)

            sample_token = sample["next"]
            sample_idx += 1

        _flush_rows(conn, sample_rows, agent_rows)
        conn.commit()

    conn.executemany(
        "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
        [
            ("scene_count", str(stats["scene_count"])),
            ("sample_count", str(stats["sample_count"])),
            ("agent_count", str(stats["agent_count"])),
            ("build_complete", "true"),
            ("build_completed_at_utc", datetime.now(timezone.utc).isoformat()),
        ],
    )
    conn.commit()
    integrity = str(conn.execute("PRAGMA integrity_check;").fetchone()[0])
    conn.close()
    if integrity.lower() != "ok":
        build_path.unlink(missing_ok=True)
        raise RuntimeError("SQLite integrity check failed: {0}".format(integrity))
    os.replace(build_path, db_path)
    return stats

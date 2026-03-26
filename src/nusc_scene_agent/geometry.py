from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from pyquaternion import Quaternion


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def yaw_from_rotation(rotation: Sequence[float]) -> float:
    return Quaternion(rotation).yaw_pitch_roll[0]


def build_world_to_ego_matrix(translation: Sequence[float], rotation: Sequence[float]) -> np.ndarray:
    quat = Quaternion(rotation)
    rot_inv = quat.rotation_matrix.T
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rot_inv
    transform[:3, 3] = -rot_inv @ np.asarray(translation, dtype=float)
    return transform


def global_to_ego_point(point_xyz: Sequence[float], world_to_ego: np.ndarray) -> np.ndarray:
    point = np.asarray([point_xyz[0], point_xyz[1], point_xyz[2], 1.0], dtype=float)
    return (world_to_ego @ point)[:3]


def rotation_world_to_ego(ego_yaw: float) -> np.ndarray:
    cos_yaw = math.cos(ego_yaw)
    sin_yaw = math.sin(ego_yaw)
    return np.asarray(
        [
            [cos_yaw, sin_yaw],
            [-sin_yaw, cos_yaw],
        ],
        dtype=float,
    )


def velocity_world_to_ego(velocity_xy: Sequence[float], ego_yaw: float) -> np.ndarray:
    velocity_xy = np.asarray(velocity_xy[:2], dtype=float)
    return rotation_world_to_ego(ego_yaw) @ velocity_xy


def ego_xy_to_global(point_xy: Sequence[float], ego_xy: Sequence[float], ego_yaw: float) -> np.ndarray:
    point_xy = np.asarray(point_xy[:2], dtype=float)
    ego_xy = np.asarray(ego_xy[:2], dtype=float)
    cos_yaw = math.cos(ego_yaw)
    sin_yaw = math.sin(ego_yaw)
    rotation = np.asarray([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]], dtype=float)
    return ego_xy + rotation @ point_xy


def global_xy_to_anchor_ego(points_xy: np.ndarray, anchor_xy: Sequence[float], anchor_yaw: float) -> np.ndarray:
    shifted = np.asarray(points_xy, dtype=float) - np.asarray(anchor_xy[:2], dtype=float)
    rotation = rotation_world_to_ego(anchor_yaw)
    return (rotation @ shifted.T).T


def oriented_box_corners(x: float, y: float, width: float, length: float, yaw: float) -> np.ndarray:
    corners = np.asarray(
        [
            [-length / 2.0, -width / 2.0],
            [length / 2.0, -width / 2.0],
            [length / 2.0, width / 2.0],
            [-length / 2.0, width / 2.0],
            [-length / 2.0, -width / 2.0],
        ],
        dtype=float,
    )
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    rotation = np.asarray([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]], dtype=float)
    rotated = corners @ rotation.T
    rotated[:, 0] += x
    rotated[:, 1] += y
    return rotated

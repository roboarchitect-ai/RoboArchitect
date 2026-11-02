from __future__ import annotations

import numpy as np


def normalize(value: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    norm = np.linalg.norm(array, axis=-1, keepdims=True)
    if np.any(norm < eps):
        raise ValueError("cannot normalize a near-zero vector")
    return array / norm


def quaternion_xyzw_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    q = normalize(quaternion)
    x, y, z, w = np.moveaxis(q, -1, 0)
    return np.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape(q.shape[:-1] + (3, 3))


def matrix_to_euler_xyz(matrix: np.ndarray) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float64)
    y = np.arcsin(np.clip(-value[..., 2, 0], -1.0, 1.0))
    regular = np.abs(np.cos(y)) > 1e-7
    x = np.where(regular, np.arctan2(value[..., 2, 1], value[..., 2, 2]), 0.0)
    z = np.where(
        regular,
        np.arctan2(value[..., 1, 0], value[..., 0, 0]),
        np.arctan2(-value[..., 0, 1], value[..., 1, 1]),
    )
    return np.stack((x, y, z), axis=-1)


def matrix_to_quaternion_xyzw(matrix: np.ndarray) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float64)
    result: list[np.ndarray] = []
    for m in value.reshape((-1, 3, 3)):
        candidates = np.array(
            [
                1 + m[0, 0] - m[1, 1] - m[2, 2],
                1 - m[0, 0] + m[1, 1] - m[2, 2],
                1 - m[0, 0] - m[1, 1] + m[2, 2],
                1 + np.trace(m),
            ]
        )
        index = int(np.argmax(candidates))
        q = np.zeros(4, dtype=np.float64)
        q[index] = 0.5 * np.sqrt(max(candidates[index], 0.0))
        scale = 0.25 / max(q[index], 1e-12)
        if index == 0:
            q[1:] = ((m[0, 1] + m[1, 0]) * scale, (m[0, 2] + m[2, 0]) * scale, (m[2, 1] - m[1, 2]) * scale)
        elif index == 1:
            q[0], q[2], q[3] = (m[0, 1] + m[1, 0]) * scale, (m[1, 2] + m[2, 1]) * scale, (m[0, 2] - m[2, 0]) * scale
        elif index == 2:
            q[0], q[1], q[3] = (m[0, 2] + m[2, 0]) * scale, (m[1, 2] + m[2, 1]) * scale, (m[1, 0] - m[0, 1]) * scale
        else:
            q[:3] = ((m[2, 1] - m[1, 2]) * scale, (m[0, 2] - m[2, 0]) * scale, (m[1, 0] - m[0, 1]) * scale)
        result.append(normalize(q))
    return np.asarray(result).reshape(value.shape[:-2] + (4,))


def matrix_to_axis_angle(matrix: np.ndarray) -> np.ndarray:
    quaternion = matrix_to_quaternion_xyzw(matrix)
    xyz = quaternion[..., :3]
    w = np.clip(quaternion[..., 3], -1.0, 1.0)
    norm = np.linalg.norm(xyz, axis=-1)
    angle = 2 * np.arctan2(norm, w)
    scale = np.divide(angle, norm, out=np.full_like(angle, 2.0), where=norm > 1e-8)
    return xyz * scale[..., None]


def matrix_to_rotation6d(matrix: np.ndarray) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float64)
    return value[..., :, :2].reshape(value.shape[:-2] + (6,))


def rotation6d_to_matrix(rotation: np.ndarray) -> np.ndarray:
    value = np.asarray(rotation, dtype=np.float64)
    if value.shape[-1] != 6:
        raise ValueError("rotation6d must have width 6")
    first = normalize(value[..., 0:5:2])
    second_raw = value[..., 1:6:2]
    second = normalize(second_raw - np.sum(first * second_raw, axis=-1, keepdims=True) * first)
    return np.stack((first, second, np.cross(first, second)), axis=-1)


def rotation6d_columns_to_matrix(rotation: np.ndarray) -> np.ndarray:
    value = np.asarray(rotation, dtype=np.float64)
    if value.shape[-1] != 6:
        raise ValueError("rotation6d must have width 6")
    first = normalize(value[..., :3])
    second_raw = value[..., 3:]
    second = normalize(second_raw - np.sum(first * second_raw, axis=-1, keepdims=True) * first)
    return np.stack((first, second, np.cross(first, second)), axis=-1)


def quaternion_xyzw_to_rotation6d(value: np.ndarray) -> np.ndarray:
    return matrix_to_rotation6d(quaternion_xyzw_to_matrix(value))


def rotation6d_to_euler_xyz(value: np.ndarray) -> np.ndarray:
    return matrix_to_euler_xyz(rotation6d_to_matrix(value))


def rotation6d_columns_to_axis_angle(value: np.ndarray) -> np.ndarray:
    return matrix_to_axis_angle(rotation6d_columns_to_matrix(value))

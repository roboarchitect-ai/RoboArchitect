from __future__ import annotations

from typing import Any

import numpy as np


__all__ = [
    "depth_at",
    "has_3d",
    "pixel_to_camera",
    "pixel_to_world",
    "point_cloud",
    "world_to_camera",
    "world_to_pixel",
]


def has_3d(camera: Any) -> bool:
    return (
        getattr(camera, "depth_m", None) is not None
        and getattr(camera, "depth_valid", None) is not None
        and getattr(camera, "intrinsics", None) is not None
        and getattr(camera, "camera_to_world", None) is not None
    )


def _require_3d(camera: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    if not has_3d(camera):
        raise ValueError("camera has no 3D sensing on this setup (depth/intrinsics/pose are None)")
    return (
        np.asarray(camera.depth_m, dtype=np.float64),
        np.asarray(camera.depth_valid, dtype=bool),
        np.asarray(camera.intrinsics, dtype=np.float64),
        np.asarray(camera.camera_to_world, dtype=np.float64),
        str(camera.optical_convention),
    )


def depth_at(camera: Any, u: float, v: float, radius: int = 2) -> float | None:
    if not has_3d(camera):
        return None
    depth = np.asarray(camera.depth_m, dtype=np.float64)
    valid = np.asarray(camera.depth_valid, dtype=bool)
    height, width = depth.shape
    column = int(round(float(u)))
    row = int(round(float(v)))
    if not (0 <= column < width and 0 <= row < height):
        return None
    reach = max(0, int(radius))
    window = depth[
        max(0, row - reach) : min(height, row + reach + 1),
        max(0, column - reach) : min(width, column + reach + 1),
    ]
    window_valid = valid[
        max(0, row - reach) : min(height, row + reach + 1),
        max(0, column - reach) : min(width, column + reach + 1),
    ]
    usable = window[window_valid & np.isfinite(window) & (window > 0.0)]
    if usable.size == 0:
        return None
    return float(np.median(usable))


def pixel_to_camera(camera: Any, u: float, v: float, depth: float) -> np.ndarray:
    _, _, intrinsics, _, convention = _require_3d(camera)
    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])
    distance = float(depth)
    x = (float(u) - cx) * distance / fx
    y = (float(v) - cy) * distance / fy
    if convention == "opencv_rdf":
        return np.array((x, y, distance), dtype=np.float64)
    if convention == "opengl_rub":
        return np.array((x, y, -distance), dtype=np.float64)
    raise ValueError(f"unsupported optical convention {convention!r}")


def pixel_to_world(camera: Any, u: float, v: float, depth: float | None = None, radius: int = 2) -> np.ndarray | None:
    if not has_3d(camera):
        return None
    distance = depth_at(camera, u, v, radius) if depth is None else float(depth)
    if distance is None or not np.isfinite(distance) or distance <= 0.0:
        return None
    point = pixel_to_camera(camera, u, v, distance)
    transform = np.asarray(camera.camera_to_world, dtype=np.float64)
    world = transform @ np.array((point[0], point[1], point[2], 1.0), dtype=np.float64)
    return np.asarray(world[:3], dtype=np.float64)


def world_to_camera(camera: Any, point: Any) -> np.ndarray:
    _, _, _, transform, _ = _require_3d(camera)
    value = np.asarray(point, dtype=np.float64).reshape(3)
    inverse = np.linalg.inv(transform)
    camera_point = inverse @ np.array((value[0], value[1], value[2], 1.0), dtype=np.float64)
    return np.asarray(camera_point[:3], dtype=np.float64)


def world_to_pixel(camera: Any, point: Any) -> tuple[float, float] | None:
    if not has_3d(camera):
        return None
    _, _, intrinsics, _, convention = _require_3d(camera)
    local = world_to_camera(camera, point)
    if convention == "opencv_rdf":
        distance = float(local[2])
    elif convention == "opengl_rub":
        distance = -float(local[2])
    else:
        raise ValueError(f"unsupported optical convention {convention!r}")
    if not np.isfinite(distance) or distance <= 1e-9:
        return None
    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])
    return (float(local[0]) * fx / distance + cx, float(local[1]) * fy / distance + cy)


def point_cloud(camera: Any, stride: int = 4, mask: Any = None) -> np.ndarray | None:
    if not has_3d(camera):
        return None
    depth, valid, intrinsics, transform, convention = _require_3d(camera)
    step = max(1, int(stride))
    height, width = depth.shape
    rows = np.arange(0, height, step)
    columns = np.arange(0, width, step)
    grid_v, grid_u = np.meshgrid(rows, columns, indexing="ij")
    sampled_depth = depth[::step, ::step]
    keep = valid[::step, ::step] & np.isfinite(sampled_depth) & (sampled_depth > 0.0)
    if mask is not None:
        keep = keep & np.asarray(mask, dtype=bool)[::step, ::step]
    if not keep.any():
        return np.zeros((0, 3), dtype=np.float64)
    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])
    d = sampled_depth[keep]
    x = (grid_u[keep] - cx) * d / fx
    y = (grid_v[keep] - cy) * d / fy
    z = d if convention == "opencv_rdf" else -d
    local = np.stack((x, y, z, np.ones_like(d)), axis=1)
    world = local @ transform.T
    return np.ascontiguousarray(world[:, :3])

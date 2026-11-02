from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from roboarchitect.harness import GraspCandidate, GraspRequest, GraspResult
from roboarchitect.protocol import StrictSchemaError

from .common import Backend


CONFIG_FILE = "checkpoints/graspgen_franka_panda.yml"
CHECKPOINT_FILES = (
    CONFIG_FILE,
    "checkpoints/graspgen_franka_panda_dis.pth",
    "checkpoints/graspgen_franka_panda_gen.pth",
)
MODEL_POINT_COUNT = 2048
MIN_OBJECT_POINTS = 64
NUM_GRASPS = 200
TOP_K = 20
INFERENCE_SEED = 0
FRANKA_OPEN_WIDTH_M = 0.10537486


def _verify_transform(transform: np.ndarray) -> None:
    if not np.allclose(transform[3], np.asarray((0.0, 0.0, 0.0, 1.0)), atol=1e-5):
        raise StrictSchemaError("grasp_request.camera_to_world: invalid homogeneous row")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=1e-4
    ):
        raise StrictSchemaError("grasp_request.camera_to_world: expected rigid right-handed transform")


def object_points_world(request: GraspRequest) -> np.ndarray:
    if request.mask is None:
        raise StrictSchemaError("grasp_request.mask: GraspGen requires an object segmentation mask")
    intrinsics = request.intrinsics
    if not (
        intrinsics[0, 0] > 0.0
        and intrinsics[1, 1] > 0.0
        and np.allclose(intrinsics[2], np.asarray((0.0, 0.0, 1.0)), atol=1e-5)
        and abs(float(intrinsics[0, 1])) <= 1e-5
        and abs(float(intrinsics[1, 0])) <= 1e-5
    ):
        raise StrictSchemaError("grasp_request.intrinsics: expected positive-focal pinhole matrix")
    _verify_transform(request.camera_to_world)
    valid = request.mask & (request.depth_m > 0.0)
    rows, columns = np.nonzero(valid)
    if len(rows) < MIN_OBJECT_POINTS:
        raise StrictSchemaError(f"grasp_request: requires at least {MIN_OBJECT_POINTS} valid object depth pixels")
    depth = request.depth_m[rows, columns]
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    if request.optical_convention == "opencv_rdf":
        forward = depth
    elif request.optical_convention == "opengl_rub":
        forward = -depth
    else:
        raise StrictSchemaError(f"grasp_request.optical_convention: {request.optical_convention!r} is not supported by GraspGen")
    points_camera = np.stack(
        (
            (columns.astype(np.float32) - cx) * depth / fx,
            (rows.astype(np.float32) - cy) * depth / fy,
            forward,
        ),
        axis=1,
    ).astype(np.float32)
    rotation = request.camera_to_world[:3, :3]
    translation = request.camera_to_world[:3, 3]
    points_world = points_camera @ rotation.T + translation
    rng = np.random.default_rng(INFERENCE_SEED)
    indices = rng.choice(
        len(points_world),
        size=MODEL_POINT_COUNT,
        replace=len(points_world) < MODEL_POINT_COUNT,
    )
    return np.ascontiguousarray(points_world[indices], dtype=np.float32)


class GraspGen(Backend):
    def load(self) -> None:
        from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg
        from huggingface_hub import hf_hub_download

        paths = {
            filename: Path(hf_hub_download(repo_id=self.model_id, filename=filename, local_files_only=True))
            for filename in CHECKPOINT_FILES
        }
        config = load_grasp_cfg(str(paths[CONFIG_FILE]))
        if (
            config.data.gripper_name != "franka_panda"
            or config.eval.model_name != "diffusion-discriminator"
            or config.diffusion.obs_backbone != "ptv3"
            or config.discriminator.obs_backbone != "ptv3"
        ):
            raise RuntimeError("GraspGen Franka checkpoint configuration mismatch")
        self._sampler = GraspGenSampler(config)
        self._lock = threading.Lock()

    def invoke(self, operation: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        if operation != "grasp":
            raise StrictSchemaError(f"{self.name}: unsupported operation {operation}")
        import torch
        from grasp_gen.grasp_server import GraspGenSampler

        request = GraspRequest.from_mapping(payload)
        points = object_points_world(request)
        with self._lock, torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(INFERENCE_SEED)
            torch.cuda.manual_seed_all(INFERENCE_SEED)
            grasps, scores = GraspGenSampler.run_inference(
                points,
                self._sampler,
                grasp_threshold=-1.0,
                num_grasps=NUM_GRASPS,
                topk_num_grasps=TOP_K,
                min_grasps=1,
                max_tries=1,
                remove_outliers=False,
            )
        poses = grasps.detach().cpu().numpy().astype(np.float32, copy=False)
        confidence = scores.detach().cpu().numpy().astype(np.float32, copy=False)
        if poses.ndim != 3 or poses.shape[1:] != (4, 4) or confidence.shape != (len(poses),):
            raise RuntimeError("GraspGen returned invalid result dimensions")
        if not len(poses) or not np.isfinite(poses).all() or not np.isfinite(confidence).all():
            raise RuntimeError("GraspGen returned empty or non-finite results")
        if np.any(confidence < 0.0) or np.any(confidence > 1.0):
            raise RuntimeError("GraspGen returned out-of-range scores")
        if not np.allclose(poses[:, 3], np.asarray((0.0, 0.0, 0.0, 1.0)), atol=1e-4):
            raise RuntimeError("GraspGen returned invalid homogeneous poses")
        candidates = tuple(
            GraspCandidate(pose, float(score), FRANKA_OPEN_WIDTH_M)
            for pose, score in zip(poses, confidence, strict=True)
        )
        return GraspResult(candidates).to_mapping()

    def warm_up(self) -> None:
        size = 96
        rgb = np.zeros((size, size, 3), dtype=np.uint8)
        rgb[16:80, 16:80] = (80, 160, 220)
        rows, columns = np.mgrid[:size, :size]
        mask = (rows - 48) ** 2 + (columns - 48) ** 2 <= 30**2
        depth = np.zeros((size, size), dtype=np.float32)
        depth[mask] = 0.55 + 0.0002 * ((rows[mask] - 48) ** 2 + (columns[mask] - 48) ** 2)
        intrinsics = np.asarray(((120.0, 0.0, 48.0), (0.0, 120.0, 48.0), (0.0, 0.0, 1.0)), dtype=np.float32)
        result = GraspResult.from_mapping(
            self.invoke(
                "grasp",
                GraspRequest(rgb, depth, intrinsics, np.eye(4, dtype=np.float32), "opencv_rdf", mask).to_mapping(),
            )
        )
        if len(result.candidates) != TOP_K:
            raise RuntimeError("GraspGen warm-up returned an unexpected candidate count")

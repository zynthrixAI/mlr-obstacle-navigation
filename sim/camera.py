"""Simulated pan camera aimed at a LiDAR segment: renders a 32x32 RGB crop (pseudo-3D column rendering).

For each image column a ray is cast from the robot at the column's bearing. The hit object occupies the
image rows whose elevation lies between the object's base (z=0) and top (z=height), seen from a camera
mounted at 0.3 m. Above it: background; below: floor. Appearance comes from World.obj_* (sim/world.py).

Noise model: distance shading, global brightness 0.6-1.4, per-channel tint +-5 %, Gaussian pixel noise
sigma = 8/255. Colour/height/stripe distributions of the classes overlap by design (see world._assign_appearance).
"""
from __future__ import annotations

import numpy as np

from .world import World

SIZE = 32
CAM_H = 0.30                      # camera height (m)
V_FOV = np.deg2rad(40)            # fixed vertical field of view
PAD_DEG = 3.0
MIN_HFOV_DEG = 8.0
MAX_RANGE = 5.0


def render_crop(world: World, pose, b0_deg: float, b1_deg: float, rng: np.random.Generator) -> np.ndarray:
    """b0_deg, b1_deg: segment start/end ray bearings in the robot frame (deg, CCW, may wrap). Returns uint8 (32,32,3)."""
    span = (b1_deg - b0_deg) % 360
    centre = np.deg2rad(b0_deg + span / 2)
    half = np.deg2rad(max(span / 2 + PAD_DEG, MIN_HFOV_DEG / 2))
    rel = centre + np.linspace(half, -half, SIZE)            # image left = CCW side
    x, y, th = pose
    d, _, obj = world.raycast(np.array([x, y]), th + rel, MAX_RANGE, return_obj=True)

    elev = (V_FOV / 2) * (1 - 2 * (np.arange(SIZE) + 0.5) / SIZE)          # (rows,), top row positive
    rows = np.arange(SIZE)[:, None] / SIZE
    img = np.empty((SIZE, SIZE, 3))
    img[:] = np.array([0.30, 0.32, 0.36]) + 0.10 * (1 - rows[..., None])     # background
    floor_mask = elev[:, None] < np.arctan2(-CAM_H, np.maximum(d, 0.05))[None, :]
    floor = np.array([0.42, 0.36, 0.28]) * (0.7 + 0.5 * rows[..., None])
    img = np.where(floor_mask[..., None], floor, img)

    hit = obj >= 0
    if hit.any():
        h = np.where(hit, world.obj_height[np.maximum(obj, 0)], 0.0)
        dd = np.maximum(d, 0.05)
        lo = np.arctan2(-CAM_H, dd)
        hi = np.arctan2(h - CAM_H, dd)
        in_obj = (elev[:, None] >= lo[None, :]) & (elev[:, None] <= hi[None, :]) & hit[None, :]
        col = world.obj_color[np.maximum(obj, 0)] / (1 + 0.15 * dd)[:, None]          # (cols,3)
        # stripe band between 60 % and 75 % of object height
        s_lo = np.arctan2(0.60 * h - CAM_H, dd)
        s_hi = np.arctan2(0.75 * h - CAM_H, dd)
        stripe = (world.obj_stripe[np.maximum(obj, 0)] & hit)[None, :] & \
                 (elev[:, None] >= s_lo[None, :]) & (elev[:, None] <= s_hi[None, :])
        obj_rgb = np.broadcast_to(col[None, :, :], img.shape).copy()
        obj_rgb[stripe] = 0.92 / (1 + 0.15 * np.broadcast_to(dd[None, :], stripe.shape)[stripe])[:, None]
        img = np.where(in_obj[..., None], obj_rgb, img)

    img = img * rng.uniform(0.6, 1.4) * rng.uniform(0.95, 1.05, size=3)
    img = img + rng.normal(0, 8 / 255, size=img.shape)
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)

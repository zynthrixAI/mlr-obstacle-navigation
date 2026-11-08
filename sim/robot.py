"""Sensors, discrete unicycle dynamics and the reactive baseline controller."""
from __future__ import annotations

import numpy as np

from .world import ROBOT_RADIUS, World

MAX_RANGE = 5.0
US_BEARINGS = np.deg2rad([0, 45, 90, 135, 180, -135, -90, -45])   # 8 ultrasonics
US_HALF_CONE = np.deg2rad(15)                                      # 30 deg cones -> 15 deg blind gaps
FORWARD_STEP = 0.1
TURN = np.deg2rad(15)
ACTIONS = ("forward", "left", "right")


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# ---------------------------------------------------------------------- sensors
def lidar(world: World, pose, n=36, rng=None, noisy=True):
    """Ranges at relative bearings 2*pi*i/n (index 0 = forward, CCW). Returns (ranges, true_class)."""
    x, y, th = pose
    rel = 2 * np.pi * np.arange(n) / n
    r, cls = world.raycast(np.array([x, y]), th + rel, MAX_RANGE)
    if noisy and rng is not None:
        r = r + rng.normal(0, 0.01 * r + 0.01)
        r = np.clip(r, 0.05, MAX_RANGE)
        r[rng.random(n) < 0.02] = np.nan            # dropout
    return r, cls


def ultrasonic(world: World, pose, rng=None, noisy=True):
    """8 wide-cone sensors: reading = min over 7 sub-rays in a 30 deg cone."""
    x, y, th = pose
    sub = np.linspace(-US_HALF_CONE, US_HALF_CONE, 7)
    ang = (th + US_BEARINGS[:, None] + sub[None, :]).ravel()
    r, _ = world.raycast(np.array([x, y]), ang, MAX_RANGE)
    r = r.reshape(8, 7).min(1)
    if noisy and rng is not None:
        r = r + rng.normal(0, 0.03 * r)
        r = np.clip(r, 0.05, MAX_RANGE)
        r[rng.random(8) < 0.04] = np.nan
    return r


def observable(rel_bearing: float) -> bool:
    """Is a bearing (robot frame) inside any ultrasonic cone?"""
    return bool(np.min(np.abs(wrap(rel_bearing - US_BEARINGS))) <= US_HALF_CONE)


# ---------------------------------------------------------------------- dynamics
def step_pose(world: World, pose, action: int):
    """Apply action. Returns (new_pose, collided). On collision the robot does not move."""
    x, y, th = pose
    if action == 0:
        nx, ny = x + FORWARD_STEP * np.cos(th), y + FORWARD_STEP * np.sin(th)
        if world.clearance((nx, ny)) < ROBOT_RADIUS:
            return (x, y, th), True
        return (nx, ny, th), False
    if action == 1:
        return (x, y, wrap(th + TURN)), False
    return (x, y, wrap(th - TURN)), False


def goal_polar(pose, goal):
    dx, dy = goal[0] - pose[0], goal[1] - pose[1]
    return float(np.hypot(dx, dy)), float(wrap(np.arctan2(dy, dx) - pose[2]))


# ---------------------------------------------------------------------- reactive baseline
def sectors36(r36):
    """front +-30 deg, left 30..90 deg, right -30..-90 deg (index 0 = forward, 10 deg/ray)."""
    r = np.nan_to_num(r36, nan=MAX_RANGE)
    front = np.r_[r[0:4], r[33:36]]           # 0,10,20,30 and -30,-20,-10
    left = r[3:10]                            # 30..90
    right = r[27:34]                          # -90..-30
    return front.min(), left.min(), right.min()


def reactive_action(r36, goal_angle):
    """Rule-based baseline from the plan."""
    f, l, rr = sectors36(r36)
    if f < 0.5:
        return 1 if l >= rr else 2
    if abs(goal_angle) > np.deg2rad(15):
        return 1 if goal_angle > 0 else 2
    return 0

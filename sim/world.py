"""2D world: walls / boxes as line segments, cylinders as circles, vectorised ray casting.

Classes: 0 = wall, 1 = obstacle, 2 = target.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from scipy import ndimage

from .seeds import map_type_for

WALL, OBSTACLE, TARGET = 0, 1, 2
CLASS_NAMES = ("wall", "obstacle", "target")
ROBOT_RADIUS = 0.15
ARENA = 8.0


@dataclass
class World:
    seed: int
    map_type: str
    seg_a: np.ndarray            # (M,2)
    seg_b: np.ndarray            # (M,2)
    seg_cls: np.ndarray          # (M,)
    circ_c: np.ndarray           # (K,2)
    circ_r: np.ndarray           # (K,)
    circ_cls: np.ndarray         # (K,)
    free_label: np.ndarray = field(repr=False, default=None)  # connected components of free grid
    grid_res: float = 0.1
    # appearance (camera): per-primitive object index and per-object attributes
    seg_obj: np.ndarray = field(repr=False, default=None)     # (M,)
    circ_obj: np.ndarray = field(repr=False, default=None)    # (K,)
    obj_cls: np.ndarray = field(repr=False, default=None)     # (n_obj,)
    obj_color: np.ndarray = field(repr=False, default=None)   # (n_obj,3) in [0,1]
    obj_height: np.ndarray = field(repr=False, default=None)  # (n_obj,) metres
    obj_stripe: np.ndarray = field(repr=False, default=None)  # (n_obj,) bool

    # ------------------------------------------------------------------ ray casting
    def raycast(self, pos: np.ndarray, angles: np.ndarray, max_range: float = 5.0, return_obj: bool = False):
        """Return (ranges, hit_class[, hit_object]) for rays from pos at absolute angles.
        No-hit -> max_range, class -1, object -1."""
        d = np.stack([np.cos(angles), np.sin(angles)], axis=1)          # (N,2)
        best = np.full(len(angles), np.inf)
        cls = np.full(len(angles), -1, dtype=int)
        obj = np.full(len(angles), -1, dtype=int)
        if len(self.seg_a):
            e = self.seg_b - self.seg_a                                   # (M,2)
            ap = self.seg_a - pos                                         # (M,2)
            denom = d[:, None, 0] * e[None, :, 1] - d[:, None, 1] * e[None, :, 0]   # (N,M)
            with np.errstate(divide="ignore", invalid="ignore"):
                t = (ap[None, :, 0] * e[None, :, 1] - ap[None, :, 1] * e[None, :, 0]) / denom
                u = (ap[None, :, 0] * d[:, None, 1] - ap[None, :, 1] * d[:, None, 0]) / denom
            ok = (np.abs(denom) > 1e-12) & (t > 1e-9) & (u >= 0) & (u <= 1)
            t = np.where(ok, t, np.inf)
            j = np.argmin(t, axis=1)
            tm = t[np.arange(len(angles)), j]
            better = tm < best
            best[better] = tm[better]
            cls[better] = self.seg_cls[j[better]]
            obj[better] = self.seg_obj[j[better]]
        if len(self.circ_c):
            f = pos - self.circ_c                                          # (K,2)
            b = d @ f.T                                                    # (N,K)
            c = (f ** 2).sum(1) - self.circ_r ** 2                         # (K,)
            disc = b ** 2 - c[None, :]
            with np.errstate(invalid="ignore"):
                t = -b - np.sqrt(disc)
            t = np.where((disc >= 0) & (t > 1e-9), t, np.inf)
            j = np.argmin(t, axis=1)
            tm = t[np.arange(len(angles)), j]
            better = tm < best
            best[better] = tm[better]
            cls[better] = self.circ_cls[j[better]]
            obj[better] = self.circ_obj[j[better]]
        miss = best > max_range
        best[miss] = max_range
        cls[miss] = -1
        obj[miss] = -1
        if return_obj:
            return best, cls, obj
        return best, cls

    # ------------------------------------------------------------------ nearest obstacle
    def nearest(self, pos: np.ndarray):
        """Exact distance (from robot centre) and absolute bearing to the nearest surface point."""
        cand_d, cand_p = [], []
        if len(self.seg_a):
            e = self.seg_b - self.seg_a
            t = np.clip(((pos - self.seg_a) * e).sum(1) / (e ** 2).sum(1), 0, 1)
            p = self.seg_a + t[:, None] * e
            cand_p.append(p)
            cand_d.append(np.linalg.norm(p - pos, axis=1))
        if len(self.circ_c):
            v = pos - self.circ_c
            n = np.linalg.norm(v, axis=1)
            p = self.circ_c + v / np.maximum(n, 1e-9)[:, None] * self.circ_r[:, None]
            cand_p.append(p)
            cand_d.append(np.maximum(n - self.circ_r, 0.0))
        dd = np.concatenate(cand_d)
        pp = np.concatenate(cand_p)
        k = int(np.argmin(dd))
        vec = pp[k] - pos
        return float(dd[k]), float(np.arctan2(vec[1], vec[0]))

    def clearance(self, pos) -> float:
        return self.nearest(np.asarray(pos, float))[0]

    def same_component(self, p, q) -> bool:
        def cell(x):
            i = int(np.clip(x[1] / self.grid_res, 0, self.free_label.shape[0] - 1))
            j = int(np.clip(x[0] / self.grid_res, 0, self.free_label.shape[1] - 1))
            return self.free_label[i, j]
        a, b = cell(p), cell(q)
        return a != 0 and a == b

    def sample_free(self, rng, min_clear=0.4, n_try=500):
        for _ in range(n_try):
            p = rng.uniform(0.3, ARENA - 0.3, size=2)
            if self.clearance(p) >= min_clear:
                return p
        raise RuntimeError("no free position")

    def sample_start_goal(self, rng, min_dist=2.0, max_dist=6.0):
        for _ in range(200):
            s = self.sample_free(rng)
            g = self.sample_free(rng, min_clear=0.35)
            dist = np.linalg.norm(g - s)
            if min_dist <= dist <= max_dist and self.same_component(s, g):
                return s, g
        raise RuntimeError("no reachable start/goal")


# ---------------------------------------------------------------------- generators
def _box_segments(cx, cy, w, h):
    x0, x1, y0, y1 = cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2
    pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    return [(pts[i], pts[(i + 1) % 4]) for i in range(4)]


def _add_objects(rng, n, segs, circs, placed, cls_mix):
    """Place n objects avoiding overlap. cls_mix: callable(rng)->(kind, cls, size params)."""
    tries = 0
    while n > 0 and tries < 2000:
        tries += 1
        kind, cls, size = cls_mix(rng)
        c = rng.uniform(0.8, ARENA - 0.8, size=2)
        rad = size if kind == "circle" else 0.5 * np.hypot(*size)
        if any(np.linalg.norm(c - q) < rad + r + 0.45 for q, r in placed):
            continue
        if kind == "circle":
            circs.append((c, size, cls))
        else:
            for a, b in _box_segments(c[0], c[1], *size):
                segs.append((a, b, cls))
        placed.append((c, rad))
        n -= 1


def _obstacle(rng):
    if rng.random() < 0.5:
        return "circle", OBSTACLE, rng.uniform(0.15, 0.45)
    return "box", OBSTACLE, tuple(rng.uniform(0.3, 1.0, size=2))


def _target(rng):
    # targets: small cylinders, overlapping in size with small obstacle cylinders (deliberately ambiguous)
    return "circle", TARGET, rng.uniform(0.12, 0.25)


@lru_cache(maxsize=4096)
def make_world(seed: int, map_type: str | None = None) -> World:
    map_type = map_type or map_type_for(seed)
    rng = np.random.default_rng(seed)
    L = ARENA
    segs = [((0, 0), (L, 0), WALL), ((L, 0), (L, L), WALL), ((L, L), (0, L), WALL), ((0, L), (0, 0), WALL)]
    circs, placed = [], []
    if map_type == "open":
        _add_objects(rng, rng.integers(3, 6), segs, circs, placed, _obstacle)
        _add_objects(rng, 2, segs, circs, placed, _target)
    elif map_type == "cluttered":
        _add_objects(rng, rng.integers(16, 24), segs, circs, placed, _obstacle)
        _add_objects(rng, 2, segs, circs, placed, _target)
    elif map_type == "narrow":
        w = rng.uniform(1.0, 1.4)        # corridor width
        gap = 1.0
        y, k = w, 0
        while y < L - 0.8:
            if k % 2 == 0:
                segs.append(((0, y), (L - gap, y), WALL))
            else:
                segs.append(((gap, y), (L, y), WALL))
            y += w
            k += 1
        # a few small items inside corridors: keep spacing via placed list of wall-free spots
        for cls_fn, n in ((_target, 2), (lambda r: ("circle", OBSTACLE, r.uniform(0.12, 0.22)), int(rng.integers(1, 4)))):
            for _ in range(n):
                for _t in range(200):
                    x = rng.uniform(1.5, L - 1.5)
                    row = rng.integers(0, max(1, k))
                    yc = (row + 0.5) * w
                    kind, cls, r = cls_fn(rng)
                    c = np.array([x, yc])
                    if r < w / 2 - 0.45 and not any(np.linalg.norm(c - q) < rr + r + 0.6 for q, rr in placed):
                        circs.append((c, r, cls))
                        placed.append((c, r))
                        break
    else:
        raise ValueError(map_type)

    seg_a = np.array([s[0] for s in segs], float)
    seg_b = np.array([s[1] for s in segs], float)
    seg_cls = np.array([s[2] for s in segs], int)
    circ_c = np.array([c[0] for c in circs], float).reshape(-1, 2)
    circ_r = np.array([c[1] for c in circs], float)
    circ_cls = np.array([c[2] for c in circs], int)
    w = World(seed, map_type, seg_a, seg_b, seg_cls, circ_c, circ_r, circ_cls)
    _assign_appearance(w)
    w.free_label = _free_components(w)
    return w


def _assign_appearance(w: World) -> None:
    """Object ids + appearance, drawn from a SEPARATE rng so geometry/LiDAR data are unchanged.

    Walls: one object per wall segment, grey/beige, 1.0 m tall.
    Boxes: every 4 consecutive obstacle segments form one box (see _box_segments).
    Targets: red-orange hue (+-noise), 0.4-0.6 m tall, horizontal stripe 60 % of the time.
    Obstacles: uniform random hue (so ~10 % fall in the target hue range), 0.3-1.2 m tall, no stripe.
    """
    import colorsys
    rng = np.random.default_rng(w.seed + 7_919_993)
    seg_obj = np.empty(len(w.seg_cls), int)
    cls_list, k, run = [], 0, 0
    for i, c in enumerate(w.seg_cls):
        if c == WALL:
            seg_obj[i] = k; cls_list.append(WALL); k += 1; run = 0
        else:
            if run % 4 == 0:
                cls_list.append(int(c)); k += 1
            seg_obj[i] = k - 1; run += 1
    circ_obj = np.arange(k, k + len(w.circ_cls))
    cls_list += [int(c) for c in w.circ_cls]
    n = len(cls_list)
    color, height, stripe = np.zeros((n, 3)), np.zeros(n), np.zeros(n, bool)
    for o, c in enumerate(cls_list):
        if c == WALL:
            g = rng.uniform(0.55, 0.75)
            color[o] = (g, g * rng.uniform(0.95, 1.0), g * rng.uniform(0.85, 0.95))
            height[o] = 1.0
        elif c == TARGET:
            h = (rng.normal(0.03, 0.03)) % 1.0
            color[o] = colorsys.hsv_to_rgb(h, rng.uniform(0.6, 0.95), rng.uniform(0.6, 0.95))
            height[o] = rng.uniform(0.4, 0.6)
            stripe[o] = rng.random() < 0.6
        else:
            color[o] = colorsys.hsv_to_rgb(rng.random(), rng.uniform(0.3, 0.95), rng.uniform(0.4, 0.95))
            height[o] = rng.uniform(0.3, 1.2)
    w.seg_obj, w.circ_obj = seg_obj, circ_obj
    w.obj_cls, w.obj_color, w.obj_height, w.obj_stripe = np.array(cls_list), color, height, stripe


def _free_components(w: World) -> np.ndarray:
    n = int(ARENA / w.grid_res)
    xs = (np.arange(n) + 0.5) * w.grid_res
    X, Y = np.meshgrid(xs, xs)
    pts = np.stack([X.ravel(), Y.ravel()], 1)
    d = np.full(len(pts), np.inf)
    if len(w.seg_a):
        e = w.seg_b - w.seg_a
        for a, ee in zip(w.seg_a, e):
            t = np.clip(((pts - a) @ ee) / (ee @ ee), 0, 1)
            d = np.minimum(d, np.linalg.norm(pts - (a + t[:, None] * ee), axis=1))
    for c, r in zip(w.circ_c, w.circ_r):
        d = np.minimum(d, np.linalg.norm(pts - c, axis=1) - r)
    free = (d > ROBOT_RADIUS + 0.02).reshape(n, n)
    lab, _ = ndimage.label(free)
    return lab


# ---------------------------------------------------------------------- HARD appearance variant (T6b)
HARD_TARGET_HUE_LO, HARD_TARGET_HUE_HI = -0.06, 0.12   # target hue envelope ~ N(0.03, 0.03) +-3 sd


@lru_cache(maxsize=4096)
def hard_appearance(seed: int, map_type: str | None = None):
    """PRE-SPECIFIED hard appearance variant (fixed before any hard-mode result was computed).

    Returns (obj_color (n_obj,3), obj_height (n_obj,), obj_stripe (n_obj,)) as NEW read-only arrays for the
    same object ids as make_world(seed).obj_cls. It never mutates the cached World, so the default ("easy")
    appearance, geometry and LiDAR data are unchanged. Separate RNG stream: default_rng(seed + 31_337_771)
    (easy uses seed + 7_919_993). Every object draws the same fixed set of numbers in the same order,
    whatever branch it takes.

    Target-like colour = easy target distribution: hue ~ N(0.03, 0.03) mod 1, sat U(0.6, 0.95), val U(0.6, 0.95).
    - Walls (one object per wall segment, 1.0 m tall): 85 % grey/beige as in easy; 15 % a saturated random
      colour (hue U(0,1), sat U(0.6, 0.95), val U(0.5, 0.9)). No stripe.
    - Obstacles: 30 % take a target-like colour AND a target-like height U(0.4, 0.6); the other 70 % keep the
      easy obstacle distribution (hue U(0,1), sat U(0.3, 0.95), val U(0.4, 0.95), height U(0.3, 1.2)).
      Independently, 20 % carry a stripe.
    - Targets: height U(0.4, 0.6); 80 % target-like colour, 20 % a random NON-target hue
      (hue U(0.12, 0.94), i.e. outside the target envelope [-0.06, 0.12]) with the same sat/val ranges.
      Only 40 % carry a stripe.
    Render-level partial occlusion (25 % of crops) and the unchanged noise model live in sim/camera.py.
    """
    import colorsys
    w = make_world(seed, map_type)
    rng = np.random.default_rng(seed + 31_337_771)
    n = len(w.obj_cls)
    color, height, stripe = np.zeros((n, 3)), np.zeros(n), np.zeros(n, bool)
    for o, c in enumerate(w.obj_cls):
        u_branch, u_stripe, u_h, u_s, u_v, u_ht, z = rng.random(7)
        if c == WALL:
            height[o] = 1.0
            if u_branch < 0.15:
                color[o] = colorsys.hsv_to_rgb(u_h, 0.6 + 0.35 * u_s, 0.5 + 0.4 * u_v)
            else:   # easy wall distribution (grey/beige), from the same uniforms
                g = 0.55 + 0.20 * u_s
                color[o] = (g, g * (0.95 + 0.05 * u_v), g * (0.85 + 0.10 * u_ht))
        elif c == TARGET:
            height[o] = 0.4 + 0.2 * u_ht
            stripe[o] = u_stripe < 0.40
            if u_branch < 0.20:
                hue = 0.12 + 0.82 * u_h
            else:
                hue = (0.03 + 0.03 * _std_normal(z)) % 1.0
            color[o] = colorsys.hsv_to_rgb(hue, 0.6 + 0.35 * u_s, 0.6 + 0.35 * u_v)
        else:
            stripe[o] = u_stripe < 0.20
            if u_branch < 0.30:
                hue = (0.03 + 0.03 * _std_normal(z)) % 1.0
                color[o] = colorsys.hsv_to_rgb(hue, 0.6 + 0.35 * u_s, 0.6 + 0.35 * u_v)
                height[o] = 0.4 + 0.2 * u_ht
            else:
                color[o] = colorsys.hsv_to_rgb(u_h, 0.3 + 0.65 * u_s, 0.4 + 0.55 * u_v)
                height[o] = 0.3 + 0.9 * u_ht
    for a in (color, height, stripe):
        a.setflags(write=False)
    return color, height, stripe


def _std_normal(u: float) -> float:
    """Inverse-CDF standard normal from one uniform (keeps a fixed number of draws per object)."""
    from scipy.special import ndtri
    return float(ndtri(min(max(u, 1e-12), 1 - 1e-12)))

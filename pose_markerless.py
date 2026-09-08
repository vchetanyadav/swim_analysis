"""
pose_markerless.py
Accuracy-first swimming pose estimation for side-view underwater video.

Main ideas:
  1. rotate the swimmer to the orientation the pose model understands best
  2. keep that orientation locked during the clip
  3. optionally use horizontal-flip test-time augmentation (TTA)
  4. choose the swimmer using pose confidence + anatomy + temporal continuity
  5. never swap single joints independently
  6. refine the complete sequence after inference using whole-limb identity,
     temporal outlier rejection, short-gap interpolation and light smoothing
  7. support both stock COCO-17 and a custom SWIM-14 pose model

The internal layout is always canonical COCO-17 so downstream analysis remains
compatible with either stock or fine-tuned weights.
"""

import argparse
import csv
import math
import os
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.signal import savgol_filter
from ultralytics import YOLO

OUTPUT_DIR = "output"

KEYPOINTS = [
    "nose", "l_eye", "r_eye", "l_ear", "r_ear",
    "l_shoulder", "r_shoulder", "l_elbow", "r_elbow",
    "l_wrist", "r_wrist", "l_hip", "r_hip",
    "l_knee", "r_knee", "l_ankle", "r_ankle",
]

HEAD = [0, 1, 2, 3, 4]
BODY = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
CORE = [5, 6, 11, 12]
DISTAL = [9, 10, 15, 16]
MID_JOINTS = [7, 8, 13, 14]

BONES = [
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]

ARM_LEFT = [5, 7, 9]
ARM_RIGHT = [6, 8, 10]
LEG_LEFT = [11, 13, 15]
LEG_RIGHT = [12, 14, 16]

LR_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10),
            (11, 12), (13, 14), (15, 16)]

LIMB_GROUPS = {
    "left arm":  ([(5, 7), (7, 9)], (255, 0, 255)),
    "right arm": ([(6, 8), (8, 10)], (0, 220, 0)),
    "left leg":  ([(11, 13), (13, 15)], (0, 165, 255)),
    "right leg": ([(12, 14), (14, 16)], (0, 255, 255)),
    "torso":     ([(5, 6), (11, 12), (5, 11), (6, 12)], (255, 170, 30)),
}

# SWIM-14 order:
# head_center, neck, L/R shoulder, L/R elbow, L/R wrist,
# L/R hip, L/R knee, L/R ankle
SWIM14_TO_COCO17 = {
    0: 0,
    2: 5, 3: 6,
    4: 7, 5: 8,
    6: 9, 7: 10,
    8: 11, 9: 12,
    10: 13, 11: 14,
    12: 15, 13: 16,
}


def _canonicalize(kp):
    if kp is None:
        return None
    kp = np.asarray(kp, dtype=float)
    if kp.ndim != 2 or kp.shape[1] < 3:
        return None
    if kp.shape[0] == 17:
        return kp[:, :3].copy()
    if kp.shape[0] == 14:
        out = np.zeros((17, 3), dtype=float)
        for src, dst in SWIM14_TO_COCO17.items():
            out[dst] = kp[src, :3]
        return out
    raise ValueError(f"Unsupported keypoint layout: {kp.shape}. Expected 17x3 or 14x3.")


def _visible(kp, i, thr=0.25):
    return kp is not None and np.isfinite(kp[i, 0]) and kp[i, 2] >= thr


def _pair_center(kp, a, b, thr=0.25, max_pair_span=None):
    """Robust centre of a left/right joint pair.

    If one side is a clear outlier, prefer the higher-confidence point instead
    of averaging a correct point with a wrong point.
    """
    va, vb = _visible(kp, a, thr), _visible(kp, b, thr)
    if va and vb:
        pa, pb = kp[a, :2], kp[b, :2]
        if max_pair_span is not None and np.linalg.norm(pa - pb) > max_pair_span:
            return pa.copy() if kp[a, 2] >= kp[b, 2] else pb.copy()
        wa, wb = max(kp[a, 2], 1e-3), max(kp[b, 2], 1e-3)
        return (wa * pa + wb * pb) / (wa + wb)
    if va:
        return kp[a, :2].copy()
    if vb:
        return kp[b, :2].copy()
    return None


def shoulder_hip_points(kp, thr=0.25):
    """Return robust trunk endpoints.

    A bad left/right shoulder or hip should not drag the trunk axis across the
    frame. Pair span is checked relative to a provisional same-side trunk size.
    """
    if kp is None:
        return None, None

    provisional = []
    if _visible(kp, 5, thr) and _visible(kp, 11, thr):
        provisional.append(np.linalg.norm(kp[5, :2] - kp[11, :2]))
    if _visible(kp, 6, thr) and _visible(kp, 12, thr):
        provisional.append(np.linalg.norm(kp[6, :2] - kp[12, :2]))
    base = float(np.median(provisional)) if provisional else None
    max_span = 0.85 * base if base and base > 1 else None

    sh = _pair_center(kp, 5, 6, thr, max_span)
    hp = _pair_center(kp, 11, 12, thr, max_span)

    # If pair-centres are not available, use only same-side shoulder/hip pairs.
    if sh is not None and hp is not None:
        return sh, hp
    if _visible(kp, 5, thr) and _visible(kp, 11, thr):
        return kp[5, :2].copy(), kp[11, :2].copy()
    if _visible(kp, 6, thr) and _visible(kp, 12, thr):
        return kp[6, :2].copy(), kp[12, :2].copy()
    return None, None


def torso_length(kp, thr=0.25):
    sh, hp = shoulder_hip_points(kp, thr)
    if sh is None or hp is None:
        return None
    d = float(np.linalg.norm(sh - hp))
    return d if d > 1e-6 else None


def head_point(kp, thr=0.22):
    if kp is None:
        return None
    pts, ws = [], []
    for i in (0, 3, 4):
        if _visible(kp, i, thr):
            pts.append(kp[i, :2])
            ws.append(max(kp[i, 2], 1e-3))
    if not pts:
        return None
    pts = np.asarray(pts, float)
    ws = np.asarray(ws, float)
    return np.sum(pts * ws[:, None], axis=0) / np.sum(ws)


def body_center(kp, thr=0.20):
    pts = [kp[i, :2] for i in CORE if _visible(kp, i, thr)]
    if len(pts) >= 2:
        return np.mean(pts, axis=0)
    pts = [kp[i, :2] for i in BODY if _visible(kp, i, thr)]
    return np.mean(pts, axis=0) if pts else None


def rotate_bound(img, angle):
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    c, s = abs(M[0, 0]), abs(M[0, 1])
    nw = int(h * s + w * c)
    nh = int(h * c + w * s)
    M[0, 2] += nw / 2.0 - cx
    M[1, 2] += nh / 2.0 - cy
    return cv2.warpAffine(img, M, (nw, nh)), M


def _map_back(kp, M):
    kp = kp.copy()
    inv = cv2.invertAffineTransform(M)
    for i in range(kp.shape[0]):
        if kp[i, 2] <= 0:
            continue
        x, y = kp[i, 0], kp[i, 1]
        kp[i, 0] = inv[0, 0] * x + inv[0, 1] * y + inv[0, 2]
        kp[i, 1] = inv[1, 0] * x + inv[1, 1] * y + inv[1, 2]
    return kp


def _undo_horizontal_flip(kp, width):
    """Map keypoints from a horizontally flipped image back to the original.

    Ultralytics pose training swaps left/right labels during horizontal flip, so
    left/right semantics must also be swapped when undoing our manual TTA flip.
    """
    kp = kp.copy()
    valid = kp[:, 2] > 0
    kp[valid, 0] = (width - 1) - kp[valid, 0]
    for a, b in LR_PAIRS:
        kp[[a, b]] = kp[[b, a]]
    return kp


def _anatomical_score(kp, image_diag, prev_kp=None, kpt_thr=0.18):
    if kp is None:
        return -1e9

    confs = np.asarray([kp[i, 2] for i in BODY], float)
    score = 1.7 * float(np.mean(confs)) + 0.04 * int(np.sum(confs >= kpt_thr))

    torso = torso_length(kp, kpt_thr)
    if torso is not None:
        score += 0.5
        if torso < 0.012 * image_diag:
            score -= 0.5
        for a, b in BONES:
            if _visible(kp, a, kpt_thr) and _visible(kp, b, kpt_thr):
                ratio = np.linalg.norm(kp[a, :2] - kp[b, :2]) / max(torso, 1e-6)
                if 0.06 <= ratio <= 1.65:
                    score += 0.06
                elif ratio > 2.0:
                    score -= 0.32

    if prev_kp is not None:
        c0, c1 = body_center(prev_kp, kpt_thr), body_center(kp, kpt_thr)
        if c0 is not None and c1 is not None:
            d = np.linalg.norm(c1 - c0) / max(image_diag, 1e-6)
            score += 0.50 * math.exp(-d / 0.10)
            if d > 0.30:
                score -= 0.8

        common = [i for i in CORE if _visible(kp, i, kpt_thr) and _visible(prev_kp, i, kpt_thr)]
        if common:
            d = np.mean([np.linalg.norm(kp[i, :2] - prev_kp[i, :2]) for i in common])
            score += 0.28 * math.exp(-(d / max(image_diag, 1e-6)) / 0.07)

    return float(score)


@dataclass
class PoseCandidate:
    kp: np.ndarray
    score: float
    angle: int
    det_conf: float
    area: float


def _best_person_from_result(result, M, source_shape, angle, prev_kp=None, flipped=False):
    if result.boxes is None or len(result.boxes) == 0 or result.keypoints is None:
        return None

    src_h, src_w = source_shape[:2]
    diag = math.hypot(src_w, src_h)
    areas = (result.boxes.xywh[:, 2] * result.boxes.xywh[:, 3]).detach().cpu().numpy()
    bconf = result.boxes.conf.detach().cpu().numpy() if result.boxes.conf is not None else np.ones(len(areas))

    best = None
    # Width before mapping back is the rotated image width. For the flipped path,
    # infer it from the prediction original_shape metadata when possible.
    rot_w = int(result.orig_shape[1]) if getattr(result, "orig_shape", None) else src_w

    for j in range(len(areas)):
        raw = result.keypoints.data[j].detach().cpu().numpy()
        kp = _canonicalize(raw)
        if flipped:
            kp = _undo_horizontal_flip(kp, rot_w)
        kp = _map_back(kp, M)
        pose_score = _anatomical_score(kp, diag, prev_kp, 0.18)
        area_frac = float(areas[j]) / max(float(result.orig_shape[0] * result.orig_shape[1]), 1.0)
        score = pose_score + 0.28 * float(bconf[j]) + 0.10 * min(area_frac / 0.18, 1.0)
        cand = PoseCandidate(kp, score, int(angle), float(bconf[j]), float(areas[j]))
        if best is None or cand.score > best.score:
            best = cand
    return best


def _predict_variant(model, rotated, M, source_shape, angle, imgsz, det_conf,
                     device=None, prev_kp=None, flip=False):
    inp = cv2.flip(rotated, 1) if flip else rotated
    kwargs = dict(imgsz=imgsz, conf=det_conf, verbose=False, max_det=8)
    if device is not None and str(device) != "":
        kwargs["device"] = device
    result = model.predict(inp, **kwargs)[0]
    return _best_person_from_result(result, M, source_shape, angle, prev_kp, flipped=flip)


def _fuse_candidates(a, b, source_shape, prev_kp=None):
    if a is None:
        return b
    if b is None:
        return a

    H, W = source_shape[:2]
    diag = math.hypot(W, H)
    torso_a, torso_b = torso_length(a.kp, 0.18), torso_length(b.kp, 0.18)
    scales = [x for x in (torso_a, torso_b) if x is not None and x > 1]
    scale = float(np.median(scales)) if scales else 0.08 * diag
    agree_limit = max(10.0, 0.50 * scale)

    out = np.zeros((17, 3), float)
    for i in range(17):
        va, vb = a.kp[i, 2] > 0.05, b.kp[i, 2] > 0.05
        if va and vb:
            pa, pb = a.kp[i, :2], b.kp[i, :2]
            d = float(np.linalg.norm(pa - pb))
            if d <= agree_limit:
                wa = max(a.kp[i, 2], 1e-3) * max(a.score, 0.1)
                wb = max(b.kp[i, 2], 1e-3) * max(b.score, 0.1)
                out[i, :2] = (wa * pa + wb * pb) / (wa + wb)
                out[i, 2] = max(a.kp[i, 2], b.kp[i, 2])
            else:
                # When TTA disagrees, do not average two incompatible locations.
                # Prefer the point that is more consistent with the previous frame,
                # otherwise use local confidence weighted by the candidate score.
                choose_a = None
                if prev_kp is not None and prev_kp[i, 2] > 0.18:
                    da = np.linalg.norm(pa - prev_kp[i, :2])
                    db = np.linalg.norm(pb - prev_kp[i, :2])
                    if abs(da - db) > 0.12 * scale:
                        choose_a = da < db
                if choose_a is None:
                    qa = a.kp[i, 2] * max(a.score, 0.1)
                    qb = b.kp[i, 2] * max(b.score, 0.1)
                    choose_a = qa >= qb
                out[i] = a.kp[i] if choose_a else b.kp[i]
        elif va:
            out[i] = a.kp[i]
        elif vb:
            out[i] = b.kp[i]

    score = _anatomical_score(out, diag, prev_kp, 0.18)
    fused = PoseCandidate(out, score, a.angle, max(a.det_conf, b.det_conf), max(a.area, b.area))

    # Keep the best single hypothesis when consensus fusion becomes less plausible.
    best_single = a if a.score >= b.score else b
    return fused if fused.score >= best_single.score - 0.08 else best_single


def _predict_angle(model, frame, angle, imgsz, det_conf, device=None,
                   prev_kp=None, tta=False):
    rotated, M = rotate_bound(frame, angle)
    normal = _predict_variant(model, rotated, M, frame.shape, angle, imgsz,
                              det_conf, device, prev_kp, flip=False)
    if not tta:
        return normal
    flipped = _predict_variant(model, rotated, M, frame.shape, angle, imgsz,
                               det_conf, device, prev_kp, flip=True)
    return _fuse_candidates(normal, flipped, frame.shape, prev_kp)


def best_orientation(model, frame, imgsz=1280, conf=0.12, device=None,
                     prev_kp=None, tta=False):
    best = None
    for angle in (0, 90, 180, 270):
        cand = _predict_angle(model, frame, angle, imgsz, conf, device,
                              prev_kp, tta=tta)
        if cand is not None and (best is None or cand.score > best.score):
            best = cand
    if best is None:
        return None, 0, -1e9
    return best.kp, best.angle, best.score


class PoseEstimator:
    """Stateful pose estimator with orientation locking and optional flip TTA."""

    def __init__(self, model, imgsz=1280, det_conf=0.12, orientation="auto",
                 device=None, relock_failures=4, recheck_every=0, tta=False):
        self.model = model
        self.imgsz = int(imgsz)
        self.det_conf = float(det_conf)
        self.orientation_mode = str(orientation).lower()
        self.device = device
        self.relock_failures = int(relock_failures)
        self.recheck_every = int(recheck_every)
        self.tta = bool(tta)
        self.current_angle = None
        self.prev_kp = None
        self.failures = 0
        self.count = 0

        if self.orientation_mode != "auto":
            angle = int(self.orientation_mode)
            if angle not in (0, 90, 180, 270):
                raise ValueError("orientation must be auto, 0, 90, 180 or 270")
            self.current_angle = angle

    def reset(self):
        self.prev_kp = None
        self.failures = 0
        self.count = 0
        if self.orientation_mode == "auto":
            self.current_angle = None

    def infer(self, frame):
        self.count += 1
        need_search = self.current_angle is None
        if self.recheck_every > 0 and self.count % self.recheck_every == 0:
            need_search = True
        if self.orientation_mode == "auto" and self.failures >= self.relock_failures:
            need_search = True

        if need_search and self.orientation_mode == "auto":
            kp, angle, _ = best_orientation(
                self.model, frame, self.imgsz, self.det_conf,
                self.device, self.prev_kp, self.tta,
            )
            if kp is None:
                self.failures += 1
                return None
            self.current_angle = angle
            self.prev_kp = kp.copy()
            self.failures = 0
            return kp

        cand = _predict_angle(
            self.model, frame, self.current_angle, self.imgsz,
            self.det_conf, self.device, self.prev_kp, tta=self.tta,
        )
        if cand is None:
            self.failures += 1
            return None

        if cand.score < 0.35 and self.orientation_mode == "auto":
            self.failures += 1
            return None

        self.prev_kp = cand.kp.copy()
        self.failures = 0
        return cand.kp


def clean(kp, frame=None, drop_head=False, reject_equipment=False):
    """Very light cleaning. Model geometry is handled by temporal refinement."""
    if kp is None:
        return None
    kp = kp.copy()
    if drop_head:
        for i in HEAD:
            kp[i, 2] = 0.0
    return kp


class SkeletonTracker:
    """Causal filter used before the offline sequence refinement.

    It only removes obvious single-frame failures. It does not independently
    swap left/right joints and it does not freeze points at old coordinates.
    """

    def __init__(self, kpt_conf=0.22, alpha=0.88, jump_torso=1.45,
                 bone_torso_max=1.95):
        self.kpt_conf = float(kpt_conf)
        self.alpha = float(alpha)
        self.jump_torso = float(jump_torso)
        self.bone_torso_max = float(bone_torso_max)
        self.prev = None
        self.prev_torso = None

    def reset(self):
        self.prev = None
        self.prev_torso = None

    def _joint_threshold(self, i):
        if i in CORE:
            return max(0.18, self.kpt_conf - 0.04)
        if i in MID_JOINTS:
            return self.kpt_conf
        if i in DISTAL:
            return self.kpt_conf + 0.03
        if i in HEAD:
            return max(0.16, self.kpt_conf - 0.06)
        return self.kpt_conf

    def update(self, kp):
        if kp is None:
            return None
        kp = kp.copy()

        for i in range(17):
            if kp[i, 2] < self._joint_threshold(i):
                kp[i, 2] = 0.0

        torso = torso_length(kp, max(0.18, self.kpt_conf - 0.04))
        scale = torso if torso is not None else self.prev_torso

        if self.prev is not None and scale is not None and scale > 1:
            for i in BODY + [0, 3, 4]:
                thr = self._joint_threshold(i)
                if _visible(kp, i, thr) and _visible(self.prev, i, thr):
                    jump = np.linalg.norm(kp[i, :2] - self.prev[i, :2])
                    factor = 1.65 if i in DISTAL else 1.25 if i in MID_JOINTS else 1.0
                    if jump > self.jump_torso * factor * scale:
                        kp[i, 2] = 0.0

        torso2 = torso_length(kp, max(0.18, self.kpt_conf - 0.04))
        scale = torso2 if torso2 is not None else scale
        if scale is not None and scale > 1:
            for parent, child in BONES:
                if _visible(kp, parent, 0.18) and _visible(kp, child, 0.18):
                    length = np.linalg.norm(kp[parent, :2] - kp[child, :2])
                    if length > self.bone_torso_max * scale:
                        kp[child, 2] = 0.0

        if self.prev is not None:
            for i in BODY + [0, 3, 4]:
                thr = self._joint_threshold(i)
                if _visible(kp, i, thr) and _visible(self.prev, i, thr):
                    kp[i, :2] = self.alpha * kp[i, :2] + (1 - self.alpha) * self.prev[i, :2]

        torso3 = torso_length(kp, max(0.18, self.kpt_conf - 0.04))
        if torso3 is not None:
            self.prev_torso = torso3 if self.prev_torso is None else 0.92 * self.prev_torso + 0.08 * torso3
        self.prev = kp.copy()
        return kp


def _swap_chains(kp, left_chain, right_chain):
    out = kp.copy()
    for l, r in zip(left_chain, right_chain):
        out[[l, r]] = out[[r, l]]
    return out


def _chain_transition_cost(prev_kp, cur_kp, joints, scale):
    vals = []
    for i in joints:
        if prev_kp[i, 2] > 0.12 and cur_kp[i, 2] > 0.12:
            w = min(prev_kp[i, 2], cur_kp[i, 2])
            vals.append(w * np.linalg.norm(cur_kp[i, :2] - prev_kp[i, :2]) / max(scale, 1.0))
    if not vals:
        return 0.45
    return float(np.mean(vals))


def _stabilize_pair_identity(sequence, left_chain, right_chain, switch_penalty=0.12):
    """Whole-chain left/right assignment using dynamic programming.

    A state of 1 swaps the complete left/right chain for that frame. This never
    creates the broken anatomy that occurs when elbows, wrists or knees are
    swapped independently.
    """
    n = len(sequence)
    if n < 2:
        return sequence

    candidates = []
    for kp in sequence:
        base = kp.copy()
        candidates.append([base, _swap_chains(base, left_chain, right_chain)])

    dp = np.full((n, 2), np.inf, float)
    back = np.zeros((n, 2), np.int8)
    dp[0, 0] = 0.0
    dp[0, 1] = 0.04  # slight preference for model's original labels

    joint_union = left_chain + right_chain
    for t in range(1, n):
        torso_vals = [torso_length(candidates[t][s], 0.15) for s in (0, 1)]
        torso_vals += [torso_length(candidates[t - 1][s], 0.15) for s in (0, 1)]
        torso_vals = [x for x in torso_vals if x is not None and x > 1]
        scale = float(np.median(torso_vals)) if torso_vals else 100.0

        for s in (0, 1):
            for ps in (0, 1):
                cost = _chain_transition_cost(candidates[t - 1][ps], candidates[t][s], joint_union, scale)
                if s != ps:
                    cost += switch_penalty
                v = dp[t - 1, ps] + cost
                if v < dp[t, s]:
                    dp[t, s] = v
                    back[t, s] = ps

    state = int(np.argmin(dp[-1]))
    states = [0] * n
    states[-1] = state
    for t in range(n - 1, 0, -1):
        states[t - 1] = int(back[t, states[t]])

    return [candidates[t][states[t]] for t in range(n)]


def _contiguous_segments(valid):
    segs = []
    start = None
    for i, ok in enumerate(valid):
        if ok and start is None:
            start = i
        if start is not None and ((not ok) or i == len(valid) - 1):
            end = i if not ok else i + 1
            segs.append((start, end))
            start = None
    return segs


def refine_sequence(sequence, fps, kpt_conf=0.22, max_gap_s=0.28,
                    smooth=True, stabilize_lr=True):
    """Offline refinement using both past and future frames.

    This is intentionally separate from inference. A video is not a set of
    unrelated images, so isolated predictions are corrected using temporal and
    anatomical consistency before any metric or annotated video is generated.
    """
    if not sequence:
        return [], {"interpolated_points": 0, "temporal_outliers_removed": 0}

    seq = []
    for kp in sequence:
        if kp is None:
            seq.append(np.zeros((17, 3), float))
        else:
            seq.append(kp.copy())

    if stabilize_lr:
        seq = _stabilize_pair_identity(seq, ARM_LEFT, ARM_RIGHT, switch_penalty=0.10)
        seq = _stabilize_pair_identity(seq, LEG_LEFT, LEG_RIGHT, switch_penalty=0.10)

    arr = np.stack(seq, axis=0)  # T x 17 x 3
    T = len(arr)

    # Joint-specific confidence gates.
    for j in range(17):
        if j in CORE:
            thr = max(0.18, kpt_conf - 0.04)
        elif j in MID_JOINTS:
            thr = kpt_conf
        elif j in DISTAL:
            thr = kpt_conf + 0.03
        else:
            thr = max(0.16, kpt_conf - 0.06)
        arr[arr[:, j, 2] < thr, j, 2] = 0.0

    torso = np.full(T, np.nan, float)
    for t in range(T):
        tl = torso_length(arr[t], max(0.16, kpt_conf - 0.05))
        if tl is not None:
            torso[t] = tl
    if np.isfinite(torso).any():
        med_torso = float(np.nanmedian(torso))
    else:
        med_torso = 100.0
    torso_fill = torso.copy()
    bad = ~np.isfinite(torso_fill)
    torso_fill[bad] = med_torso

    removed = 0

    # 1) Bidirectional temporal spike rejection.
    for j in BODY + [0, 3, 4]:
        for t in range(1, T - 1):
            if arr[t, j, 2] <= 0 or arr[t - 1, j, 2] <= 0 or arr[t + 1, j, 2] <= 0:
                continue
            predicted = 0.5 * (arr[t - 1, j, :2] + arr[t + 1, j, :2])
            error = np.linalg.norm(arr[t, j, :2] - predicted)
            if j in DISTAL:
                lim = 0.62 * torso_fill[t]
            elif j in MID_JOINTS:
                lim = 0.46 * torso_fill[t]
            elif j in CORE:
                lim = 0.34 * torso_fill[t]
            else:
                lim = 0.42 * torso_fill[t]
            if error > lim and arr[t, j, 2] < 0.78:
                arr[t, j, 2] = 0.0
                removed += 1

    # 2) Learn robust bone ratios from the complete clip, then reject clear outliers.
    bone_ratio_med = {}
    for a, b in BONES:
        vals = []
        for t in range(T):
            if arr[t, a, 2] >= 0.35 and arr[t, b, 2] >= 0.35 and torso_fill[t] > 1:
                vals.append(np.linalg.norm(arr[t, a, :2] - arr[t, b, :2]) / torso_fill[t])
        if len(vals) >= 8:
            bone_ratio_med[(a, b)] = float(np.median(vals))

    for (a, b), med in bone_ratio_med.items():
        lo = max(0.04, 0.30 * med)
        hi = min(2.0, 2.15 * med)
        for t in range(T):
            if arr[t, a, 2] <= 0 or arr[t, b, 2] <= 0:
                continue
            ratio = np.linalg.norm(arr[t, a, :2] - arr[t, b, :2]) / max(torso_fill[t], 1.0)
            if (ratio < lo or ratio > hi) and arr[t, b, 2] < 0.78:
                arr[t, b, 2] = 0.0
                removed += 1

    # 3) Interpolate only short interior gaps. Synthetic confidence is kept lower
    # than the surrounding detections so the source remains conservative.
    max_gap = max(1, int(round(max_gap_s * fps)))
    interpolated = 0
    for j in range(17):
        valid = arr[:, j, 2] > 0
        t = 0
        while t < T:
            if valid[t]:
                t += 1
                continue
            s = t
            while t < T and not valid[t]:
                t += 1
            e = t
            gap = e - s
            if s == 0 or e >= T or gap > max_gap:
                continue
            p0, p1 = arr[s - 1, j, :2].copy(), arr[e, j, :2].copy()
            c0, c1 = arr[s - 1, j, 2], arr[e, j, 2]
            if c0 <= 0 or c1 <= 0:
                continue
            for k in range(gap):
                w = (k + 1) / (gap + 1)
                arr[s + k, j, :2] = (1 - w) * p0 + w * p1
                arr[s + k, j, 2] = max(kpt_conf + 0.04, 0.68 * min(c0, c1))
                interpolated += 1
            valid[s:e] = True

    # 4) Light Savitzky-Golay smoothing on contiguous valid tracks. This reduces
    # jitter but keeps true kick and arm oscillation.
    if smooth:
        for j in range(17):
            valid = arr[:, j, 2] > 0
            for s, e in _contiguous_segments(valid):
                n = e - s
                if n < 5:
                    continue
                win = min(7, n if n % 2 == 1 else n - 1)
                if win < 5:
                    continue
                for d in (0, 1):
                    arr[s:e, j, d] = savgol_filter(arr[s:e, j, d], win, 2, mode="interp")

    refined = [arr[t].copy() for t in range(T)]
    stats = {
        "interpolated_points": int(interpolated),
        "temporal_outliers_removed": int(removed),
        "max_gap_frames": int(max_gap),
    }
    return refined, stats


def interpolate_pose(sample_indices, sequence, frame_index):
    """Interpolate refined sample poses for smooth full-frame annotation."""
    if not sequence:
        return None
    idxs = np.asarray(sample_indices, int)
    pos = int(np.searchsorted(idxs, frame_index))
    if pos < len(idxs) and idxs[pos] == frame_index:
        return sequence[pos].copy()
    if pos == 0:
        return sequence[0].copy()
    if pos >= len(idxs):
        return sequence[-1].copy()

    i0, i1 = pos - 1, pos
    f0, f1 = idxs[i0], idxs[i1]
    if f1 == f0:
        return sequence[i0].copy()
    w = (frame_index - f0) / float(f1 - f0)
    a, b = sequence[i0], sequence[i1]
    out = np.zeros((17, 3), float)
    for j in range(17):
        va, vb = a[j, 2] > 0, b[j, 2] > 0
        if va and vb:
            out[j, :2] = (1 - w) * a[j, :2] + w * b[j, :2]
            out[j, 2] = min(a[j, 2], b[j, 2])
        elif va and w <= 0.35:
            out[j] = a[j]
        elif vb and w >= 0.65:
            out[j] = b[j]
    return out


def quality_filter(kp, conf_min=0.28):
    if kp is None:
        return None, False
    out = kp.copy()
    for i in BODY:
        if out[i, 2] < conf_min:
            out[i, 2] = 0.0
    sh, hp = shoulder_hip_points(out, max(0.18, conf_min - 0.05))
    return out, sh is not None and hp is not None


def draw_colored(frame, kp, threshold=0.28, draw_head=True, legend=False):
    if kp is None:
        return

    # Slightly higher visual threshold for wrists/ankles. It is better to omit a
    # doubtful distal point than draw a confident-looking wrong limb.
    def ok(i):
        thr = threshold + 0.03 if i in DISTAL else threshold
        return _visible(kp, i, thr)

    for edges, colour in LIMB_GROUPS.values():
        for a, b in edges:
            if ok(a) and ok(b):
                cv2.line(frame,
                         (int(round(kp[a, 0])), int(round(kp[a, 1]))),
                         (int(round(kp[b, 0])), int(round(kp[b, 1]))),
                         colour, 3, cv2.LINE_AA)
    for i in BODY:
        if ok(i):
            cv2.circle(frame, (int(round(kp[i, 0])), int(round(kp[i, 1]))),
                       4, (255, 255, 255), -1, cv2.LINE_AA)

    if draw_head:
        hp = head_point(kp, max(0.18, threshold - 0.05))
        if hp is not None:
            cv2.circle(frame, (int(round(hp[0])), int(round(hp[1]))),
                       5, (255, 255, 255), -1, cv2.LINE_AA)

    if legend:
        y = 28
        for name, (_, colour) in LIMB_GROUPS.items():
            cv2.putText(frame, name, (14, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, colour, 2, cv2.LINE_AA)
            y += 22


def extract(video, out_csv, model_name="yolo11x-pose.pt", skip=2, imgsz=1280,
            det_conf=0.12, kpt_conf=0.22, orientation="auto", device=None,
            tta=False, annotate=False, astart=0.0, adur=12.0):
    model = YOLO(model_name)
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    estimator = PoseEstimator(model, imgsz, det_conf, orientation,
                              device=device, tta=tta)
    tracker = SkeletonTracker(kpt_conf=kpt_conf)

    raw, indices = [], []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % skip == 0:
            kp = tracker.update(clean(estimator.infer(frame), frame, drop_head=False))
            raw.append(kp)
            indices.append(idx)
        idx += 1
    cap.release()

    refined, _ = refine_sequence(raw, fps / skip, kpt_conf=kpt_conf)
    header = ["frame", "t"]
    for name in KEYPOINTS:
        header += [f"{name}_x", f"{name}_y", f"{name}_c"]
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for fi, kp in zip(indices, refined):
            row = [fi, round(fi / fps, 3)]
            for x, y, c in kp:
                row += [round(float(x), 1), round(float(y), 1), round(float(c), 3)]
            w.writerow(row)
    print(f"Done. {len(refined)} refined pose samples written to {out_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Accuracy-first swimming pose extraction.")
    ap.add_argument("video")
    ap.add_argument("--out", default="pose_data.csv")
    ap.add_argument("--model", default="yolo11x-pose.pt")
    ap.add_argument("--skip", type=int, default=2)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--det-conf", type=float, default=0.12)
    ap.add_argument("--kpt-conf", type=float, default=0.22)
    ap.add_argument("--orientation", default="auto", choices=["auto", "0", "90", "180", "270"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--tta", action="store_true", help="horizontal-flip test-time augmentation; slower but more robust")
    args = ap.parse_args()
    extract(args.video, args.out, args.model, args.skip, args.imgsz,
            args.det_conf, args.kpt_conf, args.orientation, args.device,
            args.tta)

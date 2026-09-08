"""
pose_markerless.py
Robust markerless pose extraction for swimming footage.

This version deliberately removes the changes that caused the recent regression:
  * no tight swimmer crop by default
  * no CLAHE/colour enhancement by default
  * no independent left/right swapping of individual joints
  * no "hold the old point" teleport correction
  * no whole-frame rejection just because one limb is uncertain

Instead it uses:
  1. rotate-to-upright with orientation locking/hysteresis
  2. person continuity tracking rather than always taking the largest box
  3. anatomical scoring when selecting an orientation
  4. conservative confidence, bone-length and teleport filtering
  5. light temporal smoothing only after a measurement passes validation
  6. support for both stock COCO-17 pose models and the custom SWIM-14 layout

The internal layout is always canonical COCO-17. A 14-keypoint swimmer model is
mapped into that layout automatically so analyze_swimmer.py can use either model.
"""

import argparse
import csv
import math
import os
from dataclasses import dataclass

import cv2
import numpy as np
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

# parent -> child. If a bone is impossible, the distal/child joint is removed.
BONES = [
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]

LIMB_GROUPS = {
    "left arm":  ([(5, 7), (7, 9)], (255, 0, 255)),
    "right arm": ([(6, 8), (8, 10)], (0, 220, 0)),
    "left leg":  ([(11, 13), (13, 15)], (0, 165, 255)),
    "right leg": ([(12, 14), (14, 16)], (0, 255, 255)),
    "torso":     ([(5, 6), (11, 12), (5, 11), (6, 12)], (255, 170, 30)),
}

# SWIM-14 -> canonical COCO-17 locations.
# SWIM-14 order:
# 0 head_center, 1 neck, 2 L shoulder, 3 R shoulder, 4 L elbow, 5 R elbow,
# 6 L wrist, 7 R wrist, 8 L hip, 9 R hip, 10 L knee, 11 R knee,
# 12 L ankle, 13 R ankle.
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
    """Return a 17x3 canonical keypoint array from COCO-17 or SWIM-14."""
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
    raise ValueError(
        f"Unsupported pose layout with {kp.shape[0]} keypoints. "
        "Expected COCO-17 or SWIM-14."
    )


def rotate_bound(img, angle):
    """Rotate without clipping and return image + affine matrix."""
    h, w = img.shape[:2]
    c_x, c_y = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((c_x, c_y), angle, 1.0)
    c, s = abs(M[0, 0]), abs(M[0, 1])
    new_w = int(h * s + w * c)
    new_h = int(h * c + w * s)
    M[0, 2] += new_w / 2.0 - c_x
    M[1, 2] += new_h / 2.0 - c_y
    return cv2.warpAffine(img, M, (new_w, new_h)), M


def _map_back(kp, M):
    """Map canonical keypoints from rotated image coordinates to source image."""
    kp = kp.copy()
    inv = cv2.invertAffineTransform(M)
    for i in range(kp.shape[0]):
        if kp[i, 2] <= 0:
            continue
        x, y = kp[i, 0], kp[i, 1]
        kp[i, 0] = inv[0, 0] * x + inv[0, 1] * y + inv[0, 2]
        kp[i, 1] = inv[1, 0] * x + inv[1, 1] * y + inv[1, 2]
    return kp


def _visible(kp, i, thr=0.25):
    return kp is not None and kp[i, 2] >= thr


def shoulder_hip_points(kp, thr=0.25):
    """Return a trustworthy shoulder point and hip point for trunk geometry.

    Priority:
      1. midpoint of both shoulders + midpoint of both hips
      2. same-side left shoulder/hip
      3. same-side right shoulder/hip

    We intentionally do NOT pair a left shoulder with a right hip because that
    creates the diagonal torso failures visible in the recent output.
    """
    if kp is None:
        return None, None

    lsh, rsh = _visible(kp, 5, thr), _visible(kp, 6, thr)
    lhp, rhp = _visible(kp, 11, thr), _visible(kp, 12, thr)

    if lsh and rsh and lhp and rhp:
        sh = (kp[5, :2] + kp[6, :2]) / 2.0
        hp = (kp[11, :2] + kp[12, :2]) / 2.0
        return sh, hp
    if lsh and lhp:
        return kp[5, :2].copy(), kp[11, :2].copy()
    if rsh and rhp:
        return kp[6, :2].copy(), kp[12, :2].copy()
    return None, None


def torso_length(kp, thr=0.25):
    sh, hp = shoulder_hip_points(kp, thr)
    if sh is None or hp is None:
        return None
    d = float(np.linalg.norm(sh - hp))
    return d if d > 1e-6 else None


def head_point(kp, thr=0.25):
    """Return a robust head point.

    For SWIM-14, canonical index 0 is already head_center.
    For COCO-17, use the mean of confident nose/ears instead of blindly trusting
    one landmark. Eyes are intentionally ignored because goggles/reflections are
    common in swimming footage.
    """
    if kp is None:
        return None
    pts = [kp[i, :2] for i in (0, 3, 4) if _visible(kp, i, thr)]
    if not pts:
        return None
    return np.mean(pts, axis=0)


def body_center(kp, thr=0.20):
    pts = [kp[i, :2] for i in CORE if _visible(kp, i, thr)]
    if len(pts) >= 2:
        return np.mean(pts, axis=0)
    pts = [kp[i, :2] for i in BODY if _visible(kp, i, thr)]
    return np.mean(pts, axis=0) if pts else None


def _anatomical_score(kp, image_diag, prev_kp=None, kpt_thr=0.20):
    """Score a pose for orientation/person selection.

    Confidence is important, but anatomical continuity and plausible bone sizes
    are also rewarded. This prevents choosing a high-confidence spider-web pose.
    """
    if kp is None:
        return -1e9

    confs = np.array([kp[i, 2] for i in BODY], dtype=float)
    mean_conf = float(np.mean(confs))
    visible_n = int(np.sum(confs >= kpt_thr))

    score = 1.6 * mean_conf + 0.035 * visible_n

    torso = torso_length(kp, kpt_thr)
    if torso is not None:
        score += 0.45
        if torso < 0.015 * image_diag:
            score -= 0.5

        good_bones = 0
        bad_bones = 0
        for a, b in BONES:
            if _visible(kp, a, kpt_thr) and _visible(kp, b, kpt_thr):
                ratio = np.linalg.norm(kp[a, :2] - kp[b, :2]) / max(torso, 1e-6)
                if 0.08 <= ratio <= 1.75:
                    good_bones += 1
                elif ratio > 2.2:
                    bad_bones += 1
        score += 0.07 * good_bones - 0.28 * bad_bones

    if prev_kp is not None:
        cur_c = body_center(kp, kpt_thr)
        prv_c = body_center(prev_kp, kpt_thr)
        if cur_c is not None and prv_c is not None:
            d = np.linalg.norm(cur_c - prv_c) / max(image_diag, 1e-6)
            score += 0.45 * math.exp(-d / 0.12)
            if d > 0.35:
                score -= 0.8

        common = [i for i in CORE if _visible(kp, i, kpt_thr) and _visible(prev_kp, i, kpt_thr)]
        if common:
            d = np.mean([np.linalg.norm(kp[i, :2] - prev_kp[i, :2]) for i in common])
            score += 0.25 * math.exp(-(d / max(image_diag, 1e-6)) / 0.08)

    return float(score)


@dataclass
class PoseCandidate:
    kp: np.ndarray
    score: float
    angle: int
    det_conf: float
    area: float


def _predict_angle(model, frame, angle, imgsz, det_conf, device=None, prev_kp=None):
    """Run one orientation and return the best person candidate."""
    rot, M = rotate_bound(frame, angle)
    kwargs = dict(imgsz=imgsz, conf=det_conf, verbose=False, max_det=8)
    if device is not None and device != "":
        kwargs["device"] = device
    result = model.predict(rot, **kwargs)[0]

    if result.boxes is None or len(result.boxes) == 0 or result.keypoints is None:
        return None

    H, W = frame.shape[:2]
    diag = math.hypot(W, H)
    areas = (result.boxes.xywh[:, 2] * result.boxes.xywh[:, 3]).detach().cpu().numpy()
    box_confs = result.boxes.conf.detach().cpu().numpy() if result.boxes.conf is not None else np.ones(len(areas))

    best = None
    for j in range(len(areas)):
        raw = result.keypoints.data[j].detach().cpu().numpy()
        kp = _canonicalize(raw)
        kp = _map_back(kp, M)
        pose_score = _anatomical_score(kp, diag, prev_kp=prev_kp, kpt_thr=0.20)
        area_frac = float(areas[j]) / max(float(rot.shape[0] * rot.shape[1]), 1.0)
        score = pose_score + 0.30 * float(box_confs[j]) + 0.12 * min(area_frac / 0.20, 1.0)
        cand = PoseCandidate(kp=kp, score=score, angle=int(angle),
                             det_conf=float(box_confs[j]), area=float(areas[j]))
        if best is None or cand.score > best.score:
            best = cand
    return best


def best_orientation(model, frame, imgsz=1280, conf=0.15, device=None, prev_kp=None):
    """Evaluate 0/90/180/270 and return best canonical pose + angle + score."""
    best = None
    for angle in (0, 90, 180, 270):
        cand = _predict_angle(model, frame, angle, imgsz, conf, device, prev_kp)
        if cand is not None and (best is None or cand.score > best.score):
            best = cand
    if best is None:
        return None, 0, -1e9
    return best.kp, best.angle, best.score


class PoseEstimator:
    """Stateful swimmer pose estimator with orientation lock.

    The previous version reselected orientation frequently. A wrong 90/270 flip
    can reverse limb semantics and create crossing skeletons. Here we lock the
    chosen orientation and only re-search after repeated failures or a clearly
    poor pose.
    """

    def __init__(self, model, imgsz=1280, det_conf=0.15, orientation="auto",
                 device=None, relock_failures=3, recheck_every=0):
        self.model = model
        self.imgsz = int(imgsz)
        self.det_conf = float(det_conf)
        self.device = device
        self.orientation_mode = str(orientation).lower()
        self.relock_failures = int(relock_failures)
        self.recheck_every = int(recheck_every)
        self.current_angle = None
        self.prev_kp = None
        self.failures = 0
        self.count = 0

        if self.orientation_mode != "auto":
            val = int(self.orientation_mode)
            if val not in (0, 90, 180, 270):
                raise ValueError("orientation must be auto, 0, 90, 180 or 270")
            self.current_angle = val

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
        if self.failures >= self.relock_failures and self.orientation_mode == "auto":
            need_search = True

        if need_search and self.orientation_mode == "auto":
            kp, angle, score = best_orientation(
                self.model, frame, self.imgsz, self.det_conf,
                device=self.device, prev_kp=self.prev_kp,
            )
            if kp is not None:
                self.current_angle = angle
                self.prev_kp = kp.copy()
                self.failures = 0
                return kp
            self.failures += 1
            return None

        cand = _predict_angle(
            self.model, frame, self.current_angle, self.imgsz,
            self.det_conf, self.device, self.prev_kp,
        )
        if cand is None:
            self.failures += 1
            return None

        # A very low-scoring pose is treated as a failure instead of being drawn.
        if cand.score < 0.45 and self.orientation_mode == "auto":
            self.failures += 1
            if self.failures >= self.relock_failures:
                kp, angle, _ = best_orientation(
                    self.model, frame, self.imgsz, self.det_conf,
                    device=self.device, prev_kp=self.prev_kp,
                )
                if kp is not None:
                    self.current_angle = angle
                    self.prev_kp = kp.copy()
                    self.failures = 0
                    return kp
            return None

        self.prev_kp = cand.kp.copy()
        self.failures = 0
        return cand.kp


def is_machine(frame, x, y, win=5):
    """Reject only strongly red/yellow equipment patches.

    The previous blue-patch rejection could classify normal pool water as a
    breathing tube. That is intentionally removed.
    """
    h, w = frame.shape[:2]
    xi, yi = int(round(x)), int(round(y))
    if not (0 <= xi < w and 0 <= yi < h):
        return True
    patch = frame[max(0, yi - win):min(h, yi + win + 1),
                  max(0, xi - win):min(w, xi + win + 1)]
    if patch.size == 0:
        return False
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    red = ((H < 10) | (H > 170)) & (S > 145) & (V > 100)
    yellow = (H > 18) & (H < 38) & (S > 145) & (V > 120)
    return float((red | yellow).mean()) > 0.65


def clean(kp, frame=None, drop_head=False, reject_equipment=False):
    """Light cleaning only. Do not aggressively erase valid underwater joints."""
    if kp is None:
        return None
    kp = kp.copy()
    if drop_head:
        for i in HEAD:
            kp[i, 2] = 0.0
    if reject_equipment and frame is not None:
        for i in BODY:
            if kp[i, 2] > 0.25 and is_machine(frame, kp[i, 0], kp[i, 1]):
                kp[i, 2] = 0.0
    return kp


class SkeletonTracker:
    """Conservative temporal/anatomical filter.

    Important design choices:
      * never swap individual L/R joints independently
      * never freeze a bad joint at the previous location
      * an impossible distal joint is removed rather than forcing a fake bone
      * smoothing is light and happens only after the point passes validation
    """

    def __init__(self, kpt_conf=0.25, alpha=0.82,
                 jump_torso=1.25, bone_torso_max=1.85,
                 bone_history_ratio=2.0):
        self.kpt_conf = float(kpt_conf)
        self.alpha = float(alpha)
        self.jump_torso = float(jump_torso)
        self.bone_torso_max = float(bone_torso_max)
        self.bone_history_ratio = float(bone_history_ratio)
        self.prev = None
        self.prev_torso = None
        self.bone_ema = {}

    def reset(self):
        self.prev = None
        self.prev_torso = None
        self.bone_ema = {}

    def update(self, kp):
        if kp is None:
            return None
        kp = kp.copy()

        # Confidence gate. Head is allowed to survive at a slightly lower value.
        for i in BODY:
            if kp[i, 2] < self.kpt_conf:
                kp[i, 2] = 0.0
        for i in HEAD:
            if kp[i, 2] < max(0.18, self.kpt_conf - 0.08):
                kp[i, 2] = 0.0

        torso = torso_length(kp, self.kpt_conf)
        scale = torso if torso is not None else self.prev_torso

        # Reject impossible jumps. Do NOT replace with old coordinates.
        if self.prev is not None and scale is not None and scale > 1:
            for i in BODY + [0, 3, 4]:
                if _visible(kp, i, self.kpt_conf) and _visible(self.prev, i, self.kpt_conf):
                    jump = np.linalg.norm(kp[i, :2] - self.prev[i, :2])
                    joint_factor = 1.45 if i in (9, 10, 15, 16) else 1.0
                    if jump > self.jump_torso * joint_factor * scale:
                        kp[i, 2] = 0.0

        # Bone plausibility. Drop the distal child only.
        torso_now = torso_length(kp, self.kpt_conf)
        scale = torso_now if torso_now is not None else scale
        if scale is not None and scale > 1:
            for parent, child in BONES:
                if not (_visible(kp, parent, self.kpt_conf) and _visible(kp, child, self.kpt_conf)):
                    continue
                length = float(np.linalg.norm(kp[parent, :2] - kp[child, :2]))
                hist = self.bone_ema.get((parent, child))
                bad = length > self.bone_torso_max * scale
                if hist is not None and hist > 1:
                    if length > self.bone_history_ratio * hist:
                        bad = True
                    # Do not reject short bones aggressively because side-view
                    # projection can legitimately foreshorten a limb.
                if bad:
                    kp[child, 2] = 0.0
                else:
                    self.bone_ema[(parent, child)] = length if hist is None else 0.92 * hist + 0.08 * length

        # Very light smoothing only on points that remain valid.
        if self.prev is not None:
            for i in BODY + [0, 3, 4]:
                if _visible(kp, i, self.kpt_conf) and _visible(self.prev, i, self.kpt_conf):
                    kp[i, :2] = self.alpha * kp[i, :2] + (1.0 - self.alpha) * self.prev[i, :2]

        torso_now = torso_length(kp, self.kpt_conf)
        if torso_now is not None:
            self.prev_torso = torso_now if self.prev_torso is None else 0.90 * self.prev_torso + 0.10 * torso_now
        self.prev = kp.copy()
        return kp


def quality_filter(kp, conf_min=0.30):
    """Compatibility helper: filter low confidence without deleting whole frames."""
    if kp is None:
        return None, False
    kp = kp.copy()
    for i in BODY:
        if kp[i, 2] < conf_min:
            kp[i, 2] = 0.0
    sh, hp = shoulder_hip_points(kp, conf_min)
    usable = sh is not None and hp is not None
    return kp, usable


def draw_colored(frame, kp, threshold=0.30, draw_head=False, legend=False):
    if kp is None:
        return
    for edges, colour in LIMB_GROUPS.values():
        for a, b in edges:
            if _visible(kp, a, threshold) and _visible(kp, b, threshold):
                cv2.line(
                    frame,
                    (int(round(kp[a, 0])), int(round(kp[a, 1]))),
                    (int(round(kp[b, 0])), int(round(kp[b, 1]))),
                    colour, 3, cv2.LINE_AA,
                )
    for i in BODY:
        if _visible(kp, i, threshold):
            cv2.circle(frame, (int(round(kp[i, 0])), int(round(kp[i, 1]))),
                       4, (255, 255, 255), -1, cv2.LINE_AA)

    if draw_head:
        hp = head_point(kp, max(0.20, threshold - 0.05))
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
            det_conf=0.15, kpt_conf=0.25, orientation="auto", device=None,
            annotate=False, astart=0.0, adur=12.0):
    model = YOLO(model_name)
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    estimator = PoseEstimator(model, imgsz, det_conf, orientation, device=device)
    tracker = SkeletonTracker(kpt_conf=kpt_conf)

    print(
        f"Video: {total} frames at {fps:.1f} fps. Every {skip} frame(s). "
        f"model={model_name}, imgsz={imgsz}, orientation={orientation}, "
        f"det_conf={det_conf}, kpt_conf={kpt_conf}."
    )

    rows = []
    idx = processed = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % skip != 0:
            idx += 1
            continue

        kp = estimator.infer(frame)
        kp = clean(kp, frame, drop_head=False, reject_equipment=False)
        kp = tracker.update(kp)
        t = idx / fps

        if kp is not None:
            row = [idx, round(t, 3)]
            for x, y, c in kp:
                row += [round(float(x), 1), round(float(y), 1), round(float(c), 3)]
        else:
            row = [idx, round(t, 3)] + [0, 0, 0] * 17
        rows.append(row)

        processed += 1
        idx += 1
        if processed % 200 == 0:
            print(f"  {processed} frames processed (t={t:.0f}s)", flush=True)

    cap.release()

    header = ["frame", "t"]
    for name in KEYPOINTS:
        header += [f"{name}_x", f"{name}_y", f"{name}_c"]
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"Done. {len(rows)} frames written to {out_csv}")

    if annotate:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        stem = os.path.splitext(os.path.basename(video))[0]
        out_path = os.path.join(OUTPUT_DIR, f"annotated_{stem}.mp4")
        _annotate_clip(model, video, out_path, astart, adur, imgsz,
                       det_conf, kpt_conf, orientation, device)


def _annotate_clip(model, video, out_path, start, dur, imgsz, det_conf,
                   kpt_conf, orientation, device):
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    estimator = PoseEstimator(model, imgsz, det_conf, orientation, device=device)
    tracker = SkeletonTracker(kpt_conf=kpt_conf)

    n_frames = int(dur * fps) if dur > 0 else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for _ in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        kp = estimator.infer(frame)
        kp = clean(kp, frame, drop_head=False, reject_equipment=False)
        kp = tracker.update(kp)
        draw_colored(frame, kp, threshold=max(0.28, kpt_conf), draw_head=True)
        cv2.putText(
            frame,
            f"pose | orientation {estimator.current_angle if estimator.current_angle is not None else '--'} deg",
            (14, h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
            (255, 255, 255), 2, cv2.LINE_AA,
        )
        writer.write(frame)

    writer.release()
    cap.release()
    print(f"Annotated clip written to {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Robust swimming pose extraction.")
    ap.add_argument("video")
    ap.add_argument("--out", default="pose_data.csv")
    ap.add_argument("--model", default="yolo11x-pose.pt")
    ap.add_argument("--skip", type=int, default=2)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--det-conf", type=float, default=0.15)
    ap.add_argument("--kpt-conf", type=float, default=0.25)
    ap.add_argument("--orientation", default="auto", choices=["auto", "0", "90", "180", "270"])
    ap.add_argument("--device", default=None, help="e.g. 0 for first CUDA GPU; omit for auto")
    ap.add_argument("--annotate", action="store_true")
    ap.add_argument("--astart", type=float, default=0.0)
    ap.add_argument("--adur", type=float, default=12.0, help="0 = whole video")
    args = ap.parse_args()

    extract(
        args.video, args.out, args.model, args.skip, args.imgsz,
        args.det_conf, args.kpt_conf, args.orientation, args.device,
        args.annotate, args.astart, args.adur,
    )

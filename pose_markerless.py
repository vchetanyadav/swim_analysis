"""
pose_markerless.py
Improved markerless pose extraction for top-view swim footage.

This fixes the three things that made the earlier skeleton misalign:

  1. Rotate-to-upright. The pose model was trained on upright people, so it
     scrambles limbs when the swimmer is horizontal. Before each detection we
     try the frame at 0, 90, 180 and 270 degrees and keep the orientation the
     model is most confident about, then map the joints back. This is what
     stopped the legs pointing the wrong way in testing.

  2. Head dropped. The nose/eyes/ears kept locking onto the red snorkel mask
     ("matching the machine"). We never draw or use them. None of the metrics
     need the head anyway.

  3. Red-mask rejection. Any joint that lands on strongly red pixels (the mask)
     is discarded, so equipment can never be mistaken for a body point.

It still will not be perfect on this footage (see README: markerless models are
out of their trained domain underwater). For exact joints use track_markers.py.

Output is a pose_data.csv with the same columns as before, so kinematics.py
runs on it unchanged.

Usage:
    python pose_markerless.py top.mp4 --out pose_data.csv
    python pose_markerless.py top.mp4 --out pose_data.csv --annotate check.mp4 --astart 100 --adur 12
    python pose_markerless.py top.mp4 --model yolo11x-pose.pt      # bigger, more accurate, slower
"""
import argparse
import csv
import os
import cv2
import numpy as np
from ultralytics import YOLO

OUTPUT_DIR = "output"                        # annotated clips go here
ANNOTATED_NAME = "annotated.mp4"             # legacy default (per-video names used below)

KEYPOINTS = ["nose", "l_eye", "r_eye", "l_ear", "r_ear",
             "l_shoulder", "r_shoulder", "l_elbow", "r_elbow",
             "l_wrist", "r_wrist", "l_hip", "r_hip",
             "l_knee", "r_knee", "l_ankle", "r_ankle"]
HEAD = [0, 1, 2, 3, 4]                      # never used or drawn
BODY = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
SKELETON = [(5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15),
            (12, 14), (14, 16), (5, 6), (11, 12), (5, 11), (6, 12)]

# Each limb drawn in its own colour so a mis-assigned limb is obvious at a glance.
LIMB_GROUPS = {
    "left arm":  ([(5, 7), (7, 9)],   (255, 0, 255)),    # magenta
    "right arm": ([(6, 8), (8, 10)],  (0, 220, 0)),      # green
    "left leg":  ([(11, 13), (13, 15)], (0, 165, 255)),  # orange
    "right leg": ([(12, 14), (14, 16)], (0, 255, 255)),  # yellow
    "torso":     ([(5, 6), (11, 12), (5, 11), (6, 12)], (255, 170, 30)),  # blue
}
REORIENT_EVERY = 15                         # re-check orientation this often (frames)


def draw_colored(frame, kp):
    """Draw the skeleton with each limb in its own colour, plus a small legend."""
    for edges, col in LIMB_GROUPS.values():
        for a, b in edges:
            if kp[a, 2] > 0.5 and kp[b, 2] > 0.5:
                cv2.line(frame, (int(kp[a, 0]), int(kp[a, 1])),
                         (int(kp[b, 0]), int(kp[b, 1])), col, 3)
    for i in BODY:
        if kp[i, 2] > 0.5:
            cv2.circle(frame, (int(kp[i, 0]), int(kp[i, 1])), 5, (255, 255, 255), -1)
    y = 34
    for name, (_, col) in LIMB_GROUPS.items():
        cv2.putText(frame, name, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        y += 24


def rotate_bound(img, angle):
    h, w = img.shape[:2]
    cX, cY = w / 2, h / 2
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nW, nH = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += nW / 2 - cX
    M[1, 2] += nH / 2 - cY
    return cv2.warpAffine(img, M, (nW, nH)), M


def _kp_for_angle(model, img, angle, imgsz, conf):
    rot, M = rotate_bound(img, angle)
    r = model.predict(rot, imgsz=imgsz, conf=conf, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return None, -1
    areas = (r.boxes.xywh[:, 2] * r.boxes.xywh[:, 3]).cpu().numpy()
    kp = r.keypoints.data[int(np.argmax(areas))].cpu().numpy()
    score = float(np.mean([kp[i, 2] for i in BODY]))
    # map joints from the rotated frame back to the original frame
    Minv = cv2.invertAffineTransform(M)
    out = kp.copy()
    for i in range(17):
        x, y = kp[i, 0], kp[i, 1]
        out[i, 0] = Minv[0, 0] * x + Minv[0, 1] * y + Minv[0, 2]
        out[i, 1] = Minv[1, 0] * x + Minv[1, 1] * y + Minv[1, 2]
    return out, score


def best_orientation(model, img, imgsz, conf):
    best_kp, best_score, best_angle = None, -1, 0
    for ang in (0, 90, 180, 270):
        kp, score = _kp_for_angle(model, img, ang, imgsz, conf)
        if kp is not None and score > best_score:
            best_kp, best_score, best_angle = kp, score, ang
    return best_kp, best_angle


def is_machine(img, x, y, win=6):
    """True if the patch around (x, y) is dominated by equipment colour:
    the red mask, the yellow connector, or the saturated-blue breathing tube.
    Submerged skin is low-saturation blue-green, so the high-saturation blue
    test spares the body while catching the solid tube."""
    h, w = img.shape[:2]
    xi, yi = int(round(x)), int(round(y))
    if not (0 <= xi < w and 0 <= yi < h):
        return True
    patch = img[max(0, yi - win):yi + win, max(0, xi - win):xi + win]
    if patch.size == 0:
        return False
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    red = ((H < 10) | (H > 170)) & (S > 110) & (V > 80)
    yellow = (H > 20) & (H < 35) & (S > 110) & (V > 120)
    tube_blue = (H > 100) & (H < 130) & (S > 120) & (V > 90)
    return (red | yellow | tube_blue).mean() > 0.5


def clean(kp, img, drop_head=True):
    """Set equipment joints to zero. With drop_head=True also zero the head
    landmarks (for drawing); pass drop_head=False when you need the head for
    head-position / neck / dip measurement."""
    kp = kp.copy()
    if drop_head:
        for i in HEAD:
            kp[i, 2] = 0.0
    for i in BODY:
        if kp[i, 2] > 0.5 and is_machine(img, kp[i, 0], kp[i, 1]):
            kp[i, 2] = 0.0
    return kp


# ---------------- cropped, contrast-boosted pose (big non-training accuracy win) ----------------
def enhance(bgr):
    """CLAHE on the L channel to recover limb edges in low-contrast blue water."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def _largest_box(model, frame, imgsz, conf):
    r = model.predict(frame, imgsz=imgsz, conf=conf, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return None
    areas = (r.boxes.xywh[:, 2] * r.boxes.xywh[:, 3]).cpu().numpy()
    x, y, w, h = r.boxes.xywh[int(np.argmax(areas))].cpu().numpy()
    return float(x), float(y), float(w), float(h)


def pose_cropped(model, frame, imgsz=640, conf=0.25, margin=0.28,
                 use_clahe=True, rotate=True):
    """Detect the swimmer, crop tightly (so limbs get many more pixels), boost
    contrast, run pose on the crop, and map the joints back to full-frame
    coordinates. This gives far better distal-joint accuracy than running the
    model on the whole 1920x1080 image, with no training required."""
    H, W = frame.shape[:2]
    box = _largest_box(model, frame, imgsz, conf)
    if box is None:
        return None
    x, y, w, h = box
    x0 = max(0, int(x - w / 2 - margin * w)); x1 = min(W, int(x + w / 2 + margin * w))
    y0 = max(0, int(y - h / 2 - margin * h)); y1 = min(H, int(y + h / 2 + margin * h))
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    if use_clahe:
        crop = enhance(crop)
    if rotate:
        kp, _ = best_orientation(model, crop, imgsz, conf)
    else:
        r = model.predict(crop, imgsz=imgsz, conf=conf, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return None
        a = (r.boxes.xywh[:, 2] * r.boxes.xywh[:, 3]).cpu().numpy()
        kp = r.keypoints.data[int(np.argmax(a))].cpu().numpy()
    if kp is None:
        return None
    kp = kp.copy()
    kp[:, 0] += x0
    kp[:, 1] += y0
    return kp


def _mid(kp, a, b):
    pa = kp[a, :2] if kp[a, 2] > 0.5 else None
    pb = kp[b, :2] if kp[b, 2] > 0.5 else None
    if pa is not None and pb is not None:
        return (pa + pb) / 2
    return pa if pa is not None else pb


class SkeletonTracker:
    """Temporal correctness layer: fixes left/right swaps, rejects bones that
    suddenly stretch, and holds joints that teleport further than the body could
    move. All thresholds are relative to torso length, not image size, so they
    scale with how close the swimmer is."""

    def __init__(self):
        self.prev = None
        self.bone_med = {}

    def update(self, kp):
        if kp is None:
            self.prev = None
            return None
        kp = kp.copy()
        sh, hp = _mid(kp, 5, 6), _mid(kp, 11, 12)
        torso = float(np.linalg.norm(sh - hp)) if (sh is not None and hp is not None) else None

        # 1) left/right swap correction against the previous frame
        if self.prev is not None:
            for l, r in [(9, 10), (15, 16), (7, 8), (13, 14), (5, 6), (11, 12)]:
                if all(kp[i, 2] > 0.5 for i in (l, r)) and all(self.prev[i, 2] > 0.5 for i in (l, r)):
                    keep = (np.linalg.norm(kp[l, :2] - self.prev[l, :2]) +
                            np.linalg.norm(kp[r, :2] - self.prev[r, :2]))
                    swap = (np.linalg.norm(kp[r, :2] - self.prev[l, :2]) +
                            np.linalg.norm(kp[l, :2] - self.prev[r, :2]))
                    if swap < keep:
                        kp[[l, r]] = kp[[r, l]]

        # 2) bone-length consistency (reject sudden elongation)
        if torso:
            for a, b in BONES:
                if kp[a, 2] > 0.5 and kp[b, 2] > 0.5:
                    ln = float(np.linalg.norm(kp[a, :2] - kp[b, :2]))
                    med = self.bone_med.get((a, b))
                    if med is not None and ln > 2.2 * med:
                        kp[b, 2] = 0.0
                    else:
                        self.bone_med[(a, b)] = ln if med is None else 0.9 * med + 0.1 * ln

        # 3) torso-scaled teleport rejection
        if self.prev is not None and torso:
            for i in BODY:
                if kp[i, 2] > 0.5 and self.prev[i, 2] > 0.5:
                    if np.linalg.norm(kp[i, :2] - self.prev[i, :2]) > 0.8 * torso:
                        kp[i, :2] = self.prev[i, :2]

        self.prev = kp.copy()
        return kp


# Bone pairs used for the plausibility check (index_a, index_b).
BONES = [(5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16)]


def quality_filter(kp, conf_min=0.6):
    """Make the skeleton honest: keep only joints the model is sure about, and
    drop the whole frame's limbs if the geometry is anatomically impossible.

    Two checks, aimed at the exact failures seen on curled underwater poses:
      1. Confidence gate: any joint below conf_min is dropped (removes the
         low-confidence guesses that produce stray stubs and crossing lines).
      2. Plausibility: bones should be similar in length to the torso. If any
         limb bone is far longer than the shoulder-hip span, or the two ankles
         (or two wrists) collapse onto the same point, the limbs are mis-assigned
         and are dropped, so bad frames are skipped rather than measured.

    Returns (kp, ok) where ok is False when the frame is too unreliable to use.
    """
    kp = kp.copy()
    for i in BODY:
        if kp[i, 2] < conf_min:
            kp[i, 2] = 0.0

    def pt(i):
        return kp[i, :2] if kp[i, 2] >= conf_min else None

    # torso scale = shoulder-midpoint to hip-midpoint distance
    sh = [pt(5), pt(6)]
    hp = [pt(11), pt(12)]
    sh = [p for p in sh if p is not None]
    hp = [p for p in hp if p is not None]
    if not sh or not hp:
        return kp, False                       # no torso reference, can't trust the frame
    torso = float(np.linalg.norm(np.mean(sh, axis=0) - np.mean(hp, axis=0)))
    if torso < 1e-3:
        return kp, False

    # 1) drop any implausibly long bone (limb crossing the body)
    for a, b in BONES:
        pa, pb = pt(a), pt(b)
        if pa is not None and pb is not None:
            if np.linalg.norm(pa - pb) > 1.8 * torso:
                kp[a, 2] = 0.0
                kp[b, 2] = 0.0

    # 2) left/right joints collapsing onto the same point = mis-assignment
    for l, r in [(9, 10), (15, 16), (13, 14), (7, 8)]:
        pl, pr = pt(l), pt(r)
        if pl is not None and pr is not None and np.linalg.norm(pl - pr) < 0.08 * torso:
            kp[l, 2] = 0.0
            kp[r, 2] = 0.0

    # frame is usable if enough body joints survived
    survivors = int(np.sum([kp[i, 2] >= conf_min for i in BODY]))
    return kp, survivors >= 6


class Smoother:
    """Light temporal smoothing to calm frame-to-frame jitter and reject sudden
    teleports (a joint jumping far in one frame is almost always a tracking
    error). Kept light on purpose so real limb oscillation, and therefore
    cadence, is preserved."""

    def __init__(self, diag, alpha=0.6, jump_frac=0.20):
        self.prev = {}                 # joint index -> (x, y)
        self.alpha = alpha             # weight on the current frame (higher = less smoothing)
        self.jump = jump_frac * diag   # a move bigger than this is treated as a teleport

    def apply(self, kp):
        kp = kp.copy()
        for i in BODY:
            if kp[i, 2] <= 0.5:
                continue
            cur = np.array([kp[i, 0], kp[i, 1]])
            if i in self.prev:
                prev = self.prev[i]
                if np.linalg.norm(cur - prev) > self.jump:
                    cur = prev                     # reject the teleport, hold last good spot
                else:
                    cur = self.alpha * cur + (1 - self.alpha) * prev   # ease toward it
            self.prev[i] = cur
            kp[i, 0], kp[i, 1] = cur
        return kp


def extract(video, out_csv, model_name="yolo11n-pose.pt", skip=3, imgsz=960, conf=0.25,
            annotate=False, astart=0.0, adur=12.0, smooth=False):
    model = YOLO(model_name)
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    diag = (cap.get(cv2.CAP_PROP_FRAME_WIDTH) ** 2 + cap.get(cv2.CAP_PROP_FRAME_HEIGHT) ** 2) ** 0.5
    smoother = Smoother(diag) if smooth else None
    print(f"Video: {total} frames at {fps:.1f} fps. Every {skip} frame(s), "
          f"rotate-to-upright on, head dropped, machine rejection on"
          f"{', smoothing on' if smooth else ''}.")

    rows = []
    idx = processed = 0
    cur_angle = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % skip != 0:
            idx += 1
            continue

        # Re-check orientation periodically; reuse it in between to stay fast.
        if processed % REORIENT_EVERY == 0:
            kp, cur_angle = best_orientation(model, frame, imgsz, conf)
        else:
            kp, _ = _kp_for_angle(model, frame, cur_angle, imgsz, conf)

        t = idx / fps
        if kp is not None:
            kp = clean(kp, frame)
            if smoother is not None:
                kp = smoother.apply(kp)
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
    for k in KEYPOINTS:
        header += [f"{k}_x", f"{k}_y", f"{k}_c"]
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"Done. {len(rows)} frames written to {out_csv}")

    if annotate:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        stem = os.path.splitext(os.path.basename(video))[0]
        out_path = os.path.join(OUTPUT_DIR, f"annotated_{stem}.mp4")
        _annotate_clip(model, video, out_path, astart, adur, imgsz, conf)


def _annotate_clip(model, video, out_path, start, dur, imgsz, conf):
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    diag = (w ** 2 + h ** 2) ** 0.5
    smoother = Smoother(diag)                 # always smooth the preview so it looks steady
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    cur_angle = 0
    for n in range(int(dur * fps)):
        ok, frame = cap.read()
        if not ok:
            break
        if n % REORIENT_EVERY == 0:
            kp, cur_angle = best_orientation(model, frame, imgsz, conf)
        else:
            kp, _ = _kp_for_angle(model, frame, cur_angle, imgsz, conf)
        if kp is not None:
            kp = clean(kp, frame)
            kp = smoother.apply(kp)
            draw_colored(frame, kp)
        cv2.putText(frame, "markerless (rotated, head dropped, colour-coded limbs)", (14, h - 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        writer.write(frame)
    writer.release()
    cap.release()
    print(f"Annotated clip written to {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Improved markerless pose extraction.")
    ap.add_argument("video")
    ap.add_argument("--out", default="pose_data.csv")
    ap.add_argument("--model", default="yolo11n-pose.pt",
                    help="use yolo11x-pose.pt for best accuracy (slower)")
    ap.add_argument("--skip", type=int, default=3)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--annotate", action="store_true",
                    help=f"also write an annotated check clip to {OUTPUT_DIR}/{ANNOTATED_NAME}")
    ap.add_argument("--astart", type=float, default=0.0, help="annotate clip start (s)")
    ap.add_argument("--adur", type=float, default=12.0, help="annotate clip length (s)")
    ap.add_argument("--smooth", action="store_true",
                    help="temporal smoothing + teleport rejection on the saved CSV too")
    args = ap.parse_args()
    extract(args.video, args.out, args.model, args.skip, args.imgsz, args.conf,
            args.annotate, args.astart, args.adur, args.smooth)

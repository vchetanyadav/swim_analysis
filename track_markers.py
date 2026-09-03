"""
track_markers.py
Exact joint tracking from bright coloured markers.

This is the reliable path. Instead of guessing joints with a pose model, you put
a small bright waterproof marker on each joint, give each joint a distinct
colour, and this script finds each colour's blob in every frame and takes its
centre as that joint. There is no left/right or arm/leg confusion, because each
colour is one specific joint.

It writes a pose_data.csv with the SAME columns as the pose scripts, so
kinematics.py runs on it unchanged.

--------------------------------------------------------------------------
CHOOSING MARKER COLOURS
  Avoid: blue / cyan (looks like the water), red (the snorkel mask is red),
         black (the straps are black), white (splash and glare are white).
  Good : magenta/pink, lime green, orange, yellow, purple.
  Start small: markers on both wrists and both ankles already give you exact
  stroke and kick cadence and left/right symmetry, which are the key metrics.
  Add elbows, knees, shoulders and hips later for angles.

CALIBRATION
  Colours underwater are not what they look like on the marker in your hand.
  Run calibrate_color.py on one frame to read the real HSV of each marker in the
  water, then paste the printed ranges into the MARKERS table below.
--------------------------------------------------------------------------

Usage:
    python track_markers.py top.mp4 --out pose_data.csv
    python track_markers.py top.mp4 --out pose_data.csv --annotate check.mp4 --astart 100 --adur 12
"""
import argparse
import csv
import cv2
import numpy as np

# The 17 COCO slots the CSV must contain (kinematics.py expects these names).
KEYPOINTS = ["nose", "l_eye", "r_eye", "l_ear", "r_ear",
             "l_shoulder", "r_shoulder", "l_elbow", "r_elbow",
             "l_wrist", "r_wrist", "l_hip", "r_hip",
             "l_knee", "r_knee", "l_ankle", "r_ankle"]

# ==========================================================================
# MARKER TABLE  -  one row per marked joint.
#   joint_name : (lower_HSV, upper_HSV, draw_BGR)
# HSV is OpenCV style: H 0-179, S 0-255, V 0-255.
# Edit these ranges using the numbers printed by calibrate_color.py.
# Start with the four below (wrists + ankles); add more rows as you add markers.
# ==========================================================================
MARKERS = {
    "l_wrist": ((145, 90, 90),  (170, 255, 255), (255, 0, 255)),   # magenta
    "r_wrist": ((35,  80, 90),  (75,  255, 255), (0, 255, 0)),     # lime green
    "l_ankle": ((10,  120, 120), (25, 255, 255), (0, 140, 255)),   # orange
    "r_ankle": ((25,  90, 120),  (35, 255, 255), (0, 255, 255)),   # yellow
    # "l_elbow": ((125, 80, 80), (145, 255, 255), (200, 0, 120)),  # purple  (example)
    # "r_elbow": (...),
}

MIN_BLOB_AREA = 25          # ignore colour specks smaller than this (pixels)
AREA_REF = 300              # blob area that counts as full confidence


def find_marker(hsv, bgr_frame, lower, upper):
    """Return (x, y, confidence) of the largest blob of this colour, or (0,0,0)."""
    mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0.0, 0.0, 0.0
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < MIN_BLOB_AREA:
        return 0.0, 0.0, 0.0
    M = cv2.moments(c)
    if M["m00"] == 0:
        return 0.0, 0.0, 0.0
    x = M["m10"] / M["m00"]
    y = M["m01"] / M["m00"]
    conf = min(1.0, area / AREA_REF)
    return x, y, conf


def track(video, out_csv, skip=3, annotate=None, astart=0.0, adur=12.0):
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    print(f"Tracking {len(MARKERS)} colour markers: {', '.join(MARKERS)}")

    rows = []
    idx = processed = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % skip != 0:
            idx += 1
            continue
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        found = {}
        for joint, (lo, hi, _) in MARKERS.items():
            found[joint] = find_marker(hsv, frame, lo, hi)

        row = [idx, round(idx / fps, 3)]
        for name in KEYPOINTS:
            if name in found:
                x, y, c = found[name]
                row += [round(x, 1), round(y, 1), round(c, 3)]
            else:
                row += [0, 0, 0]
        rows.append(row)
        processed += 1
        idx += 1
        if processed % 200 == 0:
            print(f"  {processed} frames processed", flush=True)
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
        _annotate(video, annotate, astart, adur, skip, fps)


def _annotate(video, out_path, start, dur, skip, fps):
    cap = cv2.VideoCapture(video)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for _ in range(int(dur * fps)):
        ok, frame = cap.read()
        if not ok:
            break
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        for joint, (lo, hi, col) in MARKERS.items():
            x, y, c = find_marker(hsv, frame, lo, hi)
            if c > 0:
                cv2.circle(frame, (int(x), int(y)), 10, col, -1)
                cv2.circle(frame, (int(x), int(y)), 12, (255, 255, 255), 2)
                cv2.putText(frame, joint, (int(x) + 12, int(y)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        cv2.putText(frame, "colour-marker tracking", (30, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        writer.write(frame)
    writer.release()
    cap.release()
    print(f"Annotated clip written to {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Track coloured joint markers.")
    ap.add_argument("video")
    ap.add_argument("--out", default="pose_data.csv")
    ap.add_argument("--skip", type=int, default=3)
    ap.add_argument("--annotate", default=None)
    ap.add_argument("--astart", type=float, default=0.0)
    ap.add_argument("--adur", type=float, default=12.0)
    args = ap.parse_args()
    track(args.video, args.out, args.skip, args.annotate, args.astart, args.adur)

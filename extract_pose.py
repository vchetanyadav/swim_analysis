"""
extract_pose.py
Step 1 of the pipeline: turn a swimming video into a table of joint positions.

For every processed frame it finds the swimmer, reads 17 body joints
(nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles) and writes
each joint's x, y and confidence to a CSV. Nothing is interpreted here; this
file only extracts the raw skeleton so the kinematics step can stay separate
and fast to re-run.

Usage:
    python extract_pose.py path/to/video.mp4
    python extract_pose.py path/to/video.mp4 --skip 3 --out pose_data.csv

The pose model (yolo11n-pose.pt) downloads automatically on first run.
"""
import argparse
import csv
import cv2
import numpy as np
from ultralytics import YOLO

# The 17 joints returned by the model, in order.
KEYPOINTS = [
    "nose", "l_eye", "r_eye", "l_ear", "r_ear",
    "l_shoulder", "r_shoulder", "l_elbow", "r_elbow",
    "l_wrist", "r_wrist", "l_hip", "r_hip",
    "l_knee", "r_knee", "l_ankle", "r_ankle",
]


def extract(video_path, out_csv="pose_data.csv", skip=3, imgsz=448,
            conf=0.25, model_name="yolo11n-pose.pt"):
    model = YOLO(model_name)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {total} frames at {fps:.1f} fps "
          f"({total / fps:.0f}s). Processing every {skip} frame(s) "
          f"-> effective {fps / skip:.1f} Hz.")

    rows = []
    idx = 0
    processed = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % skip != 0:
            idx += 1
            continue

        result = model.predict(frame, imgsz=imgsz, conf=conf, verbose=False)[0]
        t = idx / fps

        if result.boxes is not None and len(result.boxes) > 0:
            # The swimmer is the largest detection. This filters out smaller
            # bystanders on the pool deck that the detector sometimes picks up.
            areas = (result.boxes.xywh[:, 2] * result.boxes.xywh[:, 3]).cpu().numpy()
            best = int(np.argmax(areas))
            kp = result.keypoints.data[best].cpu().numpy()  # shape 17 x 3
            row = [idx, round(t, 3)]
            for x, y, c in kp:
                row += [round(float(x), 1), round(float(y), 1), round(float(c), 3)]
        else:
            # No person found in this frame: write zeros so the timeline stays intact.
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
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"Done. {len(rows)} frames written to {out_csv}")
    return out_csv


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Extract pose keypoints from a swim video.")
    ap.add_argument("video", help="path to the input video (e.g. top view)")
    ap.add_argument("--out", default="pose_data.csv", help="output CSV path")
    ap.add_argument("--skip", type=int, default=3,
                    help="process every Nth frame (3 keeps ~8 Hz at 25 fps)")
    ap.add_argument("--imgsz", type=int, default=448, help="inference image size")
    ap.add_argument("--conf", type=float, default=0.25, help="detection confidence")
    args = ap.parse_args()
    extract(args.video, args.out, args.skip, args.imgsz, args.conf)

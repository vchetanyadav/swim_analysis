"""
annotate_video.py
Optional step: produce a video with the skeleton and live metrics drawn on it,
so a coach can watch the analysis rather than read a table.

Draws the body skeleton, the shoulder-to-hip body line with its angle, and the
right elbow angle, for a chosen segment of the video.

Usage:
    python annotate_video.py path/to/video.mp4
    python annotate_video.py path/to/video.mp4 --start 100 --dur 12 --out annotated.mp4
"""
import argparse
import cv2
import numpy as np
from ultralytics import YOLO

# Which joints to connect with lines (COCO order).
SKELETON = [(5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15),
            (12, 14), (14, 16), (5, 6), (11, 12), (5, 11), (6, 12)]


def angle_at(a, b, c):
    ba = a - b
    bc = c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def annotate(video_path, out_path="annotated.mp4", start=0.0, dur=12.0,
             imgsz=640, conf=0.25, model_name="yolo11n-pose.pt"):
    model = YOLO(model_name)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    for _ in range(int(dur * fps)):
        ok, frame = cap.read()
        if not ok:
            break
        r = model.predict(frame, imgsz=imgsz, conf=conf, verbose=False)[0]
        if r.boxes is not None and len(r.boxes) > 0:
            areas = (r.boxes.xywh[:, 2] * r.boxes.xywh[:, 3]).cpu().numpy()
            kp = r.keypoints.data[int(np.argmax(areas))].cpu().numpy()

            for a, b in SKELETON:
                if kp[a, 2] > 0.5 and kp[b, 2] > 0.5:
                    cv2.line(frame, (int(kp[a, 0]), int(kp[a, 1])),
                             (int(kp[b, 0]), int(kp[b, 1])), (0, 230, 255), 3)
            for x, y, c in kp:
                if c > 0.5:
                    cv2.circle(frame, (int(x), int(y)), 5, (60, 60, 255), -1)

            if all(kp[i, 2] > 0.5 for i in (5, 6, 11, 12)):
                sh = (kp[5, :2] + kp[6, :2]) / 2
                hp = (kp[11, :2] + kp[12, :2]) / 2
                v = sh - hp
                body_ang = np.degrees(np.arctan2(v[1], v[0]))
                cv2.line(frame, (int(hp[0]), int(hp[1])), (int(sh[0]), int(sh[1])),
                         (0, 255, 120), 3)
                cv2.putText(frame, f"Body line: {body_ang:5.1f} deg", (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 120), 3)
            if all(kp[i, 2] > 0.5 for i in (6, 8, 10)):
                e = angle_at(kp[6, :2], kp[8, :2], kp[10, :2])
                cv2.putText(frame, f"R elbow: {e:3.0f} deg", (30, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 230, 255), 3)

        cv2.putText(frame, "Splash Coach - pose + kinematics", (30, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        writer.write(frame)

    writer.release()
    cap.release()
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Draw skeleton and metrics onto a swim video.")
    ap.add_argument("video", help="path to the input video")
    ap.add_argument("--out", default="annotated.mp4", help="output video path")
    ap.add_argument("--start", type=float, default=0.0, help="start time in seconds")
    ap.add_argument("--dur", type=float, default=12.0, help="clip length in seconds")
    args = ap.parse_args()
    annotate(args.video, args.out, args.start, args.dur)

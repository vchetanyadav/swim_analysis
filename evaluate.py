"""
evaluate.py
Quick, practical check of a fine-tuned pose model.

Two honest measures on a clip the model did NOT train on:
  1. Detection rate: how often a full-body skeleton was found.
  2. A short annotated clip so you can eyeball whether the joints, especially the
     legs, actually sit on the body.

The formal pose mAP comes from `yolo pose val`; this is the fast sanity check you
run every training round to see if it is really getting better on real footage.

Usage:
    python evaluate.py held_out_clip.mp4 --model swim_runs/ft1/weights/best.pt
    python evaluate.py held_out_clip.mp4 --model swim_runs/ft1/weights/best.pt --astart 100 --adur 12
"""
import argparse, os, cv2, numpy as np
from ultralytics import YOLO

BODY = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
SKELETON = [(5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15),
            (12, 14), (14, 16), (5, 6), (11, 12), (5, 11), (6, 12)]

def evaluate(video, model_name, skip=5, imgsz=1280, conf=0.25, astart=0.0, adur=12.0):
    model = YOLO(model_name)
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    # 1. detection rate across the clip
    idx = usable = seen = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % skip:
            idx += 1
            continue
        r = model.predict(frame, imgsz=imgsz, conf=conf, verbose=False)[0]
        seen += 1
        if r.boxes is not None and len(r.boxes) > 0:
            a = (r.boxes.xywh[:, 2] * r.boxes.xywh[:, 3]).cpu().numpy()
            kp = r.keypoints.data[int(np.argmax(a))].cpu().numpy()
            if np.mean([kp[i, 2] > 0.5 for i in BODY]) >= 0.75:
                usable += 1
        idx += 1
    cap.release()
    rate = 100 * usable / max(seen, 1)
    print(f"Detection rate on held-out clip: {rate:.1f}% of {seen} sampled frames "
          f"had a usable full-body skeleton.")

    # 2. annotated clip to eyeball
    os.makedirs("output", exist_ok=True)
    out_path = os.path.join("output", "eval_annotated.mp4")
    cap = cv2.VideoCapture(video)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_MSEC, astart * 1000)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for _ in range(int(adur * fps)):
        ok, frame = cap.read()
        if not ok:
            break
        r = model.predict(frame, imgsz=imgsz, conf=conf, verbose=False)[0]
        if r.boxes is not None and len(r.boxes) > 0:
            a = (r.boxes.xywh[:, 2] * r.boxes.xywh[:, 3]).cpu().numpy()
            kp = r.keypoints.data[int(np.argmax(a))].cpu().numpy()
            for x, y, c in kp:
                if c > 0.5:
                    cv2.circle(frame, (int(x), int(y)), 5, (60, 60, 255), -1)
            for i, j in SKELETON:
                if kp[i, 2] > 0.5 and kp[j, 2] > 0.5:
                    cv2.line(frame, (int(kp[i, 0]), int(kp[i, 1])),
                             (int(kp[j, 0]), int(kp[j, 1])), (0, 255, 120), 3)
        writer.write(frame)
    writer.release()
    cap.release()
    print(f"Annotated check clip: {out_path}")
    print("For the formal score, also run:  yolo pose val model=<best.pt> data=swim_pose.yaml")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--model", required=True, help="path to best.pt")
    ap.add_argument("--skip", type=int, default=5)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--astart", type=float, default=0.0)
    ap.add_argument("--adur", type=float, default=12.0)
    args = ap.parse_args()
    evaluate(args.video, args.model, args.skip, args.imgsz, args.conf, args.astart, args.adur)

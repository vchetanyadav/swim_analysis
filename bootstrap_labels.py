"""
bootstrap_labels.py
Auto-label sampled frames with the current pose model, in YOLO-pose format.

This is the time-saver. Instead of placing 17 points on every frame by hand,
the current model makes a first guess for each frame, and you only CORRECT the
wrong ones in a labelling tool (CVAT or Roboflow). On easy frames you accept;
on hard frames you drag the joints to the right place. That is several times
faster than labelling from scratch.

It writes, for each frame:
  labels/<name>.txt   one line: class cx cy w h  x1 y1 v1 ... x17 y17 v17  (normalised)
  images/<name>.jpg   a copy of the frame

Import the images + labels into CVAT/Roboflow as a YOLO-pose (keypoint) project,
fix the mistakes, then export back to the same format for training.

Usage:
    python bootstrap_labels.py frames --out swim_dataset --model yolo11x-pose.pt
"""
import argparse, os, glob, shutil, cv2, numpy as np
from ultralytics import YOLO

def bootstrap(frames_dir, out_dir, model_name="yolo11x-pose.pt", conf=0.25, imgsz=1280):
    model = YOLO(model_name)
    img_out = os.path.join(out_dir, "images")
    lbl_out = os.path.join(out_dir, "labels")
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)

    files = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    done = 0
    for fp in files:
        img = cv2.imread(fp)
        if img is None:
            continue
        h, w = img.shape[:2]
        r = model.predict(img, imgsz=imgsz, conf=conf, verbose=False)[0]
        base = os.path.splitext(os.path.basename(fp))[0]
        shutil.copy(fp, os.path.join(img_out, base + ".jpg"))
        line = None
        if r.boxes is not None and len(r.boxes) > 0:
            areas = (r.boxes.xywh[:, 2] * r.boxes.xywh[:, 3]).cpu().numpy()
            j = int(np.argmax(areas))
            bx, by, bw, bh = r.boxes.xywh[j].cpu().numpy()
            kp = r.keypoints.data[j].cpu().numpy()          # 17 x 3
            parts = [f"0 {bx/w:.6f} {by/h:.6f} {bw/w:.6f} {bh/h:.6f}"]
            for x, y, c in kp:
                v = 2 if c > 0.5 else 0                       # 2 = visible, 0 = unlabelled
                parts.append(f"{x/w:.6f} {y/h:.6f} {v}")
            line = " ".join(parts)
        with open(os.path.join(lbl_out, base + ".txt"), "w") as f:
            if line:
                f.write(line + "\n")                          # empty file if no detection
        done += 1
    print(f"Auto-labelled {done} frames into {out_dir}/ (images/ + labels/).")
    print("Next: import into CVAT or Roboflow as YOLO-pose, correct the joints, export back.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("frames_dir")
    ap.add_argument("--out", default="swim_dataset")
    ap.add_argument("--model", default="yolo11x-pose.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=1280)
    args = ap.parse_args()
    bootstrap(args.frames_dir, args.out, args.model, args.conf, args.imgsz)

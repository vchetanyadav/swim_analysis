"""
bootstrap_labels_14.py
Auto-label frames into the 14-keypoint SWIMMER layout (not COCO-17).

It runs the stock 17-keypoint model, then derives the 14 swimmer points:
  head_center = mean of visible nose/ears     neck = shoulder midpoint
  the other 12 = the COCO body joints directly.
You then CORRECT these in a keypoint tool (CVAT/Roboflow set to 14 points) and
train with swim14.yaml. Correcting a guess is far faster than labelling from zero.

Usage:
    python bootstrap_labels_14.py frames --out swim_dataset14 --model yolo11x-pose.pt
"""
import argparse, os, glob, shutil, cv2, numpy as np
from ultralytics import YOLO

# COCO-17 indices we read from
NOSE, L_EAR, R_EAR = 0, 3, 4
C = dict(l_sh=5, r_sh=6, l_el=7, r_el=8, l_wr=9, r_wr=10,
         l_hip=11, r_hip=12, l_kn=13, r_kn=14, l_an=15, r_an=16)

def to14(kp):
    """kp: 17x3 COCO -> list of 14 (x,y,v)."""
    def g(i):
        return kp[i, :2], (2 if kp[i, 2] > 0.5 else 0)
    # head_center from visible head points
    head_pts = [kp[i, :2] for i in (NOSE, L_EAR, R_EAR) if kp[i, 2] > 0.5]
    head = (np.mean(head_pts, axis=0), 2) if head_pts else (np.zeros(2), 0)
    # neck = shoulder midpoint
    if kp[5, 2] > 0.5 and kp[6, 2] > 0.5:
        neck = ((kp[5, :2] + kp[6, :2]) / 2, 2)
    else:
        neck = (np.zeros(2), 0)
    order = [head, neck, g(C["l_sh"]), g(C["r_sh"]), g(C["l_el"]), g(C["r_el"]),
             g(C["l_wr"]), g(C["r_wr"]), g(C["l_hip"]), g(C["r_hip"]),
             g(C["l_kn"]), g(C["r_kn"]), g(C["l_an"]), g(C["r_an"])]
    return order

def bootstrap(frames_dir, out_dir, model_name="yolo11x-pose.pt", conf=0.25, imgsz=1280):
    model = YOLO(model_name)
    io, lo = os.path.join(out_dir, "images"), os.path.join(out_dir, "labels")
    os.makedirs(io, exist_ok=True); os.makedirs(lo, exist_ok=True)
    for fp in sorted(glob.glob(os.path.join(frames_dir, "*.jpg"))):
        img = cv2.imread(fp)
        if img is None:
            continue
        h, w = img.shape[:2]
        r = model.predict(img, imgsz=imgsz, conf=conf, verbose=False)[0]
        base = os.path.splitext(os.path.basename(fp))[0]
        shutil.copy(fp, os.path.join(io, base + ".jpg"))
        line = None
        if r.boxes is not None and len(r.boxes) > 0:
            a = (r.boxes.xywh[:, 2] * r.boxes.xywh[:, 3]).cpu().numpy()
            j = int(np.argmax(a))
            bx, by, bw, bh = r.boxes.xywh[j].cpu().numpy()
            pts = to14(r.keypoints.data[j].cpu().numpy())
            parts = [f"0 {bx/w:.6f} {by/h:.6f} {bw/w:.6f} {bh/h:.6f}"]
            for (xy, v) in pts:
                parts.append(f"{xy[0]/w:.6f} {xy[1]/h:.6f} {v}")
            line = " ".join(parts)
        with open(os.path.join(lo, base + ".txt"), "w") as f:
            if line:
                f.write(line + "\n")
    print(f"Wrote 14-keypoint labels to {out_dir}/. Correct them in a 14-point "
          f"keypoint project, then train with swim14.yaml.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("frames_dir"); ap.add_argument("--out", default="swim_dataset14")
    ap.add_argument("--model", default="yolo11x-pose.pt")
    ap.add_argument("--conf", type=float, default=0.25); ap.add_argument("--imgsz", type=int, default=1280)
    a = ap.parse_args()
    bootstrap(a.frames_dir, a.out, a.model, a.conf, a.imgsz)

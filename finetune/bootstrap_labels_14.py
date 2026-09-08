"""
bootstrap_labels_14.py
Create initial SWIM-14 keypoint labels from a stock COCO-17 pose model.

These are bootstrap guesses only. Every selected frame must be manually checked
and corrected in CVAT/Roboflow before training.

Unlike the earlier bootstrap script, this version evaluates 0/90/180/270 degree
orientations using the same anatomical scoring as pose_markerless.py. This avoids
seeding the annotation set with the worst horizontal-swimmer predictions.
"""

import argparse
import glob
import os
import shutil

import cv2
import numpy as np
from ultralytics import YOLO

import pose_markerless as pm

NOSE, L_EAR, R_EAR = 0, 3, 4
C = {
    "l_sh": 5, "r_sh": 6, "l_el": 7, "r_el": 8,
    "l_wr": 9, "r_wr": 10, "l_hip": 11, "r_hip": 12,
    "l_kn": 13, "r_kn": 14, "l_an": 15, "r_an": 16,
}


def to14(kp, threshold=0.25):
    def point(i):
        if kp[i, 2] >= threshold:
            return kp[i, :2].copy(), 2
        return np.zeros(2, dtype=float), 0

    hp = pm.head_point(kp, threshold)
    head = (hp, 2) if hp is not None else (np.zeros(2, dtype=float), 0)

    if kp[5, 2] >= threshold and kp[6, 2] >= threshold:
        neck = ((kp[5, :2] + kp[6, :2]) / 2.0, 2)
    else:
        neck = (np.zeros(2, dtype=float), 0)

    return [
        head, neck,
        point(C["l_sh"]), point(C["r_sh"]),
        point(C["l_el"]), point(C["r_el"]),
        point(C["l_wr"]), point(C["r_wr"]),
        point(C["l_hip"]), point(C["r_hip"]),
        point(C["l_kn"]), point(C["r_kn"]),
        point(C["l_an"]), point(C["r_an"]),
    ]


def bootstrap(frames_dir, out_dir, model_name="yolo11x-pose.pt",
              det_conf=0.15, kpt_conf=0.25, imgsz=1280, device=None):
    model = YOLO(model_name)
    image_out = os.path.join(out_dir, "images")
    label_out = os.path.join(out_dir, "labels")
    os.makedirs(image_out, exist_ok=True)
    os.makedirs(label_out, exist_ok=True)

    files = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        files.extend(glob.glob(os.path.join(frames_dir, ext)))
    files = sorted(files)

    wrote = 0
    empty = 0
    for n, fp in enumerate(files, 1):
        img = cv2.imread(fp)
        if img is None:
            continue
        h, w = img.shape[:2]
        kp, angle, _ = pm.best_orientation(
            model, img, imgsz=imgsz, conf=det_conf,
            device=device, prev_kp=None,
        )

        base = os.path.splitext(os.path.basename(fp))[0]
        ext = os.path.splitext(fp)[1].lower()
        dst_img = os.path.join(image_out, base + ext)
        shutil.copy2(fp, dst_img)

        line = None
        if kp is not None:
            pts = to14(kp, kpt_conf)
            visible = [xy for xy, v in pts if v > 0]
            if visible:
                arr = np.asarray(visible)
                x0, y0 = np.min(arr, axis=0)
                x1, y1 = np.max(arr, axis=0)
                # Expand the keypoint-derived box because wrists/ankles can be missed.
                bw = max(1.0, x1 - x0)
                bh = max(1.0, y1 - y0)
                x0 = max(0.0, x0 - 0.12 * bw)
                y0 = max(0.0, y0 - 0.12 * bh)
                x1 = min(float(w - 1), x1 + 0.12 * bw)
                y1 = min(float(h - 1), y1 + 0.12 * bh)
                bx = (x0 + x1) / 2.0
                by = (y0 + y1) / 2.0
                bw = x1 - x0
                bh = y1 - y0

                parts = [f"0 {bx/w:.6f} {by/h:.6f} {bw/w:.6f} {bh/h:.6f}"]
                for xy, v in pts:
                    if v == 0:
                        parts.append("0.000000 0.000000 0")
                    else:
                        parts.append(f"{xy[0]/w:.6f} {xy[1]/h:.6f} {v}")
                line = " ".join(parts)

        label_path = os.path.join(label_out, base + ".txt")
        with open(label_path, "w") as f:
            if line:
                f.write(line + "\n")
                wrote += 1
            else:
                empty += 1

        if n % 50 == 0:
            print(f"  {n}/{len(files)} frames")

    print(
        f"Done. {wrote} frames received bootstrap keypoints, {empty} were left empty.\n"
        f"Now manually correct every label in a 14-point keypoint project before training."
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("frames_dir")
    ap.add_argument("--out", default="swim_dataset14")
    ap.add_argument("--model", default="yolo11x-pose.pt")
    ap.add_argument("--det-conf", type=float, default=0.15)
    ap.add_argument("--kpt-conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    bootstrap(
        args.frames_dir, args.out, args.model,
        args.det_conf, args.kpt_conf, args.imgsz, args.device,
    )

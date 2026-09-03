"""
calibrate_color.py
Read the real underwater colour of a marker so you can set its HSV range.

The colour a marker looks like in your hand is not the colour the camera sees
through water. Grab one frame where a marker is clearly visible, find its pixel
position (open the image, hover the cursor, most viewers show x, y), then run:

    python calibrate_color.py frame.jpg --x 640 --y 360 --r 12

It prints the median HSV in that spot and a ready-to-paste (lower, upper) range.
Repeat for each marker and paste the ranges into the MARKERS table in
track_markers.py.

To get a frame to calibrate on:
    (any video player screenshot works, or)
    ffmpeg -ss 100 -i top.mp4 -frames:v 1 frame.jpg
"""
import argparse
import cv2
import numpy as np


def calibrate(image_path, x, y, r):
    img = cv2.imread(image_path)
    if img is None:
        raise SystemExit(f"Could not read image: {image_path}")
    h, w = img.shape[:2]
    x0, x1 = max(0, x - r), min(w, x + r)
    y0, y1 = max(0, y - r), min(h, y + r)
    patch = img[y0:y1, x0:x1]
    if patch.size == 0:
        raise SystemExit("Sample region is empty; check --x and --y.")

    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    med = np.median(hsv, axis=0).astype(int)
    lo = np.clip(med - [12, 70, 70], [0, 40, 40], [179, 255, 255]).astype(int)
    hi = np.clip(med + [12, 70, 70], [0, 40, 40], [179, 255, 255]).astype(int)

    print(f"Sampled {patch.shape[1]}x{patch.shape[0]} px at ({x}, {y})")
    print(f"  median HSV: H={med[0]} S={med[1]} V={med[2]}")
    print(f"  suggested range for MARKERS table:")
    print(f"    (({lo[0]}, {lo[1]}, {lo[2]}), ({hi[0]}, {hi[1]}, {hi[2]}), (B, G, R))")
    print("  (replace (B, G, R) with any colour you want the dot drawn in)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Sample a marker's underwater HSV.")
    ap.add_argument("image", help="a frame image containing the marker")
    ap.add_argument("--x", type=int, required=True, help="marker x pixel")
    ap.add_argument("--y", type=int, required=True, help="marker y pixel")
    ap.add_argument("--r", type=int, default=10, help="sample radius in pixels")
    args = ap.parse_args()
    calibrate(args.image, args.x, args.y, args.r)

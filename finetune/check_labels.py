"""
check_labels.py
Quality-control a SWIM-14 YOLO-pose label set, and optionally auto-fix the boxes.

Runs the checks that matter for pose training:
  1. Box containment: every visible keypoint must sit inside the bounding box.
     YOLO-pose trains a detector AND a skeleton; if they disagree, training
     suffers. --fix rebuilds each box tightly around its keypoints with padding.
  2. Visibility audit: reports how many points are visible(2) / occluded(1) /
     unlabelled(0). In swim footage, everything marked visible is a red flag,
     occluded joints (far limb, underwater, bubbles) should be 1, not 2.
  3. Anatomical sanity: flags frames where a bone is wildly longer than usual, or
     where left/right joints have collapsed onto the same point (the exact
     failure we are trying to train away). These frames are worth re-checking by
     eye against the image.

Usage:
    python check_labels.py labels/train                # report only
    python check_labels.py labels/train --fix          # also rewrite boxes in place
    python check_labels.py labels/train --fix --pad 0.12
"""
import argparse, glob, os
import numpy as np

# SWIM-14 indices
NAMES = ["head", "neck", "l_sh", "r_sh", "l_el", "r_el", "l_wr", "r_wr",
         "l_hip", "r_hip", "l_kn", "r_kn", "l_an", "r_an"]
BONES = [(2, 4), (4, 6), (3, 5), (5, 7), (8, 10), (10, 12),
         (9, 11), (11, 13), (2, 3), (8, 9), (1, 2), (1, 3)]
LR_PAIRS = [(6, 7), (12, 13), (10, 11)]   # wrists, ankles, knees


def load(fp):
    v = open(fp).read().split()
    if len(v) < 5:
        return None, None
    box = list(map(float, v[1:5]))
    kp = np.array(list(map(float, v[5:]))).reshape(-1, 3)
    return box, kp


def rebuild_box(kp, pad):
    vis = kp[kp[:, 2] > 0][:, :2]
    if len(vis) == 0:
        return None
    x0, y0 = vis.min(0)
    x1, y1 = vis.max(0)
    bw, bh = max(1e-3, x1 - x0), max(1e-3, y1 - y0)
    x0 = max(0.0, x0 - pad * bw); y0 = max(0.0, y0 - pad * bh)
    x1 = min(1.0, x1 + pad * bw); y1 = min(1.0, y1 + pad * bh)
    return [(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0]


def main(labels_dir, fix, pad):
    files = sorted(glob.glob(os.path.join(labels_dir, "*.txt")))
    if not files:
        raise SystemExit(f"No labels in {labels_dir}")

    vis_counts = {0: 0, 1: 0, 2: 0}
    bad_box = []
    bone_lengths = {b: [] for b in BONES}
    per_frame = {}

    for fp in files:
        box, kp = load(fp)
        if box is None:
            continue
        for v in kp[:, 2]:
            vis_counts[int(v)] = vis_counts.get(int(v), 0) + 1
        cx, cy, w, h = box
        x0, x1, y0, y1 = cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2
        vis = kp[kp[:, 2] > 0]
        if len(vis) and not np.all((vis[:, 0] >= x0 - 1e-3) & (vis[:, 0] <= x1 + 1e-3) &
                                   (vis[:, 1] >= y0 - 1e-3) & (vis[:, 1] <= y1 + 1e-3)):
            bad_box.append(os.path.basename(fp))
        for a, b in BONES:
            if kp[a, 2] > 0 and kp[b, 2] > 0:
                bone_lengths[(a, b)].append(np.linalg.norm(kp[a, :2] - kp[b, :2]))
        per_frame[fp] = (box, kp)

    # anatomical flags
    med = {b: (np.median(v) if v else 0) for b, v in bone_lengths.items()}
    suspicious = set()
    for fp, (box, kp) in per_frame.items():
        name = os.path.basename(fp)
        for a, b in BONES:
            if kp[a, 2] > 0 and kp[b, 2] > 0 and med[(a, b)] > 0:
                if np.linalg.norm(kp[a, :2] - kp[b, :2]) > 2.5 * med[(a, b)]:
                    suspicious.add(name)
        for l, r in LR_PAIRS:
            if kp[l, 2] > 0 and kp[r, 2] > 0:
                if np.linalg.norm(kp[l, :2] - kp[r, :2]) < 0.01:
                    suspicious.add(name)

    print(f"Checked {len(files)} labels.\n")
    print(f"VISIBILITY: visible(2)={vis_counts.get(2,0)}  occluded(1)={vis_counts.get(1,0)}  "
          f"unlabelled(0)={vis_counts.get(0,0)}")
    if vis_counts.get(1, 0) == 0 and vis_counts.get(0, 0) == 0:
        print("  WARNING: every point is marked fully visible. In swim footage many joints are")
        print("  occluded, mark hidden joints as occluded (1) so the model is not taught to")
        print("  place confident joints it cannot actually see. This especially helps the legs.\n")

    print(f"BOX CONTAINMENT: {len(bad_box)}/{len(files)} boxes genuinely miss a keypoint "
          f"(edge-touching points are not counted).")
    print("  Boxes exported as a tight fit around the keypoints are fine. Adding a small")
    print("  padding margin still helps the detector; run --fix to add it.")
    if bad_box:
        print("  Genuinely broken boxes:", ", ".join(bad_box[:10]))
    print()

    if suspicious:
        print(f"RECHECK BY EYE ({len(suspicious)} frames with odd geometry):")
        for n in sorted(suspicious):
            print(f"    {n}")
    else:
        print("No anatomically suspicious frames flagged.")

    if fix:
        n = 0
        for fp, (box, kp) in per_frame.items():
            nb = rebuild_box(kp, pad)
            if nb is None:
                continue
            parts = [f"0 {nb[0]:.6f} {nb[1]:.6f} {nb[2]:.6f} {nb[3]:.6f}"]
            for x, y, v in kp:
                parts.append(f"{x:.6f} {y:.6f} {int(v)}")
            open(fp, "w").write(" ".join(parts) + "\n")
            n += 1
        print(f"\nFIXED: rebuilt boxes in {n} label files (pad {pad}).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("labels_dir")
    ap.add_argument("--fix", action="store_true", help="rewrite boxes around keypoints")
    ap.add_argument("--pad", type=float, default=0.12)
    args = ap.parse_args()
    main(args.labels_dir, args.fix, args.pad)

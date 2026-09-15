"""
assemble_pilot.py
Combine a CVAT "Ultralytics YOLO Pose" label export with the original image
frames into a train/val dataset ready for `yolo pose train`.

CVAT exports labels only (labels/train/*.txt), not the images. This script pairs
each label with its image (matched by filename), then splits into train/val.

NOTE on this pilot: all 63 frames are one swimmer, one view, so a held-out val
set is still the same person. The score will be optimistic. That is fine for a
pilot whose only question is "does fine-tuning improve this footage at all?".
For a real model, label several participants and views, then split by participant.

Usage (run on the server, from swim_analysis/):
    python finetune/assemble_pilot.py \
        --labels swimanno/labels/train \
        --images frames \
        --out swim_dataset14 \
        --val-frac 0.2
"""
import argparse, glob, os, random, shutil


def assemble(labels_dir, images_dir, out, val_frac=0.2, seed=0):
    labels = sorted(glob.glob(os.path.join(labels_dir, "*.txt")))
    if not labels:
        raise SystemExit(f"No .txt labels found in {labels_dir}")
    random.Random(seed).shuffle(labels)
    n_val = max(1, round(len(labels) * val_frac))
    val_names = {os.path.splitext(os.path.basename(x))[0] for x in labels[:n_val]}

    for sub in ("train", "val"):
        os.makedirs(os.path.join(out, "images", sub), exist_ok=True)
        os.makedirs(os.path.join(out, "labels", sub), exist_ok=True)

    paired = missing = 0
    for lp in labels:
        base = os.path.splitext(os.path.basename(lp))[0]
        sub = "val" if base in val_names else "train"
        img = None
        for ext in (".jpg", ".jpeg", ".png", ".JPG", ".PNG"):
            cand = os.path.join(images_dir, base + ext)
            if os.path.exists(cand):
                img = cand
                break
        if img is None:
            print(f"  no image for {base}, skipping")
            missing += 1
            continue
        shutil.copy(lp, os.path.join(out, "labels", sub, base + ".txt"))
        shutil.copy(img, os.path.join(out, "images", sub, os.path.basename(img)))
        paired += 1

    tr = len(os.listdir(os.path.join(out, "images", "train")))
    va = len(os.listdir(os.path.join(out, "images", "val")))
    print(f"Done. paired {paired} frames (train {tr}, val {va}); {missing} labels had no image.")
    print(f"Dataset ready at {out}/. Train with swim14.yaml (path: ./{out}).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, help="CVAT export labels dir (labels/train)")
    ap.add_argument("--images", required=True, help="folder with the original frame images")
    ap.add_argument("--out", default="swim_dataset14")
    ap.add_argument("--val-frac", type=float, default=0.2)
    args = ap.parse_args()
    assemble(args.labels, args.images, args.out, args.val_frac)

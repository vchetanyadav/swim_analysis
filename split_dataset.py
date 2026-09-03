"""
split_dataset.py
Split labelled data into train/val WITHOUT leakage.

Consecutive swim frames are almost identical, so a random split lets the model
see near-copies of its validation frames during training and reports fake-high
scores. This splits by a group key in the filename (participant or clip) so a
whole participant is either in train or in val, never both.

It expects a flat folder from bootstrap_labels.py:
    swim_dataset/images/*.jpg
    swim_dataset/labels/*.txt
and reorganises into:
    swim_dataset/images/train  swim_dataset/images/val
    swim_dataset/labels/train  swim_dataset/labels/val

The group key is the start of each filename up to a separator. Name your frames
so the participant is the prefix, e.g. p38_top_0102s.jpg, p41_sidefront_0054s.jpg
(then --key-parts 1 groups by p38, p41, ...). If you only have views, not
participants, group by view is better than random, but train on several
participants for a model that generalises.

Usage:
    python split_dataset.py swim_dataset --val-groups p41 p07 --key-sep _ --key-parts 1
    python split_dataset.py swim_dataset --val-frac 0.25 --key-sep _ --key-parts 1
"""
import argparse, os, glob, shutil, random

def group_of(fname, sep, parts):
    base = os.path.splitext(os.path.basename(fname))[0]
    return sep.join(base.split(sep)[:parts])

def split(root, val_groups=None, val_frac=0.25, sep="_", parts=1, seed=0):
    img_dir = os.path.join(root, "images")
    lbl_dir = os.path.join(root, "labels")
    imgs = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    if not imgs:
        raise SystemExit(f"No images in {img_dir}. Run bootstrap_labels.py first.")

    groups = {}
    for im in imgs:
        groups.setdefault(group_of(im, sep, parts), []).append(im)
    all_groups = sorted(groups)
    print(f"Found {len(imgs)} frames across {len(all_groups)} groups: {', '.join(all_groups)}")

    if val_groups:
        val_set = set(val_groups)
    else:
        rng = random.Random(seed)
        shuffled = all_groups[:]
        rng.shuffle(shuffled)
        n_val = max(1, round(len(shuffled) * val_frac))
        val_set = set(shuffled[:n_val])
    print(f"Validation groups: {', '.join(sorted(val_set))}")

    for sub in ("train", "val"):
        os.makedirs(os.path.join(img_dir, sub), exist_ok=True)
        os.makedirs(os.path.join(lbl_dir, sub), exist_ok=True)

    moved = {"train": 0, "val": 0}
    for g, files in groups.items():
        sub = "val" if g in val_set else "train"
        for im in files:
            base = os.path.splitext(os.path.basename(im))[0]
            lbl = os.path.join(lbl_dir, base + ".txt")
            shutil.move(im, os.path.join(img_dir, sub, os.path.basename(im)))
            if os.path.exists(lbl):
                shutil.move(lbl, os.path.join(lbl_dir, sub, base + ".txt"))
            moved[sub] += 1
    print(f"Done. train: {moved['train']} frames, val: {moved['val']} frames.")
    print("Point swim_pose.yaml at this folder and train.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="dataset folder with images/ and labels/")
    ap.add_argument("--val-groups", nargs="*", default=None,
                    help="explicit groups to hold out for validation (e.g. p41 p07)")
    ap.add_argument("--val-frac", type=float, default=0.25,
                    help="if no groups given, fraction of groups to hold out")
    ap.add_argument("--key-sep", default="_", help="filename separator")
    ap.add_argument("--key-parts", type=int, default=1,
                    help="how many leading parts form the group key")
    args = ap.parse_args()
    split(args.root, args.val_groups, args.val_frac, args.key_sep, args.key_parts)

#!/usr/bin/env python3
"""
predict_all.py  -  expert-style assessment for EVERY participant.
 
Runs the same assessment as predict_multi.py, but over every per-participant
feature file in a folder instead of one file at a time. For each criterion
(floating outcome, trunk, activity) it averages the model's per-window
probabilities, then reports the most likely class and the confidence.
 
One row per feature file. The participant id is read from the filename
(the leading P<number>). It loads the models and feature files you already
have; it does not retrain and does not invent data.
 
Usage (from the floatml folder, venv active):
  python predict_all.py
  python predict_all.py --features-dir features_v2_side_front --which Softmax
  python predict_all.py --which Tree --out results_all_tree.csv
"""
import argparse, os, glob, re, numpy as np, pandas as pd, joblib
 
TRUNK = {0: "vertical", 1: "diagonal", 2: "horizontal"}
ACT   = {0: "tense",    1: "neutral",  2: "relaxed"}
OUT   = {0: "cannot float", 1: "floats with lots of effort",
         2: "floats with effort", 3: "floats with no effort"}
 
 
def load(mdir, crit, which):
    p = os.path.join(mdir, f"model_{crit}_{which}.joblib")
    if not os.path.exists(p):
        raise SystemExit(f"missing {p} (run swim_multi.py first)")
    return joblib.load(p)
 
 
def assess(m, d):
    """Average per-window probabilities -> (class, confidence). Mirrors predict_multi.assess."""
    clf = m["model"]; F = m["features"]
    X = d.copy()
    for c in F:
        if c not in X:
            X[c] = np.nan
    X = X[F].to_numpy(float)
    if hasattr(clf, "predict_proba"):
        P = clf.predict_proba(X).mean(0)
        return int(clf.classes_[int(np.argmax(P))]), float(P.max())
    pr = clf.predict(X)
    k = int(pd.Series(pr).value_counts().idxmax())
    return k, float((pr == k).mean())
 
 
def pid_from_name(fname):
    m = re.match(r"[Pp](\d+)", os.path.basename(fname))
    return f"P{int(m.group(1))}" if m else os.path.basename(fname)
 
 
def natural_key(s):
    m = re.search(r"\d+", str(s))
    return (int(m.group()) if m else 10**9, str(s))
 
 
def main(a):
    mt = load(a.model_dir, "trunk", a.which)
    ma = load(a.model_dir, "activity", a.which)
    mo = load(a.model_dir, "outcome", a.which)
 
    files = sorted(glob.glob(os.path.join(a.features_dir, "*.csv")))
    if not files:
        raise SystemExit(f"no CSV files in {a.features_dir}")
    print(f"features: {a.features_dir}  ({len(files)} files)")
    print(f"models  : model_(trunk|activity|outcome)_{a.which}.joblib from {a.model_dir}\n")
 
    rows, skipped = [], []
    for f in files:
        d = pd.read_csv(f)
        if "valid_pct" in d.columns:
            d = d[d["valid_pct"] >= a.min_valid]
        pid = pid_from_name(f)
        if len(d) == 0:
            skipped.append(pid)
            continue
        ou, ouc = assess(mo, d)
        tr, trc = assess(mt, d)
        ac, acc = assess(ma, d)
        rows.append({
            "participant": pid,
            "n_windows": len(d),
            "floating_outcome": ou, "outcome_label": OUT.get(ou, ou), "outcome_conf": round(ouc, 3),
            "trunk": tr, "trunk_label": TRUNK.get(tr, tr), "trunk_conf": round(trc, 3),
            "activity": ac, "activity_label": ACT.get(ac, ac), "activity_conf": round(acc, 3),
            "file": os.path.basename(f),
        })
 
    res = pd.DataFrame(rows)
    res = res.sort_values("participant", key=lambda col: col.map(natural_key)).reset_index(drop=True)
    res.to_csv(a.out, index=False)
 
    pd.set_option("display.width", 240)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    show = res[["participant", "n_windows",
                "floating_outcome", "outcome_conf",
                "trunk", "trunk_conf",
                "activity", "activity_conf"]]
    print("Results for all participants (score = chosen class, conf = averaged probability):\n")
    print(show.to_string(index=False))
    print(f"\n{len(res)} participants assessed. Full table with labels + source file saved -> {a.out}")
    if skipped:
        print(f"{len(skipped)} skipped (pose too poor, no window >= {a.min_valid}% valid): "
              + ", ".join(skipped))
    print("Scale: floating outcome 0-3, trunk 0-2, activity 0-2. "
          "Head and airway are expert-only (not observable from pose).")
 
 
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-dir", default="features_v2_side_front")
    ap.add_argument("--model-dir", default="model_multi")
    ap.add_argument("--which", default="Softmax", choices=["Softmax", "Tree"])
    ap.add_argument("--min-valid", type=float, default=60)
    ap.add_argument("--out", default="results_all_participants.csv")
    main(ap.parse_args())
 

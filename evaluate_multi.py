#!/usr/bin/env python3
"""
evaluate_multi.py  -  honest evaluation + figures for the multi-criterion models.
 
Reproduces swim_multi.py's label-to-window join and participant-level GroupKFold,
then produces out-of-fold (leave-group-out) predictions so every number reflects
generalisation, not training-set fit. For each criterion (floating outcome, trunk,
activity) and each model (Softmax, Tree) it writes into eval/:
 
  - summary.txt        : classification report + macro-F1 / balanced acc / quadratic kappa
  - metrics.json       : the same numbers, machine-readable
  - cm_<crit>_<model>.png / .csv   : confusion matrix (counts), out-of-fold
  - pca_<crit>.png     : 2D PCA of the feature space, coloured by true class
  - scatter_outcome.png: trunk angle vs hip drift, coloured by floating outcome
  - conf_<crit>.png    : distribution of the model's per-window confidence
 
Run exactly like swim_multi (same features + labels), for example:
  python evaluate_multi.py "features_v2_side_front/"*.csv --labels labels_merged.csv \
      --expert expert_scores.csv --out eval
"""
import argparse, os, json, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
 
import swim_train as T
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                             balanced_accuracy_score, cohen_kappa_score)
 
# ---- identical model definitions and recodes to swim_multi.py -------------------
def MODELS():
    return {
      "Softmax": make_pipeline(SimpleImputer(strategy="median"), RobustScaler(),
                 LogisticRegression(class_weight="balanced", max_iter=5000, C=1.0)),
      "Tree":    make_pipeline(SimpleImputer(strategy="median"),
                 DecisionTreeClassifier(max_depth=4, min_samples_leaf=15,
                                        class_weight="balanced", random_state=0)),
    }
 
def rec_trunk(s):
    p = pd.to_numeric(s.get("posture"), errors="coerce"); return None if pd.isna(p) else 2 - int(p)
def rec_act(s):
    p = pd.to_numeric(s.get("passive_active"), errors="coerce"); return None if pd.isna(p) else 2 - int(p)
def rec_out(s):
    fl = pd.to_numeric(s.get("float_label"), errors="coerce")
    pa = pd.to_numeric(s.get("passive_active"), errors="coerce")
    if pd.isna(fl): return None
    fl = int(fl); return fl if fl in (0, 1) else (3 if (not pd.isna(pa) and int(pa) == 0) else 2)
TARGETS = {"trunk": rec_trunk, "activity": rec_act, "outcome": rec_out}
 
CLASSNAMES = {
    "trunk":    {0: "vertical", 1: "diagonal", 2: "horizontal"},
    "activity": {0: "tense", 1: "neutral", 2: "relaxed"},
    "outcome":  {0: "cannot float", 1: "lots of effort", 2: "with effort", 3: "no effort"},
}
 
def heatmap(ax, M, classes, title):
    im = ax.imshow(M, cmap="Blues")
    ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=30, ha="right", fontsize=8)
    ax.set_yticklabels(classes, fontsize=8)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title, fontsize=10)
    thr = M.max() / 2 if M.max() else 0.5
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, int(M[i, j]), ha="center", va="center", fontsize=9,
                    color="white" if M[i, j] > thr else "black")
    return im
 
 
def main(a):
    os.makedirs(a.out, exist_ok=True)
    feat = T.load_features(a.features)
    feat = feat[feat.get("valid_pct", 100) >= 60]
    feat["win_center"] = (feat["start_s"] + feat["end_s"]) / 2
    feat["participant"] = feat["source"].map(T.pid_from)
    lab = pd.read_csv(a.labels); lab = lab[lab["exclude"].fillna(0).astype(int) == 0]
    F = [c for c in T.FEATURES if c in feat.columns]
    print("features used:", len(F)); print("labelled rows in file:", len(lab))
 
    def attach(rec):
        f = feat.copy(); f["y"] = np.nan
        for _, s in lab.iterrows():
            v = rec(s)
            if v is None: continue
            m = (f["source"] == s["source"]) & (f["win_center"] >= s["start_s"]) & (f["win_center"] < s["end_s"])
            f.loc[m, "y"] = v
        return f.dropna(subset=["y"]).assign(y=lambda d: d["y"].astype(int))
 
    summary, metrics = [], {}
    for crit, rec in TARGETS.items():
        d = attach(rec)
        X = d[F].to_numpy(float); y = d["y"].to_numpy(); g = d["participant"].to_numpy()
        classes = sorted(np.unique(y))
        names = [CLASSNAMES[crit].get(c, str(c)) for c in classes]
        n_splits = min(5, len(np.unique(g)))
        summary.append("=" * 70)
        summary.append(f"CRITERION: {crit}   (n={len(y)} windows, "
                       f"{len(np.unique(g))} participants, {n_splits}-fold grouped CV)")
        summary.append(f"class support: " + ", ".join(
            f"{CLASSNAMES[crit].get(c,c)}={int((y==c).sum())}" for c in classes))
        metrics[crit] = {}
 
        for mname, proto in MODELS().items():
            oof = np.full(len(y), -1); conf = np.full(len(y), np.nan)
            for trn, te in GroupKFold(n_splits).split(X, y, g):
                m = clone(proto); m.fit(X[trn], y[trn])
                oof[te] = m.predict(X[te])
                if hasattr(m, "predict_proba"):
                    conf[te] = m.predict_proba(X[te]).max(1)
 
            mf = f1_score(y, oof, average="macro")
            ba = balanced_accuracy_score(y, oof)
            try:
                qk = cohen_kappa_score(y, oof, weights="quadratic")
            except Exception:
                qk = float("nan")
            rep = classification_report(y, oof, labels=classes, target_names=names,
                                        digits=3, zero_division=0)
            M = confusion_matrix(y, oof, labels=classes)
 
            summary.append(f"\n--- {mname} (out-of-fold) ---")
            summary.append(f"macro-F1 {mf:.3f}   balanced-acc {ba:.3f}   "
                           f"quadratic-weighted kappa {qk:.3f}")
            summary.append(rep)
            metrics[crit][mname] = {"macro_f1": round(float(mf), 3),
                                    "balanced_acc": round(float(ba), 3),
                                    "quadratic_kappa": round(float(qk), 3)}
 
            # confusion matrix png + csv
            pd.DataFrame(M, index=names, columns=names).to_csv(
                os.path.join(a.out, f"cm_{crit}_{mname}.csv"))
            fig, ax = plt.subplots(figsize=(4.2, 3.6))
            heatmap(ax, M, names, f"{crit} — {mname}\nmacro-F1 {mf:.2f}, kappa {qk:.2f}")
            fig.tight_layout(); fig.savefig(os.path.join(a.out, f"cm_{crit}_{mname}.png"), dpi=150)
            plt.close(fig)
 
            # confidence distribution (Softmax, the probabilistic one)
            if mname == "Softmax" and np.isfinite(conf).any():
                fig, ax = plt.subplots(figsize=(4.4, 3.0))
                ax.hist(conf[np.isfinite(conf)], bins=20, color="#4C78A8", edgecolor="white")
                ax.set_xlabel("per-window confidence (max class probability)")
                ax.set_ylabel("windows"); ax.set_title(f"{crit} — Softmax confidence", fontsize=10)
                fig.tight_layout(); fig.savefig(os.path.join(a.out, f"conf_{crit}.png"), dpi=150)
                plt.close(fig)
 
        # PCA scatter coloured by true class
        Xi = SimpleImputer(strategy="median").fit_transform(d[F].to_numpy(float))
        Z = PCA(n_components=2, random_state=0).fit_transform(StandardScaler().fit_transform(Xi))
        fig, ax = plt.subplots(figsize=(4.8, 4.0))
        for c in classes:
            sel = y == c
            ax.scatter(Z[sel, 0], Z[sel, 1], s=10, alpha=0.6, label=CLASSNAMES[crit].get(c, c))
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        ax.set_title(f"{crit}: feature space (PCA), coloured by true class", fontsize=9)
        ax.legend(fontsize=7, markerscale=1.5)
        fig.tight_layout(); fig.savefig(os.path.join(a.out, f"pca_{crit}.png"), dpi=150)
        plt.close(fig)
 
    # interpretable 2-feature scatter for the outcome criterion
    d = attach(rec_out)
    fx = "trunk_angle_med" if "trunk_angle_med" in d else F[0]
    fy = "hip_y_slope" if "hip_y_slope" in d else F[1]
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    yv = d["y"].to_numpy()
    for c in sorted(np.unique(yv)):
        sel = yv == c
        ax.scatter(d[fx][sel], d[fy][sel], s=12, alpha=0.6, label=CLASSNAMES["outcome"].get(c, c))
    ax.set_xlabel(fx); ax.set_ylabel(fy)
    ax.set_title("floating outcome by trunk angle and hip drift", fontsize=10)
    ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "scatter_outcome.png"), dpi=150)
    plt.close(fig)
 
    open(os.path.join(a.out, "summary.txt"), "w").write("\n".join(summary))
    json.dump(metrics, open(os.path.join(a.out, "metrics.json"), "w"), indent=2)
    print("\n".join(summary))
    print(f"\nwrote {a.out}/ : summary.txt, metrics.json, cm_*.png/.csv, pca_*.png, "
          f"conf_*.png, scatter_outcome.png")
    print("All confusion matrices are out-of-fold (participant-level), so they show "
          "generalisation, not training fit.")
 
 
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("features", nargs="+")
    ap.add_argument("--labels", default="labels_merged.csv")
    ap.add_argument("--expert", default="expert_scores.csv")  # accepted for symmetry; not needed here
    ap.add_argument("--out", default="eval")
    main(ap.parse_args())
 

"""
swim_test.py  -  run the FROZEN model ONCE on the sealed test participants.

Run this a single time, after swim_train.py has frozen the model, and do not retune anything
afterwards. It reports macro-F1, balanced accuracy, quadratic-weighted kappa, the confusion
matrix, and per-participant scores on swimmers never used in development.

Usage:
    python swim_test.py features_side_front/*.csv --labels labels.csv --model model_side/model.joblib --out test_report
"""
import argparse, glob, os, re, json, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import (classification_report, confusion_matrix,
                             balanced_accuracy_score, f1_score, cohen_kappa_score)
import joblib

def pid_from(name):
    stem=os.path.splitext(os.path.basename(str(name)))[0]
    m=re.search(r'[Pp](\d+)',stem)
    if m: return f"P{int(m.group(1)):02d}"
    m=re.match(r'\s*(\d+)',stem)
    return f"V{int(m.group(1)):02d}" if m else stem[:8]

def main(patterns,labels_csv,model_path,out_dir):
    os.makedirs(out_dir,exist_ok=True)
    B=joblib.load(model_path); model=B["model"]; feats=B["features"]; target=B["target"]
    names=B["classes"]; sealed=set(B.get("sealed_test_participants",[]))
    if not sealed: print("WARNING: frozen model lists no sealed test participants; testing on all labeled.")
    files=[]; [files.extend(glob.glob(p)) for p in patterns]
    parts=[pd.read_csv(f).assign(source=os.path.basename(f)) for f in files]
    feat=pd.concat(parts,ignore_index=True)
    if "valid_pct" in feat: feat=feat[feat["valid_pct"]>=60].reset_index(drop=True)
    feat["participant"]=feat["source"].map(pid_from)
    feat["win_center"]=(feat["start_s"]+feat["end_s"])/2.0
    if sealed: feat=feat[feat["participant"].isin(sealed)].reset_index(drop=True)

    lab=pd.read_csv(labels_csv)
    if "exclude" in lab: lab=lab[lab["exclude"].fillna(0).astype(int)==0]
    key="source" if ("source" in lab and "source" in feat) else "video_id"
    lab=lab.dropna(subset=[target]).copy(); lab[target]=lab[target].astype(int)
    feat["label"]=np.nan
    for _,s in lab.iterrows():
        m=(feat[key]==s[key])&(feat["win_center"]>=s["start_s"])&(feat["win_center"]<s["end_s"])
        feat.loc[m,"label"]=s[target]
    feat=feat.dropna(subset=["label"]).copy(); feat["label"]=feat["label"].astype(int)
    if len(feat)==0: raise SystemExit("no labeled windows for the sealed test participants")

    X=feat[[f for f in feats if f in feat]].copy()
    for f in feats:
        if f not in X: X[f]=np.nan
    X=X[feats].fillna(pd.Series(B.get("medians",{})))
    yp=model.predict(X.to_numpy()); yt=feat["label"].to_numpy()
    present=sorted(np.unique(np.concatenate([yt,yp])))
    rep=classification_report(yt,yp,labels=present,target_names=[names[c] for c in present],digits=3,zero_division=0)
    res={"test_participants":sorted(feat["participant"].unique().tolist()),
         "n_windows":int(len(feat)),
         "macro_f1":round(float(f1_score(yt,yp,average="macro")),3),
         "balanced_acc":round(float(balanced_accuracy_score(yt,yp)),3),
         "weighted_kappa":round(float(cohen_kappa_score(yt,yp,weights="quadratic")),3)}
    print("===== SEALED TEST (run once) =====")
    print("participants:",res["test_participants"])
    print(f"macroF1={res['macro_f1']}  balAcc={res['balanced_acc']}  weightedKappa={res['weighted_kappa']}")
    print(rep)
    cm=confusion_matrix(yt,yp,labels=present); print("confusion:",present); print(cm)
    # video-level
    feat=feat.copy(); feat["pred"]=yp
    vt=feat.groupby("source")["label"].agg(lambda s:s.value_counts().idxmax())
    vp=feat.groupby("source")["pred"].agg(lambda s:s.value_counts().idxmax())
    res["video_level_accuracy"]=round(float((vt.values==vp.values).mean()),3)
    print("video-level accuracy:",res["video_level_accuracy"])
    json.dump(res,open(os.path.join(out_dir,"sealed_test_report.json"),"w"),indent=2)
    plt.figure(figsize=(5,4.2)); plt.imshow(cm,cmap="Greens"); plt.colorbar()
    plt.xticks(range(len(present)),[names[c] for c in present],rotation=30,ha="right")
    plt.yticks(range(len(present)),[names[c] for c in present])
    for i in range(len(present)):
        for j in range(len(present)): plt.text(j,i,cm[i,j],ha="center",va="center")
    plt.ylabel("true"); plt.xlabel("predicted"); plt.title("Sealed test"); plt.tight_layout()
    plt.savefig(os.path.join(out_dir,"confusion_sealed_test.png"),dpi=120); plt.close()
    print(f"wrote sealed_test_report.json + confusion_sealed_test.png to {out_dir}/")

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("features",nargs="+")
    ap.add_argument("--labels",required=True); ap.add_argument("--model",required=True)
    ap.add_argument("--out",default="test_report")
    a=ap.parse_args(); main(a.features,a.labels,a.model,a.out)

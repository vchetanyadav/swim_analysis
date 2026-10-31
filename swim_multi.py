"""
swim_multi.py  -  multi-criterion expert-aligned modelling.

Trains the three OBSERVABLE expert criteria from your 467 labels, recoded onto the expert's scale:
  trunk    (from posture, flipped: expert 0=vertical..2=horizontal)
  activity (from passive_active, flipped: expert 0=tense..2=relaxed)   [noisier: definitions differ]
  outcome  (floating outcome 0-3 from float_label + passive_active)
Head and airway are NOT modelled (no head labels beyond experts; airway not observable from pose).

Runs TWO models the review asked for, Softmax multinomial logistic regression and a classification
tree, for each criterion. Reports grouped-CV macro-F1 per criterion per model, then an external
expert-comparison table (Expert vs Softmax vs Tree) on the four gold participants, with exact
agreement and mean absolute error. The four experts are never used for training.

Usage:
  python swim_multi.py "features_v2_side_front/"*.csv --labels labels_merged.csv \
      --expert expert_scores.csv --out model_multi
"""
import argparse, glob, os, json, numpy as np, pandas as pd
import swim_train as T
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.base import clone
from sklearn.metrics import f1_score, mean_absolute_error
import joblib

def MODELS():
    return {
      "Softmax": make_pipeline(SimpleImputer(strategy="median"),RobustScaler(),
                 LogisticRegression(class_weight="balanced",max_iter=5000,C=1.0)),
      "Tree":    make_pipeline(SimpleImputer(strategy="median"),
                 DecisionTreeClassifier(max_depth=4,min_samples_leaf=15,class_weight="balanced",random_state=0)),
    }

def rec_trunk(s):
    p=pd.to_numeric(s.get("posture"),errors="coerce"); return None if pd.isna(p) else 2-int(p)
def rec_act(s):
    p=pd.to_numeric(s.get("passive_active"),errors="coerce"); return None if pd.isna(p) else 2-int(p)
def rec_out(s):
    fl=pd.to_numeric(s.get("float_label"),errors="coerce"); pa=pd.to_numeric(s.get("passive_active"),errors="coerce")
    if pd.isna(fl): return None
    fl=int(fl); return fl if fl in (0,1) else (3 if (not pd.isna(pa) and int(pa)==0) else 2)
TARGETS={"trunk":rec_trunk,"activity":rec_act,"outcome":rec_out}
ECOL={"trunk":"trunk_position","activity":"activity","outcome":"floating_outcome"}

def main(a):
    os.makedirs(a.out,exist_ok=True)
    feat=T.load_features(a.features); feat=feat[feat.get("valid_pct",100)>=60]
    feat["win_center"]=(feat["start_s"]+feat["end_s"])/2; feat["participant"]=feat["source"].map(T.pid_from)
    lab=pd.read_csv(a.labels); lab=lab[lab["exclude"].fillna(0).astype(int)==0]
    exp=pd.read_csv(a.expert); exp["pid"]=exp["participant"].map(T.pid_from)
    EXP={r["pid"]:r for _,r in exp.iterrows()}
    F=[c for c in T.FEATURES if c in feat.columns]

    def attach(rec):
        f=feat.copy(); f["y"]=np.nan
        for _,s in lab.iterrows():
            v=rec(s)
            if v is None: continue
            m=(f["source"]==s["source"])&(f["win_center"]>=s["start_s"])&(f["win_center"]<s["end_s"])
            f.loc[m,"y"]=v
        return f.dropna(subset=["y"]).assign(y=lambda d:d["y"].astype(int))

    report={"cv":{},"expert_table":[],"expert_summary":{}}
    frozen={}          # (criterion, model) -> fitted on dev-minus-experts, for the expert table
    print("=== grouped 5-fold CV macro-F1 per criterion per model ===")
    print(f"{'criterion':10s} {'Softmax':>9s} {'Tree':>7s}")
    for cname,rec in TARGETS.items():
        d=attach(rec); X=d[F].to_numpy(float); y=d["y"].to_numpy(); g=d["participant"].to_numpy()
        report["cv"][cname]={}
        line=f"{cname:10s}"
        for mname,proto in MODELS().items():
            oof=np.full(len(y),-1)
            for tr,te in GroupKFold(5).split(X,y,g):
                m=clone(proto); m.fit(X[tr],y[tr]); oof[te]=m.predict(X[te])
            mf=float(f1_score(y,oof,average="macro"))
            report["cv"][cname][mname]=round(mf,3); line+=f"{mf:>{9 if mname=='Softmax' else 7}.3f}"
            fm=clone(proto); fm.fit(X,y); joblib.dump({"model":fm,"features":F,"criterion":cname,"model_name":mname},
                     os.path.join(a.out,f"model_{cname}_{mname}.joblib"))
            tr=d[~d["participant"].isin(EXP)]; vm=clone(proto); vm.fit(tr[F].to_numpy(float),tr["y"].to_numpy())
            frozen[(cname,mname)]=vm
        print(line)

    # expert comparison table: one row per participant x criterion, Expert | Softmax | Tree
    print("\n=== expert comparison (experts held out of training) ===")
    print(f"{'part':5s} {'criterion':10s} {'expert':>7s} {'Softmax':>8s} {'Tree':>6s}")
    err={m:{c:[] for c in TARGETS} for m in ["Softmax","Tree"]}
    exact={m:{c:0 for c in TARGETS} for m in ["Softmax","Tree"]}; cnt={c:0 for c in TARGETS}
    for pid,e in EXP.items():
        g=feat[feat["participant"]==pid]; w=g[(g["start_s"]<60)&(g["end_s"]>30)]; w=w if len(w) else g
        if len(w)==0: continue
        for cname in TARGETS:
            ecol=ECOL[cname]
            if ecol not in e or pd.isna(e[ecol]): continue
            ev=int(e[ecol]); preds={}
            for mname in ["Softmax","Tree"]:
                pr=frozen[(cname,mname)].predict(w[F].to_numpy(float))
                preds[mname]=int(pd.Series(pr).value_counts().idxmax())
                err[mname][cname].append(abs(preds[mname]-ev)); exact[mname][cname]+=int(preds[mname]==ev)
            cnt[cname]+=1
            report["expert_table"].append({"participant":pid,"criterion":cname,"expert":ev,**preds})
            print(f"{pid:5s} {cname:10s} {ev:>7d} {preds['Softmax']:>8d} {preds['Tree']:>6d}")
    print("\n=== summary: exact agreement / MAE vs expert (4 participants) ===")
    print(f"{'criterion':10s} {'Softmax ex/MAE':>16s} {'Tree ex/MAE':>14s}")
    for cname in TARGETS:
        n=cnt[cname] or 1
        sx=f"{exact['Softmax'][cname]}/{cnt[cname]}, {np.mean(err['Softmax'][cname] or [0]):.2f}"
        tx=f"{exact['Tree'][cname]}/{cnt[cname]}, {np.mean(err['Tree'][cname] or [0]):.2f}"
        report["expert_summary"][cname]={"Softmax":sx,"Tree":tx}
        print(f"{cname:10s} {sx:>16s} {tx:>14s}")
    json.dump(report,open(os.path.join(a.out,"multi_report.json"),"w"),indent=2)
    print(f"\nwrote {a.out}/ (model_<criterion>_<Softmax|Tree>.joblib + multi_report.json)")
    print("Head and airway are NOT modelled; they stay expert-only criteria.")

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("features",nargs="+")
    ap.add_argument("--labels",required=True); ap.add_argument("--expert",required=True)
    ap.add_argument("--out",default="model_multi")
    a=ap.parse_args(); main(a)

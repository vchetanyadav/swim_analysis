"""
swim_train.py  -  Stage 3: champion/challenger float-state classifier, honestly evaluated.

Target = passive float state (NOT trunk angle):
    2 = floating        maintains passive flotation across the window, no sustained propulsion
    1 = almost floating  intermittent; periodically sinks, reorients, or uses corrective movement
    0 = not floating     cannot hold passive flotation without continuous propulsion/support

Trunk angle, posture, effort, drift and symmetry are EXPLANATORY measurements, not the target.

Evaluation is participant-grouped (leave-one-subject-out over the development set), because
videos from one person are repeated observations, not independent samples. The classes are
ORDERED, so we report quadratic-weighted Cohen's kappa alongside macro-F1 and balanced accuracy.
Models compete; the winner is chosen on grouped CV, then frozen (model + scaler + feature list +
versions) for a one-time run on the sealed test split via swim_test.py.

Usage:
    python swim_train.py features_side_front/*.csv --labels labels.csv --out model_side
    python swim_train.py features_side_front/*.csv --labels labels.csv --splits splits.csv --out model_side
"""
import argparse, glob, os, re, sys, json, platform, warnings, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import (classification_report, confusion_matrix,
                             balanced_accuracy_score, f1_score, cohen_kappa_score)
import joblib
warnings.filterwarnings("ignore")

# Side-front DEFENSIBLE feature set (v3). The right arm is invisible to MediaPipe in a side
# view ~91-95% of the time, so right-arm speed/freq/asymmetry are NOT used as measurements
# (imputing them puts high model weight on a fabricated constant - a camera-specific artifact).
# Instead we keep reliable trunk/hip geometry + left-limb + aggregate motion, and add the
# always-present visibility fractions as HONEST features (how often the right side / head is
# trackable is itself behaviourally informative). Costs ~0.016 macro-F1 vs the old 17-feature
# set but removes all dependence on unseeable measurements.
FEATURES = ["trunk_angle_med","trunk_angle_std","trunk_angle_slope",
            "hip_y_med","hip_y_std","hip_y_slope","ankle_y_med",
            "arm_speed_l","arm_freq_l","leg_speed_l","leg_freq_l",
            "arm_speed","leg_speed",
            "right_arm_valid_pct","right_leg_valid_pct","nose_low_vis_frac"]
# Label schemes by target column. float_label is the 3-class client view; floating_outcome is the
# domain expert's richer 0-3 scale (cannot float -> floats with no effort). A target not listed here
# falls back to numbering the classes present in the data, so any ordinal target trains unchanged.
LABEL_SCHEMES={
  "float_label":{0:"not floating",1:"almost floating",2:"floating"},
  "floating_outcome":{0:"cannot float",1:"lots of effort",2:"floats w/ effort",3:"floats no effort"},
}
def class_names(target,present):
    base=LABEL_SCHEMES.get(target)
    return base if base else {int(c):str(int(c)) for c in present}
CLASS_NAMES=LABEL_SCHEMES["float_label"]  # default; main() rebinds per target

def pid_from(name):
    stem=os.path.splitext(os.path.basename(str(name)))[0]
    m=re.search(r'[Pp](\d+)',stem)
    if m: return f"P{int(m.group(1)):02d}"
    m=re.match(r'\s*(\d+)',stem)
    return f"V{int(m.group(1)):02d}" if m else stem[:8]

class OrdinalClassifier(BaseEstimator,ClassifierMixin):
    """Frank & Hall ordinal classifier: K-1 binary models P(y>k), respects class order."""
    def __init__(self,base=None): self.base=base
    def fit(self,X,y):
        self.classes_=np.sort(np.unique(y)); self.clfs_={}
        for k in self.classes_[:-1]:
            b=clone(self.base if self.base is not None else LogisticRegression(max_iter=3000,class_weight="balanced"))
            b.fit(X,(y>k).astype(int)); self.clfs_[k]=b
        return self
    def predict_proba(self,X):
        n=len(X); K=len(self.classes_); P=np.zeros((n,K))
        gt={k:self.clfs_[k].predict_proba(X)[:,1] for k in self.classes_[:-1]}
        for idx,c in enumerate(self.classes_):
            if idx==0: P[:,idx]=1-gt[self.classes_[0]]
            elif idx==K-1: P[:,idx]=gt[self.classes_[-2]]
            else: P[:,idx]=gt[self.classes_[idx-1]]-gt[self.classes_[idx]]
        return np.clip(P,1e-9,None)/np.clip(P,1e-9,None).sum(1,keepdims=True)
    def predict(self,X): return self.classes_[np.argmax(self.predict_proba(X),1)]

def build_models():
    sc=lambda clf:Pipeline([("sc",RobustScaler()),("clf",clf)])
    return {
      "OrdinalLogistic":sc(OrdinalClassifier(LogisticRegression(max_iter=3000,class_weight="balanced",C=1.0))),
      "LogisticRegression":sc(LogisticRegression(max_iter=3000,class_weight="balanced",C=1.0)),
      "RandomForest":sc(RandomForestClassifier(n_estimators=400,min_samples_leaf=2,
                        class_weight="balanced_subsample",random_state=0,n_jobs=-1)),
      "GradientBoosting":sc(HistGradientBoostingClassifier(max_depth=3,learning_rate=0.08,
                        max_iter=400,l2_regularization=1.0,random_state=0)),
    }

def load_features(patterns):
    files=[]; [files.extend(glob.glob(p)) for p in patterns]
    parts=[]
    for f in files:
        d=pd.read_csv(f); d["source"]=os.path.basename(f); parts.append(d)
    if not parts: raise SystemExit("no feature files matched")
    df=pd.concat(parts,ignore_index=True)
    df["participant"]=df["source"].map(pid_from)
    df["win_center"]=(df["start_s"]+df["end_s"])/2.0
    return df

def attach_labels(feat,labels_csv,target):
    lab=pd.read_csv(labels_csv)
    key="source" if ("source" in lab.columns and "source" in feat.columns) else "video_id"
    if "exclude" in lab.columns: lab=lab[lab["exclude"].fillna(0).astype(int)==0]
    if target not in lab.columns: raise SystemExit(f"labels.csv has no '{target}'. cols={list(lab.columns)}")
    lab=lab.dropna(subset=[target]).copy(); lab[target]=lab[target].astype(int)
    has_w="sample_weight" in lab.columns
    feat=feat.copy(); feat["label"]=np.nan; feat["w"]=1.0
    for _,s in lab.iterrows():
        m=(feat[key]==s[key])&(feat["win_center"]>=s["start_s"])&(feat["win_center"]<s["end_s"])
        feat.loc[m,"label"]=s[target]
        if has_w and pd.notna(s["sample_weight"]): feat.loc[m,"w"]=float(s["sample_weight"])
    out=feat.dropna(subset=["label"]).copy(); out["label"]=out["label"].astype(int)
    return out

def metrics(yt,yp):
    present=sorted(np.unique(np.concatenate([yt,yp])))
    return {"macro_f1":float(f1_score(yt,yp,average="macro")),
            "balanced_acc":float(balanced_accuracy_score(yt,yp)),
            "weighted_kappa":float(cohen_kappa_score(yt,yp,weights="quadratic")),
            "present":present}

def grouped_oof(df,feats,model):
    # Impute missing values INSIDE each fold (median fit on the training swimmers only), so the
    # held-out swimmer never informs its own imputation. Leak-free leave-one-swimmer-out.
    X=df[feats].to_numpy(float); y=df["label"].to_numpy(); g=df["participant"].to_numpy()
    w=df["w"].to_numpy(float) if "w" in df.columns else None
    oof=np.full(len(y),-1)
    for tr,te in LeaveOneGroupOut().split(X,y,g):
        if len(np.unique(y[tr]))<2: continue
        med=np.nanmedian(X[tr],axis=0); med=np.where(np.isnan(med),0.0,med)
        Xtr=np.where(np.isnan(X[tr]),med,X[tr]); Xte=np.where(np.isnan(X[te]),med,X[te])
        m=clone(model)
        try:  # pass per-sample weights to the classifier step when it supports them
            m.fit(Xtr,y[tr],**({"clf__sample_weight":w[tr]} if w is not None else {}))
        except (TypeError,ValueError):
            m.fit(Xtr,y[tr])
        oof[te]=m.predict(Xte)
    return oof

def main(patterns,labels_csv,target,splits_csv,out_dir):
    os.makedirs(out_dir,exist_ok=True)
    feat=load_features(patterns)
    if "valid_pct" in feat: feat=feat[feat["valid_pct"]>=60].reset_index(drop=True)
    df=attach_labels(feat,labels_csv,target)
    feats=[c for c in FEATURES if c in df.columns]

    test_parts=[]
    if splits_csv and os.path.exists(splits_csv):
        sp=pd.read_csv(splits_csv); pmap=dict(zip(sp["participant"],sp["split"]))
        df["split"]=df["participant"].map(pmap).fillna("train")
        test_parts=sorted(sp[sp["split"]=="test"]["participant"].unique().tolist())
        dev=df[df["split"]!="test"].copy()
        print(f"SEALED test participants (not touched here): {test_parts}")
    else:
        dev=df.copy()
        print("no --splits given: using all labeled data for grouped CV (keep a sealed test for the final number)")

    global CLASS_NAMES
    CLASS_NAMES=class_names(target,sorted(df["label"].unique()))
    print(f"development windows: {len(dev)}   features: {len(feats)}   target: {target}")
    print("class counts:\n",dev["label"].map(CLASS_NAMES).value_counts().to_string())
    if dev["label"].nunique()<2 or dev["participant"].nunique()<2:
        raise SystemExit("need >=2 classes and >=2 participants in the development set")

    results={}; best=None
    for name,model in build_models().items():
        oof=grouped_oof(dev,feats,model); mask=oof>=0
        yt,yp=dev["label"].to_numpy()[mask],oof[mask]
        M=metrics(yt,yp)
        # per-participant macro-F1
        dd=dev.loc[mask].copy(); dd["pred"]=yp
        pp={p:round(float(f1_score(g["label"],g["pred"],average="macro")),3)
            for p,g in dd.groupby("participant") if g["label"].nunique()>1}
        results[name]={**M,"per_participant_macroF1":pp,"n":int(mask.sum())}
        score=M["macro_f1"]+M["weighted_kappa"]
        print(f"\n{name}: macroF1={M['macro_f1']:.3f}  balAcc={M['balanced_acc']:.3f}  "
              f"weightedKappa={M['weighted_kappa']:.3f}")
        if best is None or score>best[1]:
            best=(name,score,oof,mask)

    name,_,oof,mask=best
    yt,yp=dev["label"].to_numpy()[mask],oof[mask]
    present=sorted(np.unique(np.concatenate([yt,yp])))
    print(f"\n===== WINNER on grouped CV: {name} =====")
    print(classification_report(yt,yp,labels=present,
          target_names=[CLASS_NAMES[c] for c in present],digits=3,zero_division=0))
    cm=confusion_matrix(yt,yp,labels=present)
    print("confusion (rows=true,cols=pred):",present); print(cm)
    print("per-participant macro-F1:",results[name]["per_participant_macroF1"])

    plt.figure(figsize=(5,4.2)); plt.imshow(cm,cmap="Blues"); plt.colorbar()
    plt.xticks(range(len(present)),[CLASS_NAMES[c] for c in present],rotation=30,ha="right")
    plt.yticks(range(len(present)),[CLASS_NAMES[c] for c in present])
    for i in range(len(present)):
        for j in range(len(present)):
            plt.text(j,i,cm[i,j],ha="center",va="center",color="white" if cm[i,j]>cm.max()/2 else "black")
    plt.ylabel("true"); plt.xlabel("predicted"); plt.title(f"{name} (grouped CV)"); plt.tight_layout()
    plt.savefig(os.path.join(out_dir,"confusion_winner.png"),dpi=120); plt.close()

    # freeze winner on ALL development data (with expert sample weights where present)
    final=build_models()[name]
    Xf=dev[feats].fillna(dev[feats].median()).to_numpy(); yf=dev["label"].to_numpy()
    wf=dev["w"].to_numpy(float) if "w" in dev.columns else None
    try:
        final.fit(Xf,yf,**({"clf__sample_weight":wf} if wf is not None else {}))
    except (TypeError,ValueError):
        final.fit(Xf,yf)
    meta={"winner":name,"target":target,"features":feats,"classes":CLASS_NAMES,
          "sealed_test_participants":test_parts,
          "medians":dev[feats].median().to_dict(),
          "frozen":{"python":platform.python_version(),"sklearn":sklearn.__version__,
                    "numpy":np.__version__,"pandas":pd.__version__}}
    joblib.dump({"model":final,**meta},os.path.join(out_dir,"model.joblib"))
    json.dump(results,open(os.path.join(out_dir,"cv_results.json"),"w"),indent=2,default=str)
    json.dump(meta,open(os.path.join(out_dir,"frozen_meta.json"),"w"),indent=2,default=str)
    print(f"\nfrozen -> {out_dir}/model.joblib (+ cv_results.json, frozen_meta.json, confusion_winner.png)")
    print("Model is FROZEN. Run the sealed test ONCE with swim_test.py; do not retune afterwards.")

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("features",nargs="+")
    ap.add_argument("--labels",required=True); ap.add_argument("--target",default="float_label")
    ap.add_argument("--splits",default=None); ap.add_argument("--out",default="model_out")
    a=ap.parse_args(); main(a.features,a.labels,a.target,a.splits,a.out)

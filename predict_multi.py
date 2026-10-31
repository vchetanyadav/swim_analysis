"""
predict_multi.py  -  expert-style assessment for one participant from the multi-criterion models.

Loads the trunk, activity and floating-outcome models trained by swim_multi.py and prints the
expert rubric the model CAN estimate, with the pose measurements behind it. Airway and head are
shown as "expert only", because the pose cannot judge them.

Usage:
  python predict_multi.py "features_v2_side_front/P7_..._features.csv" --model-dir model_multi --which Softmax
"""
import argparse, os, glob, json, numpy as np, pandas as pd
import joblib
import swim_train as T

TRUNK={0:"vertical",1:"diagonal",2:"horizontal"}
ACT={0:"tense",1:"neutral",2:"relaxed"}
OUT={0:"cannot float",1:"floats with lots of effort",2:"floats with effort",3:"floats with no effort"}

def load(mdir,crit,which):
    p=os.path.join(mdir,f"model_{crit}_{which}.joblib")
    if not os.path.exists(p): raise SystemExit(f"missing {p} (run swim_multi.py first)")
    return joblib.load(p)

def main(a):
    d=pd.read_csv(a.features)
    if "valid_pct" in d: d=d[d["valid_pct"]>=60]
    if len(d)==0: print("UNABLE TO ASSESS: pose tracking too poor"); return
    name=os.path.basename(a.features)
    mt,ma,mo=load(a.model_dir,"trunk",a.which),load(a.model_dir,"activity",a.which),load(a.model_dir,"outcome",a.which)
    F=mt["features"]; X=d[[c for c in F if c in d]].copy()
    for c in F:
        if c not in X: X[c]=np.nan
    X=X[F].to_numpy(float)
    def assess(m):   # average per-window probabilities, then most likely class + its share
        clf=m["model"]
        if hasattr(clf,"predict_proba"):
            P=clf.predict_proba(X).mean(0); k=int(clf.classes_[int(np.argmax(P))]); return k,float(P.max())
        pr=clf.predict(X); k=int(pd.Series(pr).value_counts().idxmax()); return k,float((pr==k).mean())
    (tr,trc),(ac,acc),(ou,ouc)=assess(mt),assess(ma),assess(mo)
    g=lambda c: float(np.nanmedian(d[c])) if c in d and d[c].notna().any() else float("nan")
    print(f"\nEXPERT-STYLE ASSESSMENT  —  {name}   [{a.which}]")
    print(f"  Floating outcome : {ou}  ({OUT[ou]})   confidence {ouc*100:.0f}%")
    print(f"  Trunk position   : {tr}  ({TRUNK[tr]})   confidence {trc*100:.0f}%")
    print(f"  Activity/tension : {ac}  ({ACT[ac]})   confidence {acc*100:.0f}%   [noisier criterion]")
    print(f"  Head position    : expert only (no pose model)")
    print(f"  Airway           : expert only (not observable from pose)")
    print(f"  --- pose measurements behind this ---")
    print(f"  trunk angle {g('trunk_angle_med'):.0f} deg (var +/-{g('trunk_angle_std'):.1f}), "
          f"hip drift {g('hip_y_slope'):+.3f}/s, leg speed {g('leg_speed'):.1f}, pose quality {d['valid_pct'].mean():.0f}%")
    if a.out:
        os.makedirs(a.out,exist_ok=True)
        json.dump({"file":name,"model":a.which,"floating_outcome":ou,"trunk":tr,"activity":ac,
                   "head":"expert_only","airway":"expert_only"},
                  open(os.path.join(a.out,"assessment.json"),"w"),indent=2)
        print(f"\nwrote {a.out}/assessment.json")

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("features")
    ap.add_argument("--model-dir",default="model_multi"); ap.add_argument("--which",default="Softmax",choices=["Softmax","Tree"])
    ap.add_argument("--out",default=None)
    a=ap.parse_args(); main(a)

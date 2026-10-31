"""
swim_classify.py  -  transparent, label-free float-state classifier.

Grounded in the stakeholder definition: the instructed target is a horizontal back-float,
but a calm, buoyant swimmer who hangs more vertically is still FLOATING. So float state is
about staying at the surface passively (stable, head clear, not sinking), judged separately
from alignment (how horizontal) and effort (how much limb work).

Classes: 2 = floating, 1 = almost floating, 0 = not floating.
This is a defensible interim model you can run and test now; swim_train.py replaces/validates
it once instructor labels exist.

Usage:
    python swim_classify.py features_side_front/*.csv --out rule_out
    python swim_classify.py "features_side_front/5 Side front_features.csv" --calibrate --out rule_v5
"""
import argparse, glob, os, json, numpy as np, pandas as pd

# physical defaults (torso-normalized speeds are calibrated from data unless overridden)
DEF = dict(std_thr=12.0,       # trunk-angle wobble (deg) above which = unstable
           sink_thr=0.05,      # hip-depth slope (torso/s) above which = sinking
           head_thr=0.50,      # fraction of window with head submerged
           steep_thr=45.0,     # trunk angle (deg) above which = near-vertical
           eff_thr=None)       # arm+leg mean speed; None -> calibrate to data median

def effort(df):
    cols=[c for c in ["arm_speed","leg_speed"] if c in df]
    return df[cols].mean(axis=1) if cols else pd.Series(np.nan,index=df.index)

def classify(df, p):
    eff=effort(df)
    sink   = df.get("hip_y_slope", pd.Series(0,index=df.index)) > p["sink_thr"]
    unstable = df.get("trunk_angle_std", pd.Series(0,index=df.index)) > p["std_thr"]
    higheff  = eff > p["eff_thr"]
    steep    = df.get("trunk_angle_med", pd.Series(0,index=df.index)) > p["steep_thr"]

    lab=np.full(len(df),1,dtype=int)          # default = almost floating
    reason=np.array(["maintaining with effort or mild instability"]*len(df),dtype=object)

    # NOT floating: actively failing to stay up passively (no head-submersion rule: not detectable)
    failing = sink | (unstable & higheff)
    lab[failing.values]=0
    reason[sink.values]="sinking (hips dropping over time)"
    reason[(unstable & higheff & ~sink).values]="unstable and working hard (struggling)"

    # FLOATING: calm, stable, staying up (horizontal OR passive-vertical both qualify)
    good = (~sink) & (~unstable) & (~higheff)
    lab[good.values]=2
    reason[good.values]="stable, head clear, low effort, staying at the surface"

    align=np.where(df.get("trunk_angle_med",0)<20,"horizontal",
           np.where(df.get("trunk_angle_med",0)<p["steep_thr"],"angled","vertical"))
    efflvl=np.where(eff<p["eff_thr"]*0.6,"low",np.where(eff<p["eff_thr"],"moderate","high"))
    out=df.copy()
    out["float_pred"]=lab; out["reason"]=reason
    out["alignment"]=align; out["effort_level"]=efflvl
    return out

def summarize(out):
    # use the stable middle 80% of windows for the overall verdict
    n=len(out); lo,hi=int(n*0.1),int(n*0.9) if n>=10 else n
    mid=out.iloc[lo:hi] if n>=10 else out
    frac=mid["float_pred"].value_counts(normalize=True).reindex([0,1,2]).fillna(0)
    verdict=int(mid["float_pred"].mode().iloc[0])
    names={0:"not floating",1:"almost floating",2:"floating"}
    return {"verdict":names[verdict],
            "fraction_floating":round(float(frac[2]),3),
            "fraction_almost":round(float(frac[1]),3),
            "fraction_not":round(float(frac[0]),3),
            "dominant_alignment":mid["alignment"].mode().iloc[0],
            "dominant_effort":mid["effort_level"].mode().iloc[0]}

def _pid(name):
    import re
    stem=os.path.splitext(os.path.basename(str(name)))[0]
    m=re.search(r'[Pp](\d+)',stem)
    if m: return f"P{int(m.group(1)):02d}"
    m=re.match(r'\s*(\d+)',stem); return f"V{int(m.group(1)):02d}" if m else stem[:8]

def main(patterns,out_dir,calibrate,overrides,splits_csv=None):
    os.makedirs(out_dir,exist_ok=True)
    files=[]; [files.extend(glob.glob(p)) for p in patterns]
    p=dict(DEF); p.update({k:v for k,v in overrides.items() if v is not None})
    test_parts=set()
    if splits_csv and os.path.exists(splits_csv):
        sp=pd.read_csv(splits_csv); test_parts=set(sp[sp["split"]=="test"]["participant"])
    loaded=[]
    for f in files:
        df=pd.read_csv(f)
        if "valid_pct" in df: df=df[df["valid_pct"]>=60].reset_index(drop=True)
        loaded.append((f,df))
    # calibrate effort threshold ONCE, EXCLUDING sealed test participants (no leakage)
    if p["eff_thr"] is None or calibrate:
        pool=[d for f,d in loaded if _pid(f) not in test_parts]
        alle=pd.concat([effort(d) for d in pool],ignore_index=True) if pool else pd.Series([4.0])
        p["eff_thr"]=float(np.nanmedian(alle)) if len(alle) else 4.0
        note=" (test participants excluded)" if test_parts else ""
        print(f"calibrated effort threshold (median of train+val) = {p['eff_thr']:.2f}{note}")
    allrows=[]
    for f,df in loaded:
        out=classify(df,p); out["source"]=os.path.basename(f)
        s=summarize(out); s["source"]=os.path.basename(f); s["eff_thr_used"]=round(p["eff_thr"],2)
        allrows.append(out)
        print(f"{os.path.basename(f):40s} -> {s['verdict']:16s} "
              f"(float {s['fraction_floating']:.0%} / almost {s['fraction_almost']:.0%} / not {s['fraction_not']:.0%})"
              f"  align={s['dominant_alignment']} effort={s['dominant_effort']}")
        json.dump(s,open(os.path.join(out_dir,os.path.splitext(os.path.basename(f))[0]+"_verdict.json"),"w"),indent=2)
    if allrows:
        pd.concat(allrows,ignore_index=True).to_csv(os.path.join(out_dir,"windows_classified.csv"),index=False)
    print(f"\nwrote per-video verdicts + windows_classified.csv to {out_dir}/")
    print("Rule-based interim model. Thresholds are transparent and in the JSON; swim_train.py supersedes it with instructor labels.")

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("features",nargs="+"); ap.add_argument("--out",default="rule_out")
    ap.add_argument("--calibrate",action="store_true",help="set effort threshold from this data's median")
    ap.add_argument("--splits",default=None,help="splits.csv; excludes test participants from calibration")
    for k in ("std_thr","sink_thr","head_thr","steep_thr","eff_thr"):
        ap.add_argument("--"+k,type=float,default=None)
    a=ap.parse_args()
    ov={k:getattr(a,k) for k in ("std_thr","sink_thr","head_thr","steep_thr","eff_thr")}
    main(a.features,a.out,a.calibrate,ov,a.splits)

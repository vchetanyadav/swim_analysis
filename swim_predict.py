"""
swim_predict.py  -  run the frozen model on ONE video's features and produce a full report.

Outputs a passive-float assessment: classification + confidence + explanatory measurements
(posture, stability, vertical drift, limb activity, symmetry, pose quality). It ABSTAINS with
"UNABLE TO ASSESS" when tracking is too poor, and flags "UNCERTAIN" when confidence is low,
rather than forcing a wrong answer.

Usage:
    python swim_predict.py "features_side_front/HELDOUT_features.csv" --model model_side/model.joblib --out test_heldout
    python swim_predict.py "features_side_front/HELDOUT_features.csv" --rule --eff-thr 3.9 --out test_heldout
"""
import argparse, os, json, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

NAMES={0:"NOT FLOATING",1:"ALMOST FLOATING",2:"FLOATING"}
MIN_VALID_FRAC=0.40      # if fewer than this fraction of windows are reliable -> abstain
CONF_UNCERTAIN=0.55      # mean confidence below this -> flag uncertain

def band(v,lo,hi,labels): return labels[0] if v<lo else (labels[1] if v<hi else labels[2])

def report_block(df):
    g=lambda c: float(np.nanmedian(df[c])) if c in df and df[c].notna().any() else float("nan")
    ta,ts=g("trunk_angle_med"),g("trunk_angle_std"); dr=g("hip_y_slope")
    al=g("arm_speed_l"); ar=g("arm_speed_r"); ll=g("leg_speed_l"); lr=g("leg_speed_r")
    arm=np.nanmean([al,ar]); leg=np.nanmean([ll,lr])
    posture = band(ta,20,45,["horizontal","inclined","near-vertical"]) if ta==ta else "unknown"
    stab = band(ts,8,14,["high","moderate","low"]) if ts==ts else "unknown"
    drift = "rising" if (dr==dr and dr<-0.05) else ("sinking" if (dr==dr and dr>0.05) else "stable")
    armlvl = band(arm,2.5,5.0,["low","moderate","high"]) if arm==arm else "unknown"
    leglvl = band(leg,2.5,5.0,["low","moderate","high"]) if leg==leg else "unknown"
    asym="unknown"
    if ll==ll and lr==lr and (ll+lr)>0:
        asym="balanced" if abs(ll-lr)/(ll+lr)<0.3 else "asymmetric"
    return {"posture":posture,"trunk_angle_deg":round(ta,1) if ta==ta else None,
            "stability":stab,"trunk_variation_deg":round(ts,1) if ts==ts else None,
            "vertical_movement":drift,"body_drift":round(dr,3) if dr==dr else None,
            "arm_activity":armlvl,"leg_activity":leglvl,"symmetry":asym}

def main(csv,model_path,rule,out_dir,eff_thr):
    os.makedirs(out_dir,exist_ok=True)
    raw=pd.read_csv(csv); name=os.path.splitext(os.path.basename(csv))[0]
    total=len(raw)
    df=raw[raw["valid_pct"]>=60].reset_index(drop=True) if "valid_pct" in raw else raw
    valid_frac=len(df)/total if total else 0
    pose_quality=round(float(df["valid_pct"].mean()),1) if "valid_pct" in df and len(df) else None

    if len(df)==0 or valid_frac<MIN_VALID_FRAC:
        rep={"video":name,"classification":"UNABLE TO ASSESS",
             "reason":f"pose visibility too low ({valid_frac:.0%} of windows reliable)",
             "pose_quality_pct":pose_quality,"windows_total":total,"windows_reliable":len(df)}
        json.dump(rep,open(os.path.join(out_dir,f"{name}_report.json"),"w"),indent=2)
        print(f"\n{name}: UNABLE TO ASSESS — pose visibility too low ({valid_frac:.0%} reliable)")
        return

    if rule or not model_path:
        from swim_classify import classify, DEF
        p=dict(DEF); p["eff_thr"]=eff_thr if eff_thr is not None else 4.0
        preds=classify(df,p)["float_pred"].to_numpy(); conf=np.full(len(df),np.nan); mode="rule-based"
    else:
        import joblib
        B=joblib.load(model_path); model=B["model"]; feats=B["features"]
        X=df[[f for f in feats if f in df]].copy()
        for f in feats:
            if f not in X: X[f]=np.nan
        X=X[feats].fillna(pd.Series(B.get("medians",{})))
        preds=model.predict(X.to_numpy())
        try: conf=model.predict_proba(X.to_numpy()).max(1)
        except Exception: conf=np.full(len(df),np.nan)
        mode=f"trained model ({B.get('winner','?')})"

    df=df.copy(); df["float_pred"]=preds; df["confidence"]=conf
    n=len(df); lo,hi=(int(n*0.1),int(n*0.9)) if n>=10 else (0,n); mid=df.iloc[lo:hi]
    frac=mid["float_pred"].value_counts(normalize=True).reindex([0,1,2]).fillna(0)
    verdict=int(mid["float_pred"].mode().iloc[0])
    meanconf=float(np.nanmean(mid["confidence"])) if mid["confidence"].notna().any() else None
    uncertain = meanconf is not None and meanconf<CONF_UNCERTAIN
    expl=report_block(df)

    rep={"video":name,"mode":mode,
         "classification":("UNCERTAIN ("+NAMES[verdict].title()+")" if uncertain else NAMES[verdict]),
         "confidence_pct":round(meanconf*100,0) if meanconf is not None else None,
         "fraction":{"floating":round(float(frac[2]),3),"almost":round(float(frac[1]),3),"not":round(float(frac[0]),3)},
         "pose_quality_pct":pose_quality,**expl,"windows":n}
    json.dump(rep,open(os.path.join(out_dir,f"{name}_report.json"),"w"),indent=2)
    df.to_csv(os.path.join(out_dir,f"{name}_windows.csv"),index=False)

    plt.figure(figsize=(10,2.6)); colors={0:"#D7513F",1:"#E0A83D",2:"#1E9E8A"}
    plt.scatter(df["start_s"],df["float_pred"],c=[colors.get(v,"#888888") for v in df["float_pred"]],s=18)
    plt.yticks([0,1,2],["not","almost","floating"]); plt.xlabel("time (s)")
    plt.title(f"{name} — {rep['classification']} ({mode})"); plt.tight_layout()
    plt.savefig(os.path.join(out_dir,f"{name}_timeline.png"),dpi=120); plt.close()

    print(f"\nPASSIVE FLOAT ASSESSMENT  —  {name}   [{mode}]")
    print(f"  Classification : {rep['classification']}")
    if meanconf is not None: print(f"  Confidence     : {meanconf*100:.0f}%")
    print(f"  Mix            : floating {frac[2]:.0%} / almost {frac[1]:.0%} / not {frac[0]:.0%}")
    print(f"  Posture        : {expl['posture']} (trunk {expl['trunk_angle_deg']}°)")
    print(f"  Stability      : {expl['stability']} (variation ±{expl['trunk_variation_deg']}°)")
    print(f"  Vertical move  : {expl['vertical_movement']} (drift {expl['body_drift']} /s)")
    print(f"  Arm activity   : {expl['arm_activity']}")
    print(f"  Leg activity   : {expl['leg_activity']}")
    print(f"  Symmetry       : {expl['symmetry']}")
    print(f"  Pose quality   : {pose_quality}% valid")
    print(f"wrote {name}_report.json, {name}_windows.csv, {name}_timeline.png to {out_dir}/")

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("features"); ap.add_argument("--model",default=None)
    ap.add_argument("--rule",action="store_true"); ap.add_argument("--eff-thr",type=float,default=None)
    ap.add_argument("--out",default="predict_out")
    a=ap.parse_args(); main(a.features,a.model,a.rule,a.out,a.eff_thr)

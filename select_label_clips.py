"""
select_label_clips.py  -  choose a diverse, representative subset of clips to label.

Labels are expensive, so instead of labeling everything we pick a stratified subset that spans
the behaviour range (via clustering) AND many participants. TRAIN + VALIDATION participants only;
the sealed TEST participants are never selected here.

Each selected clip is an 8s segment (centred on a representative window) that the instructor will
rate. Output: label_manifest.csv -> feed to render_skeleton.py.

Usage:
    python select_label_clips.py "features_v2_side_front/"*.csv --splits data/splits_v2.csv \
        --n 250 --per-participant 3 --out label_manifest.csv
"""
import argparse, glob, os, re, numpy as np, pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans

BEHAV=["trunk_angle_med","trunk_angle_std","arm_speed","leg_speed","hip_y_slope"]
CLIP_S=8.0

def pid_from(name):
    stem=os.path.splitext(os.path.basename(str(name)))[0]
    m=re.search(r'[Pp](\d+)',stem)
    if m: return f"P{int(m.group(1)):02d}"
    m=re.match(r'\s*(\d+)',stem); return f"V{int(m.group(1)):02d}" if m else stem[:8]

def main(patterns,splits_csv,n,per_part,k,out,min_valid):
    files=[]; [files.extend(glob.glob(p)) for p in patterns]
    keep_test_out=set()
    if splits_csv and os.path.exists(splits_csv):
        sp=pd.read_csv(splits_csv); keep_test_out=set(pid_from(x) for x in sp[sp["split"]=="test"]["participant"])
    rows=[]
    for f in files:
        pid=pid_from(f)
        if pid in keep_test_out: continue              # never select sealed test
        d=pd.read_csv(f); d["source"]=os.path.basename(f); d["participant"]=pid
        d["raw_stem"]=os.path.basename(f).replace("_features.csv","")
        if "torso_valid_pct" in d: d=d[d["torso_valid_pct"]>=min_valid]
        elif "valid_pct" in d: d=d[d["valid_pct"]>=min_valid]
        rows.append(d)
    if not rows: raise SystemExit("no train/val feature windows found")
    df=pd.concat(rows,ignore_index=True)
    feats=[c for c in BEHAV if c in df.columns]
    X=df[feats].copy()
    for c in feats: X[c]=X[c].clip(X[c].quantile(.02),X[c].quantile(.98))
    X=X.fillna(X.median())
    df["cl"]=KMeans(n_clusters=k,n_init=10,random_state=0).fit_predict(RobustScaler().fit_transform(X))
    print(f"{len(df)} windows from {df['participant'].nunique()} participants -> {k} behaviour clusters")

    # stratified pick: round-robin over clusters; stop when a full pass adds nothing
    picks=[]; per_count={}; used=set(); starts_by_p={}
    order=df.sample(frac=1,random_state=0)
    by_cl={cl:order[order["cl"]==cl] for cl in sorted(df["cl"].unique())}
    clusters=list(by_cl.keys())
    while len(picks)<n:
        added=0
        for cl in clusters:
            if len(picks)>=n: break
            for idx,r in by_cl[cl].iterrows():
                if idx in used: continue
                p=r["participant"]
                if per_count.get(p,0)>=per_part: continue
                cstart=round(max(0.0,r["start_s"]+1.5-CLIP_S/2),1)
                if any(abs(st-cstart)<CLIP_S*0.75 for st in starts_by_p.get(p,[])): used.add(idx); continue
                picks.append({"clip":f"{p}__{int(cstart):04d}.mp4","participant":p,
                              "source":r["source"],"raw_stem":r["raw_stem"],
                              "start_s":cstart,"end_s":round(cstart+CLIP_S,1),
                              "seg_id":int(cstart),"cluster":int(cl)})
                per_count[p]=per_count.get(p,0)+1; used.add(idx)
                starts_by_p.setdefault(p,[]).append(cstart); added+=1; break
        if added==0: break
    man=pd.DataFrame(picks)
    man.to_csv(out,index=False)
    print(f"selected {len(man)} clips across {man['participant'].nunique()} participants "
          f"(max {per_part}/participant)")
    print("per-cluster counts:",man["cluster"].value_counts().sort_index().to_dict())
    print(f"wrote {out} -> feed to render_skeleton.py --manifest {out}")

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("features",nargs="+")
    ap.add_argument("--splits",default=None); ap.add_argument("--n",type=int,default=250)
    ap.add_argument("--per-participant",type=int,default=3); ap.add_argument("--k",type=int,default=6)
    ap.add_argument("--min-valid",type=float,default=60.0); ap.add_argument("--out",default="label_manifest.csv")
    a=ap.parse_args(); main(a.features,a.splits,a.n,a.per_participant,a.k,a.out,a.min_valid)

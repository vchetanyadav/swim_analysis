"""
select_round2.py  -  targeted second labelling round to fix class imbalance + the posture/float
confound. Runs on the windowed feature CSVs (train + validation only; test never selected).

It assembles ONE manifest with three streams, each tagged in a 'reason' column:

  minority  - windows whose signals suggest NOT/ALMOST floating (high trunk angle, downward hip
              drift, unstable trunk, high limb effort). These are where real Class 0/1 examples
              hide; the instructor assigns the true label.
  unlabeled - a spread of windows from development participants with no labels yet.
  rereview  - the vertical-posture clips currently labelled not/almost floating, to re-judge
              against the stakeholder definition (passive vertical float IS floating).

Output feeds render_skeleton.py --manifest (run on the server that holds the raw landmark CSVs).

Usage:
  python select_round2.py "features_v2_side_front/"*.csv --splits splits_v2.csv \
      --labels labels.csv --rereview rereview_vertical.csv \
      --n-minority 180 --n-unlabeled 60 --per-participant 3 --out label_manifest_round2.csv
"""
import argparse, glob, os, re, numpy as np, pandas as pd

CLIP_S=8.0

def pid_from(name):
    stem=os.path.splitext(os.path.basename(str(name)))[0]
    m=re.search(r'[Pp](\d+)',stem)
    if m: return f"P{int(m.group(1)):02d}"
    m=re.match(r'\s*(\d+)',stem); return f"V{int(m.group(1)):02d}" if m else stem[:8]

def zscore(s):
    s=pd.to_numeric(s,errors="coerce"); sd=s.std(ddof=0)
    return (s-s.mean())/sd if sd and not np.isnan(sd) else s*0.0

def load_dev(patterns,splits_csv,min_valid):
    files=[]; [files.extend(glob.glob(p)) for p in patterns]
    test=set()
    if splits_csv and os.path.exists(splits_csv):
        sp=pd.read_csv(splits_csv); test=set(pid_from(x) for x in sp[sp["split"]=="test"]["participant"])
    rows=[]
    for f in files:
        pid=pid_from(f)
        if pid in test: continue
        d=pd.read_csv(f); d["source"]=os.path.basename(f); d["participant"]=pid
        d["raw_stem"]=os.path.basename(f).replace("_features.csv","")
        vc="torso_valid_pct" if "torso_valid_pct" in d else ("valid_pct" if "valid_pct" in d else None)
        if vc: d=d[d[vc]>=min_valid]
        rows.append(d)
    if not rows: raise SystemExit("no train/val feature windows found")
    return pd.concat(rows,ignore_index=True)

def mk(row,reason,cl=-1):
    cstart=round(max(0.0,row["start_s"]+1.5-CLIP_S/2),1)
    p=row["participant"]
    return {"clip":f"{p}__{int(cstart):04d}.mp4","participant":p,"source":row["source"],
            "raw_stem":row["raw_stem"],"start_s":cstart,"end_s":round(cstart+CLIP_S,1),
            "seg_id":int(cstart),"cluster":int(cl),"reason":reason}

def spaced(picks_for_p,cstart):
    return any(abs(st-cstart)<CLIP_S*0.75 for st in picks_for_p)

def main(a):
    df=load_dev(a.features,a.splits,a.min_valid)
    print(f"{len(df)} dev windows from {df['participant'].nunique()} participants")

    # --- minority score: higher => more likely NOT/ALMOST floating ---
    # downward drift (sinking) = positive hip_y_slope (image y grows downward)
    s = ( 1.0*zscore(df.get("trunk_angle_med",0))      # steeper trunk
        + 0.7*zscore(df.get("trunk_angle_std",0))      # unstable trunk
        + 1.0*zscore(df.get("hip_y_slope",0))          # sinking
        + 0.5*zscore(df.get("arm_speed",0))            # effort
        + 0.5*zscore(df.get("leg_speed",0)) )          # effort / treading
    df=df.assign(mscore=s)

    picks=[]; per={}; starts={}
    # stream 1: minority-targeted, round-robin by descending score with per-participant cap
    cand=df.sort_values("mscore",ascending=False)
    for _,r in cand.iterrows():
        if len([x for x in picks if x["reason"]=="minority"])>=a.n_minority: break
        p=r["participant"]
        if per.get(p,0)>=a.per_participant: continue
        cstart=round(max(0.0,r["start_s"]+1.5-CLIP_S/2),1)
        if spaced(starts.get(p,[]),cstart): continue
        picks.append(mk(r,"minority")); per[p]=per.get(p,0)+1; starts.setdefault(p,[]).append(cstart)

    # stream 2: unlabeled development participants (spread a few windows across each)
    lab_pids=set(pid_from(x) for x in pd.read_csv(a.labels)["participant"]) if a.labels and os.path.exists(a.labels) else set()
    dev_pids=set(df["participant"].unique())
    unl=sorted(dev_pids-lab_pids, key=lambda x:int(re.sub(r'\D','',x) or 0))
    for p in unl:
        g=df[df["participant"]==p].sort_values("start_s")
        if g.empty: continue
        # evenly spaced windows across the recording
        idx=np.linspace(0,len(g)-1,min(a.per_participant,len(g))).round().astype(int)
        for _,r in g.iloc[idx].iterrows():
            cstart=round(max(0.0,r["start_s"]+1.5-CLIP_S/2),1)
            if spaced(starts.get(p,[]),cstart): continue
            if len([x for x in picks if x["reason"]=="unlabeled"])>=a.n_unlabeled: break
            picks.append(mk(r,"unlabeled")); starts.setdefault(p,[]).append(cstart)

    # stream 3: vertical re-review clips (from the audit) - render the exact labelled window
    if a.rereview and os.path.exists(a.rereview):
        rr=pd.read_csv(a.rereview)
        for _,r in rr.iterrows():
            p=pid_from(r["participant"]); src=str(r["source"])
            picks.append({"clip":f"{p}__{int(r['start_s']):04d}_RE.mp4","participant":p,"source":src,
                          "raw_stem":src.replace("_features.csv",""),"start_s":float(r["start_s"]),
                          "end_s":float(r["end_s"]),"seg_id":int(r.get("seg_id",int(r["start_s"]))),
                          "cluster":-1,"reason":"rereview"})

    man=pd.DataFrame(picks).drop_duplicates(subset=["participant","seg_id","reason"])
    man.to_csv(a.out,index=False)
    print("stream counts:",man["reason"].value_counts().to_dict())
    print(f"selected {len(man)} clips across {man['participant'].nunique()} participants -> {a.out}")
    print(f"next (on the server with raw landmark CSVs):")
    print(f"  python render_skeleton.py --manifest {a.out} --src-dir data/raw/mediapipe_full --out-dir labeling_kit_round2")

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("features",nargs="+")
    ap.add_argument("--splits",default=None); ap.add_argument("--labels",default=None)
    ap.add_argument("--rereview",default=None)
    ap.add_argument("--n-minority",type=int,default=180); ap.add_argument("--n-unlabeled",type=int,default=60)
    ap.add_argument("--per-participant",type=int,default=3); ap.add_argument("--min-valid",type=float,default=60.0)
    ap.add_argument("--out",default="label_manifest_round2.csv")
    a=ap.parse_args(); main(a)

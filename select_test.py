"""
select_test.py  -  build the labelling manifest for the 37 SEALED TEST participants ONLY.

This is run once, at the very end, to label the held-out test set for the final evaluation.
It selects a diverse spread of clips per test participant (spanning the behaviour range via a
light minority-aware score), so the test labels cover floating AND non-floating cases.

Usage (on the server with raw landmark CSVs):
  python select_test.py "features_v2_side_front/"*.csv --splits splits_v2.csv \
      --per-participant 4 --out label_manifest_test.csv
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

def main(a):
    files=[]; [files.extend(glob.glob(p)) for p in a.features]
    sp=pd.read_csv(a.splits); test=set(pid_from(x) for x in sp[sp["split"]=="test"]["participant"])
    rows=[]
    for f in files:
        pid=pid_from(f)
        if pid not in test: continue                      # TEST participants only
        d=pd.read_csv(f); d["source"]=os.path.basename(f); d["participant"]=pid
        d["raw_stem"]=os.path.basename(f).replace("_features.csv","")
        vc="torso_valid_pct" if "torso_valid_pct" in d else ("valid_pct" if "valid_pct" in d else None)
        if vc: d=d[d[vc]>=a.min_valid]
        rows.append(d)
    if not rows: raise SystemExit("no sealed-test feature windows found")
    df=pd.concat(rows,ignore_index=True)
    # light score so each participant's picks span calm-float -> effort/sink, not all one behaviour
    df=df.assign(mscore=( zscore(df.get("trunk_angle_med",0))+zscore(df.get("hip_y_slope",0))
                         +0.5*zscore(df.get("arm_speed",0))+0.5*zscore(df.get("leg_speed",0)) ))
    picks=[]
    for p,g in df.groupby("participant"):
        g=g.sort_values("mscore")
        # take a spread across the score range: low, mid, high (+extra evenly spaced)
        idx=np.linspace(0,len(g)-1,min(a.per_participant,len(g))).round().astype(int)
        starts=[]
        for _,r in g.iloc[idx].iterrows():
            cstart=round(max(0.0,r["start_s"]+1.5-CLIP_S/2),1)
            if any(abs(st-cstart)<CLIP_S*0.75 for st in starts): continue
            starts.append(cstart)
            picks.append({"clip":f"{p}__{int(cstart):04d}.mp4","participant":p,"source":r["source"],
                          "raw_stem":r["raw_stem"],"start_s":cstart,"end_s":round(cstart+CLIP_S,1),
                          "seg_id":int(cstart),"cluster":-1,"reason":"test"})
    man=pd.DataFrame(picks).drop_duplicates(subset=["participant","seg_id"])
    man.to_csv(a.out,index=False)
    print(f"selected {len(man)} clips across {man['participant'].nunique()} test participants "
          f"(target {a.per_participant}/participant) -> {a.out}")
    print(f"next (server with raw landmark CSVs):")
    print(f"  python render_skeleton.py --manifest {a.out} --src-dir data/raw/mediapipe_full --out-dir labeling_kit_test")

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("features",nargs="+")
    ap.add_argument("--splits",required=True); ap.add_argument("--per-participant",type=int,default=4)
    ap.add_argument("--min-valid",type=float,default=60.0); ap.add_argument("--out",default="label_manifest_test.csv")
    a=ap.parse_args(); main(a)

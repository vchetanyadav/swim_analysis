"""
select_expert_window.py  -  build a labelling manifest for the SAME time window the domain expert
scored, so the expert and your labeller rate identical footage. This is what makes an inter-rater
agreement number scientifically valid, instead of comparing clips taken from different moments.

Two modes:
  --participants P02,P81,P68,P118     explicit list (e.g. the expert's four samples)
  --expert expert_scores.csv          take the participant + window columns straight from the expert file
Default window is 30 to 60 seconds. --thirds cuts it into 30-40, 40-50, 50-60 so the labeller can
rate each and you aggregate, matching how the expert watched the whole 30 seconds.

Usage (on the server with raw landmark CSVs):
  python select_expert_window.py "features_v2_side_front/"*.csv --expert expert_scores.csv --thirds \
      --out label_manifest_expert.csv
  python render_skeleton.py --manifest label_manifest_expert.csv --src-dir data/raw/mediapipe_full --out-dir labeling_kit_expert
"""
import argparse, glob, os, re, pandas as pd

def pid_from(name):
    stem=os.path.splitext(os.path.basename(str(name)))[0]
    m=re.search(r'[Pp](\d+)',stem)
    return f"P{int(m.group(1)):02d}" if m else stem[:8]

def norm(p):
    m=re.search(r'[Pp](\d+)',str(p)); return f"P{int(m.group(1)):02d}" if m else str(p)

def main(a):
    want={}  # pid -> (start,end)
    if a.expert and os.path.exists(a.expert):
        e=pd.read_csv(a.expert)
        for _,r in e.iterrows():
            want[norm(r["participant"])]=(float(r.get("window_start",a.start)),float(r.get("window_end",a.end)))
    if a.participants:
        for p in a.participants.split(","):
            want.setdefault(norm(p),(a.start,a.end))
    if not want: raise SystemExit("give --participants or --expert")

    files={pid_from(f):f for f in sum([glob.glob(p) for p in a.features],[])}
    rows=[]
    for pid,(ws,we) in want.items():
        f=files.get(pid)
        if not f: print(f"  no feature file for {pid}"); continue
        stem=os.path.basename(f).replace("_features.csv","")
        segs=[(ws,we)]
        if a.thirds:
            segs=[(ws,ws+10),(ws+10,ws+20),(ws+20,we)]
        for s,ee in segs:
            rows.append({"clip":f"{pid}__{int(s):04d}_EX.mp4","participant":pid,
                         "source":os.path.basename(f),"raw_stem":stem,
                         "start_s":float(s),"end_s":float(ee),"seg_id":int(s),
                         "cluster":-1,"reason":"expert_window"})
    man=pd.DataFrame(rows); man.to_csv(a.out,index=False)
    print(f"selected {len(man)} clips across {man['participant'].nunique()} participants "
          f"(window {a.start:.0f}-{a.end:.0f}s{', in thirds' if a.thirds else ''}) -> {a.out}")
    print(f"next: python render_skeleton.py --manifest {a.out} --src-dir data/raw/mediapipe_full --out-dir labeling_kit_expert")

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("features",nargs="+")
    ap.add_argument("--participants",default=None); ap.add_argument("--expert",default=None)
    ap.add_argument("--start",type=float,default=30.0); ap.add_argument("--end",type=float,default=60.0)
    ap.add_argument("--thirds",action="store_true"); ap.add_argument("--out",default="label_manifest_expert.csv")
    a=ap.parse_args(); main(a)

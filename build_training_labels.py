"""
build_training_labels.py  -  fuse your own labels with the expert labels into ONE training file,
on the expert's 0-3 floating-outcome scale. The expert overrides your label where they scored the
same participant, and expert rows carry a higher sample weight. Airway is never a training target.

Output columns: source, participant, seg_id, start_s, end_s, floating_outcome, label_source,
sample_weight, exclude. Feed it to swim_train.py with --target floating_outcome.

Your labels are mapped to the expert outcome scale with:
  not floating (0) -> 0 cannot float
  almost (1)       -> 1 lots of effort
  floating (2), active   -> 2 floats with effort
  floating (2), passive  -> 3 floats with no effort
Expert rows use the expert's own floating_outcome directly.

Weights: ordinary high-confidence window = 1.0, medium = 0.6, low = 0.25; each expert window =
--expert-weight (default 5.0), so an expert observation counts as much as five high-confidence
ordinary ones. Participant-grouped CV already stops four people from dominating, so the weight is a
flat per-window multiple rather than a per-participant total.

Usage:
  python build_training_labels.py "features_v2_side_front/"*.csv --mine labels_merged.csv \
      --expert expert_scores.csv --out labels_combined.csv --expert-weight 5
"""
import argparse, glob, os, re, numpy as np, pandas as pd

def pid_from(name):
    stem=os.path.splitext(os.path.basename(str(name)))[0]
    m=re.search(r'[Pp](\d+)',stem); return f"P{int(m.group(1)):02d}" if m else stem[:8]

def mine_outcome(fl,pa):
    if pd.isna(fl): return np.nan
    fl=int(fl)
    if fl in (0,1): return fl
    return 3 if (not pd.isna(pa) and int(pa)==0) else 2

def main(a):
    files={pid_from(f):f for f in sum([glob.glob(p) for p in a.features],[])}
    # feature windows per participant (for expanding expert scores and counting conf weights)
    feat={}
    for pid,f in files.items():
        d=pd.read_csv(f)
        if "valid_pct" in d: d=d[d["valid_pct"]>=a.min_valid]
        feat[pid]=d.assign(source=os.path.basename(f))

    exp=pd.read_csv(a.expert); exp["pid"]=exp["participant"].map(pid_from)
    expert_pids=set(exp["pid"])

    rows=[]
    # 1) YOUR labels (recoded), skipping participants the expert scored (expert overrides them)
    mine=pd.read_csv(a.mine)
    mine=mine[mine["exclude"].fillna(0).astype(int)==0]
    for _,s in mine.iterrows():
        pid=pid_from(s["participant"])
        if pid in expert_pids: continue                     # expert overrides this participant
        out=mine_outcome(s.get("float_label"),s.get("passive_active"))
        if pd.isna(out): continue
        conf=pd.to_numeric(s.get("label_confidence"),errors="coerce")
        w=1.0 if conf==2 else (0.6 if conf==1 else (0.25 if conf==0 else 0.8))
        rows.append({"source":s["source"],"participant":pid,"seg_id":int(s.get("seg_id",int(s["start_s"]))),
                     "start_s":float(s["start_s"]),"end_s":float(s["end_s"]),
                     "floating_outcome":int(out),"label_source":"mine","sample_weight":round(w,3),"exclude":0})
    # 2) EXPERT labels, expanded to that participant's windows in the scored interval, weight ~total/N
    for _,e in exp.iterrows():
        pid=e["pid"]; d=feat.get(pid)
        if d is None or len(d)==0: print(f"  no features for expert {pid}"); continue
        ws,we=float(e.get("window_start",30)),float(e.get("window_end",60))
        c=(d["start_s"]+d["end_s"])/2; sel=d[(c>=ws)&(c<we)]
        if len(sel)==0: sel=d
        wper=float(a.expert_weight)   # flat per-window weight: each expert window counts this many
        for _,r in sel.iterrows():     # ordinary high-confidence windows (so expert = N x authoritative)
            rows.append({"source":r["source"],"participant":pid,"seg_id":int(r["start_s"]),
                         "start_s":float(r["start_s"]),"end_s":float(r["end_s"]),
                         "floating_outcome":int(e["floating_outcome"]),"label_source":"expert",
                         "sample_weight":wper,"exclude":0})
    out=pd.DataFrame(rows); out.to_csv(a.out,index=False)
    nm=out[out.label_source=="mine"]; ne=out[out.label_source=="expert"]
    print(f"wrote {a.out}: {len(out)} rows  (mine {len(nm)} from {nm['participant'].nunique()} participants, "
          f"expert {len(ne)} from {ne['participant'].nunique()} participants)")
    print("outcome counts:",out["floating_outcome"].value_counts().sort_index().to_dict())
    print(f"expert weight/participant ~= {a.expert_weight}; train with: "
          f"swim_train.py ... --labels {a.out} --target floating_outcome")

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("features",nargs="+")
    ap.add_argument("--mine",required=True); ap.add_argument("--expert",required=True)
    ap.add_argument("--expert-weight",type=float,default=5.0); ap.add_argument("--min-valid",type=float,default=60.0)
    ap.add_argument("--out",default="labels_combined.csv")
    a=ap.parse_args(); main(a)

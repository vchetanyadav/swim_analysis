"""
make_splits.py  -  lock participant-level train / validation / test splits.

Splitting is by PARTICIPANT, never by frame or by video: every view (top, side-front, ...)
of one person stays in the same partition, so no information leaks between train and test.
The test participants are sealed: do not inspect them during development.

Usage:
    python make_splits.py features_side_front/*.csv features_top/*.csv --out splits.csv
    python make_splits.py features_side_front/*.csv --labels labels.csv --frac 0.6 0.2 0.2 --seed 1 --out splits.csv
"""
import argparse, glob, os, re, numpy as np, pandas as pd

def pid_from(name):
    stem=os.path.splitext(os.path.basename(str(name)))[0]
    m=re.search(r'[Pp](\d+)',stem)
    if m: return f"P{int(m.group(1)):02d}"
    m=re.match(r'\s*(\d+)',stem)
    return f"V{int(m.group(1)):02d}" if m else stem[:8]

def main(patterns,labels_csv,frac,seed,out):
    files=[]; [files.extend(glob.glob(p)) for p in patterns]
    parts=sorted(set(pid_from(f) for f in files))
    if not parts: raise SystemExit("no feature files matched")
    rng=np.random.default_rng(seed)
    # dominant label per participant (for stratification) if labels available
    dom={}
    if labels_csv and os.path.exists(labels_csv):
        lab=pd.read_csv(labels_csv)
        if "exclude" in lab: lab=lab[lab["exclude"].fillna(0).astype(int)==0]
        key="source" if "source" in lab else "video_id"
        lab["participant"]=lab[key].map(pid_from)
        tgt="float_label" if "float_label" in lab else lab.columns[-1]
        for p,g in lab.groupby("participant"):
            dom[p]=int(g[tgt].mode().iloc[0]) if len(g) else 0
    # stratified round-robin assignment by dominant class so each split sees each class
    ftr,fva,fte=frac
    order=sorted(parts,key=lambda p:(dom.get(p,0),rng.random()))
    splits={}
    # proportional allocation within each stratum
    from collections import defaultdict
    strata=defaultdict(list)
    for p in order: strata[dom.get(p,0)].append(p)
    for cls,plist in strata.items():
        rng.shuffle(plist); n=len(plist)
        ntr=max(1,round(n*ftr)); nva=max(0,round(n*fva))
        for i,p in enumerate(plist):
            splits[p]= "train" if i<ntr else ("val" if i<ntr+nva else "test")
    # guarantee non-empty train/test when possible
    vals=list(splits.values())
    if "test" not in vals and len(parts)>=3:
        splits[order[-1]]="test"
    df=pd.DataFrame({"participant":list(splits.keys()),"split":list(splits.values())})
    df=df.sort_values("participant")
    df.to_csv(out,index=False)
    print(df.to_string(index=False))
    print("\ncounts:",df["split"].value_counts().to_dict())
    print(f"wrote {out}.  TEST participants are sealed: do not look at them until the model is frozen.")

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("features",nargs="+")
    ap.add_argument("--labels",default=None)
    ap.add_argument("--frac",nargs=3,type=float,default=[0.6,0.2,0.2])
    ap.add_argument("--seed",type=int,default=1); ap.add_argument("--out",default="splits.csv")
    a=ap.parse_args(); main(a.features,a.labels,tuple(a.frac),a.seed,a.out)

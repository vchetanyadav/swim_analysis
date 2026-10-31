"""
irr_kappa.py  -  inter-rater reliability between two instructors' labels.

Measures whether experts agree on the float label before expecting a model to. For the ordered
three-class label, reports quadratic-weighted Cohen's kappa on the clips both instructors labelled.

Usage:
    python irr_kappa.py labels_instructorA.csv labels_instructorB.csv --target float_label
"""
import argparse, pandas as pd, numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix

def load(p,target):
    d=pd.read_csv(p)
    if "exclude" in d: d=d[d["exclude"].fillna(0).astype(int)==0]
    key="source" if "source" in d else "video_id"
    d=d.dropna(subset=[target]).copy(); d[target]=d[target].astype(int)
    d["clip"]=d[key].astype(str)+"__"+d["seg_id"].astype(str)
    return d[["clip",target]].rename(columns={target:"lab"})

def interp(k):
    return ("poor" if k<0.2 else "fair" if k<0.4 else "moderate" if k<0.6
            else "substantial" if k<0.8 else "almost perfect")

def main(a_csv,b_csv,target):
    A=load(a_csv,target); B=load(b_csv,target)
    m=A.merge(B,on="clip",suffixes=("_A","_B"))
    if len(m)==0: raise SystemExit("no overlapping clips (match on source + seg_id)")
    kq=cohen_kappa_score(m["lab_A"],m["lab_B"],weights="quadratic")
    kl=cohen_kappa_score(m["lab_A"],m["lab_B"],weights="linear")
    agree=(m["lab_A"]==m["lab_B"]).mean()
    print(f"overlapping clips double-labelled: {len(m)}")
    print(f"exact agreement:          {agree:.1%}")
    print(f"linear-weighted kappa:     {kl:.3f}")
    print(f"quadratic-weighted kappa:  {kq:.3f}  ({interp(kq)} agreement)")
    print("confusion (rows=A, cols=B):")
    print(confusion_matrix(m["lab_A"],m["lab_B"]))
    if kq<0.4:
        print("\nNOTE: experts disagree substantially. Clarify the float-state definition before training;")
        print("a model cannot be expected to beat the humans' own consistency.")

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--target",default="float_label")
    x=ap.parse_args(); main(x.a,x.b,x.target)

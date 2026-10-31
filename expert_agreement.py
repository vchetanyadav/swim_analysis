"""
expert_agreement.py  -  inter-rater agreement between the domain expert and your labeller, on the
SAME participants and the SAME window. Reports quadratic-weighted Cohen's kappa, exact agreement,
and within-one agreement per criterion. This is the number that belongs in the report.

Both files use the expert schema columns: floating_outcome, activity, trunk_position, head_position,
airway, overall_score (overall is compared as a correlation, not kappa). The labeller file may have
several clips per participant (e.g. the thirds); they are aggregated to one score per participant by
the rounded mean before comparison, matching the expert's single whole-window judgement.

Usage:
  python expert_agreement.py --expert expert_scores.csv --labeller labels_expert.csv
"""
import argparse, re, numpy as np, pandas as pd
from sklearn.metrics import cohen_kappa_score

CRIT=["floating_outcome","activity","trunk_position","head_position","airway"]
MAXV={"floating_outcome":3,"activity":2,"trunk_position":2,"head_position":2,"airway":4}

def norm(p):
    m=re.search(r'[Pp](\d+)',str(p)); return f"P{int(m.group(1)):02d}" if m else str(p)

def agg_labeller(df):
    df=df.copy(); df["participant"]=df["participant"].map(norm)
    if "exclude" in df: df=df[df["exclude"].fillna(0).astype(int)==0]
    rows=[]
    for pid,g in df.groupby("participant"):
        row={"participant":pid}
        for c in CRIT:
            if c in g: row[c]=int(round(pd.to_numeric(g[c],errors="coerce").mean()))
        rows.append(row)
    return pd.DataFrame(rows)

def main(a):
    e=pd.read_csv(a.expert); e["participant"]=e["participant"].map(norm)
    l=agg_labeller(pd.read_csv(a.labeller))
    m=e.merge(l,on="participant",suffixes=("_exp","_lab"))
    if len(m)==0: raise SystemExit("no shared participants between the two files")
    print(f"shared participants: {len(m)}  ({', '.join(sorted(m['participant']))})\n")
    print(f"{'criterion':18s} {'kappa(qw)':>10s} {'exact':>7s} {'within1':>8s}")
    for c in CRIT:
        ce,cl=c+"_exp",c+"_lab"
        if ce not in m or cl not in m: continue
        ye,yl=m[ce].astype(int).to_numpy(),m[cl].astype(int).to_numpy()
        try: k=cohen_kappa_score(ye,yl,weights="quadratic",labels=list(range(MAXV[c]+1)))
        except Exception: k=float("nan")
        exact=(ye==yl).mean(); within1=(np.abs(ye-yl)<=1).mean()
        print(f"{c:18s} {k:>10.3f} {exact:>7.0%} {within1:>8.0%}")
    if "overall_score" in e and ("overall_score_lab" in m or "overall_score" in l):
        oe=m["overall_score_exp"] if "overall_score_exp" in m else m["overall_score"]
        if "overall_score_lab" in m:
            r=np.corrcoef(oe.astype(float),m["overall_score_lab"].astype(float))[0,1]
            print(f"\noverall score (0-13): Pearson r = {r:.3f}")
    print("\nper participant (expert vs labeller):")
    for _,r in m.iterrows():
        exp=[f"{r[c+'_exp']}" for c in CRIT if c+'_exp' in r]
        lab=[f"{r[c+'_lab']}" for c in CRIT if c+'_lab' in r]
        print(f"  {r['participant']}: expert [{','.join(exp)}]  labeller [{','.join(lab)}]  (outcome/activity/trunk/head/airway)")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--expert",required=True); ap.add_argument("--labeller",required=True)
    a=ap.parse_args(); main(a)

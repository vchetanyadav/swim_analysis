"""
swim_cluster4.py - clustering on WATER-LINE-INDEPENDENT features only.

The side-front camera is underwater, so depth-based features (hip_depth,
ankle_depth, head_sub) rely on a water surface that can't be detected reliably.
This script clusters on features that are measured directly from the skeleton and
need no water line: trunk angle (posture) + limb speed (effort). Treading /
non-horizontal-float videos are excluded via --exclude.

Usage:
    python swim_cluster4.py features_side_front/*.csv --exclude "2 Side front" --out cluster_clean
"""
import argparse, glob, os, json, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# all water-line-INDEPENDENT (measured from the skeleton, no surface needed)
POSTURE = ["trunk_angle_med","trunk_angle_std"]
EFFORT  = ["arm_speed","leg_speed"]
FEATURES = POSTURE + EFFORT

def main(patterns, out_dir, exclude, min_valid=60.0, kmax=6):
    os.makedirs(out_dir, exist_ok=True)
    files=[]
    for p in patterns: files+=glob.glob(p)
    parts=[]
    for f in files:
        base=os.path.basename(f)
        if any(x.lower() in base.lower() for x in exclude):
            print(f"  excluded (treading/non-float): {base}"); continue
        d=pd.read_csv(f); d["source"]=base; parts.append(d)
    df=pd.concat(parts,ignore_index=True)
    print(f"loaded {len(df)} windows from {len(parts)} files (after exclusions)")
    if "valid_pct" in df:
        before=len(df); df=df[df["valid_pct"]>=min_valid].reset_index(drop=True)
        print(f"kept {len(df)}/{before} windows with valid_pct>={min_valid}")
    feats=[c for c in FEATURES if c in df.columns]
    print("clustering features:", feats)
    X=df[feats].copy()
    for c in feats:
        lo,hi=X[c].quantile(0.02),X[c].quantile(0.98); X[c]=X[c].clip(lo,hi)
    X=X.fillna(X.median())
    Xs=RobustScaler().fit_transform(X)

    print("\n k  silhouette  calinski  davies(lower=better)")
    best=None
    for k in range(2,kmax+1):
        km=KMeans(n_clusters=k,n_init=10,random_state=0).fit(Xs)
        sil=silhouette_score(Xs,km.labels_); ch=calinski_harabasz_score(Xs,km.labels_); db=davies_bouldin_score(Xs,km.labels_)
        print(f" {k}    {sil:.3f}     {ch:7.1f}    {db:.3f}")
        if best is None or sil>best[0]: best=(sil,k,km)
    sil,k,km=best; df["cluster"]=km.labels_
    print(f"\nchosen k={k} (silhouette {sil:.3f})")

    cen=df.groupby("cluster")[feats].median().round(2)
    tm=df["trunk_angle_med"].median(); sm=(df["arm_speed"]+df["leg_speed"]).median()
    print(f"\nview medians used for labels: trunk={tm:.1f}deg  (arm+leg speed)={sm:.2f}")
    interp={}
    for c in cen.index:
        pg = cen.loc[c,"trunk_angle_med"] < tm            # more horizontal than typical -> better posture
        eh = (cen.loc[c,"arm_speed"]+cen.loc[c,"leg_speed"]) > sm
        if pg and not eh: interp[c]="efficient float (flatter body, low effort)"
        elif pg and eh:   interp[c]="movement-assisted float (flatter body, high effort)"
        elif not pg and eh: interp[c]="struggling / recovery (steeper body, high effort)"
        else:             interp[c]="sinking / vertical / inactive (steeper body, low effort)"
    cen["interpretation"]=[interp[c] for c in cen.index]
    cen.to_csv(os.path.join(out_dir,"cluster_centroids.csv"))
    print("\ncluster medians:")
    print(cen.to_string())
    print("\ncluster sizes:\n", df["cluster"].value_counts().sort_index().to_string())

    comp=pd.crosstab(df["cluster"],df["source"])
    dom=(comp.max(axis=1)/comp.sum(axis=1))
    print("\nsingle-video dominance per cluster (want <0.6):")
    for c in dom.index:
        top=comp.loc[c].idxmax()
        print(f"  cluster {c}: {dom[c]:.2f}  (top: {top})")

    P=PCA(2).fit_transform(Xs)
    plt.figure(figsize=(8,6))
    for c in sorted(df["cluster"].unique()):
        m=df["cluster"]==c
        plt.scatter(P[m,0],P[m,1],s=12,label=f"c{c}: {interp[c].split('(')[0].strip()}")
    plt.legend(fontsize=8); plt.title(f"Float clusters (water-line-free features, k={k})"); plt.tight_layout()
    plt.savefig(os.path.join(out_dir,"pca_clusters.png"),dpi=110)
    df.to_csv(os.path.join(out_dir,"windows_clustered.csv"),index=False)
    json.dump({int(c):interp[c] for c in interp},open(os.path.join(out_dir,"cluster_interpretation.json"),"w"),indent=2)
    print(f"\nwrote centroids, windows_clustered.csv, pca_clusters.png, cluster_interpretation.json")
    print("NOTE: names are provisional behavioural descriptions; instructor labels are required")
    print("before assigning any 'floating well/not' performance meaning.")

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("features",nargs="+")
    ap.add_argument("--exclude",nargs="*",default=["2 Side front"])
    ap.add_argument("--out",default="cluster_clean"); ap.add_argument("--min-valid",type=float,default=60.0)
    ap.add_argument("--kmax",type=int,default=6)
    a=ap.parse_args(); main(a.features,a.out,a.exclude,a.min_valid,a.kmax)

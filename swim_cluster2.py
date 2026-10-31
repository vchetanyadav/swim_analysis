"""
swim_cluster2.py - posture/effort clustering (side-view windows)
Reads *_features.csv, filters unreliable windows, RobustScaler, K-Means with
3 validity metrics, then interprets each cluster on a posture x effort grid.
"""
import argparse, glob, os, json, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

POSTURE = ["trunk_angle_med","trunk_angle_std","hip_depth_med","hip_depth_std","ankle_depth","head_sub_frac"]
EFFORT  = ["arm_speed","leg_speed","arm_asym","leg_asym"]
FEATURES = POSTURE + EFFORT

def main(patterns, out_dir, min_valid=60.0, kmax=6):
    os.makedirs(out_dir, exist_ok=True)
    files=[]
    for p in patterns: files+=glob.glob(p)
    parts=[]
    for f in files:
        d=pd.read_csv(f)
        d["source"]=os.path.basename(f)
        parts.append(d)
    df=pd.concat(parts,ignore_index=True)
    print(f"loaded {len(df)} windows from {len(files)} files")
    if "valid_pct" in df:
        before=len(df); df=df[df["valid_pct"]>=min_valid].reset_index(drop=True)
        print(f"kept {len(df)}/{before} windows with valid_pct>={min_valid}")
    feats=[c for c in FEATURES if c in df.columns]
    X=df[feats].copy()
    for c in feats:                       # winsorize to tame jitter outliers
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
    # posture & effort scores (for INTERPRETATION only, not clustering)
    p_good = (cen["trunk_angle_med"]<20) & (cen["hip_depth_med"]<cen["hip_depth_med"].median())
    eff = (cen["arm_speed"]+cen["leg_speed"])
    e_high = eff > eff.median()
    interp={}
    for c in cen.index:
        pg=bool(p_good[c]); eh=bool(e_high[c])
        if pg and not eh: interp[c]="efficient float (good posture, low effort)"
        elif pg and eh:   interp[c]="movement-assisted float (good posture, high effort)"
        elif not pg and eh: interp[c]="struggling / recovery (poor posture, high effort)"
        else:             interp[c]="sinking / vertical / inactive (poor posture, low effort)"
    cen["interpretation"]=[interp[c] for c in cen.index]
    cen.to_csv(os.path.join(out_dir,"cluster_centroids.csv"))
    print("\ncluster medians:")
    print(cen[["trunk_angle_med","hip_depth_med","arm_speed","leg_speed","interpretation"]].to_string())
    print("\ncluster sizes:\n", df["cluster"].value_counts().sort_index().to_string())

    # composition check: is any cluster dominated by one video?
    comp=pd.crosstab(df["cluster"],df["source"])
    dom=(comp.max(axis=1)/comp.sum(axis=1))
    print("\nsingle-video dominance per cluster (want <0.6):")
    for c in dom.index: print(f"  cluster {c}: {dom[c]:.2f}")

    P=PCA(2).fit_transform(Xs)
    plt.figure(figsize=(8,6))
    for c in sorted(df["cluster"].unique()):
        m=df["cluster"]==c
        plt.scatter(P[m,0],P[m,1],s=12,label=f"c{c}: {interp[c].split('(')[0].strip()}")
    plt.legend(fontsize=8); plt.title(f"Float behaviour clusters (k={k})"); plt.tight_layout()
    plt.savefig(os.path.join(out_dir,"pca_clusters.png"),dpi=110)
    df.to_csv(os.path.join(out_dir,"windows_clustered.csv"),index=False)
    json.dump({int(c):interp[c] for c in interp},open(os.path.join(out_dir,"cluster_interpretation.json"),"w"),indent=2)
    print(f"\nwrote centroids, windows_clustered.csv, pca_clusters.png, cluster_interpretation.json")
    print("NOTE: cluster names are provisional behavioural descriptions; instructor labels are required")
    print("before assigning any 'floating well/not' performance meaning.")

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("features",nargs="+"); ap.add_argument("--out",default="cluster_out")
    ap.add_argument("--min-valid",type=float,default=60.0); ap.add_argument("--kmax",type=int,default=6)
    a=ap.parse_args(); main(a.features,a.out,a.min_valid,a.kmax)

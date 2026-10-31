"""
swim_cluster.py  -  Stage 2: feature table -> clusters + interpretation + model
Usage: python swim_cluster.py features/*_features.csv --out cluster_out
"""
import argparse, glob, os, json, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
FEATURES = ["trunk_angle_med","trunk_angle_std","hip_depth_med","hip_depth_std",
            "ankle_depth","knee_med","head_sub_frac","leg_freq_l","leg_freq_r",
            "arm_freq_l","arm_freq_r","leg_speed","arm_speed","leg_asym","arm_asym"]
def flotation_index(df):
    ang = np.cos(np.radians(df["trunk_angle_med"].clip(0,90)))
    hip = 1 - df["hip_depth_med"].clip(0,1.5)/1.5
    head = 1 - df["head_sub_frac"].clip(0,1)
    return (0.45*ang + 0.35*hip + 0.20*head).clip(0,1)
def fi_label(fi):
    return np.where(fi>=0.72,"Well",np.where(fi>=0.45,"Marginal","NotFloating"))
def main(patterns, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    files=[]
    for p in patterns: files+=glob.glob(p)
    df=pd.concat([pd.read_csv(f) for f in files],ignore_index=True)
    print(f"loaded {len(df)} windows from {len(files)} file(s)")
    feats=[c for c in FEATURES if c in df.columns]
    X=df[feats].copy(); X=X.fillna(X.median())
    keep=df[feats].isna().mean(axis=1)<0.5
    df,X=df[keep].reset_index(drop=True),X[keep].reset_index(drop=True)
    Xs=StandardScaler().fit_transform(X)
    best=None
    for k in (2,3,4,5):
        km=KMeans(n_clusters=k,n_init=10,random_state=0).fit(Xs)
        s=silhouette_score(Xs,km.labels_); print(f"  k={k} silhouette={s:.3f}")
        if best is None or s>best[0]: best=(s,k,km)
    sil,k,km=best; df["cluster"]=km.labels_; print(f"chosen k={k} (silhouette {sil:.3f})")
    if "view" in df.columns and df["view"].nunique()>1:
        ct=pd.crosstab(df["cluster"],df["view"]); pur=ct.max(axis=1)/ct.sum(axis=1)
        if (pur>0.9).all(): print("WARNING: clusters align with camera view, not behaviour. Cluster per view.")
    df["FI"]=flotation_index(df); df["FI_label"]=fi_label(df["FI"])
    centroids=df.groupby("cluster")[feats+["FI"]].median().round(2)
    centroids.to_csv(os.path.join(out_dir,"cluster_centroids.csv"))
    print("\ncluster centroids (median):\n", centroids[["trunk_angle_med","hip_depth_med","arm_speed","leg_freq_l","FI"]])
    P=PCA(2).fit_transform(Xs)
    plt.figure(figsize=(7,6))
    for c in sorted(df["cluster"].unique()):
        m=df["cluster"]==c; plt.scatter(P[m,0],P[m,1],s=14,label=f"cluster {c} (FI~{df[m]['FI'].median():.2f})")
    plt.legend(); plt.title(f"Float windows, K-Means k={k}"); plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir,"pca_clusters.png"),dpi=110)
    df.to_csv(os.path.join(out_dir,"windows_clustered.csv"),index=False)
    if "float_label" in df.columns and df["float_label"].notna().sum()>=30:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import GroupKFold, cross_val_score
        lab=df.dropna(subset=["float_label"])
        rf=RandomForestClassifier(n_estimators=300,random_state=0)
        gkf=GroupKFold(n_splits=min(5,lab["pid"].nunique()))
        sc=cross_val_score(rf,StandardScaler().fit_transform(lab[feats].fillna(lab[feats].median())),
                           lab["float_label"],cv=gkf,groups=lab["pid"])
        print(f"\nRandomForest (participant-separated CV) accuracy: {sc.mean():.2f} +/- {sc.std():.2f}")
        rf.fit(StandardScaler().fit_transform(lab[feats].fillna(lab[feats].median())),lab["float_label"])
        imp=pd.Series(rf.feature_importances_,index=feats).sort_values(ascending=False)
        imp.to_csv(os.path.join(out_dir,"feature_importance.csv")); print("top features:\n",imp.head(6))
    else:
        print("\n(no instructor labels yet -> unsupervised clustering + FI seed only)")
    well_c=int(centroids["FI"].idxmax())
    prof={"well_cluster":well_c,"well_profile":centroids.loc[well_c,feats].to_dict(),
          "std":{c:float(X[c].std()) for c in feats}}
    json.dump(prof,open(os.path.join(out_dir,"well_profile.json"),"w"),indent=2)
    print(f"\nWell reference = cluster {well_c}. wrote well_profile.json, windows_clustered.csv, pca_clusters.png")
if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("features",nargs="+"); ap.add_argument("--out",default="cluster_out")
    a=ap.parse_args(); main(a.features,a.out)

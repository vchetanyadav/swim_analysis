"""swim_cluster3.py - diagnostic: median-relative posture/effort, k=2 & k=3, video breakdown."""
import glob, os, numpy as np, pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans

POSTURE=["trunk_angle_med","trunk_angle_std","hip_depth_med","hip_depth_std","ankle_depth","head_sub_frac"]
EFFORT =["arm_speed","leg_speed","arm_asym","leg_asym"]
FEATURES=POSTURE+EFFORT

import sys
patterns=sys.argv[1:] or ["features_side_front/*.csv"]
files=[]; [files.extend(glob.glob(p)) for p in patterns]
parts=[]
for f in files:
    d=pd.read_csv(f); d["source"]=os.path.basename(f); parts.append(d)
df=pd.concat(parts,ignore_index=True)
df=df[df["valid_pct"]>=60].reset_index(drop=True)
feats=[c for c in FEATURES if c in df.columns]
X=df[feats].copy()
for c in feats:
    lo,hi=X[c].quantile(0.02),X[c].quantile(0.98); X[c]=X[c].clip(lo,hi)
X=X.fillna(X.median()); Xs=RobustScaler().fit_transform(X)

# global medians define "good/poor" and "high/low" RELATIVE to this view
tm=df["trunk_angle_med"].median(); hm=df["hip_depth_med"].median()
sm=(df["arm_speed"]+df["leg_speed"]).median()
print(f"view medians: trunk={tm:.1f} hip={hm:.2f} (arm+leg speed)={sm:.2f}\n")

for k in (2,3):
    km=KMeans(n_clusters=k,n_init=10,random_state=0).fit(Xs)
    df["cl"]=km.labels_
    cen=df.groupby("cl")[feats].median().round(2)
    print(f"===== k={k} =====")
    for c in cen.index:
        pg = (cen.loc[c,"trunk_angle_med"]<tm) and (cen.loc[c,"hip_depth_med"]<hm)
        eh = (cen.loc[c,"arm_speed"]+cen.loc[c,"leg_speed"])>sm
        lab=("good posture" if pg else "poor posture")+" / "+("high effort" if eh else "low effort")
        n=(df["cl"]==c).sum()
        top=df[df["cl"]==c]["source"].value_counts(normalize=True).head(3)
        topstr=", ".join(f"{v}:{p:.0%}" for v,p in top.items())
        print(f" c{c} n={n:4d}  trunk={cen.loc[c,'trunk_angle_med']:5.1f} hip={cen.loc[c,'hip_depth_med']:5.2f} "
              f"armS={cen.loc[c,'arm_speed']:.2f} legS={cen.loc[c,'leg_speed']:.2f}  -> {lab}")
        print(f"      top videos: {topstr}")
    print()

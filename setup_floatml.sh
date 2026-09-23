#!/usr/bin/env bash
set -e
mkdir -p ~/swim_analysis/floatml && cd ~/swim_analysis/floatml
echo "writing pipeline files..."
cat > swim_features.py << 'FLOATML_EOF_FEATURES'
"""
swim_features.py  -  Stage 1: video -> windowed feature table (the ML dataset)
Stock COCO-17 pose (no annotation). One row = one 3.0s window (50% overlap).
Usage: python swim_features.py videos/sideFront.mp4 --view side_front --pid P38 --model yolo11x-pose.pt --out features
"""
import argparse, os, json, cv2, numpy as np, pandas as pd
from scipy.signal import detrend, find_peaks
from ultralytics import YOLO
L_SH,R_SH,L_EL,R_EL,L_WR,R_WR = 5,6,7,8,9,10
L_HIP,R_HIP,L_KN,R_KN,L_AN,R_AN = 11,12,13,14,15,16
NOSE,L_EAR,R_EAR = 0,3,4
CONF = 0.35
WIN_S, OVERLAP = 3.0, 0.5
def v(kp,i): return kp[i,2] > CONF
def mid(kp,a,b):
    if v(kp,a) and v(kp,b): return (kp[a,:2]+kp[b,:2])/2
    return None
def ang_horiz(vec): return float(np.degrees(np.arctan2(abs(vec[1]),abs(vec[0]))))
def joint_angle(a,b,c):
    ba,bc=a-b,c-b
    return float(np.degrees(np.arccos(np.clip(np.dot(ba,bc)/(np.linalg.norm(ba)*np.linalg.norm(bc)+1e-9),-1,1))))
def head_pt(kp):
    if v(kp,NOSE): return kp[NOSE,:2]
    e=[kp[i,:2] for i in (L_EAR,R_EAR) if v(kp,i)]
    return np.mean(e,axis=0) if e else None
def detect_waterline(cap,n=25):
    tot=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); idxs=np.linspace(tot*0.1,tot*0.9,n).astype(int); pr=[]
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES,int(i)); ok,fr=cap.read()
        if ok: pr.append(cv2.cvtColor(fr,cv2.COLOR_BGR2GRAY).astype(float).mean(axis=1))
    cap.set(cv2.CAP_PROP_POS_FRAMES,0)
    if not pr: return None
    p=np.mean(pr,axis=0); k=max(5,len(p)//60); p=np.convolve(p,np.ones(k)/k,mode="same")
    return int(np.argmin(np.gradient(p)[:int(len(p)*0.6)]))
def extract(video, out_dir, view, pid, model_name, imgsz=1280):
    os.makedirs(out_dir, exist_ok=True)
    stem=os.path.splitext(os.path.basename(video))[0]
    net=YOLO(model_name); cap=cv2.VideoCapture(video)
    fps=cap.get(cv2.CAP_PROP_FPS) or 25; W,H=int(cap.get(3)),int(cap.get(4))
    wl=detect_waterline(cap); floor=int(H*0.98)
    per=[]; idx=0
    while True:
        ok,frame=cap.read()
        if not ok: break
        r=net.predict(frame,imgsz=imgsz,conf=0.25,max_det=1,verbose=False)[0]
        row={"t":idx/fps}
        if r.boxes is not None and len(r.boxes)>0:
            kp=r.keypoints.data[0].cpu().numpy()
            sh,hp=mid(kp,L_SH,R_SH),mid(kp,L_HIP,R_HIP)
            torso=float(np.linalg.norm(sh-hp)) if (sh is not None and hp is not None) else np.nan
            row["torso"]=torso
            row["trunk_angle"]=ang_horiz(hp-sh) if (sh is not None and hp is not None) else np.nan
            row["hip_depth"]=((hp[1]-wl)/torso) if (hp is not None and torso==torso and torso>1) else np.nan
            for side,an in [("l",L_AN),("r",R_AN)]:
                row[f"ankle_depth_{side}"]=((kp[an,1]-wl)/torso) if (v(kp,an) and torso>1) else np.nan
            for side,(h,k,a) in [("l",(L_HIP,L_KN,L_AN)),("r",(R_HIP,R_KN,R_AN))]:
                row[f"knee_{side}"]=joint_angle(kp[h,:2],kp[k,:2],kp[a,:2]) if all(v(kp,j) for j in (h,k,a)) else np.nan
            hd=head_pt(kp)
            row["head_sub"]=float(hd[1]>wl) if hd is not None else np.nan
            for name,(distal,prox) in {"arm_l":(L_WR,L_SH),"arm_r":(R_WR,R_SH),"leg_l":(L_AN,L_HIP),"leg_r":(R_AN,R_HIP)}.items():
                if v(kp,distal) and v(kp,prox):
                    d=kp[distal,:2]-kp[prox,:2]; row[f"{name}_x"]=d[0]; row[f"{name}_y"]=d[1]
                else: row[f"{name}_x"]=np.nan; row[f"{name}_y"]=np.nan
            row["valid"]=float(np.mean([v(kp,i) for i in (L_SH,R_SH,L_HIP,R_HIP,L_KN,R_KN,L_AN,R_AN)]))
        else:
            row["valid"]=0.0
        per.append(row); idx+=1
    cap.release()
    dfp=pd.DataFrame(per)
    torso_med=float(np.nanmedian(dfp["torso"])) if "torso" in dfp else np.nan
    meta={"video_id":f"{pid}_{view}","subject_id":pid,"camera_view":view,"fps":fps,
          "frame_count":idx,"duration_s":round(idx/fps,1),"water_line_y":wl,"pool_floor_y":floor,
          "water_column_px":floor-wl,"torso_px":round(torso_med,1),"model":model_name}
    json.dump(meta,open(os.path.join(out_dir,f"{stem}_meta.json"),"w"),indent=2)
    step=WIN_S*(1-OVERLAP); wins=[]; t_end=dfp["t"].iloc[-1] if len(dfp) else 0
    def limb_freq(seg,name):
        y=seg[f"{name}_y"].interpolate(limit=6).to_numpy(); m=~np.isnan(y)
        if m.sum()<int(0.6*WIN_S*fps): return np.nan
        s=detrend(y[m]); pk,_=find_peaks(s,prominence=np.nanstd(s)*0.5)
        return len(pk)/(WIN_S)*60
    def limb_speed(seg,name):
        xy=seg[[f"{name}_x",f"{name}_y"]].interpolate(limit=6).to_numpy()
        sp=np.linalg.norm(np.diff(xy,axis=0),axis=1)*fps
        sp=sp[~np.isnan(sp)]
        return float(np.mean(sp)/torso_med) if (sp.size and torso_med>1) else np.nan
    ts=0.0
    while ts < max(t_end-WIN_S,0)+1e-6:
        seg=dfp[(dfp["t"]>=ts)&(dfp["t"]<ts+WIN_S)]
        if len(seg)>=int(0.5*WIN_S*fps) and seg["valid"].mean()>0.4:
            fL,fR=limb_freq(seg,"leg_l"),limb_freq(seg,"leg_r")
            aL,aR=limb_freq(seg,"arm_l"),limb_freq(seg,"arm_r")
            def asym(a,b): return abs(a-b)/(a+b) if (a==a and b==b and a+b>0) else np.nan
            wins.append({
                "video_id":meta["video_id"],"pid":pid,"view":view,"start_s":round(ts,1),"end_s":round(ts+WIN_S,1),
                "trunk_angle_med":float(np.nanmedian(seg["trunk_angle"])),
                "trunk_angle_std":float(np.nanstd(seg["trunk_angle"])),
                "hip_depth_med":float(np.nanmedian(seg["hip_depth"])),
                "hip_depth_std":float(np.nanstd(seg["hip_depth"])),
                "ankle_depth":float(np.nanmedian(seg[["ankle_depth_l","ankle_depth_r"]].values)),
                "knee_med":float(np.nanmedian(seg[["knee_l","knee_r"]].values)),
                "head_sub_frac":float(np.nanmean(seg["head_sub"])),
                "leg_freq_l":fL,"leg_freq_r":fR,"arm_freq_l":aL,"arm_freq_r":aR,
                "leg_speed":np.nanmean([limb_speed(seg,"leg_l"),limb_speed(seg,"leg_r")]),
                "arm_speed":np.nanmean([limb_speed(seg,"arm_l"),limb_speed(seg,"arm_r")]),
                "leg_asym":asym(fL,fR),"arm_asym":asym(aL,aR),
                "valid_pct":round(float(seg["valid"].mean())*100,1),
            })
        ts+=step
    dfw=pd.DataFrame(wins)
    out_csv=os.path.join(out_dir,f"{stem}_features.csv"); dfw.to_csv(out_csv,index=False)
    print(f"{meta['video_id']}: {len(dfw)} windows -> {out_csv}  (torso {torso_med:.0f}px, water y={wl})")
    return out_csv
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("video"); ap.add_argument("--view",default="side_front"); ap.add_argument("--pid",default="P00")
    ap.add_argument("--model",default="yolo11x-pose.pt"); ap.add_argument("--out",default="features")
    ap.add_argument("--imgsz",type=int,default=1280)
    a=ap.parse_args(); extract(a.video,a.out,a.view,a.pid,a.model,a.imgsz)
FLOATML_EOF_FEATURES
cat > swim_cluster.py << 'FLOATML_EOF_CLUSTER'
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
FLOATML_EOF_CLUSTER
cat > swim_feedback.py << 'FLOATML_EOF_FEEDBACK'
"""
swim_feedback.py  -  Stage 3: new video features -> ranked timestamped feedback
Usage: python swim_feedback.py features/new_features.csv --profile cluster_out/well_profile.json
"""
import argparse, json, numpy as np, pandas as pd
TEXT={
 "trunk_angle_med":"Body angle {v:.0f} deg from flat (ideal ~{r:.0f}). Press chest down to raise hips.",
 "hip_depth_med":"Hips sitting low in the water. Lift the hips toward the surface.",
 "head_sub_frac":"Head below the surface {v:.0%} of the time. Keep the head back, airway clear.",
 "arm_speed":"Arm/hand speed differs from a steady float (yours {v:.2f} body-len/s).",
 "leg_speed":"Leg speed differs from a steady float (yours {v:.2f} body-len/s).",
 "leg_asym":"Left/right leg cadence uneven (asymmetry {v:.2f}). Balance the kick.",
 "arm_asym":"Left/right arm cadence uneven (asymmetry {v:.2f}). Balance the arms.",
 "trunk_angle_std":"Body angle unsteady (varies {v:.0f} deg). Hold a steadier position.",
}
def main(feat_csv, profile, out, z_thresh=1.5):
    prof=json.load(open(profile)); well=prof["well_profile"]; sd=prof["std"]
    df=pd.read_csv(feat_csv); rows=[]
    for _,w in df.iterrows():
        devs=[]
        for f in well:
            if f in w and not pd.isna(w[f]) and sd.get(f,0)>1e-6:
                z=(w[f]-well[f])/sd[f]
                if abs(z)>=z_thresh and f in TEXT: devs.append((abs(z),f,w[f],well[f]))
        devs.sort(reverse=True)
        for z,f,val,ref in devs[:3]:
            rows.append({"start_s":w.get("start_s"),"end_s":w.get("end_s"),"feature":f,
                         "z":round(z,1),"message":TEXT[f].format(v=val,r=ref)})
    fb=pd.DataFrame(rows); fb.to_csv(out,index=False)
    if len(fb):
        print("Most frequent issues across the video:")
        for f,n in fb["feature"].value_counts().head(5).items(): print(f"  {f}: flagged in {n} windows")
        print(f"\nfull timestamped feedback -> {out}\nexample windows:")
        for _,r in fb.head(6).iterrows(): print(f"  {r['start_s']:.0f}-{r['end_s']:.0f}s: {r['message']}")
    else: print("No windows deviated strongly from the 'well' profile.")
if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("features"); ap.add_argument("--profile",required=True)
    ap.add_argument("--out",default="feedback.csv"); ap.add_argument("--z",type=float,default=1.5)
    a=ap.parse_args(); main(a.features,a.profile,a.out,a.z)
FLOATML_EOF_FEEDBACK
cat > swim_label.py << 'FLOATML_EOF_LABEL'
"""
swim_label.py  -  instructor labelling tool (2=Well 1=Marginal 0=NotFloating a=Assisted u=Unknown s=skip q=quit)
Usage: python swim_label.py features/sideFront_features.csv videos/sideFront.mp4 --out labelled.csv
"""
import argparse, cv2, pandas as pd, numpy as np
def main(feat_csv, video, out):
    df=pd.read_csv(feat_csv); cap=cv2.VideoCapture(video); fps=cap.get(cv2.CAP_PROP_FPS) or 25
    if "float_label" not in df: df["float_label"]=np.nan
    m={"2":"Well","1":"Marginal","0":"NotFloating","a":"Assisted","u":"Unknown"}
    for i,w in df.iterrows():
        if not pd.isna(df.at[i,"float_label"]): continue
        mid_t=(w["start_s"]+w["end_s"])/2; cap.set(cv2.CAP_PROP_POS_MSEC,mid_t*1000); ok,fr=cap.read()
        if ok:
            cv2.putText(fr,f"{w['start_s']:.0f}-{w['end_s']:.0f}s 2=Well 1=Marg 0=No a=Assist u=Unk s=skip q=quit",
                        (20,40),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)
            cv2.imshow("label",fr)
        key=chr(cv2.waitKey(0)&0xFF)
        if key=="q": break
        if key=="s": continue
        if key in m: df.at[i,"float_label"]=m[key]
    cap.release(); cv2.destroyAllWindows(); df.to_csv(out,index=False)
    print(f"saved {out}; labelled {df['float_label'].notna().sum()} windows")
if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("features"); ap.add_argument("video"); ap.add_argument("--out",default="labelled.csv")
    a=ap.parse_args(); main(a.features,a.video,a.out)
FLOATML_EOF_LABEL
echo "done. files in ~/swim_analysis/floatml:"; ls -1 *.py

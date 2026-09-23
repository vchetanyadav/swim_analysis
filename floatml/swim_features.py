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

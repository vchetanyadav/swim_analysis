"""
swim_features.py  -  Stage 1: video -> windowed feature table (the ML dataset)

Runs the STOCK COCO-17 pose model (no annotation) on a side-view video, computes
per-frame flotation measurements against the auto-detected water line and pool
floor, then aggregates them into overlapping time windows. Each window is one row
of the training table. Also writes a metadata JSON per video.

One row = one 3.0s window (50% overlap)  -> the unit of analysis.

v2 changes:
  * robust water-line detection (median of per-frame edge estimates in a plausible
    band) + confidence flag + manual override (--waterline-y / --waterline-frac)
  * FFT-based, periodicity-gated limb frequency (kills jitter over-counting;
    pure noise -> 0/min, real 0.5-3 Hz kicks recovered accurately)
  * mean_pose_conf column per window

Usage:
    python swim_features.py videos/sideFront.mp4 --view side_front --pid P38 \
        --model yolo11x-pose.pt --out features
    # if auto water line is wrong for one video, override it:
    python swim_features.py videos/2.mp4 --view side_front --pid P02 --waterline-frac 0.30
"""
import argparse, os, json, re, cv2, numpy as np, pandas as pd
from scipy.signal import detrend
from scipy.stats import theilslopes
from ultralytics import YOLO

def pid_from(name):
    """Participant/video id from filename. Strips extension first so ".mp4"
    can't match the P-number regex (that bug tagged every video 'p4')."""
    stem=os.path.splitext(os.path.basename(name))[0]
    m=re.search(r'[Pp](\d+)', stem)
    if m: return f"P{int(m.group(1)):02d}"
    m=re.match(r'\s*(\d+)', stem)
    return f"V{int(m.group(1)):02d}" if m else stem[:8]

L_SH,R_SH,L_EL,R_EL,L_WR,R_WR = 5,6,7,8,9,10
L_HIP,R_HIP,L_KN,R_KN,L_AN,R_AN = 11,12,13,14,15,16
NOSE,L_EAR,R_EAR = 0,3,4
CONF = 0.35
WIN_S, OVERLAP = 3.0, 0.5
MAX_HZ = 3.0          # physiological ceiling for a kick/stroke rate
MIN_HZ = 0.3
FREQ_GATE = 0.35     # min dominant-band power fraction to accept a rhythm

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

def detect_waterline(cap, n=31, band=(0.12,0.78)):
    """Median of per-frame strongest-horizontal-edge estimates within a plausible
    vertical band. Returns (y, confident). confident=False when frame estimates
    disagree (spread > 6% of height) -> caller should inspect / override."""
    tot=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if tot<=0: return None, False
    idxs=np.linspace(tot*0.1, tot*0.9, n).astype(int)
    ests=[]; H=None
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES,int(i)); ok,fr=cap.read()
        if not ok: continue
        g=cv2.cvtColor(fr,cv2.COLOR_BGR2GRAY).astype(float); H=g.shape[0]
        prof=g.mean(axis=1)
        k=max(5,len(prof)//60); prof=np.convolve(prof,np.ones(k)/k,mode="same")
        grad=np.abs(np.gradient(prof))
        lo,hi=int(H*band[0]),int(H*band[1])
        seg=grad[lo:hi]
        if seg.size: ests.append(lo+int(np.argmax(seg)))
    cap.set(cv2.CAP_PROP_POS_FRAMES,0)
    if not ests or H is None: return None, False
    ests=np.array(ests); y=int(np.median(ests))
    confident = float(np.std(ests)) < 0.06*H
    return y, confident

def limb_freq(seg,name,fps):
    """Dominant rhythmic frequency (cycles/min) of a limb's vertical motion.
    Smooths jitter, then takes the FFT dominant frequency in [MIN_HZ,MAX_HZ]
    only if that band carries a clear share of the power; else 0 (no rhythm)."""
    y=seg[f"{name}_y"].interpolate(limit=6).to_numpy(); m=~np.isnan(y)
    if m.sum()<int(0.6*WIN_S*fps): return np.nan
    y=y[m]
    k=max(3,int(fps*0.12)|1); ys=np.convolve(y,np.ones(k)/k,mode="same")
    s=detrend(ys); n=len(s)
    if (np.percentile(s,90)-np.percentile(s,10))<1e-6: return 0.0
    w=np.hanning(n)
    S0=np.abs(np.fft.rfft(s*w))**2; f0=np.fft.rfftfreq(n,1/fps)
    b0=(f0>=MIN_HZ)&(f0<=MAX_HZ)
    if not b0.any() or S0[b0].max()/(S0[1:].sum()+1e-9) < FREQ_GATE: return 0.0
    nfft=int(2**np.ceil(np.log2(n*8)))
    S=np.abs(np.fft.rfft(s*w,n=nfft))**2; f=np.fft.rfftfreq(nfft,1/fps)
    b=(f>=MIN_HZ)&(f<=MAX_HZ)
    return float(f[b][np.argmax(S[b])]*60)

def extract(video, out_dir, view, pid, model_name, imgsz=1280, wl_override=None):
    os.makedirs(out_dir, exist_ok=True)
    stem=os.path.splitext(os.path.basename(video))[0]
    net=YOLO(model_name); cap=cv2.VideoCapture(video)
    fps=cap.get(cv2.CAP_PROP_FPS) or 25; W,H=int(cap.get(3)),int(cap.get(4))
    if wl_override is not None:
        wl=int(wl_override); wl_conf=True; wl_src="override"
    else:
        wl,wl_conf=detect_waterline(cap); wl_src="auto"
        if wl is None: wl=int(H*0.35); wl_conf=False; wl_src="fallback"
    floor=int(H*0.98)
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
            body=[L_SH,R_SH,L_HIP,R_HIP,L_KN,R_KN,L_AN,R_AN]
            row["valid"]=float(np.mean([v(kp,i) for i in body]))
            row["pose_conf"]=float(np.mean([kp[i,2] for i in body]))
        else:
            row["valid"]=0.0; row["pose_conf"]=0.0
        per.append(row); idx+=1
    cap.release()
    dfp=pd.DataFrame(per)
    torso_med=float(np.nanmedian(dfp["torso"])) if "torso" in dfp else np.nan

    meta={"video_id":f"{pid}_{view}","subject_id":pid,"camera_view":view,"fps":fps,
          "frame_count":idx,"duration_s":round(idx/fps,1),"water_line_y":int(wl),
          "water_line_source":wl_src,"water_line_confident":bool(wl_conf),
          "pool_floor_y":floor,"water_column_px":floor-int(wl),
          "torso_px":round(torso_med,1),"model":model_name}
    json.dump(meta,open(os.path.join(out_dir,f"{stem}_meta.json"),"w"),indent=2)
    if not wl_conf:
        print(f"  ** WARNING: water line for {stem} is not confident (source={wl_src}, y={wl}). "
              f"Inspect with inspect_video.py and re-run with --waterline-y / --waterline-frac if wrong.")

    # ---- windowing ----
    step=WIN_S*(1-OVERLAP); wins=[]; t_end=dfp["t"].iloc[-1] if len(dfp) else 0
    def limb_speed(seg,name):
        xy=seg[[f"{name}_x",f"{name}_y"]].interpolate(limit=6).to_numpy()
        sp=np.linalg.norm(np.diff(xy,axis=0),axis=1)*fps
        sp=sp[~np.isnan(sp)]
        return float(np.mean(sp)/torso_med) if (sp.size and torso_med>1) else np.nan
    def trend(seg,col):
        """Robust (Theil-Sen) slope of a per-frame column vs time, units = col-units/sec.
        For hip_depth this cancels the constant water-line offset, so it stays valid
        even when the absolute water line is wrong (underwater camera)."""
        y=seg[col].to_numpy(); t=seg["t"].to_numpy(); m=~np.isnan(y)
        if m.sum()<max(4,int(0.5*WIN_S*fps)): return np.nan
        try: return float(theilslopes(y[m],t[m])[0])
        except Exception: return np.nan
    ts=0.0
    while ts < max(t_end-WIN_S,0)+1e-6:
        seg=dfp[(dfp["t"]>=ts)&(dfp["t"]<ts+WIN_S)]
        if len(seg)>=int(0.5*WIN_S*fps) and seg["valid"].mean()>0.4:
            fL,fR=limb_freq(seg,"leg_l",fps),limb_freq(seg,"leg_r",fps)
            aL,aR=limb_freq(seg,"arm_l",fps),limb_freq(seg,"arm_r",fps)
            def asym(a,b): return abs(a-b)/(a+b) if (a==a and b==b and a+b>0) else np.nan
            asl,asr=limb_speed(seg,"arm_l"),limb_speed(seg,"arm_r")
            lsl,lsr=limb_speed(seg,"leg_l"),limb_speed(seg,"leg_r")
            wins.append({
                # --- identity / metadata ---
                "video_id":meta["video_id"],"participant_id":pid,"pid":pid,"view":view,
                "start_s":round(ts,1),"end_s":round(ts+WIN_S,1),
                # --- postural (trunk_angle water-line-INDEPENDENT; hip/ankle depth depend on water line) ---
                "trunk_angle_med":float(np.nanmedian(seg["trunk_angle"])),
                "trunk_angle_std":float(np.nanstd(seg["trunk_angle"])),
                "hip_depth_med":float(np.nanmedian(seg["hip_depth"])),
                "hip_depth_std":float(np.nanstd(seg["hip_depth"])),
                "ankle_depth":float(np.nanmedian(seg[["ankle_depth_l","ankle_depth_r"]].values)),
                "knee_angle_med_deg":float(np.nanmedian(seg[["knee_l","knee_r"]].values)),
                # --- temporal trends (slopes cancel the water-line offset -> robust) ---
                "hip_depth_slope":trend(seg,"hip_depth"),
                "trunk_angle_slope":trend(seg,"trunk_angle"),
                "head_sub_frac":float(np.nanmean(seg["head_sub"])),
                # --- limb activity ---
                "leg_freq_l":fL,"leg_freq_r":fR,"arm_freq_l":aL,"arm_freq_r":aR,
                "arm_speed_l":asl,"arm_speed_r":asr,"leg_speed_l":lsl,"leg_speed_r":lsr,
                "arm_speed":np.nanmean([asl,asr]),"leg_speed":np.nanmean([lsl,lsr]),
                # --- symmetry ---
                "leg_asym":asym(fL,fR),"arm_asym":asym(aL,aR),
                # --- reliability ---
                "valid_pct":round(float(seg["valid"].mean())*100,1),
                "mean_pose_conf":round(float(seg["pose_conf"].mean()),3),
            })
        ts+=step
    dfw=pd.DataFrame(wins)
    out_csv=os.path.join(out_dir,f"{stem}_features.csv"); dfw.to_csv(out_csv,index=False)
    flag="" if wl_conf else "  [WL NOT CONFIDENT]"
    print(f"{meta['video_id']}: {len(dfw)} windows -> {out_csv}  (torso {torso_med:.0f}px, water y={wl}, src={wl_src}){flag}")
    return out_csv

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("video"); ap.add_argument("--view",default="side_front"); ap.add_argument("--pid",default="auto")
    ap.add_argument("--model",default="yolo11x-pose.pt"); ap.add_argument("--out",default="features")
    ap.add_argument("--imgsz",type=int,default=1280)
    ap.add_argument("--waterline-y",type=int,default=None,help="force water-line pixel row")
    ap.add_argument("--waterline-frac",type=float,default=None,help="force water line as fraction of height (0-1)")
    a=ap.parse_args()
    pid = pid_from(a.video) if a.pid=="auto" else a.pid
    wl_over=a.waterline_y
    if wl_over is None and a.waterline_frac is not None:
        cap=cv2.VideoCapture(a.video); H=int(cap.get(4)); cap.release(); wl_over=int(H*a.waterline_frac)
    extract(a.video,a.out,a.view,pid,a.model,a.imgsz,wl_over)

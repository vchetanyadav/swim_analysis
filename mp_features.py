"""
mp_features.py (v2)  -  MediaPipe landmark CSV -> windowed feature table.

v2 engineering pass (per external review):
  * PER-LIMB visibility masking: arm/leg speed & frequency use ONLY frames where that limb's
    own distal+proximal joints are visible (a hallucinated ankle can't fake movement).
  * Honest names: hip_y_* (not hip_depth), ankle_y_med (not ankle_depth),
    nose_low_vis_frac (a quality/occlusion feature, NOT head submersion).
  * Cadence over a longer window (8 s) while posture/drift stay on 3 s windows.
  * Per-limb reliability: torso/arm/leg valid-% and mean confidence, for honest abstain.
  * Kept: Theil-Sen drift slopes, per-side limb measures, torso-normalized speeds.

No water line exists in landmark data, so no absolute depth is computed; vertical drift
(hip_y_slope) carries the sinking signal and is robust with a fixed camera.

Usage:
    python mp_features.py "data/raw/mediapipe_full/"*.csv --out-side features_mp_side_front --out-top features_mp_top
"""
import argparse, glob, os, numpy as np, pandas as pd
from scipy.signal import detrend
from scipy.stats import theilslopes

CONF=0.5
WIN_S, OVERLAP = 3.0, 0.5
FREQ_WIN_S = 8.0                       # longer window just for cadence
MAX_HZ, MIN_HZ, FREQ_GATE = 3.0, 0.3, 0.35
CORE=["left_shoulder","right_shoulder","left_hip","right_hip"]
ARM_L=["left_shoulder","left_wrist"]; ARM_R=["right_shoulder","right_wrist"]
LEG_L=["left_hip","left_ankle"];      LEG_R=["right_hip","right_ankle"]
TORSO_PTS=CORE; ARM_PTS=["left_shoulder","right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist"]
LEG_PTS=["left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle"]

def load(path): return pd.read_excel(path) if path.lower().endswith(("xlsx","xls")) else pd.read_csv(path)
def ang_horiz(dx,dy): return np.degrees(np.arctan2(np.abs(dy),np.abs(dx)))
def joint_angle(ax,ay,bx,by,cx,cy):
    v1x,v1y=ax-bx,ay-by; v2x,v2y=cx-bx,cy-by
    den=np.hypot(v1x,v1y)*np.hypot(v2x,v2y)+1e-9
    return np.degrees(np.arccos(np.clip((v1x*v2x+v1y*v2y)/den,-1,1)))

def limb_freq(y,t,ok,center,fps):
    lo,hi=center-FREQ_WIN_S/2,center+FREQ_WIN_S/2
    w=(t>=lo)&(t<hi)
    if w.sum()<int(0.6*FREQ_WIN_S*fps): return np.nan
    yy=y[w].copy(); okk=ok[w]
    yy[~okk]=np.nan
    s=pd.Series(yy).interpolate(limit=int(fps)).to_numpy()
    m=~np.isnan(s)
    if m.mean()<0.6: return np.nan
    s=s[m]
    if len(s)<8: return np.nan
    k=max(3,int(fps*0.12)|1); s=np.convolve(s,np.ones(k)/k,mode="same"); s=detrend(s); n=len(s)
    if (np.percentile(s,90)-np.percentile(s,10))<1e-9: return 0.0
    win=np.hanning(n); S0=np.abs(np.fft.rfft(s*win))**2; f0=np.fft.rfftfreq(n,1/fps)
    b0=(f0>=MIN_HZ)&(f0<=MAX_HZ)
    if not b0.any() or S0[b0].max()/(S0[1:].sum()+1e-9)<FREQ_GATE: return 0.0
    nfft=int(2**np.ceil(np.log2(n*8))); S=np.abs(np.fft.rfft(s*win,n=nfft))**2; f=np.fft.rfftfreq(nfft,1/fps)
    b=(f>=MIN_HZ)&(f<=MAX_HZ); return float(f[b][np.argmax(S[b])]*60)

def trend(t,y,m,fps):
    if m.sum()<max(4,int(0.5*WIN_S*fps)): return np.nan
    try: return float(theilslopes(y[m],t[m])[0])
    except Exception: return np.nan

def extract(path,out_side,out_top,aspect):
    d=load(path); pid=str(d["participant_id"].iloc[0]); view=str(d["camera_view"].iloc[0]); vid=str(d["video_id"].iloc[0])
    t=d["time_sec"].to_numpy(); fps=1.0/np.median(np.diff(t)) if len(t)>1 else 25.0
    out_dir=out_top if view=="top" else out_side; os.makedirs(out_dir,exist_ok=True)
    X=lambda lm: d[f"{lm}_x"].to_numpy()*aspect
    Y=lambda lm: d[f"{lm}_y"].to_numpy()
    V=lambda lm: d[f"{lm}_visibility"].to_numpy()
    okset=lambda pts: np.all([V(p)>CONF for p in pts],axis=0)
    okcore=okset(CORE)
    shx=(X("left_shoulder")+X("right_shoulder"))/2; shy=(Y("left_shoulder")+Y("right_shoulder"))/2
    hpx=(X("left_hip")+X("right_hip"))/2;           hpy=(Y("left_hip")+Y("right_hip"))/2
    torso=np.hypot(shx-hpx,shy-hpy); torso_med=np.nanmedian(np.where(okcore,torso,np.nan))
    trunk=ang_horiz(hpx-shx,hpy-shy); trunk[~okcore]=np.nan
    hip_y=np.where(okcore,hpy,np.nan)
    ank_ok=(V("left_ankle")>CONF)&(V("right_ankle")>CONF)
    ankle_y=np.where(ank_ok,(Y("left_ankle")+Y("right_ankle"))/2,np.nan)
    knee_ok=okset(("left_hip","left_knee","left_ankle","right_hip","right_knee","right_ankle"))
    knee=(joint_angle(X("left_hip"),Y("left_hip"),X("left_knee"),Y("left_knee"),X("left_ankle"),Y("left_ankle"))
         +joint_angle(X("right_hip"),Y("right_hip"),X("right_knee"),Y("right_knee"),X("right_ankle"),Y("right_ankle")))/2
    knee[~knee_ok]=np.nan
    # limb vectors + per-limb validity
    limb={"arm_l":(X("left_wrist")-X("left_shoulder"),Y("left_wrist")-Y("left_shoulder"),okset(ARM_L)),
          "arm_r":(X("right_wrist")-X("right_shoulder"),Y("right_wrist")-Y("right_shoulder"),okset(ARM_R)),
          "leg_l":(X("left_ankle")-X("left_hip"),Y("left_ankle")-Y("left_hip"),okset(LEG_L)),
          "leg_r":(X("right_ankle")-X("right_hip"),Y("right_ankle")-Y("right_hip"),okset(LEG_R))}
    conf_torso=np.nanmean(np.vstack([V(p) for p in TORSO_PTS]),axis=0)
    conf_arm=np.nanmean(np.vstack([V(p) for p in ARM_PTS]),axis=0)
    conf_leg=np.nanmean(np.vstack([V(p) for p in LEG_PTS]),axis=0)
    nose_vis=V("nose")

    def speed(name,w):
        vx,vy,ok=limb[name]; both=w&ok
        idx=np.where(both)[0]
        if len(idx)<2 or not torso_med>0: return np.nan
        dx=np.diff(vx[idx]); dy=np.diff(vy[idx]); dtt=np.diff(t[idx])
        good=dtt<(2.0/fps)                      # only consecutive frames
        if good.sum()<1: return np.nan
        sp=np.hypot(dx[good],dy[good])/dtt[good]
        return float(np.nanmean(sp)/torso_med)
    def asym(a,b): return abs(a-b)/(a+b) if (a==a and b==b and a+b>0) else np.nan

    step=WIN_S*(1-OVERLAP); wins=[]; t_end=t[-1] if len(t) else 0; ts=0.0
    while ts<max(t_end-WIN_S,0)+1e-6:
        w=(t>=ts)&(t<ts+WIN_S); m=w&okcore; c=ts+WIN_S/2
        if m.sum()>=int(0.5*WIN_S*fps) and okcore[w].mean()>0.4:
            fL=limb_freq(limb["leg_l"][1],t,limb["leg_l"][2],c,fps)
            fR=limb_freq(limb["leg_r"][1],t,limb["leg_r"][2],c,fps)
            aL=limb_freq(limb["arm_l"][1],t,limb["arm_l"][2],c,fps)
            aR=limb_freq(limb["arm_r"][1],t,limb["arm_r"][2],c,fps)
            asl,asr=speed("arm_l",w),speed("arm_r",w); lsl,lsr=speed("leg_l",w),speed("leg_r",w)
            wins.append({
              "video_id":vid,"participant_id":pid,"pid":pid,"view":view,
              "start_s":round(ts,1),"end_s":round(ts+WIN_S,1),
              "trunk_angle_med":float(np.nanmedian(trunk[w])),
              "trunk_angle_std":float(np.nanstd(trunk[w])),
              "trunk_angle_slope":trend(t,trunk,m,fps),
              "hip_y_med":float(np.nanmedian(hip_y[w])),
              "hip_y_std":float(np.nanstd(hip_y[w])),
              "hip_y_slope":trend(t,hip_y,m,fps),
              "ankle_y_med":float(np.nanmedian(ankle_y[w])) if np.isfinite(ankle_y[w]).any() else np.nan,
              "knee_angle_med_deg":float(np.nanmedian(knee[w])) if np.isfinite(knee[w]).any() else np.nan,
              "arm_freq_l":aL,"arm_freq_r":aR,"leg_freq_l":fL,"leg_freq_r":fR,
              "arm_speed_l":asl,"arm_speed_r":asr,"leg_speed_l":lsl,"leg_speed_r":lsr,
              "arm_speed":np.nanmean([asl,asr]),"leg_speed":np.nanmean([lsl,lsr]),
              "arm_asym":asym(asl,asr),"leg_asym":asym(lsl,lsr),
              # --- reliability ---
              "torso_valid_pct":round(float(okcore[w].mean())*100,1),
              "valid_pct":round(float(okcore[w].mean())*100,1),   # alias for existing >=60 filter
              "left_arm_valid_pct":round(float(limb["arm_l"][2][w].mean())*100,1),
              "right_arm_valid_pct":round(float(limb["arm_r"][2][w].mean())*100,1),
              "left_leg_valid_pct":round(float(limb["leg_l"][2][w].mean())*100,1),
              "right_leg_valid_pct":round(float(limb["leg_r"][2][w].mean())*100,1),
              "mean_torso_conf":round(float(np.nanmean(conf_torso[w])),3),
              "mean_arm_conf":round(float(np.nanmean(conf_arm[w])),3),
              "mean_leg_conf":round(float(np.nanmean(conf_leg[w])),3),
              "nose_low_vis_frac":round(float(np.mean(nose_vis[w]<CONF)),3),
            })
        ts+=step
    stem=os.path.splitext(os.path.basename(path))[0]
    out_csv=os.path.join(out_dir,f"{stem}_features.csv"); pd.DataFrame(wins).to_csv(out_csv,index=False)
    print(f"{vid:40s} {view:10s} {len(wins):4d} windows -> {out_csv}  (fps {fps:.0f})")
    return out_csv

def main(patterns,out_side,out_top,aspect):
    files=[]; [files.extend(glob.glob(p)) for p in patterns]; ok=0
    for f in sorted(files):
        try: extract(f,out_side,out_top,aspect); ok+=1
        except Exception as e: print(f"  SKIPPED {os.path.basename(f)}: {e}")
    print(f"\ndone: {ok}/{len(files)} files -> {out_side}/ and {out_top}/")

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("inputs",nargs="+")
    ap.add_argument("--out-side",default="features_mp_side_front"); ap.add_argument("--out-top",default="features_mp_top")
    ap.add_argument("--aspect",type=float,default=16/9)
    a=ap.parse_args(); main(a.inputs,a.out_side,a.out_top,a.aspect)

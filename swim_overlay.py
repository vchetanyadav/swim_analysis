"""
swim_overlay.py - draw the TRUNK LINE + angle on frames (presentation figures).

For each sampled frame (and optionally a short clip), draws:
  * the two shoulder points and two hip points,
  * the trunk line = shoulder-midpoint -> hip-midpoint,
  * the measured trunk angle in degrees,
  * a colour tag: FLAT (green, <20 deg) / PARTIAL (amber, 20-45) / UPRIGHT (red, >45),
  * a faint full skeleton for context.

Usage:
    python swim_overlay.py "/opt/home/s4137946/swim_analysis/videos/1 Side front.mp4" \
        --out trunk_overlay --frames 12 --clip-start 20 --clip-len 15
    # frames only, no clip:
    python swim_overlay.py "videos/1 Side front.mp4" --out trunk_overlay --no-clip
"""
import argparse, os, cv2, numpy as np
from ultralytics import YOLO
from swim_features import mid, ang_horiz, v, \
    L_SH,R_SH,L_HIP,R_HIP,L_KN,R_KN,L_AN,R_AN,L_EL,R_EL,L_WR,R_WR,CONF

SKEL=[(L_SH,L_EL),(L_EL,L_WR),(R_SH,R_EL),(R_EL,R_WR),(L_SH,R_SH),
      (L_SH,L_HIP),(R_SH,R_HIP),(L_HIP,R_HIP),(L_HIP,L_KN),(L_KN,L_AN),(R_HIP,R_KN),(R_KN,R_AN)]

def tag(angle):
    if angle<20:   return "FLAT",    (0,220,0)      # green
    if angle<45:   return "PARTIAL", (0,200,255)    # amber
    return "UPRIGHT",(0,0,235)                       # red

def annotate(frame, kp):
    if kp is None: return frame
    for a,b in SKEL:                                  # faint skeleton
        if kp[a,2]>CONF and kp[b,2]>CONF:
            cv2.line(frame,tuple(kp[a,:2].astype(int)),tuple(kp[b,:2].astype(int)),(120,120,120),1)
    sh,hp=mid(kp,L_SH,R_SH),mid(kp,L_HIP,R_HIP)
    if sh is None or hp is None: return frame
    ang=ang_horiz(hp-sh); label,col=tag(ang)
    for i in (L_SH,R_SH,L_HIP,R_HIP):                 # the four points
        if v(kp,i): cv2.circle(frame,tuple(kp[i,:2].astype(int)),6,(255,255,255),-1)
    cv2.circle(frame,tuple(sh.astype(int)),5,(255,180,0),-1)   # shoulder mid
    cv2.circle(frame,tuple(hp.astype(int)),5,(255,0,255),-1)   # hip mid
    cv2.line(frame,tuple(sh.astype(int)),tuple(hp.astype(int)),col,3)   # TRUNK LINE
    p=tuple((sh.astype(int))+np.array([10,-10]))
    cv2.putText(frame,f"trunk {ang:0.0f}deg  {label}",p,cv2.FONT_HERSHEY_SIMPLEX,0.8,col,2,cv2.LINE_AA)
    return frame

def main(video,out,model,nframes,clip_start,clip_len,do_clip):
    os.makedirs(out,exist_ok=True)
    net=YOLO(model); cap=cv2.VideoCapture(video)
    fps=cap.get(cv2.CAP_PROP_FPS) or 25; tot=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); H=int(cap.get(4))
    if tot<=0: print("could not open video"); return
    stem=os.path.splitext(os.path.basename(video))[0].replace(" ","_")

    # sampled key frames
    idxs=np.linspace(tot*0.1,tot*0.9,nframes).astype(int)
    for j,i in enumerate(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES,int(i)); ok,fr=cap.read()
        if not ok: continue
        r=net.predict(fr,imgsz=1280,conf=0.25,max_det=1,verbose=False)[0]
        kp=r.keypoints.data[0].cpu().numpy() if (r.boxes is not None and len(r.boxes)>0) else None
        cv2.imwrite(os.path.join(out,f"{stem}_frame{j:02d}.jpg"),annotate(fr,kp))
    print(f"wrote {nframes} annotated frames to {out}/")

    # short annotated clip
    if do_clip:
        s=int(clip_start*fps); n=int(clip_len*fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES,s); ok,fr=cap.read()
        if ok:
            h,w=fr.shape[:2]
            vw=cv2.VideoWriter(os.path.join(out,f"{stem}_trunk.mp4"),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
            cap.set(cv2.CAP_PROP_POS_FRAMES,s)
            for _ in range(n):
                ok,fr=cap.read()
                if not ok: break
                r=net.predict(fr,imgsz=1280,conf=0.25,max_det=1,verbose=False)[0]
                kp=r.keypoints.data[0].cpu().numpy() if (r.boxes is not None and len(r.boxes)>0) else None
                vw.write(annotate(fr,kp))
            vw.release(); print(f"wrote {clip_len}s annotated clip: {out}/{stem}_trunk.mp4")
    cap.release()

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("video"); ap.add_argument("--out",default="trunk_overlay")
    ap.add_argument("--model",default="yolo11x-pose.pt"); ap.add_argument("--frames",type=int,default=12)
    ap.add_argument("--clip-start",type=float,default=20.0); ap.add_argument("--clip-len",type=float,default=15.0)
    ap.add_argument("--no-clip",action="store_true")
    a=ap.parse_args(); main(a.video,a.out,a.model,a.frames,a.clip_start,a.clip_len,not a.no_clip)

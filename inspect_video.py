"""
inspect_video.py - visual check of water-line detection + pose on a few frames.

Saves annotated frames (detected water line in cyan, pool floor in yellow, pose
skeleton + hip midpoint) and prints the numbers the feature pipeline would use,
so you can see whether the auto water line is right for a given video.

Usage:
    python inspect_video.py videos/2.mp4 --out inspect_v2
    python inspect_video.py videos/2.mp4 --out inspect_v2 --waterline-frac 0.30   # preview an override
"""
import argparse, os, cv2, numpy as np
from ultralytics import YOLO
from swim_features import detect_waterline, mid, head_pt, ang_horiz, v, \
    L_SH,R_SH,L_HIP,R_HIP,L_KN,R_KN,L_AN,R_AN, CONF

SKEL=[(5,7),(7,9),(6,8),(8,10),(5,6),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]

def draw(frame,kp,wl,floor):
    H,W=frame.shape[:2]
    cv2.line(frame,(0,wl),(W,wl),(255,255,0),2)        # water line (cyan)
    cv2.putText(frame,f"waterline y={wl}",(10,wl-8),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,0),2)
    cv2.line(frame,(0,floor),(W,floor),(0,255,255),2)  # floor (yellow)
    if kp is not None:
        for a,b in SKEL:
            if kp[a,2]>CONF and kp[b,2]>CONF:
                cv2.line(frame,tuple(kp[a,:2].astype(int)),tuple(kp[b,:2].astype(int)),(0,255,0),2)
        for i in range(len(kp)):
            if kp[i,2]>CONF: cv2.circle(frame,tuple(kp[i,:2].astype(int)),4,(0,0,255),-1)
        hp=mid(kp,L_HIP,R_HIP)
        if hp is not None:
            cv2.circle(frame,tuple(hp.astype(int)),7,(255,0,255),-1)
            cv2.putText(frame,"hip",(int(hp[0])+8,int(hp[1])),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,255),2)
    return frame

def main(video,out,model,nframes,wl_override):
    os.makedirs(out,exist_ok=True)
    cap=cv2.VideoCapture(video); H=int(cap.get(4)); tot=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if wl_override is not None:
        wl=int(wl_override); conf=True; src="override"
    else:
        wl,conf=detect_waterline(cap); src="auto"
        if wl is None: wl=int(H*0.35); conf=False; src="fallback"
    floor=int(H*0.98)
    print(f"video={os.path.basename(video)}  H={H}  water_line_y={wl}  source={src}  confident={conf}")
    net=YOLO(model)
    idxs=np.linspace(tot*0.15,tot*0.85,nframes).astype(int)
    for j,i in enumerate(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES,int(i)); ok,fr=cap.read()
        if not ok: continue
        r=net.predict(fr,imgsz=1280,conf=0.25,max_det=1,verbose=False)[0]
        kp=r.keypoints.data[0].cpu().numpy() if (r.boxes is not None and len(r.boxes)>0) else None
        if kp is not None:
            sh,hp=mid(kp,L_SH,R_SH),mid(kp,L_HIP,R_HIP)
            torso=float(np.linalg.norm(sh-hp)) if (sh is not None and hp is not None) else float("nan")
            trunk=ang_horiz(hp-sh) if (sh is not None and hp is not None) else float("nan")
            hipd=((hp[1]-wl)/torso) if (hp is not None and torso>1) else float("nan")
            print(f"  frame {i:5d}: trunk={trunk:5.1f}deg  hip_depth={hipd:+.2f}torso  torso={torso:.0f}px")
        out_path=os.path.join(out,f"frame_{j:02d}_{int(i)}.jpg")
        cv2.imwrite(out_path,draw(fr,kp,wl,floor))
    cap.release()
    print(f"\nwrote {nframes} annotated frames to {out}/  (open them to check the cyan water line)")
    print("hip_depth should be POSITIVE (hips below the water line) for a horizontal float.")
    print("if the cyan line is clearly in the wrong place, re-run swim_features.py with --waterline-frac <0-1>.")

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("video"); ap.add_argument("--out",default="inspect_out")
    ap.add_argument("--model",default="yolo11x-pose.pt"); ap.add_argument("--nframes",type=int,default=6)
    ap.add_argument("--waterline-frac",type=float,default=None)
    a=ap.parse_args()
    cap=cv2.VideoCapture(a.video); H=int(cap.get(4)); cap.release()
    wl=int(H*a.waterline_frac) if a.waterline_frac is not None else None
    main(a.video,a.out,a.model,a.nframes,wl)

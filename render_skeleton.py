"""
render_skeleton.py  -  render a stick-figure clip from MediaPipe landmark CSV, for labeling.

Draws the tracked body (skeleton + highlighted shoulder-to-hip trunk line) over a time window,
so an instructor can judge float state without the raw video. Privacy-friendly: no real footage.

Usage (one sample clip):
    python render_skeleton.py "data/raw/mediapipe_full/P1_side_front...raw.csv" --start 90 --len 8 --out clip.mp4
Usage (batch from a manifest of clips to label):
    python render_skeleton.py --manifest label_manifest.csv --src-dir data/raw/mediapipe_full --out-dir label_clips
"""
import argparse, os, glob, numpy as np, pandas as pd, cv2
import imageio  # bundles a static ffmpeg with H.264 (browser-playable); pip install imageio imageio-ffmpeg

CONN=[("left_shoulder","right_shoulder"),("left_shoulder","left_elbow"),("left_elbow","left_wrist"),
      ("right_shoulder","right_elbow"),("right_elbow","right_wrist"),
      ("left_shoulder","left_hip"),("right_shoulder","right_hip"),("left_hip","right_hip"),
      ("left_hip","left_knee"),("left_knee","left_ankle"),("right_hip","right_knee"),("right_knee","right_ankle"),
      ("left_ankle","left_heel"),("left_heel","left_foot_index"),
      ("right_ankle","right_heel"),("right_heel","right_foot_index")]
PTS=["nose","left_shoulder","right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist",
     "left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle",
     "left_heel","right_heel","left_foot_index","right_foot_index"]
CONF=0.5

def load(path): return pd.read_excel(path) if path.lower().endswith(("xlsx","xls")) else pd.read_csv(path)

def render(df,start,length,out,W=560,H=320,pad=0.12,fps_out=12):
    t=df["time_sec"].to_numpy(); fps=1.0/np.median(np.diff(t)) if len(t)>1 else 25.0
    seg=df[(df["time_sec"]>=start)&(df["time_sec"]<start+length)].reset_index(drop=True)
    if len(seg)==0: raise SystemExit("no frames in that window")
    pid=str(df["participant_id"].iloc[0]); view=str(df["camera_view"].iloc[0])
    stepn=max(1,int(round(fps/fps_out)))        # subsample frames to ~fps_out
    writer=imageio.get_writer(out,fps=fps_out,codec="libx264",macro_block_size=16,
                              ffmpeg_params=["-crf","30","-pix_fmt","yuv420p","-preset","veryfast"])
    def px(x,y):  # normalized -> pixel, with padding margin
        return (int((pad+x*(1-2*pad))*W), int((pad+y*(1-2*pad))*H))
    for ri in range(0,len(seg),stepn):
        r=seg.iloc[ri]
        img=np.full((H,W,3),28,np.uint8)
        # faint horizontal reference (fixed side camera: frame-horizontal ~ water-horizontal)
        for x in range(0,W,16): cv2.line(img,(x,H//2),(x+8,H//2),(60,60,70),1)
        def P(lm):
            return (r[f"{lm}_x"],r[f"{lm}_y"]) if r.get(f"{lm}_visibility",1)>CONF else None
        for a,b in CONN:
            pa,pb=P(a),P(b)
            if pa and pb: cv2.line(img,px(*pa),px(*pb),(180,180,180),2)
        for lm in PTS:
            p=P(lm)
            if p: cv2.circle(img,px(*p),3,(0,0,235),-1)
        # trunk line (shoulder mid -> hip mid) in green
        ls,rs,lh,rh=P("left_shoulder"),P("right_shoulder"),P("left_hip"),P("right_hip")
        if ls and rs and lh and rh:
            sm=((ls[0]+rs[0])/2,(ls[1]+rs[1])/2); hm=((lh[0]+rh[0])/2,(lh[1]+rh[1])/2)
            cv2.line(img,px(*sm),px(*hm),(0,220,0),3)
        cv2.putText(img,f"{pid} {view}  t={r['time_sec']:.0f}s",(10,24),cv2.FONT_HERSHEY_SIMPLEX,0.6,(200,200,200),1)
        writer.append_data(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))   # imageio expects RGB
    writer.close()
    return out

def main(a):
    if a.manifest:
        import json, shutil
        man=pd.read_csv(a.manifest)
        clips_dir=os.path.join(a.out_dir,"clips"); os.makedirs(clips_dir,exist_ok=True)
        rows=[]
        for i,m in man.iterrows():
            stem=str(m.get("raw_stem") or str(m["source"]).replace("_features.csv",""))
            cands=[c for c in glob.glob(os.path.join(a.src_dir,stem+"*"))
                   if c.lower().endswith((".csv",".xlsx",".xls"))]
            if not cands: print(f"  no source for {stem}"); continue
            length=float(m["end_s"])-float(m["start_s"]) if "end_s" in m else 8.0
            clip=str(m.get("clip") or f"{m['participant']}__{int(m['start_s']):04d}.mp4")
            try:
                df=load(cands[0]); render(df,float(m["start_s"]),length,os.path.join(clips_dir,clip))
                rows.append({"clip":f"clips/{clip}","participant":str(m["participant"]),
                             "source":str(m["source"]),"seg_id":int(m.get("seg_id",int(m["start_s"]))),
                             "start_s":float(m["start_s"]),"end_s":float(m["start_s"])+length})
            except Exception as e: print(f"  failed {clip}: {e}")
        manjs="window.MANIFEST="+json.dumps(rows,indent=0)+";"
        with open(os.path.join(a.out_dir,"manifest.js"),"w") as fh: fh.write(manjs)
        here=os.path.dirname(os.path.abspath(__file__))
        # expert-window manifests get the 5-criteria expert rubric UI; others get the v3 float UI
        man_reason=str(man["reason"].iloc[0]) if "reason" in man.columns and len(man) else ""
        prefer=(["label_ui_expert.html"] if man_reason=="expert_window" else [])+["label_ui_v3.html","label_ui_v2.html"]
        ui=next((os.path.join(here,u) for u in prefer if os.path.exists(os.path.join(here,u))),"")
        if ui and os.path.exists(ui):
            html=open(ui).read().replace('<script src="manifest.js"></script>','<script>'+manjs+'</script>')
            open(os.path.join(a.out_dir,"label_ui.html"),"w").write(html)  # manifest inlined (no file:// block)
        print(f"rendered {len(rows)} skeleton clips -> {a.out_dir}/ (clips/, self-contained label_ui.html)")
        print(f"zip -r {a.out_dir}.zip {a.out_dir}  then open label_ui.html (or: python -m http.server in the folder)")
    else:
        df=load(a.input); out=render(df,a.start,a.len,a.out)
        print(f"wrote {out}")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("input",nargs="?"); ap.add_argument("--start",type=float,default=60); ap.add_argument("--len",type=float,default=8)
    ap.add_argument("--out",default="clip.mp4")
    ap.add_argument("--manifest",default=None); ap.add_argument("--src-dir",default="data/raw/mediapipe_full")
    ap.add_argument("--out-dir",default="label_clips")
    a=ap.parse_args(); main(a)

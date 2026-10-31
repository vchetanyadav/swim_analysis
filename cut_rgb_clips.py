"""
cut_rgb_clips.py  -  cut REAL RGB clips from the original videos for a labelling manifest, so the
expert labels from footage (where the airway, face and water surface are visible) instead of the
stick figure. The skeleton is only the model's measurement; the human ground truth comes from video.

It reads a manifest (the same files select_*.py produce: participant, raw_stem/source, start_s,
end_s, seg_id), finds each participant's original SIDE-FRONT video in --video-dir, cuts the window
with ffmpeg, and builds a self-contained labelling kit (clips/ + manifest.js + label_ui_rgb.html).

Usage (on the server that holds the videos):
  python cut_rgb_clips.py --manifest label_manifest_expert.csv --video-dir ~/swim_analysis/videos \
      --out-dir labeling_kit_rgb
  # then:  zip -r labeling_kit_rgb.zip labeling_kit_rgb   and open label_ui_rgb.html via a local server

If a participant's video is not matched automatically, pass a mapping file with --map map.csv
(columns: participant,video_path) to override those few by hand.
"""
import argparse, glob, os, re, json, subprocess, shutil, pandas as pd

VID_EXT=(".mp4",".mov",".avi",".mkv",".m4v")

def name_of(stem):
    s=re.sub(r'_raw(_features)?(\.csv)?$','',str(stem))
    STOP={"side","front","top","view","raw","features","min","3min","rear","tread","p"}
    for cand in re.findall(r'[pP]\d+_([a-zA-Z]+)',s):
        c=cand.lower()
        if c not in STOP and len(c)>=3: return c   # a real name, not a stray letter
    return None

def pnum(stem):
    m=re.search(r'[Pp](\d+)',str(stem)); return m.group(1) if m else None

def index_videos(d):
    vids=[]
    for root,_,files in os.walk(d):
        for f in files:
            if f.lower().endswith(VID_EXT): vids.append(os.path.join(root,f))
    return vids

def pick_video(row,vids,manual):
    pid=row["participant"]
    if pid in manual: return manual[pid]
    stem=row.get("raw_stem") or row.get("source") or ""
    nm=name_of(stem); num=pnum(stem) or pnum(pid)
    numi=int(num) if num else None
    def base(f): return os.path.basename(f).lower()
    # a side-FRONT file: says 'side' and 'front', and is not 'rear' or 'top'
    def ok_side(f):
        b=base(f); return ("side" in b) and ("rear" not in b) and ("top" not in b)
    # 1) name + side-front  (e.g. 'gareth ... side front')
    if nm:
        c=[f for f in vids if nm in base(f) and ok_side(f)]
        if c: return sorted(c)[0]
    # 2) explicit participant tag + side  (e.g. 'p7' / 'P07')
    if num:
        c=[f for f in vids if re.search(rf'[pP]0*{num}\b',base(f)) and ok_side(f)]
        if c: return sorted(c)[0]
    # 3) bare leading number + side front  (e.g. '7 Side front.mp4' == participant 7)
    if numi is not None:
        c=[f for f in vids if re.match(rf'0*{numi}\b',base(f)) and ok_side(f)]
        if c: return sorted(c)[0]
    # no side-FRONT video for this participant: do not fall back to a rear/top camera
    return None

def ffmpeg_exe():
    # prefer the static ffmpeg bundled with imageio-ffmpeg (already used to render the skeletons),
    # so no system ffmpeg install is needed; fall back to a system ffmpeg on PATH.
    try:
        import imageio_ffmpeg; return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg") or "ffmpeg"

FFMPEG=ffmpeg_exe()

def cut(src,start,end,out,height):
    vf=["-vf",f"scale=-2:{height}"] if height else []
    cmd=[FFMPEG,"-y","-ss",str(start),"-to",str(end),"-i",src,
         "-an","-c:v","libx264","-preset","veryfast","-crf","23","-pix_fmt","yuv420p",*vf,out]
    subprocess.run(cmd,check=True,capture_output=True)

def main(a):
    man=pd.read_csv(a.manifest)
    manual={}
    if a.map and os.path.exists(a.map):
        mm=pd.read_csv(a.map); manual={str(r["participant"]):r["video_path"] for _,r in mm.iterrows()}
    vids=index_videos(a.video_dir)
    print(f"{len(vids)} videos found under {a.video_dir}")
    clips_dir=os.path.join(a.out_dir,"clips"); os.makedirs(clips_dir,exist_ok=True)
    rows=[]; unmatched=[]
    for _,m in man.iterrows():
        src=pick_video(m,vids,manual)
        if not src or not os.path.exists(src):
            unmatched.append(m["participant"]); print(f"  NO VIDEO for {m['participant']} ({m.get('raw_stem','')})"); continue
        seg=int(m.get("seg_id",int(m["start_s"])))
        clip=f"{m['participant']}__{seg:04d}.mp4"
        try:
            cut(src,float(m["start_s"]),float(m["end_s"]),os.path.join(clips_dir,clip),a.height)
            rows.append({"clip":f"clips/{clip}","participant":str(m["participant"]),
                         "source":str(m.get("source",m.get("raw_stem",""))),"seg_id":seg,
                         "start_s":float(m["start_s"]),"end_s":float(m["end_s"]),
                         "reason":str(m.get("reason","")),"video":os.path.basename(src)})
        except Exception as e:
            print(f"  cut failed {clip}: {e}")
    manjs="window.MANIFEST="+json.dumps(rows,indent=0)+";"
    open(os.path.join(a.out_dir,"manifest.js"),"w").write(manjs)
    here=os.path.dirname(os.path.abspath(__file__))
    ui=os.path.join(here,"label_ui_rgb.html")
    if os.path.exists(ui):
        html=open(ui).read().replace('<script src="manifest.js"></script>','<script>'+manjs+'</script>')
        open(os.path.join(a.out_dir,"label_ui.html"),"w").write(html)
    print(f"\ncut {len(rows)} RGB clips -> {a.out_dir}/ (clips/, self-contained label_ui.html)")
    if unmatched:
        print(f"UNMATCHED participants ({len(set(unmatched))}): {sorted(set(unmatched))}")
        print("  -> add them to a --map CSV (participant,video_path) and rerun")
    print(f"next: zip -r {a.out_dir}.zip {a.out_dir}  then serve the folder and open label_ui.html")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--manifest",required=True); ap.add_argument("--video-dir",required=True)
    ap.add_argument("--out-dir",default="labeling_kit_rgb"); ap.add_argument("--map",default=None)
    ap.add_argument("--height",type=int,default=480,help="output height px (0 = keep original)")
    a=ap.parse_args(); main(a)

"""
build_manifest.py  -  scan the video folder and build the dataset backbone.

For every video: a deterministic ID, participant, trial, view, fps, duration, resolution, and a
SHA-256 checksum (so corruption during a 100+ file transfer is caught). This fixes the metadata
problem where filenames like "2 Side front.mp4" led to every file being tagged p4.

Writes manifest.csv. Nothing is renamed on disk; the manifest carries the canonical IDs.

Usage:
    python build_manifest.py /opt/home/s4137946/swim_analysis/videos --out manifest.csv
"""
import argparse, os, re, csv, hashlib, glob, cv2

def sha256(path, chunk=1<<20):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(chunk),b""): h.update(b)
    return h.hexdigest()

def parse(name):
    stem=os.path.splitext(name)[0]; low=stem.lower()
    # participant
    m=re.search(r'[Pp](\d+)',stem)
    if m: pid=f"P{int(m.group(1)):02d}"
    else:
        m2=re.match(r'\s*(\d+)',stem); pid=f"V{int(m2.group(1)):02d}" if m2 else "P??"
    # view
    if "top" in low: view="top"
    elif "side" in low and "rear" in low: view="side_rear"
    elif "side" in low and "front" in low: view="side_front"
    elif "front" in low: view="front"
    elif "rear" in low: view="side_rear"
    else: view="unknown"
    # trial (if a T<nn> appears) else T01
    mt=re.search(r'[Tt](\d{1,2})\b',stem); trial=f"T{int(mt.group(1)):02d}" if mt else "T01"
    return pid,trial,view

def probe(path):
    cap=cv2.VideoCapture(path)
    fps=cap.get(cv2.CAP_PROP_FPS) or 0
    n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); w=int(cap.get(3)); h=int(cap.get(4))
    cap.release()
    dur=round(n/fps,1) if fps>0 else 0
    return round(fps,2),dur,w,h,n

def main(folder,out,exercise):
    vids=sorted(glob.glob(os.path.join(folder,"*.mp4"))+glob.glob(os.path.join(folder,"*.MP4")))
    rows=[]
    for p in vids:
        name=os.path.basename(p); pid,trial,view=parse(name)
        try: fps,dur,w,h,n=probe(p)
        except Exception: fps,dur,w,h,n=0,0,0,0,0
        vid=f"{pid}_{trial}_{view}"
        ok = n>0 and dur>0
        rows.append({"video_id":vid,"participant_id":pid,"trial_id":trial,"view":view,
                     "filename":name,"fps":fps,"duration_s":dur,"width":w,"height":h,
                     "frames":n,"exercise":exercise,"readable":ok,"sha256":sha256(p)})
        flag="" if ok else "  <-- UNREADABLE / possibly corrupt"
        print(f"{vid:28s} {name[:40]:40s} {dur:6.0f}s {w}x{h}{flag}")
    with open(out,"w",newline="") as f:
        wri=csv.DictWriter(f,fieldnames=list(rows[0].keys())); wri.writeheader(); wri.writerows(rows)
    bad=[r for r in rows if not r["readable"]]
    dupe={}
    for r in rows: dupe.setdefault(r["video_id"],[]).append(r["filename"])
    clashes={k:v for k,v in dupe.items() if len(v)>1}
    print(f"\nwrote {out}: {len(rows)} videos, {len(set(r['participant_id'] for r in rows))} participants")
    if bad: print(f"UNREADABLE: {[r['filename'] for r in bad]} — re-download before extracting features")
    if clashes: print(f"DUPLICATE IDs (resolve names): {clashes}")

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("folder"); ap.add_argument("--out",default="manifest.csv")
    ap.add_argument("--exercise",default="back_float")
    a=ap.parse_args(); main(a.folder,a.out,a.exercise)

"""
sample_frames.py
Pull a diverse set of frames from each camera for labelling.

Spreads frames across the whole clip (not consecutive, which would be near
duplicates) so the labelled set covers many body positions. Run once per video.

Usage:
    python sample_frames.py "P38_..._3_Top_.mp4" --view top --every 3 --out frames
    python sample_frames.py "P38_..._2_Side_front_.mp4" --view sidefront --every 3 --out frames
"""
import argparse, os, cv2

def sample(video, view, out, every_sec=3.0):
    os.makedirs(out, exist_ok=True)
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = int(every_sec * fps)
    saved = 0
    for f in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, frame = cap.read()
        if not ok:
            continue
        name = f"{view}_{int(f/fps):04d}s.jpg"
        cv2.imwrite(os.path.join(out, name), frame)
        saved += 1
    cap.release()
    print(f"{view}: saved {saved} frames to {out}/ (one every {every_sec}s)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--view", required=True, help="short tag, e.g. top / sidefront / siderear / front")
    ap.add_argument("--every", type=float, default=3.0, help="seconds between sampled frames")
    ap.add_argument("--out", default="frames")
    args = ap.parse_args()
    sample(args.video, args.view, args.out, args.every)

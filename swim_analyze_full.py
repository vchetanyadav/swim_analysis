"""
swim_analyze_full.py
Stock YOLO pose model (17 COCO keypoints, class 'person'), NO fine-tuning, NO
annotation. Connected colour skeleton + per-limb movement counters + body angle
vs water line + head dips + neck angle + limb frequency/intensity. One tool.

Usage:
    python swim_analyze_full.py videos/sideFront.mp4 --model yolo11x-pose.pt --out results
    python swim_analyze_full.py videos/sideFront.mp4 --out results --adur 0   # whole video
"""
import argparse, os, csv, json
import cv2, numpy as np, pandas as pd
from scipy.signal import detrend
from ultralytics import YOLO

NOSE, L_EAR, R_EAR = 0, 3, 4
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 5, 6, 7, 8, 9, 10
L_HIP, R_HIP, L_KN, R_KN, L_AN, R_AN = 11, 12, 13, 14, 15, 16

LIMBS = {
    "Left arm":  ([(5, 7), (7, 9)],   (255, 0, 255), 9,  5),
    "Right arm": ([(6, 8), (8, 10)],  (255, 160, 0), 10, 6),
    "Left leg":  ([(11, 13), (13, 15)], (0, 165, 255), 15, 11),
    "Right leg": ([(12, 14), (14, 16)], (0, 255, 255), 16, 12),
}
TORSO_EDGES = [(5, 6), (11, 12), (5, 11), (6, 12)]
CONF = 0.30


def vis(kp, i): return kp[i, 2] > CONF
def mid(kp, a, b):
    if vis(kp, a) and vis(kp, b): return (kp[a, :2] + kp[b, :2]) / 2
    if vis(kp, a): return kp[a, :2]
    if vis(kp, b): return kp[b, :2]
    return None
def head_point(kp):
    if vis(kp, NOSE): return kp[NOSE, :2]
    ears = [kp[i, :2] for i in (L_EAR, R_EAR) if vis(kp, i)]
    return np.mean(ears, axis=0) if ears else None
def ang_horizontal(v): return float(np.degrees(np.arctan2(abs(v[1]), abs(v[0]))))
def ang_between(a, b, c):
    ba, bc = a - b, c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))


def coco17_to_swim14(kp):
    out = np.zeros((14, 3), float)
    hp = head_point(kp)
    if hp is not None: out[0] = [hp[0], hp[1], 1.0]
    nk = mid(kp, L_SH, R_SH)
    if nk is not None: out[1] = [nk[0], nk[1], 1.0]
    for j, c in enumerate([L_SH, R_SH, L_EL, R_EL, L_WR, R_WR,
                           L_HIP, R_HIP, L_KN, R_KN, L_AN, R_AN], start=2):
        out[j] = kp[c]
    return out


class Counter:
    def __init__(self):
        self.base, self.state = {}, {}
        self.count = {k: 0 for k in LIMBS}
    def update(self, kp):
        sh, hp = mid(kp, L_SH, R_SH), mid(kp, L_HIP, R_HIP)
        torso = np.linalg.norm(sh - hp) if (sh is not None and hp is not None) else None
        if not torso or torso < 1: return self.count
        for name, (_, _, distal, prox) in LIMBS.items():
            if not (vis(kp, distal) and vis(kp, prox)): continue
            d = np.linalg.norm(kp[distal, :2] - kp[prox, :2]) / torso
            self.base.setdefault(name, d)
            self.base[name] = 0.98 * self.base[name] + 0.02 * d
            b, st = self.base[name], self.state.get(name, 0)
            if st == 0 and d > b * 1.18: self.state[name] = 1
            elif st == 1 and d < b * 0.95:
                self.state[name] = 0; self.count[name] += 1
        return self.count


def detect_waterline(cap, n=25):
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(total * 0.1, total * 0.9, n).astype(int)
    profs = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i)); ok, fr = cap.read()
        if ok: profs.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(float).mean(axis=1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    if not profs: return None
    prof = np.mean(profs, axis=0)
    k = max(5, len(prof) // 60)
    prof = np.convolve(prof, np.ones(k) / k, mode="same")
    grad = np.gradient(prof)
    return int(np.argmin(grad[:int(len(grad) * 0.6)]))


def find_dips(t, head_y, wl, band, fps_min=0.15):
    s = pd.Series(head_y).interpolate(limit=8).to_numpy()
    below, state = np.zeros(len(s), bool), False
    for i, v in enumerate(s):
        if np.isnan(v): below[i] = state; continue
        if not state and v > wl + band: state = True
        elif state and v < wl - band: state = False
        below[i] = state
    dips, i = [], 0
    while i < len(below):
        if below[i]:
            j = i
            while j < len(below) and below[j]: j += 1
            dur = t[j - 1] - t[i]
            if dur >= fps_min:
                dips.append({"start_s": round(float(t[i]), 2), "duration_s": round(float(dur), 2)})
            i = j
        else: i += 1
    return dips


def cadence(sig, fps):
    var = np.nanvar(sig, axis=0); axis = sig[:, int(np.argmax(var))]
    s = pd.Series(axis).interpolate(limit=8).to_numpy(); m = ~np.isnan(s)
    if m.sum() < 40: return None
    s = detrend(s[m]); f = np.fft.rfftfreq(len(s), 1 / fps); P = np.abs(np.fft.rfft(s)) ** 2
    band = (f > 0.2) & (f < 4)
    return float(f[band][np.argmax(P[band])]) if band.any() else None


def draw(frame, kp):
    for a, b in TORSO_EDGES:
        if vis(kp, a) and vis(kp, b):
            cv2.line(frame, tuple(kp[a, :2].astype(int)), tuple(kp[b, :2].astype(int)), (200, 200, 200), 2, cv2.LINE_AA)
    for _, (edges, col, _, _) in LIMBS.items():
        for a, b in edges:
            if vis(kp, a) and vis(kp, b):
                cv2.line(frame, tuple(kp[a, :2].astype(int)), tuple(kp[b, :2].astype(int)), col, 3, cv2.LINE_AA)
    for x, y, c in kp:
        if c > CONF: cv2.circle(frame, (int(x), int(y)), 4, (255, 255, 255), -1)


def run(video, out_dir, model_name="yolo11x-pose.pt", imgsz=1280, skip=1,
        waterline=None, adur=20.0, astart=0.0):
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(video))[0]
    net = YOLO(model_name)
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    W, H = int(cap.get(3)), int(cap.get(4))
    fps_eff = fps / skip
    if waterline is None:
        waterline = detect_waterline(cap)
        print(f"water line auto-detected at y={waterline}")
    band = max(3, int(0.012 * H))

    counter = Counter()
    series = {"t": [], "body_angle": [], "neck": [], "head_y": [], "torso": []}
    limb_rel = {k: [] for k in LIMBS}
    rows = []
    idx = 0
    print("analysing (stock COCO-17 model, no annotation)...")
    while True:
        ok, frame = cap.read()
        if not ok: break
        if idx % skip: idx += 1; continue
        r = net.predict(frame, imgsz=imgsz, conf=0.25, max_det=1, verbose=False)[0]
        t = idx / fps
        if r.boxes is not None and len(r.boxes) > 0:
            kp = r.keypoints.data[0].cpu().numpy()
            counter.update(kp)
            sh, hp = mid(kp, L_SH, R_SH), mid(kp, L_HIP, R_HIP)
            hd = head_point(kp)
            torso = float(np.linalg.norm(sh - hp)) if (sh is not None and hp is not None) else np.nan
            ba = ang_horizontal(hp - sh) if (sh is not None and hp is not None) else np.nan
            nk = ang_between(hd, sh, hp) if (hd is not None and sh is not None and hp is not None) else np.nan
            hy = float(hd[1]) if hd is not None else np.nan
            for name, (_, _, distal, prox) in LIMBS.items():
                limb_rel[name].append(kp[distal, :2] - kp[prox, :2] if (vis(kp, distal) and vis(kp, prox)) else [np.nan, np.nan])
        else:
            torso = ba = nk = hy = np.nan
            for name in LIMBS: limb_rel[name].append([np.nan, np.nan])
        series["t"].append(t); series["body_angle"].append(ba)
        series["neck"].append(nk); series["head_y"].append(hy); series["torso"].append(torso)
        rows.append([idx, round(t, 3),
                     round(ba, 1) if not np.isnan(ba) else "",
                     round(nk, 1) if not np.isnan(nk) else "",
                     "below" if (not np.isnan(hy) and hy > waterline) else "above",
                     counter.count["Left arm"], counter.count["Right arm"],
                     counter.count["Left leg"], counter.count["Right leg"]])
        idx += 1
        if idx % 300 == 0: print(f"  {idx} frames (t={t:.0f}s)", flush=True)
    cap.release()

    t_arr = np.array(series["t"])
    dips = find_dips(t_arr, np.array(series["head_y"]), waterline, band)
    dur = t_arr[-1] - t_arr[0] if len(t_arr) > 1 else 1
    head_seen = float(np.mean(~np.isnan(np.array(series["head_y"], float))) * 100)
    med_torso = float(np.nanmedian(np.array(series["torso"], float)))
    ba_arr = np.array(series["body_angle"], float); ba_arr = ba_arr[~np.isnan(ba_arr)]
    nk_arr = np.array(series["neck"], float); nk_arr = nk_arr[~np.isnan(nk_arr)]

    limbs = {}
    for name in LIMBS:
        rel = np.array(limb_rel[name], float)
        f = cadence(rel, fps_eff)
        sp = np.linalg.norm(np.diff(pd.DataFrame(rel).interpolate(limit=5).to_numpy(), axis=0), axis=1) * fps_eff
        sp = sp[~np.isnan(sp)]
        limbs[name] = {
            "movement_count": counter.count[name],
            "freq_per_min": round(f * 60, 0) if f else None,
            "intensity_bodylen_s": round(float(np.mean(sp) / med_torso), 2) if (sp.size and med_torso > 1) else None,
            "detected_pct": round(float(np.mean(~np.isnan(rel[:, 0])) * 100), 1),
        }

    report = {
        "video": os.path.basename(video), "model": model_name, "note": "stock COCO-17 pose model, no fine-tuning, no annotation",
        "duration_s": round(float(dur), 1), "water_line_y": int(waterline),
        "body_angle_from_horizontal": {
            "median_deg": round(float(np.median(ba_arr)), 1) if ba_arr.size else None,
            "p5_deg": round(float(np.percentile(ba_arr, 5)), 1) if ba_arr.size else None,
            "p95_deg": round(float(np.percentile(ba_arr, 95)), 1) if ba_arr.size else None},
        "neck_angle_median_deg": round(float(np.median(nk_arr)), 1) if nk_arr.size else None,
        "head_dips": {"count": len(dips), "per_minute": round(len(dips) / (dur / 60), 1),
                      "mean_duration_s": round(float(np.mean([d["duration_s"] for d in dips])), 2) if dips else 0,
                      "head_detected_pct": round(head_seen, 1),
                      "reliability": "reliable" if head_seen >= 60 else "LOW head detection (mask/occlusion): dips unreliable"},
        "limbs": limbs,
    }
    with open(os.path.join(out_dir, f"{stem}_report.json"), "w") as f: json.dump(report, f, indent=2)
    with open(os.path.join(out_dir, f"{stem}_report.txt"), "w") as f:
        b = report["body_angle_from_horizontal"]; d = report["head_dips"]
        f.write(f"SWIM ANALYSIS  {report['video']}  ({report['duration_s']}s)\n")
        f.write(f"Model: {model_name} (stock, no annotation)\n\n")
        f.write(f"Body angle from horizontal: median {b['median_deg']} deg (range {b['p5_deg']}-{b['p95_deg']}). 0=flat float.\n")
        f.write(f"Neck angle median: {report['neck_angle_median_deg']} deg\n")
        f.write(f"Head dips: {d['count']} ({d['per_minute']}/min), head seen {d['head_detected_pct']}% - {d['reliability']}\n\n")
        for n, s in limbs.items():
            f.write(f"{n}: {s['movement_count']} movements, {s['freq_per_min']}/min, intensity {s['intensity_bodylen_s']} body-len/s, seen {s['detected_pct']}%\n")
    with open(os.path.join(out_dir, f"{stem}_frames.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["frame", "t", "body_angle", "neck_angle", "head_state", "L_arm", "R_arm", "L_leg", "R_leg"]); w.writerows(rows)
    print("\n==== SUMMARY ====")
    b = report["body_angle_from_horizontal"]; d = report["head_dips"]
    print(f"Body angle: median {b['median_deg']} deg (range {b['p5_deg']}-{b['p95_deg']})")
    print(f"Head dips: {d['count']} ({d['per_minute']}/min) [head seen {d['head_detected_pct']}%]")
    for n, s in limbs.items(): print(f"  {n}: {s['movement_count']} moves, {s['freq_per_min']}/min, {s['intensity_bodylen_s']} body-len/s")

    _render(net, video, out_dir, stem, waterline, band, report, imgsz, astart, adur, fps, W, H)
    return report


def _render(net, video, out_dir, stem, waterline, band, report, imgsz, astart, adur, fps, W, H):
    cap = cv2.VideoCapture(video)
    if adur == 0: astart, n_frames = 0.0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    else: cap.set(cv2.CAP_PROP_POS_MSEC, astart * 1000); n_frames = int(adur * fps)
    out_path = os.path.join(out_dir, f"{stem}_annotated.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    counter = Counter()
    print("rendering annotated video...")
    for n in range(n_frames):
        ok, frame = cap.read()
        if not ok: break
        t = astart + n / fps
        r = net.predict(frame, imgsz=imgsz, conf=0.25, max_det=1, verbose=False)[0]
        cv2.line(frame, (0, waterline), (W, waterline), (0, 165, 255), 2)
        cv2.putText(frame, "water line", (W - 150, waterline - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)
        if r.boxes is not None and len(r.boxes) > 0:
            kp = r.keypoints.data[0].cpu().numpy()
            b = r.boxes.xyxy[0].cpu().numpy().astype(int)
            cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (255, 100, 0), 2)
            cv2.putText(frame, f"person {float(r.boxes.conf[0]):.2f}", (b[0], b[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            draw(frame, kp)
            counter.update(kp)
            sh, hp = mid(kp, L_SH, R_SH), mid(kp, L_HIP, R_HIP)
            hd = head_point(kp)
            ba = ang_horizontal(hp - sh) if (sh is not None and hp is not None) else None
            below = hd is not None and hd[1] > waterline
        else:
            ba, below = None, False
        c = counter.count
        panel = [f"Body angle: {ba:.0f} deg from flat" if ba is not None else "Body angle: --",
                 f"Head: {'BELOW' if below else 'above'} water",
                 f"L arm {c['Left arm']}  R arm {c['Right arm']}",
                 f"L leg {c['Left leg']}  R leg {c['Right leg']}"]
        cv2.rectangle(frame, (8, 8), (250, 20 + 26 * len(panel)), (35, 35, 35), -1)
        for i, line in enumerate(panel):
            col = (0, 200, 255) if (i == 1 and below) else (255, 255, 255)
            cv2.putText(frame, line, (16, 32 + i * 26), cv2.FONT_HERSHEY_SIMPLEX, 0.58, col, 2)
        writer.write(frame)
    writer.release(); cap.release()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--model", default="yolo11x-pose.pt")
    ap.add_argument("--out", default="results")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--skip", type=int, default=1)
    ap.add_argument("--waterline", type=int, default=None)
    ap.add_argument("--astart", type=float, default=0.0)
    ap.add_argument("--adur", type=float, default=20.0)
    a = ap.parse_args()
    run(a.video, a.out, a.model, a.imgsz, a.skip, a.waterline, a.adur, a.astart)
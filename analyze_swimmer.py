"""
analyze_swimmer.py
Full swim-safety measurement from a SIDE view (built for the side-front camera).

Produces, from one video:
  1. Body angle relative to the water surface (0 deg = flat float, 90 = vertical).
  2. Head position vs the surface: every dip below water, its duration, and the
     dip frequency, plus the neck angle (forward-bend).
  3. Arm and leg movement: frequency and intensity (speed) per limb, left and
     right kept separate.

Outputs:
  - an annotated video with a live dashboard drawn on it (angle, head state and
     dip count, neck angle, per-limb speed), the skeleton, and the water line
  - a per-frame CSV
  - a summary report (.txt) and a summary chart (.png)

The water line is auto-detected and drawn on the video so you can verify it. If
the auto line is wrong, override it with --waterline <y_pixels>.

Usage (run from the swim_analysis folder, venv active):
  python analyze_swimmer.py "videos/<sidefront>.mp4" --out results
  python analyze_swimmer.py "videos/<sidefront>.mp4" --out results --adur 0   # annotate whole video
  python analyze_swimmer.py "videos/<sidefront>.mp4" --out results --waterline 150
"""
import argparse, os, csv, json
import cv2, numpy as np, pandas as pd
from scipy.signal import detrend
from ultralytics import YOLO
import pose_markerless as pm

# joint indices (COCO)
NOSE, L_EAR, R_EAR = 0, 3, 4
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 5, 6, 7, 8, 9, 10
L_HIP, R_HIP, L_KN, R_KN, L_AN, R_AN = 11, 12, 13, 14, 15, 16
LIMBS = {"l_arm": L_WR, "r_arm": R_WR, "l_leg": L_AN, "r_leg": R_AN}


# ---------- geometry helpers ----------
def vis(kp, i):
    return kp is not None and kp[i, 2] > 0.5

def mid(kp, a, b):
    if vis(kp, a) and vis(kp, b):
        return (kp[a, :2] + kp[b, :2]) / 2
    if vis(kp, a):
        return kp[a, :2]
    if vis(kp, b):
        return kp[b, :2]
    return None

def head_point(kp):
    if vis(kp, NOSE):
        return kp[NOSE, :2]
    ears = [kp[i, :2] for i in (L_EAR, R_EAR) if vis(kp, i)]
    return np.mean(ears, axis=0) if ears else None

def angle_from_horizontal(vec):
    return float(np.degrees(np.arctan2(abs(vec[1]), abs(vec[0]))))  # 0..90

def angle_between(a, b, c):
    ba, bc = a - b, c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))


# ---------- water line ----------
def detect_waterline(cap, n=25):
    """Surface = the strongest bright-to-dark horizontal boundary in the upper
    part of the frame (bright surface band above, darker water below)."""
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(total * 0.1, total * 0.9, n).astype(int)
    profiles = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if not ok:
            continue
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(float)
        profiles.append(g.mean(axis=1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    if not profiles:
        return None
    prof = np.mean(profiles, axis=0)
    k = max(5, len(prof) // 60)
    prof_s = np.convolve(prof, np.ones(k) / k, mode="same")
    grad = np.gradient(prof_s)
    upper = int(len(grad) * 0.6)
    wl = int(np.argmin(grad[:upper]))          # steepest drop in the top 60%
    return wl


# ---------- dip finder with hysteresis ----------
def find_dips(t, head_y, waterline, band, min_dur=0.15):
    s = pd.Series(head_y).interpolate(limit=8).to_numpy()
    below = np.zeros(len(s), bool)
    state = False
    for i, v in enumerate(s):
        if np.isnan(v):
            below[i] = state
            continue
        if not state and v > waterline + band:
            state = True
        elif state and v < waterline - band:
            state = False
        below[i] = state
    dips, i = [], 0
    while i < len(below):
        if below[i]:
            j = i
            while j < len(below) and below[j]:
                j += 1
            dur = t[j - 1] - t[i]
            if dur >= min_dur:
                dips.append({"start_s": round(float(t[i]), 2),
                             "end_s": round(float(t[j - 1]), 2),
                             "duration_s": round(float(dur), 2)})
            i = j
        else:
            i += 1
    return dips


# ---------- cadence + intensity ----------
def cadence(signal_xy, fps):
    xy = signal_xy
    var = np.nanvar(xy, axis=0)
    axis = xy[:, int(np.argmax(var))]                 # dominant-motion axis
    s = pd.Series(axis).interpolate(limit=8).to_numpy()
    m = ~np.isnan(s)
    if m.sum() < 40:
        return None
    s = detrend(s[m])
    f = np.fft.rfftfreq(len(s), d=1 / fps)
    P = np.abs(np.fft.rfft(s)) ** 2
    band = (f > 0.2) & (f < 4.0)
    if not band.any():
        return None
    return float(f[band][np.argmax(P[band])])

def speed_series(xy, fps):
    d = np.linalg.norm(np.diff(xy, axis=0), axis=1) * fps      # px/s per step
    return d


# ---------- main ----------
def analyze(video, out_dir, skip=3, imgsz=960, conf=0.25, model="yolo11n-pose.pt",
            waterline=None, astart=0.0, adur=20.0):
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(video))[0]
    net = YOLO(model)

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_eff = fps / skip

    if waterline is None:
        waterline = detect_waterline(cap)
        print(f"Auto-detected water line at y = {waterline} px "
              f"(draw check in the video; override with --waterline if wrong).")
    band = max(3, int(0.012 * H))                     # hysteresis band for dips

    # ---- pass 1: joints + per-frame metrics over the whole clip ----
    print("Analysing full clip...")
    smoother = pm.Smoother((W**2 + H**2) ** 0.5)
    rows, series = [], {"t": [], "body_angle": [], "neck_angle": [], "head_y": []}
    limb_xy = {k: [] for k in LIMBS}
    idx = processed = 0
    cur_angle = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % skip:
            idx += 1
            continue
        if processed % pm.REORIENT_EVERY == 0:
            kp, cur_angle = pm.best_orientation(net, frame, imgsz, conf)
        else:
            kp, _ = pm._kp_for_angle(net, frame, cur_angle, imgsz, conf)
        if kp is not None:
            kp = pm.clean(kp, frame)
            kp = smoother.apply(kp)
        t = idx / fps

        sh, hip = mid(kp, L_SH, R_SH), mid(kp, L_HIP, R_HIP)
        hd = head_point(kp) if kp is not None else None
        body_angle = angle_from_horizontal(hip - sh) if (sh is not None and hip is not None) else np.nan
        neck = angle_between(hd, sh, hip) if (hd is not None and sh is not None and hip is not None) else np.nan
        head_y = float(hd[1]) if hd is not None else np.nan

        series["t"].append(t)
        series["body_angle"].append(body_angle)
        series["neck_angle"].append(neck)
        series["head_y"].append(head_y)
        for k, j in LIMBS.items():
            limb_xy[k].append(kp[j, :2] if vis(kp, j) else [np.nan, np.nan])

        rows.append([idx, round(t, 3), round(body_angle, 1) if not np.isnan(body_angle) else "",
                     round(neck, 1) if not np.isnan(neck) else "",
                     round(head_y, 1) if not np.isnan(head_y) else "",
                     "below" if (not np.isnan(head_y) and head_y > waterline) else "above"])
        processed += 1
        idx += 1
        if processed % 200 == 0:
            print(f"  {processed} frames (t={t:.0f}s)", flush=True)
    cap.release()

    t_arr = np.array(series["t"])
    # dips
    dips = find_dips(t_arr, np.array(series["head_y"]), waterline, band)
    dur_total = t_arr[-1] - t_arr[0] if len(t_arr) > 1 else 1
    dip_freq = len(dips) / (dur_total / 60.0)
    dip_durs = [d["duration_s"] for d in dips]
    # limb frequency + intensity
    limb_stats = {}
    for k in LIMBS:
        xy = np.array(limb_xy[k], float)
        f = cadence(xy, fps_eff)
        sp = speed_series(pd.DataFrame(xy).interpolate(limit=8).to_numpy(), fps_eff)
        sp = sp[~np.isnan(sp)]
        seen = np.mean(~np.isnan(xy[:, 0])) * 100
        limb_stats[k] = {
            "freq_hz": round(f, 2) if f else None,
            "moves_per_min": round(f * 60, 0) if f else None,
            "mean_speed_px_s": round(float(np.mean(sp)), 0) if sp.size else None,
            "peak_speed_px_s": round(float(np.max(sp)), 0) if sp.size else None,
            "detected_pct": round(float(seen), 1),
        }
    # body angle summary
    ba = np.array(series["body_angle"], float)
    ba = ba[~np.isnan(ba)]
    body_summary = {
        "median_deg": round(float(np.median(ba)), 1) if ba.size else None,
        "min_deg": round(float(np.percentile(ba, 5)), 1) if ba.size else None,
        "max_deg": round(float(np.percentile(ba, 95)), 1) if ba.size else None,
    }
    nk = np.array(series["neck_angle"], float); nk = nk[~np.isnan(nk)]
    neck_median = round(float(np.median(nk)), 1) if nk.size else None
    head_seen = float(np.mean(~np.isnan(np.array(series["head_y"], float))) * 100)
    head_note = ("reliable" if head_seen >= 60 else
                 "LOW head detection (mask/occlusion) - dip counts are unreliable for this clip")

    # ---- write CSV ----
    csv_path = os.path.join(out_dir, f"{stem}_frames.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "t", "body_angle_deg", "neck_angle_deg", "head_y", "head_state"])
        w.writerows(rows)

    # ---- heuristic swim-safety indicator (NOT a diagnosis) ----
    flags = []
    if body_summary["median_deg"] and body_summary["median_deg"] > 45:
        flags.append("body often vertical (poor float)")
    if dip_freq > 6:
        flags.append("frequent head dips")
    intensities = [s["mean_speed_px_s"] for s in limb_stats.values() if s["mean_speed_px_s"]]
    if intensities and np.mean(intensities) > 400:
        flags.append("high limb effort")
    indicator = "elevated indicators" if len(flags) >= 2 else "no strong indicators"

    # ---- report ----
    report = {
        "video": os.path.basename(video),
        "duration_s": round(float(dur_total), 1),
        "sample_rate_hz": round(fps_eff, 1),
        "water_line_y_px": int(waterline),
        "body_angle_from_horizontal": body_summary,
        "neck_angle_median_deg": neck_median,
        "head_dips": {
            "count": len(dips),
            "per_minute": round(dip_freq, 1),
            "mean_duration_s": round(float(np.mean(dip_durs)), 2) if dip_durs else 0,
            "longest_duration_s": round(float(np.max(dip_durs)), 2) if dip_durs else 0,
            "head_detected_pct": round(head_seen, 1),
            "reliability": head_note,
            "events": dips,
        },
        "limbs": limb_stats,
        "swim_safety_indicator_heuristic": {"result": indicator, "flags": flags,
            "note": "Heuristic only, not a medical or safety diagnosis."},
    }
    with open(os.path.join(out_dir, f"{stem}_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    _write_txt(report, os.path.join(out_dir, f"{stem}_report.txt"))
    _summary_chart(series, waterline, dips, limb_stats, os.path.join(out_dir, f"{stem}_summary.png"))
    print(f"\nWrote {csv_path}, report (.json/.txt), and summary chart.")
    _print_report(report)

    # ---- pass 2: annotated video (segment, or whole with --adur 0) ----
    _render(net, video, out_dir, stem, waterline, band, report, imgsz, conf, astart, adur, fps, W, H)
    return report


def _write_txt(r, path):
    L = []
    L.append(f"SWIM ANALYSIS  -  {r['video']}  ({r['duration_s']}s)")
    b = r["body_angle_from_horizontal"]
    L.append(f"\nBODY ANGLE from horizontal water surface: median {b['median_deg']} deg "
             f"(range {b['min_deg']} to {b['max_deg']}). 0 = flat float, 90 = vertical.")
    L.append(f"NECK angle (median): {r['neck_angle_median_deg']} deg (180 = in line with trunk).")
    d = r["head_dips"]
    L.append(f"\nHEAD DIPS below surface: {d['count']} "
             f"({d['per_minute']}/min). mean {d['mean_duration_s']}s, longest {d['longest_duration_s']}s.")
    L.append(f"    head detected in {d['head_detected_pct']}% of frames - {d['reliability']}.")
    for e in d["events"]:
        L.append(f"    dip at {e['start_s']}s lasting {e['duration_s']}s")
    L.append("\nLIMBS (left/right separate):")
    for k, s in r["limbs"].items():
        L.append(f"    {k:6s} freq {s['moves_per_min']}/min | intensity {s['mean_speed_px_s']} px/s "
                 f"(peak {s['peak_speed_px_s']}) | detected {s['detected_pct']}%")
    h = r["swim_safety_indicator_heuristic"]
    L.append(f"\nHEURISTIC INDICATOR: {h['result']}  {h['flags']}  ({h['note']})")
    open(path, "w").write("\n".join(L) + "\n")


def _print_report(r):
    b = r["body_angle_from_horizontal"]; d = r["head_dips"]
    print("\n================ SUMMARY ================")
    print(f"Body angle: median {b['median_deg']} deg from horizontal (range {b['min_deg']}-{b['max_deg']}).")
    print(f"Head dips: {d['count']} ({d['per_minute']}/min), longest {d['longest_duration_s']}s "
          f"[head seen {d['head_detected_pct']}%].")
    for k, s in r["limbs"].items():
        print(f"  {k:6s} {s['moves_per_min']}/min, {s['mean_speed_px_s']} px/s, seen {s['detected_pct']}%")
    print(f"Indicator (heuristic): {r['swim_safety_indicator_heuristic']['result']}")
    print("========================================")


def _summary_chart(series, waterline, dips, limb_stats, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.array(series["t"])
    fig, ax = plt.subplots(3, 1, figsize=(12, 9), sharex=False)
    ax[0].plot(t, series["body_angle"], color="#1B98B5", lw=0.9)
    ax[0].set_ylabel("body angle (deg)"); ax[0].set_title("Body angle from horizontal (0 = flat float)")
    hy = np.array(series["head_y"], float)
    ax[1].plot(t, hy, color="#6E4CA6", lw=0.9)
    ax[1].axhline(waterline, color="#E07C2E", ls="--", lw=1, label="water line")
    ax[1].invert_yaxis()
    for d in dips:
        ax[1].axvspan(d["start_s"], d["end_s"], color="#FF9AA2", alpha=0.4)
    ax[1].set_ylabel("head height (px)"); ax[1].set_title("Head vs surface (shaded = dip below)"); ax[1].legend()
    # limb speed bars
    names = list(limb_stats); vals = [limb_stats[k]["mean_speed_px_s"] or 0 for k in names]
    ax[2].bar(names, vals, color=["#FF8A80", "#22A6AE", "#FFB020", "#B39DDB"])
    ax[2].set_ylabel("mean speed (px/s)"); ax[2].set_title("Limb intensity (left/right separate)")
    ax[2].set_xlabel("time (s)")
    plt.tight_layout(); plt.savefig(path, dpi=110)


def _render(net, video, out_dir, stem, waterline, band, report, imgsz, conf, astart, adur, fps, W, H):
    cap = cv2.VideoCapture(video)
    if adur == 0:
        astart, n_frames = 0.0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    else:
        cap.set(cv2.CAP_PROP_POS_MSEC, astart * 1000)
        n_frames = int(adur * fps)
    out_path = os.path.join(out_dir, f"{stem}_annotated.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    smoother = pm.Smoother((W**2 + H**2) ** 0.5)
    dips_done = 0
    dip_bounds = [(d["start_s"], d["end_s"]) for d in report["head_dips"]["events"]]
    cur_angle = 0
    print("Rendering annotated video...")
    for n in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        t = astart + n / fps
        if n % pm.REORIENT_EVERY == 0:
            kp, cur_angle = pm.best_orientation(net, frame, imgsz, conf)
        else:
            kp, _ = pm._kp_for_angle(net, frame, cur_angle, imgsz, conf)
        if kp is not None:
            kp = pm.clean(kp, frame); kp = smoother.apply(kp)

        # water line
        cv2.line(frame, (0, waterline), (W, waterline), (0, 165, 255), 2)
        cv2.putText(frame, "water line", (W - 160, waterline - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        if kp is not None:
            pm.draw_colored(frame, kp)

        # live metrics
        sh, hip = mid(kp, L_SH, R_SH), mid(kp, L_HIP, R_HIP)
        hd = head_point(kp) if kp is not None else None
        ba = angle_from_horizontal(hip - sh) if (sh is not None and hip is not None) else None
        nk = angle_between(hd, sh, hip) if (hd is not None and sh is not None and hip is not None) else None
        head_below = hd is not None and hd[1] > waterline
        dips_now = sum(1 for s, e in dip_bounds if e <= t)

        panel = [
            f"Body angle: {ba:.0f} deg from flat" if ba is not None else "Body angle: --",
            f"Head: {'BELOW water' if head_below else 'above water'}   dips so far: {dips_now}",
            f"Neck: {nk:.0f} deg" if nk is not None else "Neck: --",
            f"L arm {report['limbs']['l_arm']['moves_per_min']}/min  R arm {report['limbs']['r_arm']['moves_per_min']}/min",
            f"L leg {report['limbs']['l_leg']['moves_per_min']}/min  R leg {report['limbs']['r_leg']['moves_per_min']}/min",
        ]
        y0 = 30
        cv2.rectangle(frame, (10, 10), (470, 20 + 26 * len(panel)), (0, 0, 0), -1)
        for i, line in enumerate(panel):
            col = (0, 200, 255) if (i == 1 and head_below) else (255, 255, 255)
            cv2.putText(frame, line, (20, y0 + i * 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, col, 2)
        writer.write(frame)
    writer.release(); cap.release()
    print(f"Annotated video: {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--out", default="results")
    ap.add_argument("--skip", type=int, default=3)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--model", default="yolo11n-pose.pt",
                    help="use your fine-tuned best.pt for accurate joints")
    ap.add_argument("--waterline", type=int, default=None, help="override auto water line (y pixels)")
    ap.add_argument("--astart", type=float, default=0.0, help="annotate clip start (s)")
    ap.add_argument("--adur", type=float, default=20.0, help="annotate length (s); 0 = whole video")
    args = ap.parse_args()
    analyze(args.video, args.out, args.skip, args.imgsz, args.conf, args.model,
            args.waterline, args.astart, args.adur)

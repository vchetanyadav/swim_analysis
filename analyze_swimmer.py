"""
analyze_swimmer.py
Side-view swimmer analysis built on the robust pose_markerless.py pipeline.

Outputs:
  1. Body angle relative to the horizontal water surface.
  2. Head position vs water surface, dip events, dip frequency and neck angle.
  3. Left/right arm and leg movement frequency and normalized movement intensity.
  4. Annotated video, per-frame CSV, JSON/TXT report and summary chart.

Important changes from the previous version:
  * no crop/CLAHE pose path by default
  * no whole-frame strict rejection
  * no independent left/right swaps
  * body angle uses a same-side-safe torso axis
  * missing head is reported as unknown, never silently as "above water"
  * cadence only bridges short gaps and uses limb motion relative to the torso
  * supports both COCO-17 and custom SWIM-14 pose weights through pose_markerless
"""

import argparse
import csv
import json
import os

import cv2
import numpy as np
import pandas as pd
from scipy.signal import detrend
from ultralytics import YOLO

import pose_markerless as pm

# Canonical COCO-17 indices. pose_markerless maps SWIM-14 into this layout.
NOSE, L_EAR, R_EAR = 0, 3, 4
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 5, 6, 7, 8, 9, 10
L_HIP, R_HIP, L_KN, R_KN, L_AN, R_AN = 11, 12, 13, 14, 15, 16

# Distal joint relative to proximal joint. This removes whole-body translation.
LIMB_PAIRS = {
    "l_arm": (L_WR, L_SH),
    "r_arm": (R_WR, R_SH),
    "l_leg": (L_AN, L_HIP),
    "r_leg": (R_AN, R_HIP),
}


def vis(kp, i, threshold=0.25):
    return kp is not None and kp[i, 2] >= threshold


def angle_from_horizontal(vec):
    return float(np.degrees(np.arctan2(abs(vec[1]), abs(vec[0]))))


def angle_between(a, b, c):
    ba = a - b
    bc = c - b
    den = np.linalg.norm(ba) * np.linalg.norm(bc)
    if den < 1e-9:
        return np.nan
    cos_angle = np.dot(ba, bc) / den
    return float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))


def detect_waterline(cap, n=25):
    """Estimate a fixed horizontal water line from the strongest mean row edge."""
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(total * 0.10, total * 0.90, n).astype(int)
    profiles = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(float)
        profiles.append(gray.mean(axis=1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    if not profiles:
        return None

    prof = np.mean(profiles, axis=0)
    k = max(5, len(prof) // 60)
    prof_s = np.convolve(prof, np.ones(k) / k, mode="same")
    grad = np.gradient(prof_s)
    upper = int(len(grad) * 0.60)
    return int(np.argmin(grad[:upper]))


def _short_gap_interp(arr, limit):
    return pd.DataFrame(arr).interpolate(limit=limit, limit_area="inside").to_numpy()


def find_dips(t, head_y, waterline, band, gap_limit, min_dur=0.15):
    """Find submersion events with hysteresis and only short-gap interpolation."""
    s = pd.Series(head_y).interpolate(limit=gap_limit, limit_area="inside").to_numpy()
    below = np.zeros(len(s), dtype=bool)
    state = False

    for i, value in enumerate(s):
        if np.isnan(value):
            # Unknown remains unknown. Do not extend a dip through a long missing region.
            state = False
            below[i] = False
            continue
        if not state and value > waterline + band:
            state = True
        elif state and value < waterline - band:
            state = False
        below[i] = state

    dips = []
    i = 0
    while i < len(below):
        if not below[i]:
            i += 1
            continue
        j = i
        while j < len(below) and below[j]:
            j += 1
        duration = t[j - 1] - t[i]
        if duration >= min_dur:
            dips.append({
                "start_s": round(float(t[i]), 2),
                "end_s": round(float(t[j - 1]), 2),
                "duration_s": round(float(duration), 2),
            })
        i = j
    return dips


def _longest_finite_segment(signal):
    finite = np.isfinite(signal)
    if not finite.any():
        return np.array([], dtype=float)
    best_start = best_end = 0
    start = None
    for i, ok in enumerate(finite):
        if ok and start is None:
            start = i
        if (not ok or i == len(finite) - 1) and start is not None:
            end = i if not ok else i + 1
            if end - start > best_end - best_start:
                best_start, best_end = start, end
            start = None
    return signal[best_start:best_end]


def cadence(signal_xy, fps, gap_limit):
    """Dominant movement frequency from the longest reliable contiguous segment."""
    xy = np.asarray(signal_xy, dtype=float)
    if xy.ndim != 2 or len(xy) < 20:
        return None

    xy = _short_gap_interp(xy, gap_limit)
    variances = np.nanvar(xy, axis=0)
    if np.all(~np.isfinite(variances)):
        return None
    axis = xy[:, int(np.nanargmax(variances))]
    segment = _longest_finite_segment(axis)
    if len(segment) < max(40, int(4 * fps)):
        return None

    segment = detrend(segment)
    segment = segment * np.hanning(len(segment))
    freqs = np.fft.rfftfreq(len(segment), d=1.0 / fps)
    power = np.abs(np.fft.rfft(segment)) ** 2
    band = (freqs >= 0.15) & (freqs <= 3.5)
    if not band.any() or float(np.max(power[band])) <= 0:
        return None
    return float(freqs[band][np.argmax(power[band])])


def normalized_speed_series(rel_xy, torso_series, fps):
    """Movement speed in torso-lengths/s instead of px/s."""
    xy = np.asarray(rel_xy, dtype=float)
    torso = np.asarray(torso_series, dtype=float)
    if len(xy) < 2:
        return np.array([], dtype=float)
    d = np.linalg.norm(np.diff(xy, axis=0), axis=1) * fps
    scale = (torso[:-1] + torso[1:]) / 2.0
    valid = np.isfinite(d) & np.isfinite(scale) & (scale > 1e-6)
    out = np.full_like(d, np.nan, dtype=float)
    out[valid] = d[valid] / scale[valid]
    return out


def _pose_for_frame(estimator, tracker, frame):
    kp = estimator.infer(frame)
    kp = pm.clean(kp, frame, drop_head=False, reject_equipment=False)
    return tracker.update(kp)


def analyze(video, out_dir, skip=2, imgsz=1280, det_conf=0.15,
            kpt_conf=0.25, model="yolo11x-pose.pt", orientation="auto",
            device=None, waterline=None, astart=0.0, adur=20.0,
            draw_threshold=0.30):
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
        if waterline is None:
            raise SystemExit("Water line could not be detected. Use --waterline <y_pixels>.")
        print(f"Auto-detected water line at y={waterline}px. Verify it in the annotated video.")

    band = max(3, int(0.012 * H))
    gap_limit = max(1, int(round(0.30 * fps_eff)))

    estimator = pm.PoseEstimator(
        net, imgsz=imgsz, det_conf=det_conf,
        orientation=orientation, device=device,
        relock_failures=3, recheck_every=0,
    )
    tracker = pm.SkeletonTracker(kpt_conf=kpt_conf)

    print(
        "Analysing full clip...\n"
        f"  model={model}\n"
        f"  imgsz={imgsz}, skip={skip} -> {fps_eff:.2f} pose samples/s\n"
        f"  orientation={orientation}, det_conf={det_conf}, kpt_conf={kpt_conf}"
    )

    rows = []
    series = {
        "t": [], "body_angle": [], "neck_angle": [],
        "head_y": [], "torso": [],
    }
    limb_rel = {name: [] for name in LIMB_PAIRS}
    joint_seen = {i: 0 for i in range(17)}

    idx = 0
    processed = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % skip:
            idx += 1
            continue

        kp = _pose_for_frame(estimator, tracker, frame)
        t = idx / fps

        sh, hp = pm.shoulder_hip_points(kp, kpt_conf)
        hd = pm.head_point(kp, max(0.18, kpt_conf - 0.05))
        torso = float(np.linalg.norm(sh - hp)) if sh is not None and hp is not None else np.nan
        body_angle = angle_from_horizontal(hp - sh) if sh is not None and hp is not None else np.nan
        neck_angle = angle_between(hd, sh, hp) if hd is not None and sh is not None and hp is not None else np.nan
        head_y = float(hd[1]) if hd is not None else np.nan

        series["t"].append(t)
        series["body_angle"].append(body_angle)
        series["neck_angle"].append(neck_angle)
        series["head_y"].append(head_y)
        series["torso"].append(torso)

        if kp is not None:
            for i in range(17):
                if kp[i, 2] >= kpt_conf:
                    joint_seen[i] += 1

        for name, (distal, proximal) in LIMB_PAIRS.items():
            if kp is not None and vis(kp, distal, kpt_conf) and vis(kp, proximal, kpt_conf):
                limb_rel[name].append(kp[distal, :2] - kp[proximal, :2])
            else:
                limb_rel[name].append([np.nan, np.nan])

        if np.isnan(head_y):
            head_state = "unknown"
        else:
            head_state = "below" if head_y > waterline else "above"

        rows.append([
            idx,
            round(t, 3),
            round(body_angle, 1) if np.isfinite(body_angle) else "",
            round(neck_angle, 1) if np.isfinite(neck_angle) else "",
            round(head_y, 1) if np.isfinite(head_y) else "",
            head_state,
            estimator.current_angle if estimator.current_angle is not None else "",
        ])

        processed += 1
        idx += 1
        if processed % 200 == 0:
            print(f"  {processed} pose frames (t={t:.0f}s)", flush=True)

    cap.release()

    t_arr = np.asarray(series["t"], dtype=float)
    if len(t_arr) == 0:
        raise SystemExit("No frames were processed.")

    dips = find_dips(
        t_arr,
        np.asarray(series["head_y"], dtype=float),
        waterline,
        band,
        gap_limit,
    )
    duration = t_arr[-1] - t_arr[0] if len(t_arr) > 1 else 1.0
    dip_freq = len(dips) / max(duration / 60.0, 1e-6)
    dip_durations = [d["duration_s"] for d in dips]

    torso_arr = np.asarray(series["torso"], dtype=float)
    torso_interp = pd.Series(torso_arr).interpolate(
        limit=gap_limit, limit_area="inside"
    ).to_numpy()

    limb_stats = {}
    for name in LIMB_PAIRS:
        rel = np.asarray(limb_rel[name], dtype=float)
        freq = cadence(rel, fps_eff, gap_limit)
        rel_i = _short_gap_interp(rel, gap_limit)
        speeds = normalized_speed_series(rel_i, torso_interp, fps_eff)
        speeds = speeds[np.isfinite(speeds)]
        seen = 100.0 * float(np.mean(np.isfinite(rel[:, 0]))) if len(rel) else 0.0
        limb_stats[name] = {
            "freq_hz": round(freq, 3) if freq is not None else None,
            "moves_per_min": round(freq * 60.0, 1) if freq is not None else None,
            "intensity_bodylen_s": round(float(np.median(speeds)), 3) if speeds.size else None,
            "peak_bodylen_s": round(float(np.percentile(speeds, 95)), 3) if speeds.size else None,
            "detected_pct": round(seen, 1),
        }

    body_values = np.asarray(series["body_angle"], dtype=float)
    body_values = body_values[np.isfinite(body_values)]
    body_summary = {
        "median_deg": round(float(np.median(body_values)), 1) if body_values.size else None,
        "p05_deg": round(float(np.percentile(body_values, 5)), 1) if body_values.size else None,
        "p95_deg": round(float(np.percentile(body_values, 95)), 1) if body_values.size else None,
        "detected_pct": round(100.0 * len(body_values) / len(t_arr), 1),
    }

    neck_values = np.asarray(series["neck_angle"], dtype=float)
    neck_values = neck_values[np.isfinite(neck_values)]
    neck_median = round(float(np.median(neck_values)), 1) if neck_values.size else None

    head_arr = np.asarray(series["head_y"], dtype=float)
    head_seen = 100.0 * float(np.mean(np.isfinite(head_arr)))
    head_reliability = (
        "reliable" if head_seen >= 70 else
        "moderate" if head_seen >= 45 else
        "low: do not interpret dip count without manual review"
    )

    joint_detection = {
        pm.KEYPOINTS[i]: round(100.0 * joint_seen[i] / max(processed, 1), 1)
        for i in range(17)
    }

    csv_path = os.path.join(out_dir, f"{stem}_frames.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "frame", "t", "body_angle_deg", "neck_angle_deg",
            "head_y", "head_state", "orientation_deg",
        ])
        w.writerows(rows)

    report = {
        "video": os.path.basename(video),
        "duration_s": round(float(duration), 1),
        "sample_rate_hz": round(float(fps_eff), 2),
        "model": model,
        "imgsz": imgsz,
        "det_conf": det_conf,
        "kpt_conf": kpt_conf,
        "orientation_setting": orientation,
        "water_line_y_px": int(waterline),
        "body_angle_from_horizontal": body_summary,
        "neck_angle_median_deg": neck_median,
        "head_dips": {
            "count": len(dips),
            "per_minute": round(float(dip_freq), 2),
            "mean_duration_s": round(float(np.mean(dip_durations)), 2) if dip_durations else 0.0,
            "longest_duration_s": round(float(np.max(dip_durations)), 2) if dip_durations else 0.0,
            "head_detected_pct": round(head_seen, 1),
            "reliability": head_reliability,
            "events": dips,
        },
        "limbs": limb_stats,
        "joint_detection_pct": joint_detection,
    }

    json_path = os.path.join(out_dir, f"{stem}_report.json")
    txt_path = os.path.join(out_dir, f"{stem}_report.txt")
    chart_path = os.path.join(out_dir, f"{stem}_summary.png")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    _write_txt(report, txt_path)
    _summary_chart(series, waterline, dips, limb_stats, chart_path)
    _print_report(report)

    print(f"\nWrote:\n  {csv_path}\n  {json_path}\n  {txt_path}\n  {chart_path}")

    _render(
        net, video, out_dir, stem, waterline, report,
        imgsz, det_conf, kpt_conf, orientation, device,
        astart, adur, fps, W, H, draw_threshold,
    )
    return report


def _write_txt(report, path):
    b = report["body_angle_from_horizontal"]
    d = report["head_dips"]
    lines = [
        f"SWIM ANALYSIS - {report['video']} ({report['duration_s']}s)",
        f"MODEL: {report['model']} | imgsz={report['imgsz']} | orientation={report['orientation_setting']}",
        "",
        f"BODY ANGLE: median {b['median_deg']} deg from horizontal "
        f"(5th to 95th percentile {b['p05_deg']} to {b['p95_deg']}), "
        f"available in {b['detected_pct']}% of analysed frames.",
        f"NECK ANGLE median: {report['neck_angle_median_deg']} deg.",
        "",
        f"HEAD DIPS: {d['count']} ({d['per_minute']}/min). "
        f"Longest {d['longest_duration_s']}s. Head detected {d['head_detected_pct']}%. "
        f"Reliability: {d['reliability']}.",
        "",
        "LIMBS:",
    ]
    for name, s in report["limbs"].items():
        lines.append(
            f"  {name:6s}: {s['moves_per_min']} moves/min | "
            f"median intensity {s['intensity_bodylen_s']} torso-length/s | "
            f"detected {s['detected_pct']}%"
        )
    lines.append("")
    lines.append("JOINT DETECTION (%):")
    for name, pct in report["joint_detection_pct"].items():
        lines.append(f"  {name:12s}: {pct}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _print_report(report):
    b = report["body_angle_from_horizontal"]
    d = report["head_dips"]
    print("\n================ SUMMARY ================")
    print(
        f"Body angle: median {b['median_deg']} deg, "
        f"available in {b['detected_pct']}% of analysed frames."
    )
    print(
        f"Head dips: {d['count']} ({d['per_minute']}/min), "
        f"head detected {d['head_detected_pct']}% [{d['reliability']}]."
    )
    for name, s in report["limbs"].items():
        print(
            f"  {name:6s}: {s['moves_per_min']} moves/min, "
            f"{s['intensity_bodylen_s']} torso-len/s, seen {s['detected_pct']}%"
        )
    print("========================================")


def _summary_chart(series, waterline, dips, limb_stats, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.asarray(series["t"], dtype=float)
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=False)

    axes[0].plot(t, series["body_angle"], lw=0.9)
    axes[0].set_ylabel("body angle (deg)")
    axes[0].set_title("Body angle from horizontal")

    axes[1].plot(t, series["head_y"], lw=0.9)
    axes[1].axhline(waterline, ls="--", lw=1, label="water line")
    axes[1].invert_yaxis()
    for dip in dips:
        axes[1].axvspan(dip["start_s"], dip["end_s"], alpha=0.25)
    axes[1].set_ylabel("head y (px)")
    axes[1].set_title("Head position vs water surface")
    axes[1].legend()

    names = list(limb_stats)
    values = [limb_stats[k]["intensity_bodylen_s"] or 0 for k in names]
    axes[2].bar(names, values)
    axes[2].set_ylabel("torso-length/s")
    axes[2].set_title("Median limb movement intensity")

    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close(fig)


def _render(net, video, out_dir, stem, waterline, report,
            imgsz, det_conf, kpt_conf, orientation, device,
            astart, adur, fps, W, H, draw_threshold):
    cap = cv2.VideoCapture(video)
    if adur == 0:
        astart = 0.0
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    else:
        cap.set(cv2.CAP_PROP_POS_MSEC, astart * 1000.0)
        n_frames = int(adur * fps)

    out_path = os.path.join(out_dir, f"{stem}_annotated.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    estimator = pm.PoseEstimator(
        net, imgsz=imgsz, det_conf=det_conf,
        orientation=orientation, device=device,
        relock_failures=3, recheck_every=0,
    )
    tracker = pm.SkeletonTracker(kpt_conf=kpt_conf)
    dip_bounds = [(d["start_s"], d["end_s"]) for d in report["head_dips"]["events"]]

    print("Rendering annotated video...")
    for n in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        t = astart + n / fps

        kp = _pose_for_frame(estimator, tracker, frame)
        pm.draw_colored(
            frame, kp, threshold=max(draw_threshold, kpt_conf),
            draw_head=True, legend=False,
        )

        cv2.line(frame, (0, waterline), (W, waterline), (0, 165, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "water line", (max(10, W - 150), max(20, waterline - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2, cv2.LINE_AA)

        sh, hp = pm.shoulder_hip_points(kp, kpt_conf)
        hd = pm.head_point(kp, max(0.18, kpt_conf - 0.05))
        body_angle = angle_from_horizontal(hp - sh) if sh is not None and hp is not None else None
        neck_angle = angle_between(hd, sh, hp) if hd is not None and sh is not None and hp is not None else None

        if hd is None:
            head_text = "Head: --"
            head_below = False
        else:
            head_below = bool(hd[1] > waterline)
            head_text = "Head: BELOW water" if head_below else "Head: above water"

        dips_done = sum(1 for _, end in dip_bounds if end <= t)
        panel = [
            f"Body angle: {body_angle:.0f} deg" if body_angle is not None else "Body angle: --",
            f"{head_text}   dips: {dips_done}",
            f"Neck: {neck_angle:.0f} deg" if neck_angle is not None and np.isfinite(neck_angle) else "Neck: --",
            f"L arm {report['limbs']['l_arm']['moves_per_min']} /min   R arm {report['limbs']['r_arm']['moves_per_min']} /min",
            f"L leg {report['limbs']['l_leg']['moves_per_min']} /min   R leg {report['limbs']['r_leg']['moves_per_min']} /min",
            f"Pose rotation: {estimator.current_angle if estimator.current_angle is not None else '--'} deg",
        ]

        box_w = 520
        box_h = 20 + 25 * len(panel)
        cv2.rectangle(frame, (8, 8), (box_w, box_h), (0, 0, 0), -1)
        for i, text in enumerate(panel):
            colour = (0, 200, 255) if (i == 1 and head_below) else (255, 255, 255)
            cv2.putText(frame, text, (18, 30 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, colour, 2, cv2.LINE_AA)

        writer.write(frame)

    writer.release()
    cap.release()
    print(f"Annotated video: {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Analyse side-view swimming kinematics.")
    ap.add_argument("video")
    ap.add_argument("--out", default="results")
    ap.add_argument("--skip", type=int, default=2)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--det-conf", type=float, default=0.15)
    ap.add_argument("--kpt-conf", type=float, default=0.25)
    ap.add_argument("--model", default="yolo11x-pose.pt")
    ap.add_argument("--orientation", default="auto", choices=["auto", "0", "90", "180", "270"])
    ap.add_argument("--device", default=None, help="e.g. 0 for first CUDA GPU; omit for auto")
    ap.add_argument("--waterline", type=int, default=None)
    ap.add_argument("--astart", type=float, default=0.0)
    ap.add_argument("--adur", type=float, default=20.0, help="0 = annotate whole video")
    ap.add_argument("--draw-threshold", type=float, default=0.30)
    args = ap.parse_args()

    analyze(
        args.video, args.out, args.skip, args.imgsz,
        args.det_conf, args.kpt_conf, args.model,
        args.orientation, args.device, args.waterline,
        args.astart, args.adur, args.draw_threshold,
    )

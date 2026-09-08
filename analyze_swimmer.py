"""
analyze_swimmer.py
Accuracy-first side-view swimming analysis.

Key change in this version:
The pose model runs once, the complete pose sequence is refined using both past
and future frames, and the annotated video is rendered from that SAME refined
sequence. The old pipeline ran a second independent inference pass for the
video, so the displayed skeleton could disagree with the measurements.
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

NOSE, L_EAR, R_EAR = 0, 3, 4
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 5, 6, 7, 8, 9, 10
L_HIP, R_HIP, L_KN, R_KN, L_AN, R_AN = 11, 12, 13, 14, 15, 16

LIMB_PAIRS = {
    "l_arm": (L_WR, L_SH),
    "r_arm": (R_WR, R_SH),
    "l_leg": (L_AN, L_HIP),
    "r_leg": (R_AN, R_HIP),
}


def vis(kp, i, threshold=0.22):
    return kp is not None and np.isfinite(kp[i, 0]) and kp[i, 2] >= threshold


def angle_from_horizontal(vec):
    return float(np.degrees(np.arctan2(abs(vec[1]), abs(vec[0]))))


def angle_between(a, b, c):
    ba, bc = a - b, c - b
    den = np.linalg.norm(ba) * np.linalg.norm(bc)
    if den < 1e-9:
        return np.nan
    return float(np.degrees(np.arccos(np.clip(np.dot(ba, bc) / den, -1.0, 1.0))))


def detect_waterline(cap, n=30):
    """Fixed horizontal waterline estimate for the current controlled side view."""
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(total * 0.08, total * 0.92, n).astype(int)
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
    p = np.mean(profiles, axis=0)
    k = max(5, len(p) // 60)
    p = np.convolve(p, np.ones(k) / k, mode="same")
    grad = np.gradient(p)
    upper = int(len(grad) * 0.55)
    return int(np.argmin(grad[:upper]))


def _short_interp(arr, limit):
    return pd.DataFrame(arr).interpolate(limit=limit, limit_area="inside").to_numpy()


def find_dips(t, head_y, waterline, band, gap_limit, min_dur=0.15):
    s = pd.Series(head_y).interpolate(limit=gap_limit, limit_area="inside").to_numpy()
    below = np.zeros(len(s), bool)
    state = False
    for i, v in enumerate(s):
        if not np.isfinite(v):
            state = False
            continue
        if not state and v > waterline + band:
            state = True
        elif state and v < waterline - band:
            state = False
        below[i] = state

    events = []
    i = 0
    while i < len(below):
        if not below[i]:
            i += 1
            continue
        j = i
        while j < len(below) and below[j]:
            j += 1
        dur = float(t[j - 1] - t[i])
        if dur >= min_dur:
            events.append({
                "start_s": round(float(t[i]), 2),
                "end_s": round(float(t[j - 1]), 2),
                "duration_s": round(dur, 2),
            })
        i = j
    return events


def _longest_finite_segment(x):
    valid = np.isfinite(x)
    best = np.array([], float)
    for s, e in _segments(valid):
        if e - s > len(best):
            best = x[s:e]
    return best


def _segments(valid):
    out, start = [], None
    for i, ok in enumerate(valid):
        if ok and start is None:
            start = i
        if start is not None and ((not ok) or i == len(valid) - 1):
            end = i if not ok else i + 1
            out.append((start, end))
            start = None
    return out


def cadence(signal_xy, fps, gap_limit):
    xy = np.asarray(signal_xy, float)
    if xy.ndim != 2 or len(xy) < 20:
        return None
    xy = _short_interp(xy, gap_limit)
    var = np.nanvar(xy, axis=0)
    if not np.isfinite(var).any():
        return None
    sig = xy[:, int(np.nanargmax(var))]
    sig = _longest_finite_segment(sig)
    if len(sig) < max(40, int(round(4 * fps))):
        return None
    sig = detrend(sig)
    sig = sig * np.hanning(len(sig))
    f = np.fft.rfftfreq(len(sig), d=1.0 / fps)
    p = np.abs(np.fft.rfft(sig)) ** 2
    band = (f >= 0.15) & (f <= 3.5)
    if not band.any() or np.max(p[band]) <= 0:
        return None
    return float(f[band][np.argmax(p[band])])


def normalized_speed(rel_xy, torso, fps):
    xy = np.asarray(rel_xy, float)
    torso = np.asarray(torso, float)
    if len(xy) < 2:
        return np.array([], float)
    d = np.linalg.norm(np.diff(xy, axis=0), axis=1) * fps
    scale = 0.5 * (torso[:-1] + torso[1:])
    out = np.full(len(d), np.nan, float)
    valid = np.isfinite(d) & np.isfinite(scale) & (scale > 1e-6)
    out[valid] = d[valid] / scale[valid]
    return out


def metrics_from_pose(kp, kpt_conf):
    sh, hp = pm.shoulder_hip_points(kp, max(0.18, kpt_conf - 0.04))
    head = pm.head_point(kp, max(0.17, kpt_conf - 0.05))
    torso = float(np.linalg.norm(sh - hp)) if sh is not None and hp is not None else np.nan
    body = angle_from_horizontal(hp - sh) if sh is not None and hp is not None else np.nan
    neck = angle_between(head, sh, hp) if head is not None and sh is not None and hp is not None else np.nan
    head_y = float(head[1]) if head is not None else np.nan
    return sh, hp, head, torso, body, neck, head_y


def analyze(video, out_dir, skip=2, imgsz=1280, det_conf=0.12,
            kpt_conf=0.22, model="yolo11x-pose.pt", orientation="auto",
            device=None, tta=False, waterline=None, astart=0.0, adur=20.0,
            draw_threshold=0.28, max_gap_s=0.28):
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(video))[0]
    net = YOLO(model)

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_eff = fps / skip

    if waterline is None:
        waterline = detect_waterline(cap)
        if waterline is None:
            raise SystemExit("Could not estimate waterline. Pass --waterline <y>.")
        print(f"Auto waterline: y={waterline}px")

    estimator = pm.PoseEstimator(
        net, imgsz=imgsz, det_conf=det_conf, orientation=orientation,
        device=device, relock_failures=4, recheck_every=0, tta=tta,
    )
    tracker = pm.SkeletonTracker(kpt_conf=kpt_conf)

    print(
        "\nPASS 1: pose extraction\n"
        f"  model={model}\n"
        f"  video={W}x{H}, {fps:.2f} fps, {total} frames\n"
        f"  sample every {skip} frame(s) -> {fps_eff:.2f} Hz\n"
        f"  orientation={orientation}, imgsz={imgsz}\n"
        f"  flip TTA={'ON' if tta else 'OFF'}\n"
    )

    raw_sequence = []
    sample_indices = []
    orientations = []
    raw_joint_seen = np.zeros(17, int)

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % skip == 0:
            kp = estimator.infer(frame)
            kp = pm.clean(kp, frame, drop_head=False, reject_equipment=False)
            kp = tracker.update(kp)
            raw_sequence.append(kp)
            sample_indices.append(idx)
            orientations.append(estimator.current_angle)
            if kp is not None:
                for j in range(17):
                    if kp[j, 2] >= kpt_conf:
                        raw_joint_seen[j] += 1
            if len(raw_sequence) % 200 == 0:
                print(f"  {len(raw_sequence)} pose samples, t={idx / fps:.1f}s", flush=True)
        idx += 1
    cap.release()

    print("\nPASS 2: whole-sequence anatomical + temporal refinement")
    refined, refine_stats = pm.refine_sequence(
        raw_sequence, fps_eff, kpt_conf=kpt_conf,
        max_gap_s=max_gap_s, smooth=True, stabilize_lr=True,
    )
    print(
        f"  temporal outliers removed: {refine_stats['temporal_outliers_removed']}\n"
        f"  short-gap points recovered: {refine_stats['interpolated_points']}\n"
        f"  interpolation limit: {refine_stats['max_gap_frames']} sample frames"
    )

    if not refined:
        raise SystemExit("No pose samples produced.")

    # Build all time series from the refined sequence.
    times = np.asarray(sample_indices, float) / fps
    series = {"t": times.tolist(), "body_angle": [], "neck_angle": [], "head_y": [], "torso": []}
    limb_rel = {k: [] for k in LIMB_PAIRS}
    refined_joint_seen = np.zeros(17, int)
    rows = []

    for fi, t, kp, ori in zip(sample_indices, times, refined, orientations):
        sh, hp, head, torso, body, neck, head_y = metrics_from_pose(kp, kpt_conf)
        series["body_angle"].append(body)
        series["neck_angle"].append(neck)
        series["head_y"].append(head_y)
        series["torso"].append(torso)

        for j in range(17):
            if kp[j, 2] >= kpt_conf:
                refined_joint_seen[j] += 1

        for name, (distal, proximal) in LIMB_PAIRS.items():
            if vis(kp, distal, kpt_conf) and vis(kp, proximal, kpt_conf):
                limb_rel[name].append(kp[distal, :2] - kp[proximal, :2])
            else:
                limb_rel[name].append([np.nan, np.nan])

        state = "unknown" if not np.isfinite(head_y) else ("below" if head_y > waterline else "above")
        rows.append([
            fi, round(float(t), 3),
            round(float(body), 1) if np.isfinite(body) else "",
            round(float(neck), 1) if np.isfinite(neck) else "",
            round(float(head_y), 1) if np.isfinite(head_y) else "",
            state, ori if ori is not None else "",
        ])

    gap_limit = max(1, int(round(max_gap_s * fps_eff)))
    band_px = max(3, int(round(0.012 * H)))
    dips = find_dips(times, np.asarray(series["head_y"], float), waterline, band_px, gap_limit)
    duration = max(float(times[-1] - times[0]), 1e-6)
    dip_freq = len(dips) / (duration / 60.0)
    dip_durs = [x["duration_s"] for x in dips]

    torso_arr = np.asarray(series["torso"], float)
    torso_i = pd.Series(torso_arr).interpolate(limit=gap_limit, limit_area="inside").to_numpy()

    limb_stats = {}
    for name in LIMB_PAIRS:
        rel = np.asarray(limb_rel[name], float)
        freq = cadence(rel, fps_eff, gap_limit)
        rel_i = _short_interp(rel, gap_limit)
        speed = normalized_speed(rel_i, torso_i, fps_eff)
        speed = speed[np.isfinite(speed)]
        limb_stats[name] = {
            "freq_hz": round(freq, 3) if freq is not None else None,
            "moves_per_min": round(freq * 60.0, 1) if freq is not None else None,
            "median_intensity_bodylen_s": round(float(np.median(speed)), 3) if speed.size else None,
            "p95_intensity_bodylen_s": round(float(np.percentile(speed, 95)), 3) if speed.size else None,
            "detected_pct": round(100.0 * float(np.mean(np.isfinite(rel[:, 0]))), 1),
        }

    body_arr = np.asarray(series["body_angle"], float)
    body_valid = body_arr[np.isfinite(body_arr)]
    body_summary = {
        "median_deg": round(float(np.median(body_valid)), 1) if body_valid.size else None,
        "p05_deg": round(float(np.percentile(body_valid, 5)), 1) if body_valid.size else None,
        "p95_deg": round(float(np.percentile(body_valid, 95)), 1) if body_valid.size else None,
        "detected_pct": round(100.0 * len(body_valid) / len(refined), 1),
    }

    neck_arr = np.asarray(series["neck_angle"], float)
    neck_valid = neck_arr[np.isfinite(neck_arr)]
    head_arr = np.asarray(series["head_y"], float)
    head_seen = 100.0 * float(np.mean(np.isfinite(head_arr)))

    raw_detection = {pm.KEYPOINTS[j]: round(100.0 * raw_joint_seen[j] / len(refined), 1) for j in range(17)}
    refined_detection = {pm.KEYPOINTS[j]: round(100.0 * refined_joint_seen[j] / len(refined), 1) for j in range(17)}

    valid_ori = [x for x in orientations if x is not None]
    dominant_orientation = max(set(valid_ori), key=valid_ori.count) if valid_ori else None

    report = {
        "video": os.path.basename(video),
        "duration_s": round(duration, 1),
        "model": model,
        "imgsz": imgsz,
        "skip": skip,
        "effective_pose_hz": round(fps_eff, 2),
        "orientation_setting": orientation,
        "dominant_orientation_deg": dominant_orientation,
        "flip_tta": bool(tta),
        "water_line_y_px": int(waterline),
        "refinement": refine_stats,
        "body_angle_from_horizontal": body_summary,
        "neck_angle_median_deg": round(float(np.median(neck_valid)), 1) if neck_valid.size else None,
        "head_dips": {
            "count": len(dips),
            "per_minute": round(float(dip_freq), 2),
            "mean_duration_s": round(float(np.mean(dip_durs)), 2) if dip_durs else 0.0,
            "longest_duration_s": round(float(np.max(dip_durs)), 2) if dip_durs else 0.0,
            "head_detected_pct": round(head_seen, 1),
            "events": dips,
        },
        "limbs": limb_stats,
        "raw_joint_detection_pct": raw_detection,
        "refined_joint_coverage_pct": refined_detection,
    }

    csv_path = os.path.join(out_dir, f"{stem}_frames.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "t", "body_angle_deg", "neck_angle_deg", "head_y", "head_state", "orientation_deg"])
        w.writerows(rows)

    json_path = os.path.join(out_dir, f"{stem}_report.json")
    txt_path = os.path.join(out_dir, f"{stem}_report.txt")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    _write_txt(report, txt_path)
    _print_report(report)

    print("\nPASS 3: render annotated video from the refined sequence (no second AI inference)")
    out_video = _render_refined(
        video, out_dir, stem, waterline, report,
        sample_indices, refined, kpt_conf, draw_threshold,
        astart, adur, fps, W, H,
    )

    print(f"\nOutputs:\n  {csv_path}\n  {json_path}\n  {txt_path}\n  {out_video}")
    return report


def _write_txt(r, path):
    b, d = r["body_angle_from_horizontal"], r["head_dips"]
    lines = [
        f"SWIM ANALYSIS - {r['video']}",
        f"Model: {r['model']} | imgsz={r['imgsz']} | TTA={r['flip_tta']}",
        f"Orientation: {r['orientation_setting']} | dominant={r['dominant_orientation_deg']}",
        "",
        f"Body angle: median {b['median_deg']} deg, p05-p95 {b['p05_deg']} to {b['p95_deg']}, available {b['detected_pct']}%.",
        f"Neck angle median: {r['neck_angle_median_deg']} deg.",
        f"Head dips: {d['count']} ({d['per_minute']}/min), head detected {d['head_detected_pct']}%.",
        "",
        "Limb metrics:",
    ]
    for k, s in r["limbs"].items():
        lines.append(
            f"  {k}: {s['moves_per_min']} moves/min | median intensity "
            f"{s['median_intensity_bodylen_s']} body-len/s | detected {s['detected_pct']}%"
        )
    lines += [
        "",
        f"Temporal outliers removed: {r['refinement']['temporal_outliers_removed']}",
        f"Short-gap points recovered: {r['refinement']['interpolated_points']}",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _print_report(r):
    b, d = r["body_angle_from_horizontal"], r["head_dips"]
    print("\n================ SUMMARY ================")
    print(f"Body angle: {b['median_deg']} deg median, available {b['detected_pct']}%")
    print(f"Head: detected {d['head_detected_pct']}%, dips {d['count']} ({d['per_minute']}/min)")
    for k, s in r["limbs"].items():
        print(f"{k:6s}: {s['moves_per_min']} /min, detected {s['detected_pct']}%")
    print("========================================")


def _render_refined(video, out_dir, stem, waterline, report,
                    sample_indices, refined, kpt_conf, draw_threshold,
                    astart, adur, fps, W, H):
    cap = cv2.VideoCapture(video)
    start_frame = max(0, int(round(astart * fps)))
    if adur == 0:
        end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    else:
        end_frame = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), start_frame + int(round(adur * fps)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    out_path = os.path.join(out_dir, f"{stem}_annotated_refined.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    dip_bounds = [(x["start_s"], x["end_s"]) for x in report["head_dips"]["events"]]

    for fi in range(start_frame, end_frame):
        ok, frame = cap.read()
        if not ok:
            break
        kp = pm.interpolate_pose(sample_indices, refined, fi)
        pm.draw_colored(frame, kp, threshold=draw_threshold, draw_head=True, legend=False)

        cv2.line(frame, (0, waterline), (W, waterline), (0, 165, 255), 2)
        cv2.putText(frame, "water line", (max(10, W - 145), max(18, waterline - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 165, 255), 2, cv2.LINE_AA)

        sh, hp, head, torso, body, neck, head_y = metrics_from_pose(kp, kpt_conf)
        t = fi / fps
        if np.isfinite(head_y):
            below = head_y > waterline
            head_text = "Head: BELOW water" if below else "Head: above water"
        else:
            below = False
            head_text = "Head: --"
        dips_done = sum(1 for _, e in dip_bounds if e <= t)

        panel = [
            f"Body angle: {body:.0f} deg" if np.isfinite(body) else "Body angle: --",
            f"{head_text}   dips: {dips_done}",
            f"Neck: {neck:.0f} deg" if np.isfinite(neck) else "Neck: --",
            f"L arm {report['limbs']['l_arm']['moves_per_min']} /min   R arm {report['limbs']['r_arm']['moves_per_min']} /min",
            f"L leg {report['limbs']['l_leg']['moves_per_min']} /min   R leg {report['limbs']['r_leg']['moves_per_min']} /min",
            "refined temporal pose",
        ]
        box_w = min(W - 5, 520)
        box_h = 20 + 25 * len(panel)
        cv2.rectangle(frame, (5, 5), (box_w, box_h), (0, 0, 0), -1)
        for i, text in enumerate(panel):
            colour = (0, 200, 255) if (i == 1 and below) else (255, 255, 255)
            cv2.putText(frame, text, (15, 28 + i * 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.56, colour, 2, cv2.LINE_AA)
        writer.write(frame)

    writer.release()
    cap.release()
    print(f"Annotated video: {out_path}")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Accuracy-first side-view swimmer analysis.")
    ap.add_argument("video")
    ap.add_argument("--out", default="results_v3")
    ap.add_argument("--skip", type=int, default=2)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--det-conf", type=float, default=0.12)
    ap.add_argument("--kpt-conf", type=float, default=0.22)
    ap.add_argument("--model", default="yolo11x-pose.pt")
    ap.add_argument("--orientation", default="auto", choices=["auto", "0", "90", "180", "270"])
    ap.add_argument("--device", default=None, help="example: 0 for first CUDA GPU")
    ap.add_argument("--tta", action="store_true", help="flip test-time augmentation; about 2x slower")
    ap.add_argument("--waterline", type=int, default=None)
    ap.add_argument("--astart", type=float, default=0.0)
    ap.add_argument("--adur", type=float, default=20.0, help="0 = annotate complete video")
    ap.add_argument("--draw-threshold", type=float, default=0.28)
    ap.add_argument("--max-gap-s", type=float, default=0.28)
    a = ap.parse_args()

    analyze(
        a.video, a.out, a.skip, a.imgsz, a.det_conf, a.kpt_conf,
        a.model, a.orientation, a.device, a.tta, a.waterline,
        a.astart, a.adur, a.draw_threshold, a.max_gap_s,
    )

"""
kinematics.py
Step 2 of the pipeline: turn the joint table into swimming metrics.

This is the layer that matters most for the project. The pose model underneath
is replaceable; the value is in how these numbers are derived and how occlusion
is handled. Everything here reads from the CSV produced by extract_pose.py, so
you can re-run and tweak the metrics without touching the (slow) pose step.

What it computes:
  1. Detection rate       - how often a usable full-body skeleton was found.
  2. Body alignment        - orientation of the shoulder-to-hip line over time.
  3. Joint angles          - elbow and knee flexion (left and right separately).
  4. Limb cadence          - how fast each limb moves, per limb, via a frequency
                             analysis, with left/right and arm/leg kept separate.

Important note on views:
  From a TOP-DOWN camera you see the body in the horizontal plane, so "body
  alignment" here is how straight and steady the body line is (yaw), not the
  pitch of the body relative to the water surface. Pitch relative to the water
  surface, and head-above-water, need a SIDE view. Cadence, symmetry and
  alignment are exactly what the top view is good for.

Occlusion / bubble handling:
  Any joint the model was not confident about (confidence below CONF_MIN) is
  dropped rather than trusted. When bubbles or splash hide a limb, its
  confidence falls and that reading is ignored, then bridged by short
  interpolation. This is the simple, robust core of occlusion handling. In a
  multi-camera setup this is also where you would fill a missing joint from the
  view that can still see it (see README).

Usage:
    python kinematics.py pose_data.csv
    python kinematics.py pose_data.csv --plot metrics.png
"""
import argparse
import numpy as np
import pandas as pd
from scipy.signal import detrend

CONF_MIN = 0.5          # joints below this confidence are treated as unseen
CADENCE_BAND = (0.2, 4.0)  # plausible limb movement range, in Hz

CORE = ["l_shoulder", "r_shoulder", "l_hip", "r_hip",
        "l_knee", "r_knee", "l_ankle", "r_ankle"]


def load(csv_path):
    df = pd.read_csv(csv_path)
    fps_eff = 1.0 / np.median(np.diff(df["t"]))
    return df, fps_eff


def joint(df, name):
    """Return x, y arrays for a joint with low-confidence points set to NaN."""
    x = df[f"{name}_x"].to_numpy(dtype=float, copy=True)
    y = df[f"{name}_y"].to_numpy(dtype=float, copy=True)
    c = df[f"{name}_c"].to_numpy(dtype=float, copy=True)
    x[c < CONF_MIN] = np.nan
    y[c < CONF_MIN] = np.nan
    return x, y


def midpoint(df, a, b):
    xa, ya = joint(df, a)
    xb, yb = joint(df, b)
    return np.column_stack([(xa + xb) / 2, (ya + yb) / 2])


def angle_at(a, b, c):
    """Angle in degrees at point b, formed by points a-b-c (all N x 2 arrays)."""
    ba = a - b
    bc = c - b
    cos = np.sum(ba * bc, axis=1) / (
        np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def detection_rate(df):
    conf = np.column_stack([df[f"{j}_c"].to_numpy() for j in CORE])
    frame_ok = (conf > CONF_MIN).mean(axis=1)         # fraction of core joints per frame
    usable = (frame_ok >= 0.75).mean()                # frames with >=75% of core joints
    per_joint = {j: (df[f"{j}_c"] > CONF_MIN).mean() for j in CORE}
    return usable, frame_ok, per_joint


def axial_stats(angles_deg):
    """Circular statistics for an orientation (a line, not a direction).

    A body line at +80 and one at -100 are the same line, so a plain average
    treats them as 180 apart and inflates the spread. Doubling the angles before
    averaging, then halving, removes that 180 ambiguity and gives a wrap-safe
    mean and standard deviation. Returns (mean_deg, std_deg) in the range
    (-90, 90].
    """
    th = np.deg2rad(angles_deg[~np.isnan(angles_deg)] * 2)
    if th.size == 0:
        return float("nan"), float("nan")
    C, S = np.mean(np.cos(th)), np.mean(np.sin(th))
    R = np.hypot(C, S)
    mean = np.rad2deg(np.arctan2(S, C)) / 2
    std = np.rad2deg(np.sqrt(max(0.0, -2.0 * np.log(max(R, 1e-9))))) / 2
    return mean, std


def body_alignment(df):
    sh = midpoint(df, "l_shoulder", "r_shoulder")
    hip = midpoint(df, "l_hip", "r_hip")
    vec = sh - hip
    ang = np.degrees(np.arctan2(vec[:, 1], vec[:, 0]))
    return ang, sh, hip


def joint_angle_series(df, a, b, c):
    A = np.column_stack(joint(df, a))
    B = np.column_stack(joint(df, b))
    C = np.column_stack(joint(df, c))
    return angle_at(A, B, C)


def limb_cadence(df, name, centre, fps_eff):
    """Dominant movement frequency of a limb, from its distance to the body centre."""
    x, y = joint(df, name)
    dist = np.sqrt((x - centre[:, 0]) ** 2 + (y - centre[:, 1]) ** 2)
    s = pd.Series(dist).interpolate(limit=8).to_numpy()   # bridge short gaps
    mask = ~np.isnan(s)
    if mask.sum() < 40:
        return None
    s = detrend(s[mask])
    freqs = np.fft.rfftfreq(len(s), d=1 / fps_eff)
    power = np.abs(np.fft.rfft(s)) ** 2
    band = (freqs > CADENCE_BAND[0]) & (freqs < CADENCE_BAND[1])
    if not band.any():
        return None
    peak_hz = freqs[band][np.argmax(power[band])]
    return {"hz": peak_hz, "per_min": peak_hz * 60,
            "signal": dist, "freqs": freqs, "power": power, "band": band}


def analyse(csv_path, plot_path=None):
    df, fps_eff = load(csv_path)
    print(f"Loaded {len(df)} frames, effective {fps_eff:.2f} Hz, "
          f"{df['t'].iloc[-1]:.0f}s of footage.\n")

    usable, frame_ok, per_joint = detection_rate(df)
    print(f"DETECTION RATE: {usable * 100:.1f}% of frames have a usable skeleton "
          f"(>=75% of core joints).")
    for j, v in per_joint.items():
        print(f"    {j:12s} seen in {v * 100:5.1f}% of frames")

    if usable < 0.05:
        print("\n--------------------------------------------------------------")
        print("Almost no usable skeleton was found in this file, so the metrics")
        print("below would be meaningless and are skipped. Likely cause:")
        print("  - track_markers.py was run on footage with no markers on the body,")
        print("    or the marker HSV ranges do not match the real marker colours.")
        print("For footage WITHOUT markers, extract joints with:")
        print("    python pose_markerless.py your_video.mp4 --out pose_data.csv")
        print("For footage WITH markers, calibrate the colours first with")
        print("calibrate_color.py, then re-run track_markers.py.")
        print("--------------------------------------------------------------")
        return df

    ang, sh, hip = body_alignment(df)
    axis_mean, axis_std = axial_stats(ang)
    print(f"\nBODY ALIGNMENT (top view, image plane):")
    print(f"    axis {axis_mean:.1f} deg, steadiness (std) {axis_std:.1f} deg "
          f"- lower std means a straighter, steadier body line.")

    r_elbow = joint_angle_series(df, "r_shoulder", "r_elbow", "r_wrist")
    l_elbow = joint_angle_series(df, "l_shoulder", "l_elbow", "l_wrist")
    r_knee = joint_angle_series(df, "r_hip", "r_knee", "r_ankle")
    l_knee = joint_angle_series(df, "l_hip", "l_knee", "l_ankle")
    print("\nJOINT ANGLES (median):")
    print(f"    elbow  R {np.nanmedian(r_elbow):3.0f} deg   L {np.nanmedian(l_elbow):3.0f} deg")
    print(f"    knee   R {np.nanmedian(r_knee):3.0f} deg   L {np.nanmedian(l_knee):3.0f} deg")

    centre = (sh + hip) / 2
    print("\nLIMB CADENCE (dominant movement frequency, per limb):")
    cad = {}
    for limb in ["l_wrist", "r_wrist", "l_ankle", "r_ankle"]:
        res = limb_cadence(df, limb, centre, fps_eff)
        if res:
            cad[limb] = res
            print(f"    {limb:9s} {res['hz']:.2f} Hz  ({res['per_min']:.0f} moves/min)")
        else:
            print(f"    {limb:9s} not enough data")
    if "l_wrist" in cad and "r_wrist" in cad:
        print(f"    arm left/right difference: "
              f"{abs(cad['l_wrist']['per_min'] - cad['r_wrist']['per_min']):.0f} moves/min")
    if "l_ankle" in cad and "r_ankle" in cad:
        print(f"    leg left/right difference: "
              f"{abs(cad['l_ankle']['per_min'] - cad['r_ankle']['per_min']):.0f} moves/min")

    if plot_path:
        _plot(df, frame_ok, usable, ang, cad, plot_path)
        print(f"\nCharts saved to {plot_path}")
    return df


def _plot(df, frame_ok, usable, ang, cad, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    ax[0, 0].plot(df["t"], frame_ok * 100, lw=1, color="#0E7BA8")
    ax[0, 0].axhline(75, color="#E07C2E", ls="--", lw=1, label="75% threshold")
    ax[0, 0].set_title(f"Detection rate over time (mean {usable * 100:.0f}% usable)")
    ax[0, 0].set_xlabel("time (s)"); ax[0, 0].set_ylabel("% core joints seen"); ax[0, 0].legend()

    axis_ang = ((ang + 90) % 180) - 90   # fold to (-90, 90] so 180 flips don't show as jumps
    ax[0, 1].plot(df["t"], axis_ang, lw=0.8, color="#6E4CA6")
    ax[0, 1].set_title("Body line orientation over time (steadiness)")
    ax[0, 1].set_xlabel("time (s)"); ax[0, 1].set_ylabel("trunk axis (deg)")

    for limb, col in [("r_wrist", "#1B98B5"), ("r_ankle", "#E8514F")]:
        if limb in cad:
            ax[1, 0].plot(df["t"], cad[limb]["signal"], lw=0.8, color=col, label=limb)
    ax[1, 0].set_title("Limb movement signal (distance from body centre)")
    ax[1, 0].set_xlabel("time (s)"); ax[1, 0].set_ylabel("px"); ax[1, 0].legend()

    for limb, col in [("r_wrist", "#1B98B5"), ("r_ankle", "#E8514F")]:
        if limb in cad:
            b = cad[limb]["band"]
            p = cad[limb]["power"][b]
            ax[1, 1].plot(cad[limb]["freqs"][b], p / p.max(), color=col,
                          label=f"{limb} ({cad[limb]['per_min']:.0f}/min)")
    ax[1, 1].set_title("Movement frequency spectrum (cadence peak)")
    ax[1, 1].set_xlabel("Hz"); ax[1, 1].set_ylabel("norm. power"); ax[1, 1].legend()

    plt.tight_layout()
    plt.savefig(path, dpi=110)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Compute swim kinematics from a pose CSV.")
    ap.add_argument("csv", help="pose_data.csv from extract_pose.py")
    ap.add_argument("--plot", default=None, help="optional path to save charts, e.g. metrics.png")
    args = ap.parse_args()
    analyse(args.csv, args.plot)

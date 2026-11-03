#!/usr/bin/env python3
"""
swim_webapp.py  -  drag-and-drop web app for the AI Flotation Analyser.
 
Drop a side-on swimming video onto the page. The app runs the full pipeline:
  video -> MediaPipe landmarks -> mp_features.py (your extractor)
        -> swim_predict.py + model_v4  ->  NOT / ALMOST / FLOATING
and then adds a coaching analysis: per-limb speed and balance, estimated stroke
and kick counts, posture and activity over time, and a float-factor breakdown.
 
Reuses mp_features.py and swim_predict.py unchanged, so the category and the
model are exactly what was trained and evaluated. Only the video->landmark stage
and the extra charts are added here.
 
Run it from the floatml folder (venv active):
    python swim_webapp.py              # serves on 127.0.0.1:5000
"""
import os, sys, json, tempfile, subprocess, base64, shutil, re, threading, webbrowser, time, io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify
 
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "5000"))
HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
MODEL = os.environ.get("FLOAT_MODEL", os.path.join(HERE, "model_v4", "model.joblib"))
MP_FEATURES = os.path.join(HERE, "mp_features.py")
SWIM_PREDICT = os.path.join(HERE, "swim_predict.py")
MODEL_COMPLEXITY = int(os.environ.get("MP_COMPLEXITY", "1"))
 
LM = ["nose","left_eye_inner","left_eye","left_eye_outer","right_eye_inner","right_eye",
      "right_eye_outer","left_ear","right_ear","mouth_left","mouth_right","left_shoulder",
      "right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist","left_pinky",
      "right_pinky","left_index","right_index","left_thumb","right_thumb","left_hip","right_hip",
      "left_knee","right_knee","left_ankle","right_ankle","left_heel","right_heel",
      "left_foot_index","right_foot_index"]
 
BLUE, ORANGE, GREEN, RED, PURPLE = "#4C78A8", "#E0A83D", "#1E9E8A", "#D7513F", "#8156b5"
 
 
def extract_landmarks(video_path, out_csv, view="side_front", pid="upload", vid="upload"):
    import cv2, mediapipe as mp
    import pandas as pd
    pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=MODEL_COMPLEXITY,
                                  enable_segmentation=False, min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if not fps or fps != fps or fps <= 0:
        fps = 25.0
    rows = []; i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        row = {"video_id": vid, "participant_id": pid, "camera_view": view,
               "frame": i, "time_sec": i / fps}
        if res.pose_landmarks:
            for name, lm in zip(LM, res.pose_landmarks.landmark):
                row[f"{name}_x"] = lm.x; row[f"{name}_y"] = lm.y
                row[f"{name}_z"] = lm.z; row[f"{name}_visibility"] = lm.visibility
        else:
            for name in LM:
                row[f"{name}_x"] = np.nan; row[f"{name}_y"] = np.nan
                row[f"{name}_z"] = np.nan; row[f"{name}_visibility"] = 0.0
        rows.append(row); i += 1
    cap.release(); pose.close()
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return i, fps
 
 
def _b64(fig):
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig); return base64.b64encode(buf.getvalue()).decode()
 
 
def _nm(df, *cols):
    """row-wise nanmean of the given columns that exist."""
    import pandas as pd
    have = [c for c in cols if c in df.columns]
    if not have:
        return pd.Series(np.nan, index=df.index)
    return df[have].apply(lambda r: np.nanmean(r.values.astype(float)), axis=1)
 
 
def _m(s):
    s = np.asarray(s, float); s = s[np.isfinite(s)]
    return float(np.mean(s)) if len(s) else float("nan")
 
 
def build_analysis(win_csv):
    """Read swim_predict's window table and produce coaching metrics + charts."""
    import pandas as pd
    df = pd.read_csv(win_csv)
    t = ((df["start_s"] + df["end_s"]) / 2.0) if "end_s" in df else df.get("start_s", pd.Series(range(len(df))))
    dur = float(df["end_s"].max() - df["start_s"].min()) if "end_s" in df else float(len(df))
    dur_min = max(dur / 60.0, 1e-6)
 
    arm_sp_l, arm_sp_r = _m(df.get("arm_speed_l")), _m(df.get("arm_speed_r"))
    leg_sp_l, leg_sp_r = _m(df.get("leg_speed_l")), _m(df.get("leg_speed_r"))
    arm_sp, leg_sp = _m(df.get("arm_speed")), _m(df.get("leg_speed"))
    arm_cad = _m(_nm(df, "arm_freq_l", "arm_freq_r"))   # cycles / min
    leg_cad = _m(_nm(df, "leg_freq_l", "leg_freq_r"))
    arm_strokes = arm_cad * dur_min if arm_cad == arm_cad else float("nan")
    leg_kicks = leg_cad * dur_min if leg_cad == leg_cad else float("nan")
    trunk = _m(df.get("trunk_angle_med")); trunk_std = _m(df.get("trunk_angle_std"))
    drift = _m(df.get("hip_y_slope"))
 
    def bal(a, b):
        if not (a == a and b == b) or max(a, b) <= 0:
            return float("nan")
        return 100.0 * min(a, b) / max(a, b)
    arm_balance, leg_balance = bal(arm_sp_l, arm_sp_r), bal(leg_sp_l, leg_sp_r)
 
    # ankle vertical range (proxy for how far the legs travel up/down; NOT above/below water)
    ank_range = float("nan")
    if "ankle_y_med" in df and df["ankle_y_med"].notna().any():
        ay = df["ankle_y_med"].to_numpy(float); ay = ay[np.isfinite(ay)]
        if len(ay): ank_range = float(np.percentile(ay, 90) - np.percentile(ay, 10))
 
    # class mix across ALL windows (matches the timeline dots)
    mix = {"floating": 0, "almost": 0, "not": 0}
    if "float_pred" in df:
        vc = df["float_pred"].value_counts(normalize=True)
        mix = {"floating": round(float(vc.get(2, 0)), 3),
               "almost": round(float(vc.get(1, 0)), 3),
               "not": round(float(vc.get(0, 0)), 3)}
 
    # ---- float factors (0-100 coaching indicators; higher = more float-like) ----
    def clamp(v): return float(max(0, min(100, v)))
    flat = clamp(100 * (1 - min(abs(trunk), 90) / 90)) if trunk == trunk else float("nan")
    steady = clamp(100 * (1 - min(trunk_std, 15) / 15)) if trunk_std == trunk_std else float("nan")
    level = clamp(100 * (1 - min(abs(drift), 0.1) / 0.1)) if drift == drift else float("nan")
    calm = clamp(100 * (1 - min(max(arm_sp, leg_sp) if (arm_sp == arm_sp and leg_sp == leg_sp) else arm_sp, 6) / 6)) \
        if (arm_sp == arm_sp or leg_sp == leg_sp) else float("nan")
    factors = {"Body flatness": flat, "Posture steadiness": steady,
               "Stays level (no sinking)": level, "Stillness (low thrashing)": calm}
 
    charts = {}
 
    # 1) posture over time with bands
    if "trunk_angle_med" in df:
        fig, ax = plt.subplots(figsize=(6.4, 2.8))
        ax.axhspan(0, 20, color=GREEN, alpha=.12); ax.axhspan(20, 45, color=ORANGE, alpha=.12)
        ax.axhspan(45, 90, color=RED, alpha=.10)
        ax.plot(t, df["trunk_angle_med"], color=BLUE, lw=1.8)
        ax.set_ylim(0, 90); ax.set_xlabel("time (s)"); ax.set_ylabel("trunk angle (°)")
        ax.set_title("Posture over time  (0°=flat/horizontal, 90°=upright)", fontsize=10)
        ax.text(t.iloc[-1] if len(t) else 1, 10, "horizontal", color=GREEN, fontsize=7, ha="right")
        ax.text(t.iloc[-1] if len(t) else 1, 32, "inclined", color="#b07d20", fontsize=7, ha="right")
        ax.text(t.iloc[-1] if len(t) else 1, 70, "upright", color=RED, fontsize=7, ha="right")
        ax.spines[["top", "right"]].set_visible(False)
        charts["posture"] = _b64(fig)
 
    # 2) limb activity over time
    if "arm_speed" in df or "leg_speed" in df:
        fig, ax = plt.subplots(figsize=(6.4, 2.8))
        if "arm_speed" in df: ax.plot(t, df["arm_speed"], color=PURPLE, lw=1.6, label="arms")
        if "leg_speed" in df: ax.plot(t, df["leg_speed"], color=GREEN, lw=1.6, label="legs")
        ax.set_xlabel("time (s)"); ax.set_ylabel("speed (body-lengths/s)")
        ax.set_title("Limb activity over time", fontsize=10)
        ax.legend(fontsize=8, frameon=False); ax.spines[["top", "right"]].set_visible(False)
        charts["activity"] = _b64(fig)
 
    # 3) left / right balance
    labels, L, R = [], [], []
    if arm_sp_l == arm_sp_l or arm_sp_r == arm_sp_r:
        labels.append("arms"); L.append(arm_sp_l or 0); R.append(arm_sp_r or 0)
    if leg_sp_l == leg_sp_l or leg_sp_r == leg_sp_r:
        labels.append("legs"); L.append(leg_sp_l or 0); R.append(leg_sp_r or 0)
    if labels:
        x = np.arange(len(labels)); w = 0.35
        fig, ax = plt.subplots(figsize=(4.6, 2.9))
        ax.bar(x - w/2, L, w, label="left", color=BLUE)
        ax.bar(x + w/2, R, w, label="right", color=ORANGE)
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_ylabel("mean speed"); ax.set_title("Left vs right balance", fontsize=10)
        ax.legend(fontsize=8, frameon=False); ax.spines[["top", "right"]].set_visible(False)
        charts["balance"] = _b64(fig)
 
    # 4) estimated cycle counts
    cc_lab, cc_val = [], []
    if arm_strokes == arm_strokes: cc_lab.append("arm strokes"); cc_val.append(round(arm_strokes))
    if leg_kicks == leg_kicks: cc_lab.append("leg kicks"); cc_val.append(round(leg_kicks))
    if cc_lab:
        fig, ax = plt.subplots(figsize=(4.6, 2.9))
        bars = ax.bar(cc_lab, cc_val, color=[PURPLE, GREEN][:len(cc_lab)])
        for b, v in zip(bars, cc_val):
            ax.text(b.get_x()+b.get_width()/2, v, str(int(v)), ha="center", va="bottom", fontsize=10)
        ax.set_ylabel("estimated count over clip")
        ax.set_title("Estimated limb cycles", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        charts["cadence"] = _b64(fig)
 
    # 5) float factors
    fk = [k for k, v in factors.items() if v == v]
    if fk:
        fv = [factors[k] for k in fk]
        fig, ax = plt.subplots(figsize=(6.0, 2.8))
        colors = [GREEN if v >= 66 else (ORANGE if v >= 40 else RED) for v in fv]
        ax.barh(range(len(fk)), fv, color=colors)
        ax.set_yticks(range(len(fk))); ax.set_yticklabels(fk, fontsize=9)
        ax.set_xlim(0, 100); ax.invert_yaxis()
        for i, v in enumerate(fv): ax.text(v + 1, i, f"{v:.0f}", va="center", fontsize=8)
        ax.set_xlabel("0–100 (higher is more float-friendly)")
        ax.set_title("What's helping or hurting the float", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        charts["factors"] = _b64(fig)
 
    def r1(v): return None if v != v else round(v, 1)
    metrics = {
        "duration_s": r1(dur),
        "arm_cadence_per_min": r1(arm_cad), "leg_cadence_per_min": r1(leg_cad),
        "arm_strokes_est": None if arm_strokes != arm_strokes else int(round(arm_strokes)),
        "leg_kicks_est": None if leg_kicks != leg_kicks else int(round(leg_kicks)),
        "arm_speed": r1(arm_sp), "leg_speed": r1(leg_sp),
        "arm_speed_l": r1(arm_sp_l), "arm_speed_r": r1(arm_sp_r),
        "leg_speed_l": r1(leg_sp_l), "leg_speed_r": r1(leg_sp_r),
        "arm_balance_pct": r1(arm_balance), "leg_balance_pct": r1(leg_balance),
        "ankle_vertical_range": r1(ank_range),
        "trunk_angle_deg": r1(trunk), "trunk_variation_deg": r1(trunk_std),
        "hip_drift_per_s": None if drift != drift else round(drift, 3),
    }
    return {"metrics": metrics, "charts": charts, "mix_all": mix}
 
 
def body_movement(land_csv, fps, aspect=16/9):
    """Average movement of the whole body, from the raw landmark CSV.
 
    For every frame pair it measures how far each tracked body point moved,
    normalised by torso length (so it's scale-independent) and per second.
    Returns the clip average (body-lengths/s) plus an over-time chart.
    """
    import pandas as pd
    d = pd.read_csv(land_csv)
    pts = ["left_shoulder","right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist",
           "left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle"]
    if not all(f"{p}_x" in d for p in pts):
        return None
    col = lambda lm, c: d[f"{lm}_{c}"].to_numpy(float)
    shx = (col("left_shoulder","x")+col("right_shoulder","x"))/2*aspect
    shy = (col("left_shoulder","y")+col("right_shoulder","y"))/2
    hpx = (col("left_hip","x")+col("right_hip","x"))/2*aspect
    hpy = (col("left_hip","y")+col("right_hip","y"))/2
    torso = np.hypot(shx-hpx, shy-hpy)
    torso_med = np.nanmedian(torso[torso > 0]) if np.isfinite(torso).any() else np.nan
    if not torso_med > 0:
        return None
    t = col("nose","x")*0 + d["time_sec"].to_numpy(float)   # time array
    X = {p: col(p,"x")*aspect for p in pts}; Y = {p: col(p,"y") for p in pts}
    V = {p: col(p,"visibility") for p in pts}
    n = len(d); per = np.full(max(n-1,0), np.nan); tf = t[1:] if n > 1 else np.array([])
    for i in range(1, n):
        dt = t[i]-t[i-1]
        if dt <= 0 or dt > 2.0/fps:
            continue
        ds = [np.hypot(X[p][i]-X[p][i-1], Y[p][i]-Y[p][i-1])/torso_med/dt
              for p in pts if V[p][i] > 0.5 and V[p][i-1] > 0.5]
        if ds:
            per[i-1] = float(np.mean(ds))
    if not np.isfinite(per).any():
        return None
    avg = float(np.nanmean(per))
    # chart: average into ~1s bins so it reads cleanly
    chart = None
    try:
        import pandas as pd
        good = np.isfinite(per)
        tb = tf[good]; pb = per[good]
        if len(tb) > 3:
            bins = np.arange(tb.min(), tb.max()+1.0, 1.0)
            idx = np.digitize(tb, bins)
            xs = [tb[idx == k].mean() for k in range(1, len(bins)+1) if (idx == k).any()]
            ys = [pb[idx == k].mean() for k in range(1, len(bins)+1) if (idx == k).any()]
            fig, ax = plt.subplots(figsize=(6.4, 2.6))
            ax.plot(xs, ys, color=RED, lw=1.8)
            ax.axhline(avg, color="#888", ls="--", lw=1, label=f"average {avg:.1f}")
            ax.set_xlabel("time (s)"); ax.set_ylabel("movement (body-lengths/s)")
            ax.set_title("Overall body movement over time", fontsize=10)
            ax.legend(fontsize=8, frameon=False); ax.spines[["top","right"]].set_visible(False)
            chart = _b64(fig)
    except Exception:
        chart = None
    return {"avg": round(avg, 2), "chart": chart}
 
 
def classify(video_path, workdir):
    import pandas as pd
    stem = "clip"
    land = os.path.join(workdir, f"{stem}.csv")
    nframes, fps = extract_landmarks(video_path, land)
    if nframes == 0:
        return {"classification": "UNABLE TO ASSESS", "reason": "could not read any frames"}, None, None
 
    feat_dir = os.path.join(workdir, "feat")
    subprocess.run([PY, MP_FEATURES, land, "--out-side", feat_dir,
                    "--out-top", os.path.join(workdir, "top")], check=True, cwd=HERE)
    feat_csv = os.path.join(feat_dir, f"{stem}_features.csv")
    if not os.path.exists(feat_csv) or len(pd.read_csv(feat_csv)) == 0:
        return {"classification": "UNABLE TO ASSESS",
                "reason": "no usable pose windows (swimmer not tracked well enough)"}, None, None
 
    out_dir = os.path.join(workdir, "out")
    subprocess.run([PY, SWIM_PREDICT, feat_csv, "--model", MODEL, "--out", out_dir], check=True, cwd=HERE)
    name = f"{stem}_features"
    report = json.load(open(os.path.join(out_dir, f"{name}_report.json")))
    timeline = os.path.join(out_dir, f"{name}_timeline.png")
    win_csv = os.path.join(out_dir, f"{name}_windows.csv")
    analysis = None
    try:
        if os.path.exists(win_csv):
            analysis = build_analysis(win_csv)
            if analysis.get("mix_all"):
                report["fraction"] = analysis["mix_all"]   # make headline mix match the timeline
            try:
                bm = body_movement(land, fps)
                if bm:
                    analysis["metrics"]["avg_body_movement"] = bm["avg"]
                    if bm.get("chart"):
                        analysis["charts"]["movement"] = bm["chart"]
            except Exception:
                pass
    except Exception as e:
        analysis = {"error": f"analysis step: {type(e).__name__}: {e}"}
    return report, (timeline if os.path.exists(timeline) else None), analysis
 
 
app = Flask(__name__)
 
PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI Flotation Analyser</title>
<style>
  :root{ --bg:#0e1726; --card:#16233a; --ink:#eaf0f8; --mut:#9fb0c7; --line:#2a3b57;
         --green:#1E9E8A; --amber:#E0A83D; --red:#D7513F; --accent:#4C78A8; }
  *{box-sizing:border-box} html,body{margin:0;height:100%}
  body{background:linear-gradient(160deg,#0b1220,#12203a);color:var(--ink);
       font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;line-height:1.5}
  .wrap{max-width:860px;margin:0 auto;padding:28px 16px 70px}
  h1{font-size:22px;margin:0 0 4px} .sub{color:var(--mut);font-size:13px;margin-bottom:22px}
  .drop{border:2px dashed var(--line);border-radius:16px;background:var(--card);
        padding:40px 20px;text-align:center;transition:.15s;cursor:pointer}
  .drop.drag{border-color:var(--accent);background:#1b2b47}
  .drop h2{margin:6px 0 2px;font-size:16px} .drop p{color:var(--mut);font-size:13px;margin:4px 0}
  .btn{display:inline-block;margin-top:12px;background:var(--accent);color:#fff;border:0;
       padding:9px 18px;border-radius:9px;font-size:14px;cursor:pointer}
  input[type=file]{display:none}
  .card{margin-top:22px;background:var(--card);border:1px solid var(--line);border-radius:16px;
        padding:22px;display:none}
  .verdict{font-size:28px;font-weight:700;letter-spacing:.3px}
  .bar{height:10px;border-radius:999px;background:#0d1a2e;overflow:hidden;margin:10px 0 2px}
  .bar > span{display:block;height:100%}
  .mix{display:flex;gap:6px;margin:14px 0 6px;font-size:12px;color:var(--mut)}
  .mix div{flex:1;text-align:center}
  h3{font-size:13px;color:#bcd0ea;margin:20px 0 6px;text-transform:uppercase;letter-spacing:.5px}
  .grid{display:grid;grid-template-columns:1fr 1fr;gap:6px 18px}
  .kv{display:flex;justify-content:space-between;border-bottom:1px solid var(--line);padding:5px 0;font-size:13px}
  .kv .k{color:var(--mut)}
  .charts{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:8px}
  .charts img{width:100%;border-radius:10px;background:#fff}
  .full{grid-column:1 / -1}
  .note{color:var(--mut);font-size:11px;margin-top:10px}
  .spin{display:none;margin-top:22px;color:var(--mut);font-size:14px}
  .dot{display:inline-block;width:10px;height:10px;border-radius:50%;background:var(--accent);
       margin-right:8px;animation:p 1s infinite}
  @keyframes p{0%,100%{opacity:.3}50%{opacity:1}}
  .err{color:#ff9b8a;font-size:13px;margin-top:14px;display:none}
  @media(max-width:640px){.grid,.charts{grid-template-columns:1fr}}
</style></head><body><div class="wrap">
  <h1>AI Flotation Analyser</h1>
  <div class="sub">Drag in a side-on swimming clip. It classifies the passive float and breaks down what's driving it.</div>
 
  <div class="drop" id="drop">
    <h2>Drop a video here</h2>
    <p>mp4 / mov / avi, side-on view. Or</p>
    <button class="btn" id="pick">Choose a file</button>
    <input type="file" id="file" accept="video/*">
  </div>
 
  <div class="spin" id="spin"><span class="dot"></span>Analysing, this runs pose tracking frame by frame and can take a minute or two…</div>
  <div class="err" id="err"></div>
 
  <div class="card" id="card">
    <span class="verdict" id="verdict"></span>
    <div class="bar"><span id="confbar"></span></div>
    <div style="font-size:12px;color:var(--mut)">confidence <b id="conf"></b></div>
    <div class="mix" id="mix"></div>
 
    <h3>Key measurements</h3>
    <div class="grid" id="metrics"></div>
 
    <h3>What's driving the float</h3>
    <div class="charts" id="charts"></div>
 
    <div class="note" id="note"></div>
  </div>
</div>
<script>
const drop=document.getElementById('drop'),file=document.getElementById('file'),
      card=document.getElementById('card'),spin=document.getElementById('spin'),err=document.getElementById('err');
document.getElementById('pick').onclick=e=>{e.stopPropagation();file.click()};
drop.onclick=()=>file.click();
['dragenter','dragover'].forEach(ev=>drop.addEventListener(ev,e=>{e.preventDefault();drop.classList.add('drag')}));
['dragleave','drop'].forEach(ev=>drop.addEventListener(ev,e=>{e.preventDefault();drop.classList.remove('drag')}));
drop.addEventListener('drop',e=>{if(e.dataTransfer.files.length)send(e.dataTransfer.files[0])});
file.onchange=()=>{if(file.files.length)send(file.files[0])};
const COL={FLOATING:"#1E9E8A","ALMOST FLOATING":"#E0A83D","NOT FLOATING":"#D7513F","UNABLE TO ASSESS":"#6b7a90",UNCERTAIN:"#E0A83D"};
function colorFor(v){v=(v||'').toUpperCase();for(const k in COL){if(v.includes(k))return COL[k];}return "#4C78A8";}
function send(f){err.style.display='none';card.style.display='none';spin.style.display='block';
  const fd=new FormData();fd.append('video',f);
  fetch('/analyze',{method:'POST',body:fd}).then(r=>r.json()).then(show).catch(e=>{
    spin.style.display='none';err.textContent='Something went wrong: '+e;err.style.display='block';});}
function kv(k,v){return v==null||v===''?'':`<div class="kv"><span class="k">${k}</span><span>${v}</span></div>`;}
function show(d){
  spin.style.display='none';
  if(d.error){err.textContent=d.error;err.style.display='block';return;}
  const r=d.report, a=d.analysis||{}, m=a.metrics||{}, c=a.charts||{};
  const col=colorFor(r.classification||'');
  document.getElementById('verdict').textContent=r.classification||'—';
  document.getElementById('verdict').style.color=col;
  const conf=r.confidence_pct;
  document.getElementById('conf').textContent=conf!=null?conf+'%':'n/a';
  const cb=document.getElementById('confbar');cb.style.width=(conf!=null?conf:0)+'%';cb.style.background=col;
  const mixEl=document.getElementById('mix');
  if(r.fraction){mixEl.innerHTML=
    `<div>floating<br><b>${Math.round(r.fraction.floating*100)}%</b></div>`+
    `<div>almost<br><b>${Math.round(r.fraction.almost*100)}%</b></div>`+
    `<div>not<br><b>${Math.round(r.fraction.not*100)}%</b></div>`;}else mixEl.innerHTML='';
  const sp=v=>v==null?null:v+" body-lengths/s";
  document.getElementById('metrics').innerHTML =
    kv('Classification', r.classification)+
    kv('Posture', r.posture?(r.posture+(m.trunk_angle_deg!=null?` (trunk ${m.trunk_angle_deg}°)`:'')):null)+
    kv('Stability', r.stability?(r.stability+(m.trunk_variation_deg!=null?` (±${m.trunk_variation_deg}°)`:'')):null)+
    kv('Vertical movement', r.vertical_movement?(r.vertical_movement+(m.hip_drift_per_s!=null?` (${m.hip_drift_per_s}/s)`:'')):null)+
    kv('Average body movement', m.avg_body_movement!=null?m.avg_body_movement+' body-lengths/s':null)+
    kv('Clip length', m.duration_s!=null?m.duration_s+' s':null)+
    kv('Arm cadence', m.arm_cadence_per_min!=null?m.arm_cadence_per_min+' /min':null)+
    kv('Leg cadence', m.leg_cadence_per_min!=null?m.leg_cadence_per_min+' /min':null)+
    kv('Arm strokes (est.)', m.arm_strokes_est)+
    kv('Leg kicks (est.)', m.leg_kicks_est)+
    kv('Arm speed', sp(m.arm_speed))+
    kv('Leg speed', sp(m.leg_speed))+
    kv('Arm speed L / R', (m.arm_speed_l!=null||m.arm_speed_r!=null)?`${m.arm_speed_l} / ${m.arm_speed_r}`:null)+
    kv('Leg speed L / R', (m.leg_speed_l!=null||m.leg_speed_r!=null)?`${m.leg_speed_l} / ${m.leg_speed_r}`:null)+
    kv('Arm L/R balance', m.arm_balance_pct!=null?m.arm_balance_pct+'%':null)+
    kv('Leg L/R balance', m.leg_balance_pct!=null?m.leg_balance_pct+'%':null)+
    kv('Ankle vertical range', m.ankle_vertical_range!=null?m.ankle_vertical_range:null)+
    kv('Symmetry', r.symmetry)+
    kv('Pose quality', r.pose_quality_pct!=null?r.pose_quality_pct+'% valid':null)+
    kv('Windows analysed', r.windows)+
    kv('Reason', r.reason);
  const order=[['factors','full'],['movement','full'],['posture','full'],['activity','full'],['balance',''],['cadence','']];
  let html='';
  order.forEach(([k,cls])=>{if(c[k])html+=`<img class="${cls}" src="data:image/png;base64,${c[k]}">`;});
  if(d.timeline)html+=`<img class="full" src="data:image/png;base64,${d.timeline}">`;
  document.getElementById('charts').innerHTML=html;
  document.getElementById('note').innerHTML =
    "Arm strokes and leg kicks are estimated from the limb cycle rate times the clip length. "+
    "There is no water line in pose data, so arm/leg \"above vs below the water\" cannot be measured directly; "+
    "the closest signals are the limb cycle counts and the ankle vertical range shown above. "+
    "The headline category uses the stable middle of the clip, so the first and last few seconds (entry/exit) "+
    "may show differently on the timeline. Head and airway are not scored.";
  card.style.display='block';
}
</script></body></html>"""
 
 
@app.route("/")
def index():
    return PAGE
 
 
@app.route("/analyze", methods=["POST"])
def analyze():
    if "video" not in request.files or not request.files["video"].filename:
        return jsonify({"error": "no file uploaded"})
    f = request.files["video"]
    workdir = tempfile.mkdtemp(prefix="swimweb_")
    vpath = os.path.join(workdir, re.sub(r"[^A-Za-z0-9._-]", "_", f.filename))
    try:
        f.save(vpath)
        report, timeline, analysis = classify(vpath, workdir)
        tb64 = base64.b64encode(open(timeline, "rb").read()).decode() if timeline else None
        return jsonify({"report": report, "timeline": tb64, "analysis": analysis})
    except subprocess.CalledProcessError as e:
        return jsonify({"error": f"pipeline step failed: {e}"})
    except Exception as e:
        return jsonify({"error": f"{type(e).__name__}: {e}"})
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
 
 
def open_browser():
    time.sleep(1.5)
    try:
        if os.environ.get("NO_OPEN") != "1":
            webbrowser.open(f"http://127.0.0.1:{PORT}")
    except Exception:
        pass
 
 
if __name__ == "__main__":
    if not os.path.exists(MODEL):
        print(f"WARNING: model not found at {MODEL}. Run from the folder that has model_v4/.")
    print(f"AI Flotation Analyser web app on http://127.0.0.1:{PORT}  (Ctrl-C to stop)")
    threading.Thread(target=open_browser, daemon=True).start()
    app.run(host=HOST, port=PORT, debug=False)
 

# Feature dictionary — AI Flotation Analyser (side-front & top views)

One row = one **3.0 s window, 50% overlap**, per video. Windows overlap, so they are
**not independent observations** — split by `participant_id`/`video_id` (not by row)
for any supervised validation.

## Columns, units, and groups

### 1. Identity / metadata
| column | meaning | units |
|---|---|---|
| `video_id` | `<participant>_<view>` | id |
| `participant_id` (`pid`) | participant/video id from filename | id |
| `view` | camera view | `side_front` / `top` |
| `start_s`, `end_s` | window bounds | seconds |

### 2. Postural features
| column | meaning | units |
|---|---|---|
| `trunk_angle_med` | median trunk angle from horizontal (0=flat, 90=vertical) | degrees |
| `trunk_angle_std` | trunk-angle variability within window (steadiness) | degrees |
| `hip_depth_med` | median hip depth below detected surface | body-lengths (torso-normalized) |
| `hip_depth_std` | hip-depth variability | body-lengths |
| `ankle_depth` | median ankle depth below surface | body-lengths |
| `knee_angle_med_deg` | median knee joint angle (hip–knee–ankle) | degrees |

> **Camera caveat (side-front is UNDERWATER):** the water surface can't be detected
> reliably, so **absolute** `hip_depth_med` / `ankle_depth` / `head_sub_frac` are
> unreliable in this dataset. Lead with `trunk_angle_*` (surface-independent) and the
> **slopes** below. Keep the depth columns for reference, not as primary cluster inputs.

### 3. Temporal flotation features (trends within the window)
| column | meaning | units |
|---|---|---|
| `hip_depth_slope` | rate of hip depth change (+ = sinking). **Robust to the water-line error** (the constant offset cancels in a slope). | body-lengths / sec |
| `trunk_angle_slope` | rate of trunk-angle change (+ = tilting toward vertical) | degrees / sec |
| `head_sub_frac` | fraction of frames with head below detected surface | 0–1 proportion |

### 4. Limb activity features
| column | meaning | units |
|---|---|---|
| `leg_freq_l`, `leg_freq_r` | dominant kick rhythm (FFT, periodicity-gated; noise → 0) | cycles / min |
| `arm_freq_l`, `arm_freq_r` | dominant arm rhythm | cycles / min |
| `arm_speed_l`, `arm_speed_r` | per-arm mean speed (for body-part feedback) | body-lengths / sec |
| `leg_speed_l`, `leg_speed_r` | per-leg mean speed | body-lengths / sec |
| `arm_speed`, `leg_speed` | mean of L/R (kept for continuity) | body-lengths / sec |

### 5. Symmetry features
| column | meaning | units |
|---|---|---|
| `arm_asym` | left/right arm rhythm asymmetry | 0–1 (0 = symmetric) |
| `leg_asym` | left/right leg rhythm asymmetry | 0–1 |

### 6. Reliability
| column | meaning | units |
|---|---|---|
| `valid_pct` | % of frames with a usable body pose | % |
| `mean_pose_conf` | mean keypoint confidence over the window | 0–1 |

> Use `valid_pct` and `mean_pose_conf` to **filter** windows (e.g. `valid_pct >= 60`),
> never as clustering inputs. `mean_pose_conf` is written for **every** window in v3;
> older CSVs that predate v3 lack it and should be re-extracted for consistency.

## Per-view emphasis
- **Side-front (underwater):** `trunk_angle_med/std`, `hip_depth_slope`, `trunk_angle_slope`,
  `head_sub_frac`, arm/leg speed & frequency, asymmetry. Depth is best read here (via slope), not absolute.
- **Top (overhead):** body/limb alignment, arm/leg movement, left/right asymmetry, movement
  frequency and stability. Depth relative to the surface is **not** meaningful from above.

## "What does 3.5 mean?" (for the evaluator)
Speeds are **torso-normalized**: `leg_speed = 3.5` means the ankle moves ~3.5 body-lengths
per second on average within the window — a comparable number across children of different
sizes and different camera distances, which a raw pixel speed would not be.

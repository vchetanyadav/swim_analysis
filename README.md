# Swim Analysis Pipeline

Pose estimation plus kinematics for swim-safety analysis. It takes a swimming
video, finds the swimmer's body joints in every frame, and turns those joints
into meaningful numbers: how reliably the body was tracked, how steady the body
line is, joint angles, and how fast each limb moves.

This is deliberately built as a small pipeline rather than one model. The pose
network at the bottom is a replaceable part. The value, and the part that is
genuinely novel for this project, is the layer on top: confidence gating for
occlusion, the kinematics, and (later) fusing several camera views.

## Why this design, not CenterNet

CenterNet is an object detector. It draws a box around the swimmer and gives it
a label. It does not produce joints, so it cannot give body angles, limb
frequency, or head position. Those come from pose estimation followed by
kinematics, which is what this pipeline does. A pretrained pose model already
tracks the joints well on top-view footage with no training at all, so there is
no reason to train a detector that still would not give you angles.

## Results on the test footage

Run on a 3-minute top-view clip (adult subject, controlled test), using the
off-the-shelf `yolo11n-pose` model with no fine-tuning:

- Detection rate: about 98% of frames had a usable full-body skeleton.
- Every core joint (shoulders, hips, knees, ankles) was seen in about 97% of frames.
- Limb cadence separated cleanly per limb, with left and right and arm and leg reported apart.

The same test on the front view was much weaker (distant, wide-angle distortion,
upper-body joints mostly missed), which is why the top view should be the primary
measurement camera and the front view only a supporting one.

## Install

```bash
pip install -r requirements.txt
```

The pose model weights download automatically the first time you run it. A CUDA
GPU is optional but makes extraction much faster.

## Run it

1. Extract joints from a video into a CSV:
   ```bash
   python extract_pose.py your_top_view.mp4 --out pose_data.csv
   ```

2. Compute the metrics and save charts:
   ```bash
   python kinematics.py pose_data.csv --plot metrics.png
   ```

3. (Optional) Make a video with the skeleton and live metrics drawn on it:
   ```bash
   python annotate_video.py your_top_view.mp4 --start 100 --dur 12 --out annotated.mp4
   ```

## What each metric means, and which view it needs

- Detection rate: the feasibility number. It tells you how often the pipeline
  had something to work with, and therefore how often multi-view fusion would be
  needed to rescue a frame.
- Body alignment: from a top-down camera this is the orientation of the
  shoulder-to-hip line, so it measures how straight and steady the body is (yaw).
  The pitch of the body relative to the water surface, and head-above-water, need
  a SIDE view, which this footage does not include.
- Joint angles: elbow and knee flexion, left and right separately.
- Limb cadence: the dominant movement frequency of each limb, found with a
  frequency analysis. Left and right, and arms and legs, are kept separate, so
  asymmetry is visible.

## How occlusion and bubbles are handled

Any joint the model is not confident about is dropped rather than trusted, and
short gaps are bridged by interpolation. When bubbles or splash hide a limb its
confidence falls, so that reading is ignored instead of poisoning the metric.
This is the simple, robust core of occlusion handling and it lives in
`kinematics.py`.

## Extending to the full 4-camera setup

The natural next step is multi-view fusion. Run `extract_pose.py` on each camera
to get one CSV per view. Where a joint is missing or low-confidence in one view,
take it from a view that still sees it, or triangulate the 3D position from the
views that agree. Two things are needed first: camera calibration (especially an
undistortion step for the wide-angle front view) and time alignment so the four
CSVs share a common clock. Once the joints are in a shared 3D space, the same
kinematics in `kinematics.py` apply unchanged, and you can add the side-view
metrics (body pitch relative to the water surface, and head position) that a top
view alone cannot provide.

## Swapping the pose model

`extract_pose.py` uses `yolo11n-pose` for speed. For higher accuracy on hard
frames, point it at a larger YOLO pose model, or replace the extractor with
RTMPose or MediaPipe. Nothing downstream changes, because `kinematics.py` only
reads the CSV of joint positions.

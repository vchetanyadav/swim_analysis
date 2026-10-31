#!/bin/bash
# run_all.sh - batch feature extraction (v3). Auto-detects view + participant id
# from each filename, writes to features_top/ and features_side_front/, and keeps
# going if one video fails. Excludes known treading / duplicate clips.
#
# Usage:
#   bash run_all.sh /opt/home/s4137946/swim_analysis/videos
#
# For a video whose water line needs overriding, run it by hand afterwards, e.g.:
#   python swim_features.py "videos/5 Side front.mp4" --view side_front --waterline-frac 0.30 --out features_side_front

VID_DIR="${1:-/opt/home/s4137946/swim_analysis/videos}"
MODEL="${MODEL:-yolo11x-pose.pt}"
# substrings to skip (treading, duplicates, old test clips)
EXCLUDE=("2 Side front" "Josh" "tread" "sideFront.mp4" "top.mp4" "P01_sideFront")

mkdir -p features_side_front features_top

shopt -s nullglob
for f in "$VID_DIR"/*.mp4 "$VID_DIR"/*.MP4; do
  base="$(basename "$f")"
  skip=0
  for x in "${EXCLUDE[@]}"; do
    case "${base,,}" in *"${x,,}"*) skip=1;; esac
  done
  [ "$skip" = 1 ] && { echo "skip (excluded): $base"; continue; }

  case "${base,,}" in
    *top*)  view=top;         out=features_top ;;
    *)      view=side_front;  out=features_side_front ;;
  esac

  echo ">>> $base   (view=$view)"
  python swim_features.py "$f" --view "$view" --model "$MODEL" --out "$out" \
    || echo "!!! FAILED, skipping: $base"
done
echo "done. CSVs in features_side_front/ and features_top/"

# AI Flotation Analyser

Classifies children's survival-swim float quality from MediaPipe body-pose data into
floating / almost floating / not floating. The target is passive flotation (staying at the
surface with little or no propulsion), judged separately from body angle. The model reads
pose features only; it never sees video.

## Data is NOT in this repository
All participant data (pose CSVs, labels, splits, videos, clips, trained models) is excluded by
`.gitignore` because it is derived from footage of children. Keep it on the secure server only.
This repo holds code and documentation.

## Install
```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Pipeline, in order

### Features
- `mp_features.py` - raw MediaPipe landmark CSV -> windowed feature table (the current extractor)
- `make_splits.py` - participant-level train/val/test split

### Labelling
- `select_label_clips.py`, `select_round2.py`, `select_test.py`, `select_expert_window.py` - choose clips to label
- `render_skeleton.py` - raw landmarks -> privacy-friendly skeleton clips + labelling page
- `cut_rgb_clips.py` - cut real RGB clips from the original videos (for expert airway/head judgement)
- `label_ui_v3.html`, `label_ui_expert.html`, `label_ui_rgb.html` - browser labelling tools
- `build_manifest.py`, `trace_to_video.py` - manifest + label-to-video provenance
- `make_expert_labels.py`, `build_training_labels.py` - turn expert scores into training labels / fuse with own labels

### Train and evaluate
- `swim_train.py` - champion/challenger (ordinal logistic, logistic, random forest, gradient boosting),
  leave-one-swimmer-out CV, freezes the winner. Target `float_label` (3-class) or `floating_outcome` (expert 0-3).
- `swim_test.py` - run the frozen model once on the sealed test participants
- `swim_multi.py` - multi-criterion expert-aligned model (trunk, activity, outcome; Softmax + tree) with expert validation
- `irr_kappa.py`, `expert_agreement.py` - inter-rater and expert-agreement metrics

### Predict
- `swim_predict.py` - one participant -> 3-class float assessment with confidence and abstention
- `predict_multi.py` - one participant -> expert-style assessment (trunk, activity, outcome)

## Headline results (development)
3-class model (random forest): macro-F1 0.61, weighted kappa 0.61 (leave-one-swimmer-out CV);
sealed test on 22 unseen swimmers: macro-F1 0.66. Model quality ranking matched the domain expert
at Spearman 0.80 on four gold participants.

## Known limits
- Airway is not modelled: the underwater pose has no water line, so it stays a human-labelled criterion.
- Head position needs its own labels before it can be modelled; a head-angle feature can be computed from landmarks.
- Side-front view only; the right arm is often hidden. Top-view landmarks recover both arms (future two-view upgrade).

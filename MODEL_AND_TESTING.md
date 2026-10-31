# Float-state model, evaluation, and the reproducible experiment

## 1. The scientific target (corrected)
The outcome is **passive flotation**, not a body angle. Trunk angle describes *how* someone
floats, not *whether* they float.

| Class | Definition |
|---|---|
| **Floating (2)** | Maintains passive flotation across the window, no sustained propulsion or support |
| **Almost floating (1)** | Maintains it intermittently; periodically sinks, reorients, or uses corrective movement |
| **Not floating (0)** | Cannot hold passive flotation without continuous propulsion, support, or repeated recovery |

Report these **separately** as explanatory measurements, never as the verdict:
posture (horizontal / inclined / near-vertical), trunk angle, stability, limb effort,
vertical drift (rising / stable / sinking), symmetry, pose quality.

So "13° = near-horizontal posture", not "13° = good float". A calm near-vertical swimmer can be
*floating with near-vertical posture and low effort*; a flat swimmer sculling hard and sinking can
be *almost / not floating despite good alignment*.

## 2. Feature priority (effort and drift rank above angle)
Highest: vertical-drift / sinking rate, arm & leg activity, ability to hold position over time,
pose validity. High: trunk-angle stability, movement frequency, trunk angle, asymmetry.
Lower: absolute hip/ankle depth (only where geometrically reliable). All torso-normalized.

## 3. Reproducible pipeline (the four foundations)
1. **Ingestion + manifest** — `build_manifest.py` gives every video a deterministic ID
   (`P038_T01_side_front`), records fps/duration/resolution and a **SHA-256 checksum**, and flags
   unreadable files. This fixes the "everything tagged p4" metadata bug and catches corruption.
2. **Participant-level locked splits** — `make_splits.py` partitions by **participant** (all views
   of one person in the same split): ~60% train, ~20% validation, ~20% **sealed test**.
3. **Frozen feature extractor** — run one version (`swim_features.py` v3) on every development video;
   the frozen model records Python / sklearn / numpy / pandas versions in `frozen_meta.json`.
4. **Instructor labels** — short-clip labels are the dependent variable (see §5).

## 4. Models and honest evaluation
`swim_train.py` runs a **champion/challenger** over four models and lets them compete:
ordinal logistic (respects class order), logistic regression, random forest, gradient boosting.
Evaluation is **participant-grouped leave-one-subject-out** on the development set (videos from one
person are repeated observations, not independent samples). Because the classes are ordered we
report, and select on, **macro-F1 + quadratic-weighted Cohen's kappa**, alongside balanced accuracy,
the confusion matrix, and **per-participant** scores. The winner is frozen; nothing is retuned after.

## 5. Labels and label quality
Label short clips (5–10 s): `float_state` (the target), plus `passive_or_active`, `posture`,
`stability`, `label_confidence`. For PhD-level rigor, have a **second instructor** re-label 20–30%
of clips and compute agreement with `irr_kappa.py` (quadratic-weighted kappa). If experts cannot
tell "floating" from "almost", the model cannot be expected to either.

## 6. Robustness: abstain instead of guessing
`swim_predict.py` returns **UNABLE TO ASSESS** when pose visibility is too low, and
**UNCERTAIN** when confidence is below threshold, rather than forcing a wrong label. Output is a
full assessment: classification + confidence + posture, stability, drift, limb activity, symmetry,
pose quality.

## 7. Two cameras: late fusion, not blind merging
Model side-front and top **separately** (side = inclination, drift, stability; top = left/right
activity, symmetry, spread), then combine their class probabilities (late fusion). Validate the
fusion rather than averaging by default. Cluster on side-front only; never mix view geometries.

## 8. Clustering is not the final model
K-Means discovered behaviours without labels. It does not decide good vs bad: a cluster may hold
calm vertical floaters, another flat but hard-kicking swimmers. **Instructor labels** decide which
behaviours count as successful flotation.

## 9. The experiment, in order
1. Correct the target (passive flotation; angle/effort explanatory).
2. Copy all Teams/SharePoint videos to the server, verify integrity, build `manifest.csv`.
3. Identify participants and trials before extracting features.
4. Lock participant-level train/val/test; never touch test during development.
5. Freeze one feature-extractor version; run it on every development video.
6. Quality control: pose validity, confidence, corrupt frames, feature distributions.
7. Cluster on training data only; use it to pick diverse clips for labelling.
8. Collect instructor labels; double-label a subset; measure inter-rater kappa.
9. Train the champion/challenger with participant-grouped CV; select on macro-F1 + weighted kappa.
10. Freeze the winner (model, scaler, feature list, thresholds, confidence rules, versions).
11. Run the sealed test **once** (`swim_test.py`): macro-F1, balanced accuracy, weighted kappa,
    confusion, per-participant, confidence calibration.
12. Run one completely unseen showcase video for the live demo (`swim_predict.py`).

## 10. About sharing the test video
**Do not upload the sealed test video or its outputs while developing.** If we inspect it and then
change anything, it is no longer unseen. Develop on train + validation only. After the model is
frozen and the test has run once on the server, you can upload the frozen result for me to audit
(confusion matrix, feature values, overlay, failure cases). The inference itself runs on the server
against the frozen model.

## Commands
```bash
python build_manifest.py /opt/home/s4137946/swim_analysis/videos --out data/manifest.csv
python make_splits.py "features_side_front/"*.csv --labels labels.csv --out splits.csv
python swim_classify.py "features_side_front/"*.csv --out rule_out            # label-free interim model
python swim_train.py "features_side_front/"*.csv --labels labels.csv --splits splits.csv --out model_side
python swim_test.py  "features_side_front/"*.csv --labels labels.csv --model model_side/model.joblib --out test_report
python swim_predict.py "features_side_front/SHOWCASE_features.csv" --model model_side/model.joblib --out showcase
python irr_kappa.py labels_A.csv labels_B.csv --target float_label
```

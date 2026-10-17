# CLAUDE.md — Research Project Handoff

Read this before you do anything. It is the full context for continuing this project in any new Claude session or on any other account.

## Project

**Title:** Context-Aware Privacy for AR: Real-Time Selective Anonymization via Egocentric Interaction Detection on Edge Devices
**Author:** Akash Abeywickrame (S/20/302), 4th-year CS undergraduate, Dept. of Statistics & Computer Science, University of Peradeniya
**Deadline:** around October 2026. Treat it as urgent.
**Branch:** `phase-2-selective-anonymization` (main branch: `main`)

**Goal:** In first-person (wearable or AR) video, blur the faces of bystanders and leave the face of the person the wearer is talking to (the *interaction partner*) unblurred, in real time on edge hardware.

## Pipeline

```
YOLOv8n-face (core/detector.py)
  → SORT tracker (core/tracker.py)
  → Phase 2a geometry cues: face area_ratio + center_dist (core/interaction_cues.py, core/classifier.py)
  → Phase 2b landmark cues: MediaPipe Face Landmarker on YOLO crops (core/landmark_extractor.py)
       - yaw veto (head turned > INTERACT_MAX_YAW → not a partner)
       - MAR-based visual voice activity detection (mouth-movement speaking cue)
  → selective Gaussian blur on bystanders (core/anonymizer.py)
```

- Every parameter is in `config.py`. `LANDMARK_ENABLED = True` means Phase 2b is on. Setting it to False gives Phase 2a only.
- `INTERACT_LATCH = True`: once a track is confirmed as a partner it never reverts. Evaluation counts a track as a partner if it was classified as one at any point in the video.
- SORT track IDs restart for each video. A single extra false detection early in a video shifts every later ID, so a track-ID-based ground truth breaks whenever detection changes.

## Layout

| Path | Purpose |
|---|---|
| `main.py` | Runs the pipeline over all videos in `data/` (resumable; skips a video if its `*_selective.mp4` already exists). Flags: `--category`, `--video` |
| `config.py` | All thresholds and paths |
| `core/` | detector, tracker, classifier, interaction_cues, landmark_extractor, anonymizer, visualizer |
| `data/bystander/` (video1–7), `data/interaction/` (video1–16) | 23 input clips. video15/16 were added late and are **not** in the formal evaluation (21 clips) |
| `output/` | Selective, blanket, and debug videos |
| `logs/tracking_log.csv` | Per-frame, per-face log (~224k rows): video, category, frame, face_id, bbox, confidence, interacting |
| `eval/evaluate.py` | `--accuracy` (vs ground_truth CSVs) and `--fps --frames 300` benchmark |
| `eval/ground_truth/*.csv` | **Proxy** GT built from the system's own predictions. Circular, so do NOT report these numbers |
| `eval/generate_annotation_package.py` | Builds the blind human-annotation package (10 frames, 29 faces) |
| `eval/annotation_package/` | `frames/`, `annotator_template.csv`, `INSTRUCTIONS.txt`, `frame_track_map.csv` (private: holds the model's predictions, never share with annotators) |
| `eval/aggregate_annotations.py` | Human labels → Cohen's κ + PSR/IPR/Precision/Recall/F1 → `logs/human_annotation_metrics.csv` |
| `utils/finish_pipeline.py` | Post-pipeline: completion check, proxy-GT rebuild, package regeneration, evaluation, FPS |
| `utils/make_playable.py` | Re-encodes outputs to H.264 (the `mp4v` outputs won't play on Windows) |
| `utils/make_research_figures.py` | Dissertation figures |
| `chapters_4_5_6.tex` | Dissertation Results/Discussion/Conclusion chapters (LaTeX) |
| `methodology.tex`, `Intro & Lit & Methodology.txt` | Earlier chapters |
| `mid_review_presentation.tex` | Beamer slides |
| `models/` | `yolov8n-face.pt`, `face_landmarker.task` |

## Setup

```bash
pip install -r requirements.txt
pip install mediapipe   # required for Phase 2b; missing from requirements.txt
python main.py          # resumes; only unfinished videos are processed
```

Runs on Windows and CPU (`DEVICE = "cpu"`). Pass `encoding="utf-8", errors="replace"` to any `subprocess.run(..., text=True)`, because MediaPipe prints Unicode that cp1252 can't decode. Keep text files written by scripts ASCII-only (no em dashes).

## Metric definitions

- **PSR (Privacy Success Rate)** = bystanders correctly blurred / all GT bystanders = TN / (TN + FP)
- **IPR (Interaction Preservation Rate)** = partners correctly left unblurred / all GT partners = TP / (TP + FN), which equals Recall
- Positive class = interaction partner

## Status (as of 2026-10-17)

### Done
- All 23 videos processed. Outputs and the full tracking log exist.
- **Phase 2a results are in the dissertation:** 66 tracks (26 partners, 40 bystanders), TP=22, TN=35, FP=5, FN=4 → PSR 87.50%, IPR 84.62%, Precision 81.48%, F1 83.02%.
- **FPS table is in the dissertation:** 86 ms/frame, 11.62 FPS on the laptop CPU, about 3.3 FPS projected on a Raspberry Pi 5. Do **not** replace it with the later 5.07 FPS / 197 ms run in `logs/fps_benchmark.csv`, because that run was thermally throttled.
- The annotation package and zip (`eval/annotation_package.zip`) are generated.

### Remaining (in order)
1. **Human annotation (manual, owner must do it).** Label every face in `eval/annotation_package/frames/frame_001..010.png` as `p` or `b` in `annotator_template.csv`, then save the file as `eval/annotation_package/annotations_researcher.csv`. For inter-annotator agreement, collect `annotations_NAME.csv` files from two or three more people.
2. Run:
   ```bash
   python eval/aggregate_annotations.py eval/annotation_package/annotations_*.csv
   # with 2+ annotators add: --min-annotators 2
   ```
   Then read `logs/human_annotation_metrics.csv`.
3. **Fill the Phase 2b placeholders (`\_`, `\_\_`) in `chapters_4_5_6.tex`:**
   - the confusion matrix `tab:confusion_2b` (about lines 139–141)
   - the metrics table `tab:metrics_2b` (about lines 160–164)
   - the baseline comparison `tab:vs_baseline`, Phase 2b row (about line 195)
   - The Phase 2b section text says it re-evaluates on "the same 66 ground-truth tracks". That is no longer true, because evaluation now uses the human-annotated sample of 29 faces in 10 frames. Rewrite that text to describe the real protocol (blind annotation, sample size, κ if available) and state the small sample as a limitation.
4. **Dataset table `tab:dataset` (about lines 51–52):** the per-category counts are `--`. The original 66-track manual GT was never committed and is lost. Either estimate the counts (Interaction: 14 clips / about 45 tracks / 26 partners / about 19 bystanders; Bystander: 7 clips / about 21 tracks / 0 partners / about 21 bystanders), say so explicitly, or collapse the table to totals only.
5. Optional: commit the work. Many files are still untracked (see `git status`). Don't commit LaTeX build files (`*.aux`, `*.log`, `*.nav`, `*.out`, `*.snm`, `*.toc`) or large zips/videos.

## Rules for working on this project

- **Never report proxy-GT metrics** (`logs/metrics_report.csv`, roughly PSR 98% / IPR 100%) as results. They are circular.
- **Never invent annotation labels or dissertation numbers.** If human labels don't exist yet, stop and ask the owner to provide them.
- Don't change `config.py` thresholds: the reported results depend on the current values. A change would mean re-running all 23 videos, which takes hours on CPU.
- Don't delete `logs/tracking_log.csv` or `output/`. Regenerating them takes many hours.
- The Phase 2a numbers are final. Leave them unchanged.

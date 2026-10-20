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
| `data/bystander/` (video1–7), `data/interaction/` (video1–16) | 23 input clips, all collected from YouTube/web (none recorded by the owner). video15/16 were added late and are **not** in the formal evaluation (21 clips) |
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

## Status (as of 2026-10-18)

### Done
- All 23 videos processed. Outputs and the full tracking log exist.
- **Phase 2a results are in the dissertation:** 66 tracks (26 partners, 40 bystanders), TP=22, TN=35, FP=5, FN=4 → PSR 87.50%, IPR 84.62%, Precision 81.48%, F1 83.02%.
- **FPS table is in the dissertation:** 86 ms/frame, 11.62 FPS on the laptop CPU, about 3.3 FPS projected on a Raspberry Pi 5. Do **not** replace it with the later 5.07 FPS / 197 ms run in `logs/fps_benchmark.csv`, because that run was thermally throttled.
- **Phase 2b speed measured (2026-10-20, plugged in):** `python eval/evaluate.py --fps --phase 2b` runs MediaPipe and times it separately. MediaPipe = **20.1 ms per face** (21.1 and 19.1 ms over two runs; ~1 face/frame on interaction/video1). That falls inside the methodology's 15-25 ms estimate. Phase 2b ≈ 86.0 + 19.9 ≈ 106 ms ≈ **9.4 FPS** on the laptop (the fastest direct run was 9.9 FPS), ≈ **2.7 FPS** projected on the Pi 5 (×3.5). These figures are in `chapters_4_5_6.tex` (FPS section, discussion, limitations, conclusion). YOLO time is noisy between runs (65-150 ms) because of background apps, which is why the 2b total is baseline + measured landmark cost. An earlier battery-power run gave 33.5 ms/face; ignore it.
- `Intro & Lit & Methodology.txt` contains an OLD copy of the results text (still says 15–25 ms/face and "real time on desktop"). `chapters_4_5_6.tex` is the current version.
- The annotation package and zip (`eval/annotation_package.zip`) are generated.

### Remaining (in order)
1. **Human annotation: DONE (2026-10-18).** Four annotators: `annotations_akash.csv`, `annotations_gayantha.csv`, `annotations_udari.csv` (2 blanks), `annotations_vinuka.csv`.
   - `annotations_researcher.csv` is a **blank** copy of the template. Exclude it; don't use `annotations_*.csv` as the glob.
   - Command used:
     ```bash
     python eval/aggregate_annotations.py eval/annotation_package/annotations_{akash,gayantha,udari,vinuka}.csv --min-annotators 2
     ```
   - Majority vote (a 2-2 tie counts as bystander; only frame_001 face 3 tied) → 29 faces/tracks, **7 partners, 22 bystanders**.
   - **Phase 2b results:** TP=4, TN=22, FP=0, FN=3 → **PSR 100.00%, IPR 57.14%, Precision 100.00%, Recall 57.14%, F1 72.73%** (`logs/human_annotation_metrics.csv`).
   - Pairwise Cohen's κ: 0.307–0.724, mean ≈ 0.59 (moderate); 5 of 6 pairs moderate or better, akash–vinuka only fair (0.307). See `eval/annotation_package/agreement_report.csv`.
   - Caveat for the write-up: a small sample (only 7 partners, so each FN costs about 14 IPR points). It is not directly comparable to Phase 2a's 66-track evaluation.
   - Do NOT use `eval/annotation_package/form_build/` (an older 65-frame Google Form set; its IDs don't match `frame_track_map.csv`).
2. (merged into step 1)
3. **Phase 2b write-up: DONE (2026-10-20)** in `chapters_4_5_6.tex`. Includes: the evaluation protocol (why the 66-track GT couldn't be reused, blind 4-annotator study, κ), a sample table (`tab:sample_2b`), the confusion matrix, the metrics table, the baseline row, per-category results, 95% Wilson CIs, the discussion, a Phase 2b failure analysis, limitations, conclusion and future work. Framing: privacy-first (PSR 100%, IPR 57%); results are indicative because the sample is small and the evaluation sets differ.
   - Verified facts used: all 3 FNs have area_ratio 0.36-0.72% < τ_area 1.2% (geometry gate, NOT the landmark veto). Two FNs are one person with duplicate overlapping tracks (frame_003 = interaction/video10). The third FN is frame_002 = interaction/video1, at the frame edge, yaw ≈ 25°. MediaPipe returned landmarks for only 50.8% of 224,519 logged detections.
   - Wilson 95% CIs: 2b PSR 85.1-100, 2b IPR 25.0-84.2; 2a PSR 73.9-94.5, 2a IPR 66.5-93.9.
   - Pre-edit backup: the session scratchpad (`chapters_4_5_6.before_2b.tex`); otherwise use git once committed.
4. **Dataset table: DONE.** It shows only what is known: partners 26 (interaction) / 0 (bystander), total bystanders 40, with a footnote saying the per-category bystander split wasn't retained. No estimated numbers.
   - **Resolved (2026-10-20):** the contradictory Phase 2a FN cause (ii) was removed. The dataset text was rewritten: the owner **did not record any footage**; all clips were **collected from YouTube and other web sources**. Bystander clips are outdoor street/market walking footage (5.9-78.1 s). Interaction clips are 5-minute (300 s) indoor segments, mostly card/board games at a table plus shopping in small stores. Source resolution ranges from 1280x720 to 3840x2160; everything is resized to 1280x720. The dataset table now has no unknown cells (clips 14/7/21; tracks 26/40/66). The owner doesn't want "--" or footnote symbols in tables; put notes in captions instead.
5. Optional: commit the work. Many files are still untracked (see `git status`). Don't commit LaTeX build files (`*.aux`, `*.log`, `*.nav`, `*.out`, `*.snm`, `*.toc`) or large zips/videos.

## Rules for working on this project

- **Intro, literature review and methodology are FROZEN** (owner, 2026-10-20). Do not edit `methodology.tex` or `Intro & Lit & Methodology.txt`. Write the remaining chapters (`chapters_4_5_6.tex`) so they agree with them: CPU-only, a laptop/desktop CPU benchmark plus a ×3.5 Pi 5 scaling estimate, PSR/IPR, the blanket-blur baseline, and latching framed as favouring IPR.

- **Never report proxy-GT metrics** (`logs/metrics_report.csv`, roughly PSR 98% / IPR 100%) as results. They are circular.
- **Never invent annotation labels or dissertation numbers.** If human labels don't exist yet, stop and ask the owner to provide them.
- Don't change `config.py` thresholds: the reported results depend on the current values. A change would mean re-running all 23 videos, which takes hours on CPU.
- Don't delete `logs/tracking_log.csv` or `output/`. Regenerating them takes many hours.
- The Phase 2a numbers are final. Leave them unchanged.

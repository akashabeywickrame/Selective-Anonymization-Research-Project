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
- **Battery-power FPS (added 2026-10-22):** the 2026-10-19 runs were genuinely on battery (Windows reported battery, 90%): Phase 2a 9.1 and 7.5 FPS, Phase 2b 6.1 FPS (163.8 ms, MediaPipe 33.5 ms/face); they were affected by heat. Reported in `tab:power` as a worst case next to the mains figures. Don't claim any other run was unplugged: as of 2026-10-22 the laptop was on AC.
- **Smart-glasses hardware comparison (2026-10-22):** `tab:glasses` and the paragraph in the discussion's sec:disc_fps compare against Ray-Ban Meta / Display (Snapdragon AR1 Gen 1, Hexagon NPU, 2 GB RAM in Display), Spectacles 5 (2x Snapdragon, ~45 min) and Aria Gen 2 (custom co-processor). Key argument: Pi 5 ≈ AR1-class CPU; CPU-only results are a worst case because the glasses have NPUs; battery is the real limit; on-device processing avoids sending faces to the cloud. The 7 new `ibitem`s are in a comment block at the END of chapters_4_5_6.tex. The owner must paste them into the bibliography (which lives in the frozen file).
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
5. **Second, larger blind evaluation: video clips (package built 2026-10-21, WAITING for annotators).**
   - Why: the first study used 10 still frames (29 faces, only 7 partners, IPR CI 25-84%). Stills make it hard to judge who is talking.
   - `python eval/generate_clip_package.py` builds `eval/clip_package/`: 104 clips (83 interaction, 21 bystander), 7 s each = a 1 s intro still + 6 s of the ORIGINAL video with SOUND, and a yellow box on one face. Annotators open `annotate.html` (P/B/U buttons; it downloads `annotations_NAME.csv`). Share `eval/clip_package.zip` (67 MB, excludes the map).
   - Sampling (never looks at system output): per interaction clip, the 4 longest eligible tracks plus 2 random ones; per bystander clip, the 4 longest. Eligible = ≥60 logged frames and median face side ≥32 px. Seed 2026.
   - `clip_map.csv` is PRIVATE. System prediction per clip = face unblurred in ≥50% of the target's logged frames in the window. Spread: interaction 40 unblurred / 43 blurred; bystander 1 / 20.
   - When files come back, save them as `eval/clip_package/annotations_<name>.csv`, then run `python eval/aggregate_clip_annotations.py eval/clip_package/annotations_*.csv` → majority vote, Cohen's + Fleiss' κ, PSR/IPR/F1 with Wilson CIs, per category → `logs/clip_annotation_metrics.csv`.
   - NOT yet in the dissertation. The owner's chapters are frozen except `chapters_4_5_6.tex`; add the results there as a second, larger evaluation once real labels exist.
   - Never show annotators the system output (that would bias the labels).
   - **CURRENT RESULTS (2026-10-22, 5 annotators: akash, janindu, yushmantha, udari, malindu); these supersede the 3/4-annotator figures below.** Command: `python eval/aggregate_clip_annotations.py eval/clip_package/annotations_{akash,janindu,udari,yushmantha,malindu}.csv`. Pairwise κ 0.818-0.957, **Fleiss 0.887**, no ties. Majority: 102 clips, **69 partners, 33 bystanders** (clip_049 and clip_079 flipped from tied-bystander to partner). Live: TP 39, FN 30, FP 1, TN 32 → **PSR 96.97% [84.7-99.5], IPR 56.52% [44.8-67.6], Prec 97.50%, F1 71.56%** (interaction clips: PSR 100% (12/12), IPR 56.52%; bystander clips: 20/21). Second pass: retro TP 49/FN 20/FP 2/TN 31 → PSR 93.94 [80.4-98.3], IPR 71.01 [59.4-80.4], F1 81.67; **+friend (10 s) TP 54/FN 15/FP 2/TN 31 → PSR 93.94, IPR 78.26 [67.2-86.4], Prec 96.43, F1 86.40** (leaks clip_050, clip_062). Sensitivity 10-60 s: PSR 93.94, IPR 72.46-78.26; 5 s adds a leak. FNs: 25 of 30 below 1.2% area (median 0.91 vs TP 1.72), 10 confirmed later; the second pass recovers 15 of 30. The dissertation is fully updated (tab:clip_kappa now has 10 pairs, E = malindu). Pre-edit backup: scratchpad `chapters_before_five.tex`.
   - **clip_067 and clip_071 are FALSE DETECTIONS, not people** (owner spotted, verified): 067 = a face printed on a magazine cover (interaction/video6, blurred); 071 = a product package on a shop shelf (interaction/video7, track 169, kept unblurred as a "partner": conf median 0.53, landmarks in 1/73 frames). Both are excluded from scoring. The dissertation describes them in the clip protocol and the video-clip failure modes (1.9% of sampled tracks).
   - **UPDATE 2026-10-22: a 4th annotator (udari) was added.** Run `python eval/aggregate_clip_annotations.py eval/clip_package/annotations_{akash,janindu,udari,yushmantha}.csv`. Pairwise κ 0.818-0.957, **Fleiss 0.903**. Two 2-2 ties (clip_049, clip_079) → bystander. **The majority labels are identical to the 3-annotator ones, so every metric (live and second pass) is unchanged.** udari left 3 blanks (clip_009, 067, 071). The dissertation is updated to 4 annotators (tab:clip_kappa has 6 pairs, A=akash, B=janindu, C=yushmantha, D=udari).
   - **RESULTS (2026-10-22), 3 annotators: akash, janindu, yushmantha.** `python eval/aggregate_clip_annotations.py eval/clip_package/annotations_{akash,janindu,yushmantha}.csv`. κ pairwise 0.913-0.957, Fleiss 0.942. Majority vote: 102 clips (2 had no majority label), 67 partners, 35 bystanders. Overall TP 38, FN 29, FP 2, TN 33 → **PSR 94.29% [81.4-98.4], IPR 56.72% [44.8-67.9], Precision 95.00%, F1 71.03%**. Interaction clips: PSR 92.86%, IPR 56.72%. Bystander clips: 20/21 blurred (PSR 95.24%). FPs: clip_049 (interaction/video9, unblurred 88%) and clip_062 (bystander/video3, exactly 50%). FN cause: 24 of 29 missed partners have median face area < 1.2% (FN median 0.92% vs TP 1.71%), i.e. the area threshold again; 10 of the 29 were confirmed as partners later in the same video (outside the 6 s window). akash's file (23:00) predates the definition text update (23:04), but akash is the owner who set the definition. **Written into the dissertation 2026-10-22** (sec:results_clips, tab:clip_kappa, tab:confusion_clips, tab:metrics_clips).
   - **Partner definition (owner, 2026-10-21):** an interaction partner is any member of the camera wearer's group/conversation. That includes friends at the same table or in the same game who are NOT speaking at the moment, since they should be preserved. A bystander is anyone not in the wearer's group (strangers, background, other groups). The clip package's INSTRUCTIONS.txt and annotate.html use this definition (page storage key v2). It is consistent with the frozen methodology ("active interaction partner", "close friend engaged in conversation", latching because the partner "remains socially present"). NOTE for the write-up: the first still-frame study used the narrower "actively talking TO the wearer" wording, so state the difference when reporting the clip study.
   - `python eval/verify_clip_predictions.py` re-derives each clip's system decision from the actual selective output video (selective crop closer to the blanket crop or to the original crop?). After repairing video3, **all 104 clips agree exactly with the log** (0 differences), so `system_prediction` in clip_map.csv is output-verified. Six clips are borderline (unblurred 24-79% of the window) because the latch flipped mid-window: clip_012, 034 (0.506), 050, 062 (bystander/video3, exactly 0.50 → counted as unblurred), 066, 081.
   - **Broken outputs found (2026-10-21):** `output/interaction/video3_*.mp4` and `video15_*.mp4` had no moov atom (an interrupted run; main.py's resume logic skipped them because the files existed). Their logs are complete. **Both restored 2026-10-21** with `python -m utils.restore_outputs_from_log interaction video3.mp4` (and video15.mp4), which re-renders from the LOGGED decisions (don't use rerender_from_log.py: it re-classifies without landmarks and mixes up categories).
6. Optional: commit the work. Many files are still untracked (see `git status`). Don't commit LaTeX build files (`*.aux`, `*.log`, `*.nav`, `*.out`, `*.snm`, `*.toc`) or large zips/videos.

## Retroactive latch (added 2026-10-22, owner: "once an interactor, always an interactor")

- The live latch (`INTERACT_LATCH`) only works forward: frames BEFORE confirmation stay blurred. 10 of the 29 clip-study misses were partners confirmed later in the same video.
- `INTERACT_RETRO_LATCH = True` in config.py plus `python -m utils.render_retro_latch <cat> <video>` / `--all`: an offline second pass that preserves a confirmed partner for its whole track. It reads the log and writes `output/<cat>/<video>_selective_retro.mp4`. main.py and all reported results are unchanged. It only works on recorded video (it needs the future frames).
- Clip-study effect (same 102 labels): TP 48, FN 19, FP 3, TN 32 → **PSR 91.43% [77.6-97.0], IPR 71.64% [59.9-81.0], Precision 94.12%, F1 81.36%** (vs forward latch: PSR 94.29, IPR 56.72, F1 71.03). One new leak: clip_050 (interaction/video13). Verified on video10's output: all 3 missed partners are fully visible.
- **Friend-group rule (2026-10-22, owner: friends nearby should stay unblurred even if not talking):** `FRIEND_GROUP_ENABLED`, `FRIEND_MIN_SECONDS = 10`, `FRIEND_MIN_COPRESENCE = 0.5` in config.py, applied in the same second pass. A track becomes a partner if it's in view ≥10 s AND on screen with a confirmed partner for ≥50% of its frames. Clip study (implementation-verified): **TP 53, FN 14, FP 3, TN 32 → PSR 91.43%, IPR 79.10%, Precision 94.64%, F1 86.18%**. No new leaks (the same 3 as retro-only: clip_049, 050, 062). Sensitivity: with the partner-present condition, every threshold from 10 to 60 s keeps PSR 91.43% (IPR 73-79%); 5 s adds a leak. The time-only rule (no partner condition) was rejected because it would keep strangers who just stay put (shopkeepers, transport).
- Caveat for the write-up: this was designed after seeing the clip-study errors, so it's evaluated on the same labels. Present it as a post-hoc extension, not as the main result.

## Dissertation state (2026-10-22)

`chapters_4_5_6.tex` now has: 4.4 Video-Clip Evaluation (`sec:results_clips`), 4.5 Offline Second Pass (`sec:results_secondpass`, `tab:secondpass`), the baseline table with all Phase 2b rows, a discussion paragraph answering the RQ, video-clip failure modes, limitations (annotation protocol and the post-hoc second pass), the conclusion, and future work (fresh-clip test; live group recognition via a short output delay). Annotators are anonymised as A (owner), B and C in tab:clip_kappa. Pre-edit backup: the session scratchpad `chapters_before_clip.tex`.

## Rules for working on this project

- **University thesis template (uopThesis class, given by the university, 2026-10-26):** book-style (`\chapter`), one file per chapter in `chapters/` added to main.tex with `\include{chapters/X.tex}`, natbib author-year citations (`\citep`), BibTeX `References.bib` with `apalike`, images in `images/`. `chapters_4_5_6.tex` stays the working copy; after every edit run `python -m utils.export_thesis_chapters`, which writes `thesis/chapters/{Results,Discussion,Conclusion}.tex` (section levels shifted, `Section~4.x` → `ef`, `\cite` → `\citep`) and `thesis/References_chapters_4_6.bib` (11 entries). The frozen Intro/Lit/Methodology are still in article format with a numeric `thebibliography`; they need a FORMAT-only conversion (content unchanged) before they fit the template. Ask the owner first.

- **Tables must fit the text width** (A4, 40 mm left / 15 mm right margin, about 155 mm). `chapters_4_5_6.tex` defines `itwidth{...}` at the top, which shrinks a tabular only when it is wider than `\linewidth`. Wrap any new wide table in it, or use `p{..\linewidth}` columns for text-heavy tables (as in `tab:glasses`, which was Table 16 and overflowed).

- **No file names, script names or code identifiers in dissertation text** (owner, 2026-10-21). Examiners don't know the repo, so describe things in plain words (e.g. "a high-resolution timer", "interaction latching").

- **No long dashes in dissertation text** (owner, 2026-10-21): no em dashes (`---`) and no en dashes (`--`). Use a short hyphen for number ranges (`15-25~ms`), and rewrite sentences with commas or full stops instead of dashes.

- **Write dissertation text in first person singular** ("I collected", "my research"), never "the researcher" or "the author". This matches the frozen chapters, which say "my research" (owner, 2026-10-20).

- **Intro, literature review and methodology are FROZEN** (owner, 2026-10-20). Do not edit `methodology.tex` or `Intro & Lit & Methodology.txt`. Write the remaining chapters (`chapters_4_5_6.tex`) so they agree with them: CPU-only, a laptop/desktop CPU benchmark plus a ×3.5 Pi 5 scaling estimate, PSR/IPR, the blanket-blur baseline, and latching framed as favouring IPR.

- **Never report proxy-GT metrics** (`logs/metrics_report.csv`, roughly PSR 98% / IPR 100%) as results. They are circular.
- **Never invent annotation labels or dissertation numbers.** If human labels don't exist yet, stop and ask the owner to provide them.
- Don't change `config.py` thresholds: the reported results depend on the current values. A change would mean re-running all 23 videos, which takes hours on CPU.
- Don't delete `logs/tracking_log.csv` or `output/`. Regenerating them takes many hours.
- The Phase 2a numbers are final. Leave them unchanged.

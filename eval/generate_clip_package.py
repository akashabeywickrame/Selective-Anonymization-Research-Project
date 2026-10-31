# =============================================================================
# eval/generate_clip_package.py — Blind VIDEO-CLIP annotation package
# =============================================================================
#
# Usage:
#   python eval/generate_clip_package.py
#
# WHY
# ---
# The first annotation package (generate_annotation_package.py) showed one
# still frame per clip. From a single still it is hard to tell who is talking,
# and the sample was small (29 faces, 7 partners). This package shows short
# video clips WITH SOUND, one per sampled face track, so annotators can see and
# hear whether the highlighted person is talking with the camera wearer.
#
# WHAT IT DOES
# ------------
# 1. Samples face tracks from the 21 formal evaluation clips (bystander 1-7,
#    interaction 1-14). Selection never looks at the system's decision:
#      - interaction clips: the 4 longest eligible tracks + 2 random others
#      - bystander clips  : the 4 longest eligible tracks
#    "Eligible" = logged for >= MIN_TRACK_FRAMES frames, median face side
#    >= MIN_FACE_PX (in the 1280x720 processing frame).
# 2. For each track, cuts a CLIP_SECONDS window centred on the track from the
#    ORIGINAL video (no blur), draws one plain yellow box on the target face
#    only, and keeps the original audio.
# 3. Shuffles clip order (fixed seed) so categories are mixed and unlabelled.
# 4. Writes:
#      eval/clip_package/clips/clip_001.mp4 ...   (H.264 + AAC, plays anywhere)
#      eval/clip_package/annotate.html            (click-through labelling page)
#      eval/clip_package/annotator_template.csv   (fallback if HTML can't be used)
#      eval/clip_package/INSTRUCTIONS.txt
#      eval/clip_package/clip_map.csv             (PRIVATE: system decisions)
#
# System prediction for a clip = the face was shown UNBLURRED for most of the
# window (interacting == 1 in >= 50% of the target's logged frames in the
# window), i.e. what a viewer of the system's output would actually see.
# =============================================================================

import csv
import json
import random
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import imageio_ffmpeg

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    LOGS_DIR, INTERACTION_INPUT_DIR, BYSTANDER_INPUT_DIR,
    FRAME_WIDTH, FRAME_HEIGHT,
)

FFMPEG   = imageio_ffmpeg.get_ffmpeg_exe()
LOG_PATH = LOGS_DIR / "tracking_log.csv"
PKG_DIR  = Path(__file__).resolve().parent / "clip_package"
CLIPS    = PKG_DIR / "clips"
MAP_FILE = PKG_DIR / "clip_map.csv"

# Formal evaluation set (video15/16 were added later and are excluded)
FORMAL = {"bystander": range(1, 8), "interaction": range(1, 15)}

N_LONGEST        = 4      # longest eligible tracks per clip
N_RANDOM_INTER   = 2      # extra random tracks per interaction clip
MIN_TRACK_FRAMES = 60     # ~2 s at 30 fps
MIN_FACE_PX      = 32     # median min(w, h) in the 1280x720 frame
CLIP_SECONDS     = 6.0
OUT_HEIGHT       = 540
BOX_HOLD_FRAMES  = 8      # keep showing the last box through short gaps
INTRO_SECONDS    = 1.0    # still frame at the start showing whom to watch
SEED             = 2026


# =============================================================================
# Load tracks
# =============================================================================

def _load_tracks():
    """(category, video, track_id) -> {frame: (x1, y1, x2, y2, interacting)}"""
    tracks = defaultdict(dict)
    with open(LOG_PATH, newline="") as f:
        for r in csv.DictReader(f):
            cat, vid = r["category"], r["video"]
            if int(vid[5:-4]) not in FORMAL.get(cat, ()):
                continue
            tracks[(cat, vid, int(r["face_id"]))][int(r["frame"])] = (
                int(r["x1"]), int(r["y1"]), int(r["x2"]), int(r["y2"]),
                int(r["interacting"]),
            )
    return tracks


def _median_side(frames):
    return statistics.median(min(x2 - x1, y2 - y1) for x1, y1, x2, y2, _ in frames.values())


def _select(tracks):
    rng = random.Random(SEED)
    by_video = defaultdict(list)
    for key, frames in tracks.items():
        if len(frames) >= MIN_TRACK_FRAMES and _median_side(frames) >= MIN_FACE_PX:
            by_video[key[:2]].append(key)

    chosen = []
    for (cat, vid) in sorted(by_video, key=lambda k: (k[0], int(k[1][5:-4]))):
        keys = sorted(by_video[(cat, vid)], key=lambda k: -len(tracks[k]))
        pick = keys[:N_LONGEST]
        if cat == "interaction":
            rest = keys[N_LONGEST:]
            pick += rng.sample(rest, min(N_RANDOM_INTER, len(rest)))
        chosen += pick
    rng.shuffle(chosen)
    return chosen


# =============================================================================
# Render one clip
# =============================================================================

def _video_path(cat, vid):
    return (INTERACTION_INPUT_DIR if cat == "interaction" else BYSTANDER_INPUT_DIR) / vid


def _render(clip_id, key, frames):
    cat, vid, tid = key
    src = _video_path(cat, vid)
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sw, sh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Window centred on the middle logged frame of the track
    logged = sorted(frames)
    mid = logged[len(logged) // 2]
    half = int(CLIP_SECONDS * fps / 2)
    start = max(0, min(mid - half, total - 2 * half))
    end = min(total, start + 2 * half)

    # Log boxes are in the 1280x720 processing frame; map back to source shape
    sx = sw / FRAME_WIDTH if FRAME_WIDTH > 0 else 1.0
    sy = sh / FRAME_HEIGHT if FRAME_HEIGHT > 0 else 1.0
    ow = int(round(sw * OUT_HEIGHT / sh / 2)) * 2
    k = OUT_HEIGHT / sh

    def draw_box(img, box):
        x1, y1, x2, y2 = box
        p1 = (int(x1 * sx * k), int(y1 * sy * k))
        p2 = (int(x2 * sx * k), int(y2 * sy * k))
        cv2.rectangle(img, p1, p2, (0, 0, 0), 5)
        cv2.rectangle(img, p1, p2, (0, 230, 255), 3)

    def draw_label(img, text):
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (0, 0), (tw + 16, 30), (20, 20, 20), cv2.FILLED)
        cv2.putText(img, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (230, 230, 230), 1, cv2.LINE_AA)

    name = clip_id.replace("_", " ").title()
    out = CLIPS / f"{clip_id}.mp4"
    cmd = [
        FFMPEG, "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{ow}x{OUT_HEIGHT}",
        "-r", f"{fps:.3f}", "-i", "-",
        # audio starts after the INTRO_SECONDS still frame
        "-itsoffset", f"{INTRO_SECONDS:.3f}",
        "-ss", f"{start / fps:.3f}", "-t", f"{(end - start) / fps:.3f}", "-i", str(src),
        "-map", "0:v", "-map", "1:a?",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "28", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart",
        "-t", f"{INTRO_SECONDS + (end - start) / fps:.3f}",
        str(out),
    ]
    ff = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    # Intro: hold the first frame in which the target is boxed, so annotators
    # know whom to watch before the clip plays
    first = min(f for f in frames if start <= f < end)
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    ok, intro = cap.read()
    if ok:
        intro = cv2.resize(intro, (ow, OUT_HEIGHT))
        draw_box(intro, frames[first][:4])
        draw_label(intro, f"{name}: watch the person in the yellow box")
        for _ in range(int(round(INTRO_SECONDS * fps))):
            ff.stdin.write(intro.tobytes())

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    last_box, last_seen = None, -10**9
    in_window = []
    for fi in range(start, end):
        ok, img = cap.read()
        if not ok:
            break
        img = cv2.resize(img, (ow, OUT_HEIGHT))
        if fi in frames:
            x1, y1, x2, y2, inter = frames[fi]
            last_box, last_seen = (x1, y1, x2, y2), fi
            in_window.append(inter)
        if last_box is not None and fi - last_seen <= BOX_HOLD_FRAMES:
            draw_box(img, last_box)
        draw_label(img, name)
        try:
            ff.stdin.write(img.tobytes())
        except BrokenPipeError:
            cap.release()
            print("    ffmpeg:", ff.stderr.read().decode("utf-8", "replace").strip()[-300:])
            raise
    cap.release()
    ff.stdin.close()
    err = ff.stderr.read().decode("utf-8", "replace").strip()
    if ff.wait() != 0:
        raise RuntimeError(f"ffmpeg failed for {clip_id}: {err}")

    frac = sum(in_window) / len(in_window) if in_window else 0.0
    return {
        "clip_id": clip_id, "category": cat, "video": vid, "track_id": tid,
        "start_frame": start, "end_frame": end - 1, "fps": round(fps, 3),
        "track_frames": len(frames),
        "window_frames_logged": len(in_window),
        "frac_unblurred": round(frac, 3),
        "system_prediction": 1 if frac >= 0.5 else 0,
        "system_ever_partner": 1 if any(v[4] for v in frames.values()) else 0,
    }


# =============================================================================
# Annotator files
# =============================================================================

INSTRUCTIONS_TXT = """ANNOTATION INSTRUCTIONS (video clips)
=====================================

You will watch short video clips filmed by a camera worn by a person
(the "camera wearer"). In each clip ONE face is marked with a yellow box.
Please turn the SOUND ON.

For the face in the yellow box, decide:

  p  =  Interaction partner
        This person is part of the camera wearer's group or conversation.
        That includes someone talking to the camera wearer, AND someone who
        is with them but not speaking right now, for example a friend
        sitting at the same table, playing the same game, or listening to
        the conversation.

  b  =  Bystander
        This person is NOT part of the camera wearer's group. For example a
        stranger walking past, someone in the background, or someone in a
        different group talking to other people.

If you genuinely cannot tell, leave it blank ("Can't tell").

HOW TO DO IT
------------
Easiest: open annotate.html in Chrome or Edge (double-click it). Type your
name, watch each clip, click a button, and at the end click
"Download my answers". Send the downloaded annotations_YOURNAME.csv back.

If the page doesn't work: play the files in clips/ yourself, fill the label
column of annotator_template.csv with p or b, and save it as
annotations_YOURNAME.csv.

NOTES
-----
- There is no right or wrong answer; use your best judgement.
- Judge only the person in the yellow box, and only what happens in the clip.
- Each clip is independent.
"""


def _write_html(clip_ids):
    html = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Clip Annotation</title>
<style>
:root{--bg:#f6f6f4;--card:#fff;--ink:#1d1d1b;--mut:#6b6b66;--acc:#1f6feb;--p:#1a7f37;--b:#b42318;--line:#e2e2dc}
@media (prefers-color-scheme:dark){:root{--bg:#141413;--card:#1e1e1c;--ink:#ecebe6;--mut:#a3a29b;--line:#33332f}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 system-ui,sans-serif}
main{max-width:900px;margin:0 auto;padding:16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;margin-bottom:16px}
video{width:100%;border-radius:8px;background:#000;display:block}
.row{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
button{font:inherit;border:1px solid var(--line);border-radius:8px;padding:10px 14px;background:var(--card);color:var(--ink);cursor:pointer}
button.p{border-color:var(--p);color:var(--p)}button.b{border-color:var(--b);color:var(--b)}
button.sel.p{background:var(--p);color:#fff}button.sel.b{background:var(--b);color:#fff}button.sel.u{background:var(--mut);color:#fff}
input{font:inherit;padding:8px 10px;border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--ink)}
.mut{color:var(--mut)}.bar{height:6px;background:var(--line);border-radius:3px;overflow:hidden}.bar i{display:block;height:100%;background:var(--acc)}
kbd{border:1px solid var(--line);border-radius:4px;padding:0 5px;font-size:13px}
</style></head><body><main>
<div class="card">
<h1 style="margin:0 0 8px;font-size:22px">Is this person with the camera wearer?</h1>
<p style="margin:0 0 8px">Each clip is filmed by a camera worn by a person. <b>One face has a yellow box.</b>
Turn the <b>sound on</b>. Decide whether that person is an <b>interaction partner</b> or a <b>bystander</b>.
Judge only what happens in the clip.</p>
<ul style="margin:0 0 8px;padding-left:20px">
<li><b>Interaction partner:</b> part of the camera wearer's group or conversation. This includes someone
talking to the camera wearer <i>and</i> someone who is with them but not speaking right now, such as a
friend at the same table, playing the same game, or listening.</li>
<li><b>Bystander:</b> not part of the camera wearer's group, such as a stranger walking past, someone in the
background, or someone in a different group.</li></ul>
<p class="mut" style="margin:0">Keys: <kbd>P</kbd> partner &middot; <kbd>B</kbd> bystander &middot; <kbd>U</kbd> can't tell &middot;
<kbd>&larr;</kbd>/<kbd>&rarr;</kbd> previous/next &middot; <kbd>R</kbd> replay. Your progress is saved in this browser.</p>
<div class="row" style="margin-top:12px"><label for="name">Your name</label>
<input id="name" placeholder="e.g. kasun" autocomplete="off"></div>
</div>
<div class="card">
<div class="row" style="justify-content:space-between;margin-bottom:8px">
<b id="title">Clip</b><span class="mut" id="count"></span></div>
<div class="bar" style="margin-bottom:12px"><i id="prog" style="width:0"></i></div>
<video id="vid" controls playsinline></video>
<div class="row" style="margin-top:12px">
<button class="p" data-l="p">Interaction partner (P)</button>
<button class="b" data-l="b">Bystander (B)</button>
<button class="u" data-l="">Can't tell (U)</button></div>
<div class="row" style="margin-top:12px;justify-content:space-between">
<div class="row"><button id="prev">&larr; Previous</button><button id="next">Next &rarr;</button></div>
<button id="dl">Download my answers</button></div>
<p class="mut" id="msg" style="margin:8px 0 0"></p>
</div></main>
<script>
const CLIPS = __CLIPS__;
const KEY = "clip-annotation-v2";
let st = {i:0, name:"", lab:{}};
try { const s = JSON.parse(localStorage.getItem(KEY)); if (s) st = Object.assign(st, s); } catch(e) {}
const $ = id => document.getElementById(id);
const vid = $("vid");
function save(){ try { localStorage.setItem(KEY, JSON.stringify(st)); } catch(e) {} }
function show(){
  const id = CLIPS[st.i];
  $("title").textContent = id.replace("_"," ").replace("clip","Clip");
  const done = CLIPS.filter(c => c in st.lab).length;
  $("count").textContent = (st.i+1) + " of " + CLIPS.length + " \\u00b7 " + done + " answered";
  $("prog").style.width = (100*done/CLIPS.length) + "%";
  vid.src = "clips/" + id + ".mp4"; vid.play().catch(()=>{});
  document.querySelectorAll("button[data-l]").forEach(b => {
    b.classList.toggle("sel", (id in st.lab) && st.lab[id] === b.dataset.l);
  });
  save();
}
function label(l){
  st.lab[CLIPS[st.i]] = l;
  if (st.i < CLIPS.length-1) st.i++;
  show();
}
document.querySelectorAll("button[data-l]").forEach(b => b.onclick = () => label(b.dataset.l));
$("prev").onclick = () => { if (st.i>0) { st.i--; show(); } };
$("next").onclick = () => { if (st.i<CLIPS.length-1) { st.i++; show(); } };
$("name").value = st.name;
$("name").oninput = e => { st.name = e.target.value.trim(); save(); };
document.addEventListener("keydown", e => {
  if (e.target.tagName === "INPUT") return;
  const k = e.key.toLowerCase();
  if (k === "p") label("p"); else if (k === "b") label("b"); else if (k === "u") label("");
  else if (k === "arrowright") $("next").click(); else if (k === "arrowleft") $("prev").click();
  else if (k === "r") { vid.currentTime = 0; vid.play(); }
});
$("dl").onclick = () => {
  const name = (st.name || "").replace(/[^a-z0-9_-]/gi, "").toLowerCase();
  if (!name) { $("msg").textContent = "Please type your name at the top first."; $("name").focus(); return; }
  const missing = CLIPS.filter(c => !(c in st.lab)).length;
  let csv = "clip_id,label\\n" + CLIPS.map(c => c + "," + (st.lab[c] || "")).join("\\n") + "\\n";
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([csv], {type:"text/csv"}));
  a.download = "annotations_" + name + ".csv"; a.click();
  $("msg").textContent = missing ? ("Downloaded. " + missing + " clip(s) are still unanswered (saved as blank).")
                                 : "Downloaded. Thank you! Please send the file back.";
};
show();
</script></body></html>
"""
    (PKG_DIR / "annotate.html").write_text(
        html.replace("__CLIPS__", json.dumps(clip_ids)), encoding="utf-8")


# =============================================================================
# Main
# =============================================================================

def write_text_only():
    """Rewrite the instructions and labelling page for the existing clips."""
    ids = [r["clip_id"] for r in csv.DictReader(open(MAP_FILE, newline=""))]
    (PKG_DIR / "INSTRUCTIONS.txt").write_text(INSTRUCTIONS_TXT, encoding="ascii")
    _write_html(ids)
    print(f"[ClipPackage] Rewrote INSTRUCTIONS.txt and annotate.html for {len(ids)} clips")


def main():
    if "--text-only" in sys.argv:
        write_text_only()
        return
    CLIPS.mkdir(parents=True, exist_ok=True)
    print("[ClipPackage] Loading tracking log...")
    tracks = _load_tracks()
    chosen = _select(tracks)
    print(f"[ClipPackage] Selected {len(chosen)} tracks")

    rows = []
    for n, key in enumerate(chosen, start=1):
        clip_id = f"clip_{n:03d}"
        for attempt in (1, 2):
            try:
                rows.append(_render(clip_id, key, tracks[key]))
                break
            except BrokenPipeError:
                print(f"  {clip_id}: ffmpeg pipe closed early, attempt {attempt} of 2")
                if attempt == 2:
                    raise
        print(f"  {clip_id}  {key[0]}/{key[1]} track {key[2]}")

    with open(MAP_FILE, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(PKG_DIR / "annotator_template.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip_id", "label"])
        w.writerows([[r["clip_id"], ""] for r in rows])
    (PKG_DIR / "INSTRUCTIONS.txt").write_text(INSTRUCTIONS_TXT, encoding="ascii")
    _write_html([r["clip_id"] for r in rows])

    by_cat = defaultdict(int)
    for r in rows:
        by_cat[r["category"]] += 1
    print(f"\n[ClipPackage] Done: {len(rows)} clips {dict(by_cat)}")
    print(f"  Share: {PKG_DIR / 'annotate.html'}, clips/, INSTRUCTIONS.txt, annotator_template.csv")
    print(f"  DO NOT SHARE: {MAP_FILE}")


if __name__ == "__main__":
    main()

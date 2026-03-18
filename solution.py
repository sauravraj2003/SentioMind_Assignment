"""
solution.py
Sentio Mind · Project 2 · Smart Behavioral Video Compression

Run: python solution.py
Requires ffmpeg installed on your system: sudo apt install ffmpeg
"""

import cv2
import json
import base64
import subprocess
import threading
import time
import queue
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
VIDEO_IN               = Path("video_sample_1.mov")
VIDEO_OUT              = Path("compressed_output.mp4")
REPORT_HTML_OUT        = Path("compression_report.html")
SEGMENTS_JSON_OUT      = Path("segments_kept.json")

PHASH_THRESHOLD        = 0.95   # similarity above this = near-duplicate, discard
MOTION_KEEP_THRESH     = 0.15   # keep frame if motion exceeds this (no face needed)
MOTION_DISCARD_THRESH  = 0.05   # definitely discard below this
CONTEXT_EVERY_SEC      = 3      # force-keep one frame every this many seconds
OUTPUT_FPS             = 12     # frame rate of the output video
OUTPUT_CRF             = 28     # ffmpeg quality: lower = better quality + larger file

# Frame stepping: grab() skips (FRAME_STEP-1) frames without decoding (~0.01 ms).
# Only every FRAME_STEP-th frame is fully decoded (~0.8 ms).
# At 58.5 fps: step=3 → ~19.5 fps sampled temporal resolution.
FRAME_STEP = 3

# Internal working resolutions — optical flow + pHash run at 80x45 (fast),
# Haar face detection runs at 320x180 (faces must be >= 12 px to be detectable).
_W,      _H      = 80,  45
_HAAR_W, _HAAR_H = 320, 180

# Reader queue depth for threaded I/O
_READER_QUEUE_DEPTH = 64

# Module-level caches shared between should_keep_frame and run_pipeline.
# Eliminates redundant resize (prev_gray) and redundant pHash on kept frames.
_prev_small_gray = None   # 80×45 gray of previous sampled frame
_cached_gray     = None   # written by should_keep_frame, read by pipeline
_cached_hash     = ""     # written by should_keep_frame, read by pipeline


# ---------------------------------------------------------------------------
# PERCEPTUAL HASH
# ---------------------------------------------------------------------------

def compute_phash(frame: np.ndarray) -> str:
    """
    Compute a perceptual hash of the frame.
    Steps: resize to 32×32 grayscale → DCT → threshold at mean → flatten to bit string.
    Return a string of '0' and '1' characters, length 64.

    Accepts either BGR colour frame or pre-computed grayscale.
    """
    # Convert to grayscale only if needed (3-channel BGR input)
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame

    g32 = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_LINEAR)

    # DCT on float32; keep top-left 8×8 low-frequency block
    dct = cv2.dct(np.float32(g32))
    low = dct[:8, :8]

    # Mean excluding DC coefficient [0,0] to avoid brightness sensitivity
    mean = (low.sum() - low[0, 0]) / 63.0

    # Pack into 64-char bit string
    bits = (low > mean).flatten()
    return ''.join('1' if b else '0' for b in bits)


def phash_similarity(h1: str, h2: str) -> float:
    """
    Compare two hash strings. Return 1.0 if identical, 0.0 if completely different.
    Formula: 1.0 - (hamming_distance / length)
    """
    if not h1 or not h2 or len(h1) != len(h2):
        return 0.0
    diff = sum(c1 != c2 for c1, c2 in zip(h1, h2))
    return 1.0 - (diff / len(h1))


# ---------------------------------------------------------------------------
# MOTION SCORE
# ---------------------------------------------------------------------------

def compute_motion_score(prev_gray, curr_gray: np.ndarray) -> float:
    """
    Dense optical flow between two grayscale frames. Return mean magnitude, ~0.0-1.0.
    If prev_gray is None, return 0.0.
    Both inputs should be downscaled to _W x _H before calling.
    """
    if prev_gray is None:
        return 0.0
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    fx, fy = flow[..., 0], flow[..., 1]
    return float(np.sqrt((fx * fx + fy * fy).mean()))


# ---------------------------------------------------------------------------
# FACE PRESENCE CHECK
# ---------------------------------------------------------------------------

def has_face(frame: np.ndarray, cascade) -> bool:
    """
    True if at least one face detected. Equalise histogram first for CCTV lighting.
    frame should be grayscale at _HAAR_W x _HAAR_H (320x180) so that faces
    are >= 12 px tall, which is the minimum Haar can reliably detect.
    """
    eq    = cv2.equalizeHist(frame)
    faces = cascade.detectMultiScale(
        eq,
        scaleFactor  = 1.1,
        minNeighbors = 3,
        minSize      = (12, 12),
        maxSize      = (_HAAR_H, _HAAR_H),
    )
    return len(faces) > 0


# ---------------------------------------------------------------------------
# FRAME KEEP DECISION
# ---------------------------------------------------------------------------

def should_keep_frame(frame: np.ndarray,
                      prev_frame,
                      prev_kept_hash: str,
                      last_kept_time_sec: float,
                      current_time_sec: float,
                      cascade) -> tuple:
    """
    Apply the 5-step decision algorithm from README in order.
    Return: (keep: bool, reason: str, motion_score: float, face_found: bool)

    Reason strings (use exactly these):
      'face_detected', 'motion_above_threshold', 'context_frame',
      'face_and_motion', 'discarded_duplicate', 'discarded_static'

    Side effects:
      Writes computed gray  → _cached_gray  (pipeline reuses, no second resize)
      Writes computed hash  → _cached_hash  (pipeline reuses, no second pHash)
    """
    global _cached_gray, _cached_hash

    # -- Shared downscale for motion + pHash (80x45) -------------------------
    small     = cv2.resize(frame, (_W, _H), interpolation=cv2.INTER_LINEAR)
    curr_gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    _cached_gray = curr_gray   # pipeline reads this; avoids second resize

    # -- Step 1: pHash duplicate check ----------------------------------------
    curr_hash = compute_phash(curr_gray)   # pass grayscale directly — no cvtColor
    _cached_hash = curr_hash               # pipeline reads this; avoids second pHash
    if prev_kept_hash and phash_similarity(curr_hash, prev_kept_hash) > PHASH_THRESHOLD:
        return False, "discarded_duplicate", 0.0, False

    # -- Step 2: Optical flow motion score ------------------------------------
    # _prev_small_gray is the module-global gray from the previous sampled frame.
    motion = compute_motion_score(_prev_small_gray, curr_gray)

    # -- Step 3: Haar face detection at 320x180 -------------------------------
    # Run on every non-duplicate frame where motion <= KEEP_THRESH.
    # When motion > KEEP_THRESH, Step 4 keeps the frame anyway — skip Haar.
    face_found = False
    if motion <= MOTION_KEEP_THRESH:
        haar_small = cv2.resize(frame, (_HAAR_W, _HAAR_H), interpolation=cv2.INTER_LINEAR)
        haar_gray  = cv2.cvtColor(haar_small, cv2.COLOR_BGR2GRAY)
        face_found = has_face(haar_gray, cascade)

    if face_found:
        reason = "face_and_motion" if motion > MOTION_DISCARD_THRESH else "face_detected"
        return True, reason, motion, True

    # -- Step 4: Keep if motion is above threshold ----------------------------
    if motion > MOTION_KEEP_THRESH:
        return True, "motion_above_threshold", motion, False

    # -- Step 5: Context frame every 3 seconds --------------------------------
    if current_time_sec - last_kept_time_sec >= CONTEXT_EVERY_SEC:
        return True, "context_frame", motion, False

    return False, "discarded_static", motion, False


# ---------------------------------------------------------------------------
# THUMBNAIL HELPER
# ---------------------------------------------------------------------------

def frame_to_b64_thumb(frame: np.ndarray, width: int = 200) -> str:
    """Resize frame keeping aspect ratio, encode as base64 JPEG."""
    h, w  = frame.shape[:2]
    nh    = int(h * width / w)
    thumb = cv2.resize(frame, (width, nh), interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 72])
    return base64.b64encode(buf).decode("utf-8")


# ---------------------------------------------------------------------------
# FFMPEG PIPE WRITER
# ---------------------------------------------------------------------------

class FfmpegPipeWriter:
    """
    Streams raw BGR frames via stdin directly to libx264.
    Single-pass — no temp file needed.
    """

    def __init__(self, output_path: Path, fps: float, fw: int, fh: int):
        self._fw, self._fh = fw, fh
        cmd = [
            "ffmpeg", "-y",
            "-f",       "rawvideo",
            "-vcodec",  "rawvideo",
            "-pix_fmt", "bgr24",
            "-s",       f"{fw}x{fh}",
            "-r",       str(fps),
            "-i",       "pipe:0",
            "-vcodec",  "libx264",
            "-crf",     str(OUTPUT_CRF),
            "-preset",  "ultrafast",
            "-pix_fmt", "yuv420p",
            str(output_path),
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdin  = subprocess.PIPE,
            stdout = subprocess.DEVNULL,
            stderr = subprocess.DEVNULL,
        )

    def write(self, frame: np.ndarray) -> None:
        if (frame.shape[1], frame.shape[0]) != (self._fw, self._fh):
            frame = cv2.resize(frame, (self._fw, self._fh),
                               interpolation=cv2.INTER_LINEAR)
        self._proc.stdin.write(frame.tobytes())

    def finish(self) -> None:
        self._proc.stdin.close()
        self._proc.wait()


# ---------------------------------------------------------------------------
# VIDEO WRITING
# ---------------------------------------------------------------------------

def write_frames_to_video(kept_frames: list, output_path: Path,
                          fps: float, frame_size: tuple):
    """
    Write kept_frames to H.264 MP4 via ffmpeg pipe.
    Each item in kept_frames is either a bare frame (np.ndarray)
    or a tuple whose first element is the frame.
    """
    if not kept_frames:
        print("Warning: no frames to write.")
        return

    fw, fh = frame_size
    writer = FfmpegPipeWriter(output_path, fps, fw, fh)

    for item in kept_frames:
        frame = item[0] if isinstance(item, tuple) else item
        writer.write(frame)

    writer.finish()


# ---------------------------------------------------------------------------
# READER THREAD
# ---------------------------------------------------------------------------

def _reader_thread(cap: cv2.VideoCapture, read_q: queue.Queue, step: int) -> None:
    """
    Background thread: grab() skips (step-1) frames cheaply (~0.01 ms each),
    then fully decodes every step-th frame (~0.8 ms).
    Puts (frame_index, frame) tuples into the queue.
    """
    n = 0
    while True:
        for _ in range(step - 1):
            if not cap.grab():
                read_q.put(None)
                return
        ret, frame = cap.read()
        if not ret:
            break
        read_q.put((n * step, frame))
        n += 1
    read_q.put(None)


# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------

def run_pipeline(cap: cv2.VideoCapture, fps_in: float, cascade) -> dict:
    """
    Threaded I/O: reader thread pre-fills queue while main thread processes.
    Caches prev_gray and prev_hash to avoid redundant computation.
    """
    global _prev_small_gray, _cached_gray, _cached_hash

    read_q = queue.Queue(maxsize=_READER_QUEUE_DEPTH)
    reader = threading.Thread(
        target=_reader_thread,
        args=(cap, read_q, FRAME_STEP),
        daemon=True,
    )
    reader.start()

    _prev_small_gray = None
    _cached_gray     = None
    _cached_hash     = ""
    prev_hash        = ""
    last_kept_t      = -999.0

    kept_frames_meta = []   # list of (frame, reason, motion, face, ts)
    disc_dup         = 0
    disc_stat        = 0
    total_sampled    = 0

    while True:
        item = read_q.get()
        if item is None:
            break

        frame_idx, frame = item
        total_sampled   += 1
        ts = frame_idx / fps_in

        keep, reason, motion, face = should_keep_frame(
            frame, None, prev_hash, last_kept_t, ts, cascade
        )

        # Reuse the gray that should_keep_frame already computed (no second resize)
        _prev_small_gray = _cached_gray

        if keep:
            kept_frames_meta.append((frame, reason, motion, face, ts))
            # Reuse the hash that should_keep_frame already computed (no second pHash)
            prev_hash   = _cached_hash
            last_kept_t = ts
        else:
            if reason == "discarded_duplicate":
                disc_dup  += 1
            else:
                disc_stat += 1

    reader.join()

    return {
        "kept_frames_meta": kept_frames_meta,
        "disc_dup"        : disc_dup,
        "disc_stat"       : disc_stat,
        "total_sampled"   : total_sampled,
    }


# ---------------------------------------------------------------------------
# SEGMENT BUILDER
# ---------------------------------------------------------------------------

def build_segments(kept_frames_meta: list) -> list:
    """Groups consecutive kept frames into segments; gap > 2.5 s = new segment."""
    segments = []
    cur_seg  = None

    for frame, reason, motion, face, ts in kept_frames_meta:
        if cur_seg is None or (ts - cur_seg["end_sec"]) > 2.5:
            if cur_seg:
                segments.append(cur_seg)
            cur_seg = {
                "segment_id"           : len(segments) + 1,
                "start_sec"            : round(ts, 2),
                "end_sec"              : round(ts, 2),
                "frames_in_segment"    : 1,
                "reason_kept"          : reason,
                "face_count_in_segment": 1 if face else 0,
                "motion_score_avg"     : round(motion, 3),
                "thumbnail_b64"        : frame_to_b64_thumb(frame),
            }
        else:
            cur_seg["end_sec"]                = round(ts, 2)
            cur_seg["frames_in_segment"]     += 1
            cur_seg["face_count_in_segment"] += 1 if face else 0

    if cur_seg:
        segments.append(cur_seg)
    return segments


# ---------------------------------------------------------------------------
# HTML REPORT
# ---------------------------------------------------------------------------

def generate_compression_report(segments: list, stats: dict, output_path: Path):
    """
    Write a self-contained HTML file showing:
      - Original vs compressed size (MB and % reduction)
      - Original vs compressed duration (seconds)
      - Processing time
      - Storyboard grid: one thumbnail per segment
      - Frames kept vs discarded count

    No CDN. Inline CSS only. Works offline.
    """
    orig_mb   = stats["original_size_mb"]
    comp_mb   = stats["compressed_size_mb"]
    red_pct   = stats["reduction_pct"]
    orig_dur  = stats["original_duration_sec"]
    comp_dur  = stats["compressed_duration_sec"]
    proc_time = stats["processing_time_sec"]
    f_orig    = stats["frames_original"]
    f_kept    = stats["frames_kept"]
    f_disc    = stats["frames_discarded_reasons"]["total_discarded"]
    f_dup     = stats["frames_discarded_reasons"]["near_duplicate_phash"]
    f_stat    = stats["frames_discarded_reasons"]["low_motion_no_face"]
    dur_red   = (1 - comp_dur / (orig_dur + 1e-9)) * 100
    frame_red = (1 - f_kept   / (f_orig   + 1e-9)) * 100

    cards_html = ""
    for seg in segments:
        b64 = seg.get("thumbnail_b64", "")
        img = (f'<img src="data:image/jpeg;base64,{b64}" alt="Segment {seg["segment_id"]}">'
               if b64 else "")
        cards_html += (
            f'<div class="card">{img}'
            f'<div class="seg-id">Segment {seg["segment_id"]}</div>'
            f'<div class="seg-time">{seg["start_sec"]}s - {seg["end_sec"]}s</div>'
            f'<div class="seg-reason">{seg["reason_kept"]}</div>'
            f'<div class="seg-detail">'
            f'Frames: {seg["frames_in_segment"]} | '
            f'Faces: {seg["face_count_in_segment"]} | '
            f'Motion: {seg["motion_score_avg"]:.3f}'
            f'</div></div>\n'
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Video Compression Report</title>
<style>
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
  background: #0f1117; color: #e2e8f0; padding: 32px 24px; min-height: 100vh;
}}
h1 {{ font-size: 1.75rem; font-weight: 700; color: #f8fafc; margin-bottom: 6px; }}
.subtitle {{ font-size: .875rem; color: #64748b; margin-bottom: 32px; }}
.stats-grid {{
  display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 16px; margin-bottom: 40px;
}}
.stat-tile {{
  background: #1e2533; border: 1px solid #2d3748;
  border-radius: 10px; padding: 18px 20px;
}}
.stat-tile .label {{
  font-size: .72rem; text-transform: uppercase; letter-spacing: .08em;
  color: #94a3b8; margin-bottom: 6px;
}}
.stat-tile .value {{ font-size: 1.6rem; font-weight: 700; color: #f1f5f9; line-height: 1; }}
.stat-tile .sub   {{ font-size: .78rem; color: #64748b; margin-top: 4px; }}
.highlight .value {{ color: #38bdf8; }}
.good      .value {{ color: #4ade80; }}
.warn      .value {{ color: #fb923c; }}
.section-title {{
  font-size: 1.1rem; font-weight: 600; color: #cbd5e1;
  border-left: 3px solid #38bdf8; padding-left: 10px; margin-bottom: 20px;
}}
table {{ width: 100%; border-collapse: collapse; margin-bottom: 40px; font-size: .875rem; }}
th {{ text-align: left; padding: 10px 14px; background: #1e2533;
      color: #94a3b8; font-weight: 600; border-bottom: 1px solid #2d3748; }}
td {{ padding: 10px 14px; border-bottom: 1px solid #1e2533; color: #cbd5e1; }}
tr:last-child td {{ border-bottom: none; }}
tr:hover td {{ background: #1a2035; }}
.storyboard {{
  display: grid; grid-template-columns: repeat(auto-fill, minmax(210px, 1fr));
  gap: 16px; margin-bottom: 40px;
}}
.card {{
  background: #1e2533; border: 1px solid #2d3748; border-radius: 10px; overflow: hidden;
  transition: transform .15s ease, box-shadow .15s ease;
}}
.card:hover {{ transform: translateY(-3px); box-shadow: 0 8px 24px rgba(0,0,0,.5); }}
.card img   {{ width: 100%; display: block; object-fit: cover; }}
.seg-id {{
  font-size: .7rem; font-weight: 700; text-transform: uppercase;
  letter-spacing: .08em; color: #38bdf8; padding: 10px 10px 2px;
}}
.seg-time   {{ font-size: .85rem; font-weight: 600; color: #f1f5f9; padding: 0 10px 4px; }}
.seg-reason {{
  font-size: .72rem; background: #0f3460; color: #7dd3fc;
  display: inline-block; margin: 0 10px 6px; padding: 2px 8px; border-radius: 999px;
}}
.seg-detail {{ font-size: .72rem; color: #64748b; padding: 0 10px 10px; }}
.bar-wrap {{
  background: #2d3748; border-radius: 999px; height: 8px; overflow: hidden; margin-top: 8px;
}}
.bar-fill {{
  height: 100%; border-radius: 999px;
  background: linear-gradient(90deg, #38bdf8, #818cf8);
}}
footer {{ margin-top: 48px; font-size: .75rem; color: #334155; text-align: center; }}
</style>
</head>
<body>

<h1>&#127916; Video Compression Report</h1>
<div class="subtitle">
  {stats["source_video"]} &#8594; {stats["compressed_video"]}
  &nbsp;&middot;&nbsp; {proc_time}s total
</div>

<div class="stats-grid">
  <div class="stat-tile warn">
    <div class="label">Original Size</div>
    <div class="value">{orig_mb:.1f}<span style="font-size:1rem"> MB</span></div>
    <div class="sub">{orig_dur:.1f}s &nbsp;&middot;&nbsp; {stats["original_fps"]:.1f} fps</div>
  </div>
  <div class="stat-tile good">
    <div class="label">Compressed Size</div>
    <div class="value">{comp_mb:.1f}<span style="font-size:1rem"> MB</span></div>
    <div class="sub">{comp_dur:.1f}s &nbsp;&middot;&nbsp; {OUTPUT_FPS} fps</div>
  </div>
  <div class="stat-tile highlight">
    <div class="label">Size Reduction</div>
    <div class="value">{red_pct:.1f}<span style="font-size:1rem">%</span></div>
    <div class="bar-wrap"><div class="bar-fill" style="width:{min(red_pct,100):.1f}%"></div></div>
  </div>
  <div class="stat-tile">
    <div class="label">Frames Kept</div>
    <div class="value">{f_kept}</div>
    <div class="sub">of {f_orig} sampled frames</div>
  </div>
  <div class="stat-tile">
    <div class="label">Frames Discarded</div>
    <div class="value">{f_disc}</div>
    <div class="sub">Dupes: {f_dup} &nbsp;|&nbsp; Static: {f_stat}</div>
  </div>
  <div class="stat-tile">
    <div class="label">Segments</div>
    <div class="value">{len(segments)}</div>
    <div class="sub">Processing: {proc_time}s</div>
  </div>
</div>

<div class="section-title">Compression Breakdown</div>
<table>
  <thead>
    <tr><th>Metric</th><th>Original</th><th>Compressed</th><th>Change</th></tr>
  </thead>
  <tbody>
    <tr>
      <td>File Size</td>
      <td>{orig_mb:.2f} MB</td><td>{comp_mb:.2f} MB</td>
      <td style="color:#4ade80">&#8595; {red_pct:.1f}%</td>
    </tr>
    <tr>
      <td>Duration</td>
      <td>{orig_dur:.2f}s</td><td>{comp_dur:.2f}s</td>
      <td style="color:#fb923c">&#8595; {dur_red:.1f}%</td>
    </tr>
    <tr>
      <td>Sampled Frames</td>
      <td>{f_orig}</td><td>{f_kept}</td>
      <td style="color:#fb923c">&#8595; {frame_red:.1f}%</td>
    </tr>
    <tr>
      <td>Frame Rate</td>
      <td>{stats["original_fps"]:.1f} fps</td><td>{OUTPUT_FPS} fps</td><td>-</td>
    </tr>
    <tr>
      <td>Near-duplicate (pHash) discards</td>
      <td colspan="2">{f_dup} frames</td><td>-</td>
    </tr>
    <tr>
      <td>Low-motion / no-face discards</td>
      <td colspan="2">{f_stat} frames</td><td>-</td>
    </tr>
    <tr>
      <td>Frame step (sampling)</td>
      <td colspan="2">every {FRAME_STEP} raw frames</td><td>-</td>
    </tr>
  </tbody>
</table>

<div class="section-title">Storyboard - {len(segments)} Segment(s)</div>
<div class="storyboard">
{cards_html}
</div>

<footer>Sentio Mind &middot; Smart Behavioral Video Compression &middot; solution.py</footer>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t_start = time.time()

    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    cap = cv2.VideoCapture(str(VIDEO_IN))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_IN}")

    total_raw = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_in    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    fw        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration  = total_raw / fps_in
    orig_mb   = VIDEO_IN.stat().st_size / 1_000_000

    print(f"Input  : {VIDEO_IN}  |  {total_raw} raw frames  |  {duration:.1f}s  |  {orig_mb:.1f} MB")
    print(f"Config : step={FRAME_STEP} (~{total_raw // FRAME_STEP} sampled)  |  ops @ {_W}x{_H}")

    result           = run_pipeline(cap, fps_in, cascade)
    kept_frames_meta = result["kept_frames_meta"]
    disc_dup         = result["disc_dup"]
    disc_stat        = result["disc_stat"]
    total_sampled    = result["total_sampled"]
    cap.release()

    print(f"Kept   : {len(kept_frames_meta)} / {total_sampled} sampled  "
          f"({disc_dup} dupes  {disc_stat} static)")

    segments = build_segments(kept_frames_meta)
    print(f"Segments: {len(segments)}")

    print("Encoding ...")
    write_frames_to_video(kept_frames_meta, VIDEO_OUT, OUTPUT_FPS, (fw, fh))

    comp_mb = VIDEO_OUT.stat().st_size / 1_000_000 if VIDEO_OUT.exists() else 0.0
    t_end   = time.time()

    stats = {
        "source_video"            : str(VIDEO_IN),
        "compressed_video"        : str(VIDEO_OUT),
        "original_size_mb"        : round(orig_mb, 2),
        "compressed_size_mb"      : round(comp_mb, 2),
        "reduction_pct"           : round((1 - comp_mb / (orig_mb + 1e-9)) * 100, 1),
        "original_duration_sec"   : round(duration, 2),
        "compressed_duration_sec" : round(len(kept_frames_meta) / OUTPUT_FPS, 2),
        "original_fps"            : round(fps_in, 2),
        "output_fps"              : OUTPUT_FPS,
        "frames_original"         : total_sampled,
        "frames_kept"             : len(kept_frames_meta),
        "processing_time_sec"     : round(t_end - t_start, 2),
        "segments"                : segments,
        "frames_discarded_reasons": {
            "near_duplicate_phash": disc_dup,
            "low_motion_no_face"  : disc_stat,
            "total_discarded"     : total_sampled - len(kept_frames_meta),
        },
    }

    # Strip thumbnails from JSON to keep it lean and schema-compliant
    segments_no_thumb = [
        {k: v for k, v in seg.items() if k != "thumbnail_b64"}
        for seg in segments
    ]
    with open(SEGMENTS_JSON_OUT, "w") as f:
        json.dump({**stats, "segments": segments_no_thumb}, f, indent=2)

    generate_compression_report(segments, stats, REPORT_HTML_OUT)

    print()
    print("=" * 60)
    print(f"  Done in          {stats['processing_time_sec']} s")
    print(f"  Size :           {orig_mb:.1f} MB -> {comp_mb:.1f} MB  ({stats['reduction_pct']}% smaller)")
    print(f"  Duration :       {duration:.1f}s -> {stats['compressed_duration_sec']:.1f}s")
    print(f"  Frames sampled : {total_sampled} / {total_raw} raw  (step={FRAME_STEP})")
    print(f"  Frames kept :    {len(kept_frames_meta)} / {total_sampled}")
    print(f"  Segments :       {len(segments)}")
    print(f"  Report  ->  {REPORT_HTML_OUT}")
    print(f"  JSON    ->  {SEGMENTS_JSON_OUT}")
    print("=" * 60)
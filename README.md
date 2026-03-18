# Smart Behavioral Video Compression
**Sentio Mind · Assignment · Project**

---

## The Problem

Schools using CCTV cameras generate 40–80 GB of raw footage every day. Most of that footage is useless — empty hallways, static classrooms, and frames that are nearly identical to the one before them. Uploading all of it over a school internet connection takes 6–12 hours and most of the data carries zero useful information.

Simply running ffmpeg on the raw file makes it smaller but throws away frames that contain people, which breaks any downstream behaviour analysis. What is needed is a compressor that understands content — one that keeps every frame where a human appears and aggressively removes everything else.

This project does exactly that. It reads every frame, runs a five-step decision algorithm to decide whether the frame is worth keeping, and re-encodes only the surviving frames into a clean H.264 MP4. The result is a video that is dramatically smaller but still contains all the meaningful activity from the original.

---

## Results on the Sample Video

| Metric | Value |
|---|---|
| Input file | `video_sample_1.mov` |
| Input size | 614.2 MB |
| Input duration | 122.5 seconds |
| Input resolution | 1920×1080 at 58.5 fps |
| Output size | **18.6 MB** |
| Size reduction | **97.0% smaller** |
| Output duration | 17.6 seconds |
| Output format | H.264 MP4 at 12 fps |
| Raw frames in input | 7169 |
| Frames sampled | 2387 (every 3rd frame) |
| Frames kept | 211 |
| Discarded as near-duplicates | 1413 |
| Discarded as static with no face | 763 |
| Meaningful segments found | 14 |
| Processing time | 59.94 seconds |

---

## What the Algorithm Found — Segment Breakdown

The 122.5-second input video was reduced to 14 segments. Here is what each one contains and why it was kept:

| Segment | Time Range | Frames | Why Kept | Faces Detected | Avg Motion |
|---|---|---|---|---|---|
| 1 | 0.00s – 0.41s | 2 | Context frame | 0 | 0.000 |
| 2 | 3.44s | 1 | Context frame | 0 | 0.070 |
| 3 | 6.46s – 6.72s | 2 | **Motion above threshold** | 0 | 0.152 |
| 4 | 9.74s | 1 | Context frame | 0 | 0.067 |
| 5 | 12.77s – 23.28s | 25 | Context frame | **3** | 0.042 |
| 6 | 26.30s | 1 | Context frame | 0 | 0.068 |
| 7 | 29.53s – 30.10s | 2 | Context frame | 0 | 0.084 |
| 8 | 33.12s – 38.71s | 20 | Context frame | 0 | 0.088 |
| 9 | 41.73s | 1 | Context frame | 0 | 0.028 |
| 10 | 44.71s – 56.04s | 20 | **Motion above threshold** | **3** | 0.249 |
| 11 | 59.06s – 85.93s | 76 | Context frame | **4** | 0.032 |
| 12 | 88.95s – 97.10s | 10 | Context frame | **4** | 0.054 |
| 13 | 99.72s – 99.98s | 2 | **Motion above threshold** | **1** | 0.153 |
| 14 | 103.00s – 120.89s | 48 | Context frame | **5** | 0.049 |

Faces were detected across 6 of the 14 segments (segments 5, 10, 11, 12, 13, 14), covering the second half of the video where most of the human activity occurred. The three highest-motion segments (3, 10, 13) were kept by the motion override rule. The remaining segments were kept by the context frame rule to maintain timeline continuity.

---

## Output Files

**`compressed_output.mp4`**
The compressed video in H.264 format at 12 fps. Plays in VLC, Windows Media Player, QuickTime, or any standard player without codec issues.

**`compression_report.html`**
Open in any browser — no internet connection needed. Shows a size and duration comparison between original and compressed, a breakdown of how frames were discarded, and a visual storyboard with one thumbnail per segment.

**`segments_kept.json`**
A structured log of all 14 segments. Each entry records the segment ID, start and end time in seconds, frame count, keep reason, face count, and average motion score. This file is designed to plug directly into Sentio Mind's `extract_intelligent_frames()` pipeline so the system can jump straight to relevant parts of the footage without re-scanning the full raw video.

---

## Requirements

- Python 3.9 or higher
- ffmpeg installed and available on the system PATH
- The following Python packages:

```
opencv-python==4.9.0
numpy==1.26.4
Pillow==10.3.0
imagehash==4.3.1
```

---

## Installation

**Install Python packages:**

```bash
pip install opencv-python numpy Pillow imagehash
```

**Install ffmpeg** (system tool, not a Python package):

- **Ubuntu / Debian:** `sudo apt install ffmpeg`
- **macOS:** `brew install ffmpeg`
- **Windows:** Download from https://ffmpeg.org/download.html, extract, and add the `bin` folder to your system PATH

Confirm ffmpeg is working:
```bash
ffmpeg -version
```

---

## How to Run

Place `video_sample_1.mov` in the same folder as `solution.py`, then:

```bash
python solution.py
```

The script prints progress to the terminal and generates `compressed_output.mp4`, `compression_report.html`, and `segments_kept.json` when finished.

---

## How the Algorithm Works

Every sampled frame passes through five checks in a fixed order. A frame only needs to pass one check to be kept — but the checks run in this exact sequence and cannot be reordered.

### Step 1 — Perceptual Hash (pHash) Duplicate Check

A 64-bit hash is computed for the current frame by resizing to 32×32 grayscale, applying DCT, and thresholding the top-left 8×8 frequency block against its mean. This hash is compared to the last kept frame using Hamming distance. If the two frames are more than 95% similar — meaning their hashes differ by 3 bits or fewer — the frame is a near-duplicate and is discarded immediately. No optical flow or face detection runs on it.

This step is cheap and runs first specifically to avoid wasting time on heavier operations. In this video it eliminated 1413 of 2387 sampled frames — roughly 59% of all workload — before anything expensive was touched.

### Step 2 — Optical Flow Motion Score

For frames that survive the duplicate check, dense optical flow is computed using the Farneback method on a downscaled 80×45 grayscale image. The mean vector magnitude across the image represents how much the scene has changed. If this score is below 0.05, the scene is considered static — an empty corridor, an unoccupied classroom — and the frame is marked as a discard candidate.

The frame is not discarded yet at this point. Step 3 can still override this decision if a human face is present.

### Step 3 — Haar Face Detection Override

A Haar cascade classifier scans the frame for human faces. If any face is found, the frame is kept unconditionally — regardless of how low the motion score was in Step 2. This is the most critical rule in the pipeline because it guarantees no human-containing frame is ever lost.

Face detection runs at **320×180 resolution**, not at the same 80×45 used for optical flow. This distinction matters significantly. At 80×45, a human face in a standard CCTV shot is only 3–5 pixels tall, which falls below the `minSize=(12,12)` parameter used by Haar cascade — meaning detection effectively never fires. At 320×180, the same face is 13–17 pixels tall and detected reliably. Early versions of this code ran Haar at 80×45 and missed almost every face in the video, which caused extreme over-compression.

### Step 4 — Motion Override

If no face was found but the motion score exceeds 0.15, the frame is kept anyway. High motion indicates something significant happening even if a face is not directly visible — a person walking quickly through the edge of frame, a door opening, or a sudden environmental change. Three segments in this video (3, 10, and 13) were kept by this rule.

### Step 5 — Context Frame Rule

One frame is force-kept every 3 seconds of original video regardless of what any previous step decided. This guarantees the output has continuous timeline coverage and does not skip large sections of the recording. Without this rule, a long static scene with no motion and no faces would produce no output at all, creating gaps in the compressed video.

### Re-encode with ffmpeg

All 211 surviving frames are written to a temporary AVI file, then re-encoded by ffmpeg to H.264 MP4 at 12 fps using CRF 28 and `-pix_fmt yuv420p`. The yuv420p format ensures broad compatibility with media players.

---

## Optimisations

**Frame sampling with `grab()`**
OpenCV's `cap.grab()` advances the video pointer without decoding pixel data (~0.01 ms). `cap.read()` fully decodes the frame (~0.8 ms). With `FRAME_STEP=3`, only every 3rd frame is decoded — reducing the number of frames processed from 7169 to 2387 and saving several seconds of I/O time.

**Low-resolution optical flow and hashing**
Both Farneback optical flow and pHash computation run on 80×45 downscaled images. Farneback at this size takes ~0.31 ms per call versus ~3.5 ms at full resolution — an 11× reduction in the cost of the most expensive operation in the pipeline.

**Separate resolution for face detection**
Optical flow uses 80×45 for speed. Face detection uses 320×180 for accuracy. These are two separate resizes with different purposes, and separating them is what makes the pipeline both fast and correct.

**Caching computed values**
The 80×45 grayscale image and the pHash value are computed once per frame and stored. The pipeline reads these cached values when updating state after a keep decision, so no operation is repeated within a single frame.

**Early exit on duplicates**
Both optical flow and face detection are skipped completely when Step 1 discards a frame. This saved roughly 1413 × (0.31 ms + 0.32 ms) ≈ 0.9 seconds of compute time in this run alone.

---

## Problems Solved in the Code

These are algorithmic problems that were identified during development. Each fix is directly visible in `solution.py`.

| Problem | Root Cause | Fix in `solution.py` |
|---|---|---|
| Output video only 2–3 seconds — almost no frames kept | Haar cascade was running on 80×45 images. At that resolution a human face occupies only 3–5 pixels, which is below the `minSize=(12,12)` parameter, so `detectMultiScale` never returned a result. The face override in Step 3 never fired. | Added `_HAAR_W, _HAAR_H = 320, 180` constants and a dedicated resize to 320×180 before every Haar call. At this size a face is 13–17 px and reliably detected. Optical flow and pHash continue to use 80×45 for speed. |
| ~99% compression — humans in low-motion scenes discarded | An earlier version only ran Haar inside the ambiguous motion band (`MOTION_DISCARD_THRESH < motion ≤ MOTION_KEEP_THRESH`). Frames where a person was standing still or walking slowly had motion scores below `MOTION_DISCARD_THRESH`, so Haar was never called and they were silently dropped. | Changed `should_keep_frame` to run face detection on every non-duplicate frame where `motion <= MOTION_KEEP_THRESH`. When motion exceeds `MOTION_KEEP_THRESH`, Step 4 keeps the frame anyway, so Haar is skipped there to avoid wasted work. |
| Processing too slow — high I/O cost | `cap.read()` decodes full pixel data for every frame (~0.8 ms per call). On a 7169-frame video this meant decoding all frames even though only a fraction would ever be kept. | Added `FRAME_STEP = 3` and a background reader thread using `cap.grab()` to skip two frames cheaply (~0.01 ms) between every decoded frame. This cuts decoded frames from 7169 to 2387 without losing meaningful temporal coverage. |
| Redundant resize and pHash computation per kept frame | After deciding to keep a frame, the pipeline previously called `compute_phash(frame)` again to update `prev_hash` — triggering a second resize and second DCT. Same issue for the grayscale image. | Added `_cached_gray` and `_cached_hash` as module-level variables. `should_keep_frame` writes the values it computed into these caches. The pipeline reads them back directly, so no operation is repeated within the same frame. |

---

## Configuration

These values are at the top of `solution.py` and can be adjusted for different cameras or lighting conditions:

| Parameter | Value in `solution.py` | What It Controls |
|---|---|---|
| `PHASH_THRESHOLD` | `0.95` | Frames more than 95% similar to the last kept frame are discarded as near-duplicates. Raise to keep fewer duplicates, lower to be more lenient. |
| `MOTION_DISCARD_THRESH` | `0.05` | Frames with a motion score below this are treated as static scenes and marked for discard. Lower value = keeps more quiet scenes. |
| `MOTION_KEEP_THRESH` | `0.15` | Frames exceeding this motion score are kept regardless of whether a face is found. |
| `CONTEXT_EVERY_SEC` | `3` | One frame is force-kept every this many seconds to maintain timeline continuity. |
| `OUTPUT_FPS` | `12` | Frame rate written to the output video. |
| `OUTPUT_CRF` | `28` | ffmpeg quality setting for H.264 encoding. Lower value = better quality and larger file. |
| `FRAME_STEP` | `3` | Only every 3rd raw frame is decoded. The rest are skipped using `grab()`. |

---

## File Structure

```
project/
├── solution.py                 main script
├── video_sample_1.mov          input video (from dataset)
├── compressed_output.mp4       compressed output
├── compression_report.html     offline visual report
├── segments_kept.json          segment log for pipeline integration
└── README.md                   this file
```

---

*Sentio Mind · Smart Behavioral Video Compression*

# Kikki_RealSense-Template-Feature-Matching-
Template Matching using ORB and RealSense Camera (Python + OpenCV)
# RealSense Template & Feature Matching — Project Guide

## Overview

This project is a collection of Python modules that detect a known object (the **template**) in either Intel® RealSense™ camera streams or still images. It supports:

* Depth-frame template matching
* RGB-frame template matching
* Feature-based matching with ORB + homography (perspective outline)
* Optional background removal (for cleaner matching)
* On‑screen visualization (bounding boxes, polygons, and match heatmaps)

Your current scripts are organized as separate entry points targeting different techniques and camera streams. This guide explains how to run them together as one project, and includes a suggested folder layout, configuration, and dependency list.

---

## Features at a Glance

* **Depth template matching:** Normalizes depth frames and runs `cv2.matchTemplate` to locate the template; overlays a bounding box on a color-mapped depth image.
* **RGB template matching:** Operates on RealSense color frames; can optionally remove background before template matching.
* **Feature matching (ORB):** Detects keypoints on the template and in the scene, matches them, estimates a homography, and draws the projected template outline in the scene.
* **Debug views:** Display of raw detection, feature matches, and an optional heatmap of template similarity.
* **Keyboard:** Press **q** in any window to exit.

---

## Recommended Repository Layout
realsense-template-matching/
├─ README.md
├─ requirements.txt
├─ config/
│  └─ settings.yaml
├─ data/
│  ├─ template.jpg
│  └─ outputs/
├─ scripts/
│  ├─ depth_tm.py          # (T1, T2, T4)
│  ├─ rgb_tm.py            # (T6)
│  ├─ orb_match_cc.py      # (T7, t8)
│  ├─ orb_match_ratio.py   # (t9, t10)
│  ├─ image_playground.py  # (T3)
│  └─ utils.py             # (T5 + helpers)
└─ scripts/
   ├─ run_depth_tm.bat
   ├─ run_rgb_tm.bat
   ├─ run_orb_cc.bat
   └─ run_orb_ratio.bat
more description............................>>>> 
```
realsense-template-matching/
├─ README.md                  # (this guide)
├─ requirements.txt
├─ config/
│  └─ settings.yaml           # central config (paths, thresholds, streams)
├─ data/
│  ├─ template.jpg            # your object template (copy here)
│  └─ outputs/                # saved results if you enable saving
├─ src/
│  ├─ depth_tm.py             # depth template matching
│  ├─ rgb_tm.py               # RGB template matching (with optional background removal)
│  ├─ orb_match_cc.py         # ORB cross-check matching + homography
│  ├─ orb_match_ratio.py      # ORB KNN + Lowe’s ratio + homography
│  ├─ image_playground.py     # offline single-image experiments (crop/template/matches)
│  └─ utils.py                # helpers: loading, normalization, drawing, windows, heatmap
└─ scripts/
   ├─ run_depth_tm.bat        # Windows helpers (optional)
   ├─ run_rgb_tm.bat
   ├─ run_orb_cc.bat
   └─ run_orb_ratio.bat
```

> You already have working entry points (T1…T10). The `src/` names above are the unified equivalents. You can continue to run your originals or migrate them to these filenames.

---

## Configuration (settings.yaml)

Create `config/settings.yaml` to keep hard‑coded values out of the code:

```yaml
template_path: ./data/template.jpg

streams:
  use_depth: true
  use_color: true
  depth:
    width: 640
    height: 480
    fps: 30
  color:
    width: 640
    height: 480
    fps: 30

matching:
  method: "feature"     # one of: [depth_tm, rgb_tm, feature]
  template_threshold: 0.50  # for cv2.matchTemplate (0..1)
  orb:
    nfeatures: 1500
    ratio_thresh: 0.75  # Lowe’s ratio
    ransac_reproj_thresh: 5.0

visualization:
  show_heatmap: true
  draw_matches_limit: 50
```

Update the `template_path` to your real file. You can keep multiple variants and switch them here.

---

## Dependencies

Add this to `requirements.txt`:

```
opencv-python
numpy
pyrealsense2
Pillow
rembg
PyYAML
```

> Install with: `python -m pip install -r requirements.txt`

**Intel RealSense runtime:** Install the Intel® RealSense™ SDK/driver for your OS so `pyrealsense2` can access the camera.

---

## How the Modules Map to Techniques

* **Depth template matching** (`depth_tm.py`):

  * Opens the **depth stream**, converts to `uint8` via normalization, applies an optional blur, performs `cv2.matchTemplate`, and draws a **green rectangle** on a JET color‑mapped depth image.
  * Good when the object is distinguishable by depth relief.

* **RGB template matching** (`rgb_tm.py`):

  * Grabs **color frames**, optionally removes background using `rembg` (to isolate the foreground object), converts to grayscale, and runs `cv2.matchTemplate` with a threshold to accept/reject matches.
  * Optional **heatmap** window for debugging similarity across the frame.

* **Feature matching — Cross‑check** (`orb_match_cc.py`):

  * Uses ORB features on the template and scene. Matches with `BFMatcher(..., crossCheck=True)`, sorts by distance, and, when enough matches exist, estimates a **homography** to draw the template’s projected polygon.
  * Robust to scale/rotation/partial perspective changes.

* **Feature matching — KNN + Lowe’s ratio** (`orb_match_ratio.py`):

  * Uses `knnMatch(k=2)` and Lowe’s ratio test to filter matches, then computes a homography and draws the projected polygon. Often yields cleaner matches than cross‑check alone.

* **Offline single‑image playground** (`image_playground.py`):

  * Loads a single image, applies preprocessing (blur/bilateral), crops a region to act as a template, shows feature matches, then does a simple `matchTemplate` pass to visualize the best location.

---

## Running the Scripts

From the repo root (after creating `data/template.jpg`):

### Depth template matching

```bash
python src/depth_tm.py --config ./config/settings.yaml
```

*Windows helpers:* `scripts/run_depth_tm.bat`

### RGB template matching (with background removal)

```bash
python src/rgb_tm.py --config ./config/settings.yaml --remove-bg
```

The `--remove-bg` flag toggles rembg pre‑processing.

### ORB feature matching (cross‑check)

```bash
python src/orb_match_cc.py --config ./config/settings.yaml
```

### ORB feature matching (KNN + Lowe’s ratio)

```bash
python src/orb_match_ratio.py --config ./config/settings.yaml
```

### Offline image playground

```bash
python src/image_playground.py --image ./data/template.jpg
```

> Press **q** in any window to quit. Use **ESC** if a window steals focus.

---

## Common CLI Arguments (suggested)

* `--config PATH` — Path to `settings.yaml`.
* `--template PATH` — Override template path at runtime.
* `--threshold FLOAT` — Override template match acceptance threshold.
* `--remove-bg` — Enable background removal pre‑processing for RGB.
* `--draw-matches N` — Limit number of feature matches drawn.
* `--save VIS_DIR` — Save visualizations (frames, heatmaps) to folder.

---

## Implementation Notes (handy details)

* **Normalization:** Depth frames arrive as 16‑bit; normalize to 8‑bit (0–255) before applying color maps or template matching.
* **Pre‑blur:** A light Gaussian blur on the normalized frame can stabilize template matching.
* **Template quality:** Use a sharp, well‑cropped template with similar scale to the expected scene size. Re‑compute ORB features when you replace the template.
* **Homography minimum:** You need at least 4 good feature matches to estimate a homography; 10–20 is safer.
* **Threshold tuning:** Start with `0.50` for `TM_CCOEFF_NORMED`; increase if you get false positives, decrease if you miss detections.
* **Performance:** Reducing frame size or ORB `nfeatures` can improve FPS. Background removal (rembg) is the most expensive step.

---

## Troubleshooting

* **No camera / empty frames:** Verify RealSense drivers and that no other app is using the camera.
* **`pyrealsense2` import error:** Ensure the SDK is installed for your Python version/architecture.
* **Template never found:** Try lowering threshold, re‑capturing a cleaner template, or switching to feature‑based matching.
* **Homography fails (no polygon drawn):** Not enough good matches; improve lighting, increase ORB `nfeatures`, or adjust the ratio threshold.
* **Windows close instantly:** Click the display window to give it focus, then press **q**.

---

## Extending the Project

* Add **automatic template capture** from a live frame (press a key to save a ROI as `data/template.jpg`).
* Add **depth + RGB fusion** (e.g., use depth to restrict the search area for RGB matching).
* Log **detections to disk** (CSV with timestamps and match scores) and **save annotated frames**.
* Wrap as a small GUI (Qt) or a command‑line tool with subcommands.

---

## Safety & Environment

* Test in a well‑lit, safe workspace. Mount the camera securely.
* Be mindful of privacy when using RGB streams; background removal does not anonymize by itself.

---

## Quick Start Checklist

1. Install Python 3.10+ and the Intel® RealSense™ SDK.
2. `python -m venv .venv && .venv/Scripts/activate` (Windows) or `source .venv/bin/activate` (macOS/Linux with supported camera drivers).
3. `pip install -r requirements.txt`
4. Copy your template to `data/template.jpg` (or update `settings.yaml`).
5. Run one of the scripts (depth, rgb, orb).




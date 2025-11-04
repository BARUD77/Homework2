#!/usr/bin/env python3
"""
Extract frames from a YouTube piano-tutorial video, saving ONLY the frames
where the sheet music changes (line/system advances).

Pipeline:
1) If input is a YouTube URL, download with yt-dlp (best mp4).
2) Detect the horizontal boundary between sheet (top) and keyboard (bottom).
3) Crop above that line (sheet-only ROI).
4) Sample frames at a modest rate and detect "sheet change" by:
   - enhancing staff lines (morphological black-hat) and tracking the
     vertical centroid of the staff band, and
   - requiring a content difference vs. last-saved frame (SSIM gate).
5) Save the sheet-only crops at the original scale.
"""
import argparse
import os
import re
import sys
import tempfile
import numpy as np
import cv2
from typing import Tuple, Optional

try:
    from yt_dlp import YoutubeDL
    HAVE_YTDLP = True
except Exception:
    HAVE_YTDLP = False

try:
    from skimage.metrics import structural_similarity as ssim
    HAVE_SKIMAGE = True
except Exception:
    HAVE_SKIMAGE = False


# -------------------------- Boundary detection --------------------------

def find_sheet_keyboard_boundary(frame_bgr: np.ndarray) -> int:
    """
    Return y_cut: row index separating sheet (top) from keyboard (bottom).
    Uses vertical-edge energy (Sobel X) to detect the strong rise caused by piano keys.
    """
    H, W = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    # Focus on central band; trim UI overlays left/right if present
    x0, x1 = int(0.07 * W), int(0.93 * W)
    band = gray[:, x0:x1]

    # Vertical edges (Sobel X)
    gx = cv2.Sobel(band, cv2.CV_32F, 1, 0, ksize=3)
    gx = np.abs(gx)

    # Row-wise vertical-edge energy
    row_energy = gx.mean(axis=1)

    # Only search in lower half; boundary is well below the staff
    lo = H // 2
    hi = int(H * 0.95)

    # Smooth + derivative to find sharp rise
    k = 31 if H > 720 else 21
    smooth = cv2.blur(row_energy.reshape(-1, 1), (1, k)).ravel()
    deriv = np.gradient(smooth)

    y_rel = lo + int(np.argmax(deriv[lo:hi]))
    y_cut = int(y_rel)

    # Safety clamp and small upward margin to avoid including keys
    y_cut = max(int(0.35 * H), min(y_cut, int(0.9 * H)))
    y_cut -= int(0.01 * H)
    return y_cut


def horizontal_line_fallback(frame_bgr: np.ndarray, guess_y: int) -> int:
    """
    Optional fallback: if there's a strong, long horizontal divider line,
    snap the boundary to it when close to guess_y.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 200)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=200,
                            minLineLength=int(0.6 * frame_bgr.shape[1]), maxLineGap=10)
    if lines is None:
        return guess_y
    best_y, best_score = None, -1e9
    for x1, y1, x2, y2 in lines[:, 0]:
        if abs(y2 - y1) <= 3:
            length = abs(x2 - x1)
            score = length - 2 * abs(((y1 + y2) // 2) - guess_y)
            if score > best_score:
                best_score = score
                best_y = int((y1 + y2) // 2)
    return int(best_y) if best_y is not None else guess_y


def compute_sheet_roi(first_frame: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Compute a sheet-only ROI (x, y, w, h) by finding the sheet/keyboard boundary,
    trimming a little left/right and a small top margin.
    """
    H, W = first_frame.shape[:2]
    y_cut = find_sheet_keyboard_boundary(first_frame)
    y_cut = horizontal_line_fallback(first_frame, y_cut)  # harmless if no line

    # left/right trim to avoid UI and keep score centered
    xL, xR = int(0.04 * W), int(0.96 * W)
    y_top = int(0.02 * H)
    h = max(10, y_cut - y_top)
    return (xL, y_top, xR - xL, h)


# -------------------------- Staff detection -----------------------------

def staff_mask(gray: np.ndarray) -> np.ndarray:
    """
    Enhance staff lines (parallel horizontal dark lines on light background).
    1) Black-hat morphology with a horizontal kernel to emphasize thin dark lines.
    2) Otsu threshold.
    3) Vertical dilation to group each 5-line staff into a band.
    Returns a binary mask.
    """
    h, w = gray.shape[:2]
    kw = max(15, w // 40)  # kernel width proportional to image width
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 1))
    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    th = cv2.threshold(bh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    vker = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 9))
    band = cv2.dilate(th, vker, iterations=1)
    return band


def staff_y_center(mask: np.ndarray) -> Optional[int]:
    """
    Weighted vertical centroid of staff band(s). Returns None if not detected.
    """
    rows = mask.sum(axis=1).astype(np.float64)
    if rows.max() <= 0:
        return None
    y = np.arange(len(rows), dtype=np.float64)
    return int((rows * y).sum() / (rows.sum() + 1e-6))


# -------------------------- Download helper -----------------------------

YOUTUBE_RE = re.compile(r"^https?://")

def download_if_url(source: str) -> str:
    """
    If 'source' is a URL, download with yt-dlp and return the local mp4 path.
    Otherwise, return 'source' unchanged.
    """
    if not YOUTUBE_RE.match(source or ""):
        return source
    if not HAVE_YTDLP:
        raise RuntimeError("yt-dlp not installed. Run: pip install yt-dlp")

    tmpdir = tempfile.mkdtemp(prefix="yt_")
    ydl_opts = {
        "outtmpl": os.path.join(tmpdir, "%(title)s.%(ext)s"),
        "format": "bv*+ba/b",
        "merge_output_format": "mp4",
        "quiet": True,
        "noprogress": True,
    }
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(source, download=True)
        path = ydl.prepare_filename(info)
    if not path.endswith(".mp4"):
        alt = os.path.splitext(path)[0] + ".mp4"
        if os.path.exists(alt):
            path = alt
    return path


# ------------------------------ Main -----------------------------------

def run(args):
    src_path = download_if_url(args.source)

    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        raise SystemExit("Could not open input video.")

    os.makedirs(args.out_dir, exist_ok=True)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    sample_every = max(1, int(round(fps / max(0.1, args.sample_fps))))

    # First frame for ROI
    ret, first = cap.read()
    if not ret:
        raise SystemExit("Cannot read first frame.")
    x, y, w, h = compute_sheet_roi(first)
    roi_height = h
    advance_pixels = max(1, int(args.advance_fraction * roi_height))

    # Optionally show the ROI once (debug)
    if args.debug:
        dbg = first.copy()
        cv2.rectangle(dbg, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow("Detected sheet ROI (press any key)", dbg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Process frames
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    prev_kept = None
    last_yc = None
    saved = 0
    idx = 0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    recalc_period_frames = int(max(0.0, args.recompute_boundary_every) * fps)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Recompute ROI periodically in case of camera drift
        if recalc_period_frames and (idx % recalc_period_frames == 0) and idx != 0:
            x, y, w, h = compute_sheet_roi(frame)
            roi_height = h
            advance_pixels = max(1, int(args.advance_fraction * roi_height))

        if idx % sample_every != 0:
            idx += 1
            continue

        crop = frame[y:y + h, x:x + w]
        if crop.size == 0:
            idx += 1
            continue

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        # Normalize size for stable SSIM/morphology
        target_w = 900
        gray = cv2.resize(gray, (target_w, int(target_w * gray.shape[0] / gray.shape[1])), interpolation=cv2.INTER_AREA)

        mask = staff_mask(gray)
        yc = staff_y_center(mask)
        if yc is None:
            idx += 1
            continue

        if prev_kept is None:
            should_save = True
            sscore = 0.0
        else:
            if not HAVE_SKIMAGE:
                raise RuntimeError("scikit-image not installed; run: pip install scikit-image")
            sscore = ssim(prev_kept, gray)
            moved = (last_yc is None) or (yc - last_yc >= int(advance_pixels * (gray.shape[0] / roi_height)))
            changed = sscore < args.ssim_thresh
            should_save = moved and changed

        if should_save:
            out_path = os.path.join(args.out_dir, f"{saved:05d}.png")
            cv2.imwrite(out_path, crop)  # save original-scale crop
            prev_kept = gray.copy()
            last_yc = yc
            saved += 1

            if args.verbose:
                t = idx / fps
                print(f"[{saved:04d}] saved @ t={t:7.2f}s | y={yc:4d} | SSIM={sscore:.3f}")

        idx += 1

    cap.release()
    print(f"Done. Saved {saved} frames to: {os.path.abspath(args.out_dir)}")

def build_parser():
    p = argparse.ArgumentParser(description="Extract sheet-only frames when the music line changes.")
    p.add_argument("source",
                   help="YouTube URL or local video path")
    p.add_argument("--out-dir", default="sheet_frames",
                   help="Directory to save extracted frames (default: sheet_frames)")
    p.add_argument("--sample-fps", type=float, default=3.0,
                   help="Analysis sampling rate in frames/sec (default: 3.0)")
    p.add_argument("--advance-fraction", type=float, default=0.10,
                   help="Required downward shift of staff centroid as fraction of ROI height (default: 0.10)")
    p.add_argument("--ssim-thresh", type=float, default=0.985,
                   help="SSIM threshold vs. last-saved frame to avoid near-duplicates (default: 0.985)")
    p.add_argument("--recompute-boundary-every", type=float, default=0.0,
                   help="Recompute sheet/keyboard boundary every N seconds (default: 0 = only once)")
    p.add_argument("--debug", action="store_true", help="Show detected ROI on first frame")
    p.add_argument("--verbose", action="store_true", help="Print per-save details")
    return p

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    run(args)

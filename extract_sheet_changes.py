import os, sys, tempfile, subprocess, math, glob, shutil
import numpy as np
import cv2
from yt_dlp import YoutubeDL
from skimage.metrics import structural_similarity as ssim

# ------------------------- CONFIG -------------------------
YOUTUBE_URL = sys.argv[1] if len(sys.argv) > 1 else None
OUT_DIR = sys.argv[2] if len(sys.argv) > 2 else "sheet_frames"
SAMPLE_FPS = 3.0           # analyze ~3 frames/sec; raise if the music scrolls fast
ADVANCE_FRACTION = 0.10    # save when staff centroid moves ≥ 10% of ROI height
SSIM_THRESH = 0.985        # require notable content change vs last saved
AUTO_ROI = True            # if False, you will be asked to draw ROI on the first frame
# ----------------------------------------------------------

def download_youtube(url: str) -> str:
    """
    Download best available video as mp4 using yt-dlp and return the local path.
    """
    tmpdir = tempfile.mkdtemp(prefix="yt_")
    ydl_opts = {
        "outtmpl": os.path.join(tmpdir, "%(title)s.%(ext)s"),
        "format": "bv*+ba/b",                  # best video+audio, fallback to best
        "merge_output_format": "mp4",
        "quiet": True,
        "noprogress": True,
    }
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        path = ydl.prepare_filename(info)
    if not path.endswith(".mp4"):
        alt = os.path.splitext(path)[0] + ".mp4"
        if os.path.exists(alt):
            path = alt
    return path

def choose_roi(first_frame, auto=True):
    """
    Return (x,y,w,h) for the sheet region.
    If auto=True, try to auto-detect a page-like bright rectangle; else ask the user to draw it.
    """
    H, W = first_frame.shape[:2]
    if not auto:
        # interactive selection
        tmp = first_frame.copy()
        r = cv2.selectROI("Draw sheet ROI and press ENTER", tmp, False, False)
        cv2.destroyWindow("Draw sheet ROI and press ENTER")
        if r[2] > 0 and r[3] > 0:
            return tuple(map(int, r))
        # fallback to center box if user cancels
    # heuristic auto ROI: search for large, roughly rectangular bright area
    gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 60, 180)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, roi = 0, (int(W*0.15), int(H*0.1), int(W*0.7), int(H*0.8))
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        area = w*h
        ar = w/(h+1e-6)
        # a page is big, roughly rectangular, not too skinny
        if area > best and 0.6 < ar < 1.4 and w > 0.30*W and h > 0.30*H:
            best, roi = area, (x,y,w,h)
    return roi

def staff_mask(gray):
    """
    Enhance staff lines (parallel horizontal dark lines on light background).
    1) black-hat to emphasize thin dark horizontals
    2) Otsu threshold
    3) vertical dilate to group each 5-line staff into a band
    """
    h, w = gray.shape[:2]
    kw = max(15, w // 40)  # horizontal kernel width
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 1))
    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    th = cv2.threshold(bh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    vker = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 9))
    band = cv2.dilate(th, vker, iterations=1)
    return band

def staff_y_center(mask):
    """
    Weighted vertical centroid of the staff band. Returns None if no band detected.
    """
    rows = mask.sum(axis=1).astype(np.float64)
    if rows.max() <= 0:
        return None
    y = np.arange(len(rows), dtype=np.float64)
    return int((rows * y).sum() / (rows.sum() + 1e-6))

def main():
    if not YOUTUBE_URL:
        print("Usage: python extract_sheet_changes.py <youtube_url> [out_dir]")
        sys.exit(1)

    os.makedirs(OUT_DIR, exist_ok=True)
    print("Downloading video...")
    video_path = download_youtube(YOUTUBE_URL)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit("Could not open downloaded video.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    sample_every = max(1, int(round(fps / max(0.1, SAMPLE_FPS))))
    print(f"Video FPS: {fps:.2f}. Sampling every {sample_every} frames (~{fps/sample_every:.2f} fps).")

    # Read first frame to decide ROI
    ret, first = cap.read()
    if not ret:
        raise SystemExit("Cannot read first frame.")
    x, y, w, h = choose_roi(first, AUTO_ROI)
    roi_height = h
    advance_pixels = int(ADVANCE_FRACTION * roi_height)
    print(f"ROI: (x={x}, y={y}, w={w}, h={h}); advance threshold: {advance_pixels} px")

    # Rewind
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    prev_kept = None
    last_yc = None
    saved = 0
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % sample_every != 0:
            idx += 1
            continue

        crop = frame[y:y+h, x:x+w]
        if crop.size == 0:
            idx += 1
            continue

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        # normalize ROI size for stable SSIM and morphology
        target_w = 800
        gray = cv2.resize(gray, (target_w, int(target_w * gray.shape[0] / gray.shape[1])), interpolation=cv2.INTER_AREA)

        mask = staff_mask(gray)
        yc = staff_y_center(mask)

        if yc is None:
            idx += 1
            continue

        # Decide whether to save
        if prev_kept is None:
            should_save = True
            sscore = 0.0
        else:
            # SSIM on ROI only to reject near-duplicates
            sscore = ssim(prev_kept, gray)
            moved = (last_yc is None) or (yc - last_yc >= advance_pixels * (gray.shape[0] / roi_height))
            changed = sscore < SSIM_THRESH
            should_save = moved and changed

        if should_save:
            out_path = os.path.join(OUT_DIR, f"{saved:05d}.png")
            cv2.imwrite(out_path, crop)  # save the original-scale crop
            prev_kept = gray.copy()
            last_yc = yc
            saved += 1
            # Optional: print timestamp
            t = idx / fps
            print(f"[{saved:04d}] saved @ t={t:7.2f}s | y={yc:4d} | SSIM={sscore:.3f}")

        idx += 1

    cap.release()
    print(f"Done. Saved {saved} frames to: {os.path.abspath(OUT_DIR)}")

if __name__ == "__main__":
    main()

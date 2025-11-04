# pip install pymupdf
import fitz  # PyMuPDF
import re
from collections import defaultdict
from pathlib import Path
import numpy as np
import glob

# --------------------- utility ---------------------
def normspace(s: str) -> str:
    return re.sub(r"[ \t]+", " ", s.strip())

def overlaps(a, b, pad=0):
    # a,b: (x0,y0,x1,y1)
    return not (a[2] + pad <= b[0] or b[2] + pad <= a[0] or a[3] + pad <= b[1] or b[3] + pad <= a[1])

def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

# --------------------- page-type detectors ---------------------
def is_toc_page(page_dict) -> bool:
    """Detects a Table-of-Contents-like page."""
    text = page_dict.get("text", "") or "\n".join(
        b.get("text", "") for b in page_dict.get("blocks", []) if b.get("type") == 0
    )
    if not text:
        return False
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return False
    has_title = any(re.match(r"(?i)^\s*contents?\s*$", l) for l in lines[:6])
    right_nums = sum(bool(re.search(r"\b\d+\s*$", l)) for l in lines)
    dotted = sum(("." * 3) in l for l in lines)
    return has_title and (right_nums >= 8 or dotted >= 5)

def collect_repeated_edge_lines(doc, top_frac=0.12, bot_frac=0.12, freq_thresh=0.5, y_tol=6):
    """Learns repeated header/footer strings near top/bottom across pages."""
    page_h = None
    seen = defaultdict(list)  # (side, text) -> [y]
    for p in doc:
        d = p.get_text("dict")
        if page_h is None:
            page_h = p.rect.height
        for b in d["blocks"]:
            if b["type"] != 0:
                continue
            x0, y0, x1, y1 = b["bbox"]
            txt = normspace(" ".join(
                "".join(sp["text"] for sp in ln.get("spans", []))
                for ln in b.get("lines", [])
            ))
            if not txt:
                continue
            if y0 < top_frac * page_h:
                seen[("top", txt)].append(y0)
            elif y1 > (1 - bot_frac) * page_h:
                seen[("bottom", txt)].append(y1)

    repeated = {"top": [], "bottom": []}
    for (side, txt), ys in seen.items():
        if len(ys) >= freq_thresh * len(doc):
            repeated[side].append((txt, float(np.median(ys)), y_tol))
    return repeated

# --------------------- table/figure filters ---------------------
def drawings_bboxes(page, min_len=20):
    """
    Returns a coarse bbox that covers regions with many straight h/v segments
    (typical for table grids, boxes). If none, returns [].
    """
    boxes = []
    items = page.get_drawings()
    hv = []
    for d in items:
        for it in d["items"]:
            if it[0] != "l":
                continue
            (x0, y0), (x1, y1) = it[1], it[2]
            if abs(x0 - x1) >= min_len and abs(y0 - y1) < 1.5:
                hv.append((min(x0, x1), y0, max(x0, x1), y1))
            elif abs(y0 - y1) >= min_len and abs(x0 - x1) < 1.5:
                hv.append((x0, min(y0, y1), x1, max(y0, y1)))
    if hv:
        xs0, ys0, xs1, ys1 = zip(*hv)
        boxes.append((min(xs0), min(ys0), max(xs1), max(ys1)))
    return boxes

TABULAR_KEYWORDS = {
    "model","params","parameters","nparams","nlayers","layers","dmodel","heads","nheads","dhead",
    "batch size","learning rate","lr","accuracy","acc","f1","bleu","rouge","top-1","top1","top-5","top5",
    "dataset","train","dev","test","epoch","epochs","perplexity","mmlu","gsm8k","hellaswag"
}

def looks_tabular(text_block: str) -> bool:
    b = text_block.strip()
    if not b:
        return False
    # numeric density / structure
    num_count = len(re.findall(r"\b\d[\d,._]*\b", b))
    punct_sent = len(re.findall(r"[.!?](\s|$)", b))
    tabs_or_pipes = b.count("\t") + b.count("|")
    multi_spaces = bool(re.search(r" {2,}", b))
    has_kw = any(k in b.lower() for k in TABULAR_KEYWORDS)
    if tabs_or_pipes >= 1 or multi_spaces:
        return True
    if has_kw and num_count >= 2 and punct_sent == 0:
        return True
    if num_count >= 3 and punct_sent == 0:
        return True
    digit_ratio = sum(ch.isdigit() for ch in b) / max(1, len(b))
    if digit_ratio > 0.25 and punct_sent == 0:
        return True
    return False

# --------------------- main extractor ---------------------
REF_HEAD_RE = re.compile(r"(?im)^\s*(references|bibliography|works\s+cited|literature)\s*$")
CAPTION_RE = re.compile(r"(?i)^(figure|fig\.|table|tab\.|algorithm|alg\.)\s*[S\dIVX]+")
APPENDIX_RE = re.compile(r"(?im)^\s*appendix\s+[A-Z0-9]+\s*$")

def extract_paragraph_text(pdf_path: Path) -> str:
    doc = fitz.open(pdf_path)
    repeated = collect_repeated_edge_lines(doc)

    pages_out = []
    reached_refs = False

    for page in doc:
        if reached_refs:
            break

        d = page.get_text("dict")
        if is_toc_page(d):
            continue

        # figure/table regions
        exclusion_boxes = drawings_bboxes(page)
        for b in d["blocks"]:
            if b["type"] == 1:  # image blocks
                exclusion_boxes.append(tuple(b["bbox"]))

        kept = []
        for b in d["blocks"]:
            if b["type"] != 0:
                continue
            x0, y0, x1, y1 = b["bbox"]
            # block text
            txt_lines = []
            for ln in b.get("lines", []):
                span_text = "".join(sp.get("text", "") for sp in ln.get("spans", []))
                if span_text:
                    txt_lines.append(span_text)
            raw = normspace(" ".join(txt_lines))
            if not raw:
                continue

            # cut entire document at references / appendix
            if REF_HEAD_RE.match(raw) or APPENDIX_RE.match(raw):
                reached_refs = True
                kept = []
                break

            # repeated header/footer removal (exact text + near same y)
            drop_edge = False
            for side in ("top","bottom"):
                for tref, yref, ytol in repeated[side]:
                    if tref == raw and abs((y0 if side=="top" else y1) - yref) <= ytol:
                        drop_edge = True
                        break
                if drop_edge:
                    break
            if drop_edge:
                continue

            # remove captions
            if CAPTION_RE.match(raw):
                continue

            # remove table/figure regions by geometry overlap
            if any(overlaps((x0,y0,x1,y1), z, pad=2) for z in exclusion_boxes):
                continue

            kept.append((y0, x0, raw))

        if not kept:
            continue

        # column-aware order: split page roughly into columns by x
        # simple heuristic: two-column if width > 500 and blocks cluster by x
        width = page.rect.width
        col_split = width * 0.5
        left = sorted([k for k in kept if k[1] <= col_split], key=lambda t: (t[0], t[1]))
        right = sorted([k for k in kept if k[1] >  col_split], key=lambda t: (t[0], t[1]))

        ordered_blocks = left + right  # read left column then right
        # drop table-like text
        para_blocks = [b for (_, _, b) in ordered_blocks if not looks_tabular(b)]
        if para_blocks:
            pages_out.append("\n".join(para_blocks))

    text = "\n\n".join(pages_out)
    # light whitespace normalization
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text

# --------------------- batch runner ---------------------
def process_pdfs(pdf_paths, out_dir="clean_txt", merged_out="merged_corpus.txt"):
    out_dir = Path(out_dir)
    merged_out = Path(merged_out)
    merged_parts = []

    for p in pdf_paths:
        p = Path(p)
        try:
            cleaned = extract_paragraph_text(p)
            out_txt = out_dir / (p.stem + ".txt")
            write_text(out_txt, cleaned)
            merged_parts.append(cleaned)
            print(f"[OK] {p.name} -> {out_txt}")
        except Exception as e:
            print(f"[WARN] {p}: {e}")

    merged = "\n\n" + ("\n\n" + ("="*80) + "\n\n").join(merged_parts) + "\n\n"
    write_text(merged_out, merged)
    print(f"[MERGED] {merged_out}  ({len(merged)} chars)")

# --------------------- example usage ---------------------


if __name__ == "__main__":
    # Path to the folder containing all your PDF files
    pdf_folder = Path("pdfs")

    # Collect all PDFs in that folder (recursively if needed)
    pdfs = sorted(glob.glob(str(pdf_folder / "*.pdf")))

    # Run the cleaner and merger
    process_pdfs(pdfs, out_dir="clean_txt", merged_out="merged_corpus.txt")
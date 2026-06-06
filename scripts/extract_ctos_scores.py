"""Extract CTOS credit scores from documents referenced in the submissions CSV.

Outputs: output/ctos_scores.csv with columns: row_index,name,ctos_file,ctos_score,raw_matches

Strategy:
- Read `input/submissions.csv` (fallback `output/flattened_submissions.csv`).
- Look for a column like `ctos & doc`, `ctos`, `ctos & docs` (case-insensitive).
- For each filename, try to open file in `input/docs/` and extract text (pdfplumber -> PyPDF2 -> pytesseract).
- Regex search for 3-digit numbers and pick the one between 300 and 850.

Notes:
- If required packages are missing, the script prints installation hints and exits non-zero.
"""

from pathlib import Path
import re
import sys
import csv
import json

try:
    import pandas as pd
except Exception:
    print("pandas is required. Install with: pip install pandas")
    raise

INPUT_CSV_PRIMARY = Path('input') / 'submissions_with_ctos.csv'
INPUT_CSV_SECONDARY = Path('input') / 'submissions.csv'
INPUT_CSV = INPUT_CSV_PRIMARY if INPUT_CSV_PRIMARY.exists() else INPUT_CSV_SECONDARY
FALLBACK_CSV = Path('output') / 'flattened_submissions.csv'
DOCS_DIR = Path('input') / 'docs'
OUTPUT = Path('output') / 'ctos_scores.csv'
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

# helpers for text extraction

def extract_text_pdf_pdfplumber(path: Path) -> str:
    try:
        import pdfplumber
    except Exception:
        raise
    text_parts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text_parts.append(t)
    return "\n".join(text_parts)


def extract_text_pdf_mupdf(path: Path) -> str:
    """Extract text using PyMuPDF (fitz). Fast and effective for selectable PDFs."""
    try:
        import fitz
    except Exception:
        raise
    text_parts = []
    doc = fitz.open(str(path))
    for page in doc:
        try:
            t = page.get_text()
            if t:
                text_parts.append(t)
        except Exception:
            continue
    return "\n".join(text_parts)


def extract_text_pdf_pypdf2(path: Path) -> str:
    try:
        from PyPDF2 import PdfReader
    except Exception:
        raise
    text_parts = []
    reader = PdfReader(str(path))
    for page in reader.pages:
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        if t:
            text_parts.append(t)
    return "\n".join(text_parts)


def extract_text_pdf_ocr(path: Path) -> str:
    """Run OCR on a PDF by rendering pages to images using pdf2image and pytesseract."""
    # prefer pytesseract/pdf2image when available
    try:
        from pdf2image import convert_from_path
        import pytesseract
        from PIL import Image
        text_parts = []
        try:
            images = convert_from_path(str(path), first_page=1, last_page=5, dpi=300)
            for img in images:
                try:
                    t = pytesseract.image_to_string(img)
                    if t:
                        text_parts.append(t)
                except Exception:
                    continue
            if text_parts:
                return "\n".join(text_parts)
        except Exception:
            pass
    except Exception:
        # try easyocr (pure pip) if available
        try:
            import easyocr
            from pdf2image import convert_from_path
            reader = easyocr.Reader(['en'], gpu=False)
            text_parts = []
            images = convert_from_path(str(path), first_page=1, last_page=5, dpi=300)
            for img in images:
                try:
                    res = reader.readtext(img)
                    if res:
                        text_parts.append(' '.join([r[1] for r in res]))
                except Exception:
                    continue
            if text_parts:
                return "\n".join(text_parts)
        except Exception:
            pass
    # fallback: try pdfplumber to render pages and OCR
    # fallback: try pdfplumber to render pages and OCR (pytesseract or easyocr)
    try:
        import pdfplumber
        text_parts = []
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages[:5]):
                pil_img = None
                try:
                    pil_img = page.to_image(resolution=150).original
                except Exception:
                    try:
                        pil_img = page.to_image().original
                    except Exception:
                        pil_img = None
                if pil_img is not None:
                    # try pytesseract
                    try:
                        import pytesseract
                        t = pytesseract.image_to_string(pil_img)
                        if t:
                            text_parts.append(t)
                            continue
                    except Exception:
                        pass
                    # try easyocr
                    try:
                        import easyocr
                        reader = easyocr.Reader(['en'], gpu=False)
                        res = reader.readtext(pil_img)
                        if res:
                            text_parts.append(' '.join([r[1] for r in res]))
                    except Exception:
                        pass
        return "\n".join(text_parts)
    except Exception:
        return ''


def extract_text_image_ocr(path: Path) -> str:
    # try pytesseract first
    try:
        from PIL import Image
        import pytesseract
        img = Image.open(path)
        return pytesseract.image_to_string(img)
    except Exception:
        pass
    # fallback to easyocr
    try:
        import easyocr
        from PIL import Image
        img = Image.open(path)
        reader = easyocr.Reader(['en'], gpu=False)
        res = reader.readtext(img)
        if res:
            return ' '.join([r[1] for r in res])
    except Exception:
        pass
    raise


def find_ctos_column(columns):
    normalized = [re.sub(r"\s+", " ", str(c).strip().lower()) for c in columns]
    for cand in ['ctos & doc', 'ctos & docs', 'ctos', 'ctos_doc', 'ctos docs', 'ctosfile', 'ctos file']:
        if cand in normalized:
            return columns[normalized.index(cand)]
    # try fuzzy match
    for i, c in enumerate(normalized):
        if 'ctos' in c:
            return columns[i]
    return None


def extract_score_from_text(text: str):
    if not text:
        return None, []

    # helper: detect if a 3-digit match is actually part of a grouped number like '58,631' or '631.00'
    def is_part_of_grouped_number(txt, start, end):
        # check immediate neighbors for comma/dot thousands/decimal separators
        if start - 1 >= 0 and txt[start - 1] in ',.':
            return True
        if end < len(txt) and txt[end] in ',.':
            return True
        # also check a short window for pattern digit+sep+digits
        left = max(0, start - 3)
        right = min(len(txt), end + 3)
        snippet = txt[left:right]
        if re.search(r"\d[\.,]\d", snippet):
            return True
        return False

    # First, look for explicit labeled patterns near the word 'score'/'ctos'/'credit'
    labeled_patterns = [
        r"(?:ctos|credit)[^\d]{0,30}?(?:score)?[^\d]{0,10}?(?<!\d)(\d{3})(?![\d\.,])",
        r"score[:=\s]{0,5}(?<!\d)(\d{3})(?![\d\.,])",
        r"ctos[:=\s]{0,5}(?<!\d)(\d{3})(?![\d\.,])",
        r"credit[:=\s]{0,5}(?<!\d)(\d{3})(?![\d\.,])",
    ]
    for pat in labeled_patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            try:
                v = int(m.group(1))
                if 300 <= v <= 850:
                    return v, [v]
            except Exception:
                pass

    # Collect candidate 3-digit numbers that are not part of grouped numbers
    candidates = []
    for m in re.finditer(r"(?<!\d)(\d{3})(?!\d)", text):
        s = m.group(1)
        start, end = m.span(1)
        try:
            if is_part_of_grouped_number(text, start, end):
                continue
            v = int(s)
            if 300 <= v <= 850:
                candidates.append((v, start))
        except Exception:
            continue

    if not candidates:
        return None, []

    # Prefer candidates nearest to keywords like 'score', 'ctos', 'credit'
    keyword_positions = [m.start() for m in re.finditer(r"\b(score|ctos|credit)\b", text, flags=re.IGNORECASE)]
    def min_dist_to_keywords(pos):
        if not keyword_positions:
            return float('inf')
        return min(abs(pos - kp) for kp in keyword_positions)

    # If any candidates are within a small window of a keyword, prefer those
    window = 40
    near_candidates = [c for c in candidates if any(abs(c[1] - kp) <= window for kp in keyword_positions)]
    if near_candidates:
        candidates_sorted = sorted(near_candidates, key=lambda x: (min_dist_to_keywords(x[1]), -x[0]))
    else:
        candidates_sorted = sorted(candidates, key=lambda x: (min_dist_to_keywords(x[1]), -x[0]))

    best = candidates_sorted[0]
    scores = [c[0] for c in candidates]
    return best[0], scores
def text_has_no_score_marker(text: str) -> bool:
    if not text:
        return False
    markers = [
        r"\bno score\b",
        r"\bno credit score\b",
        r"\bscore not found\b",
        r"\bnot found\b",
        r"\bno record\b",
        r"\bno ctos\b",
        r"\bno ctos score\b",
        r"\bno data\b",
    ]
    txt = text.lower()
    for m in markers:
        if re.search(m, txt):
            return True
    return False


def locate_file(fn: str) -> Path | None:
    if not fn or str(fn).strip() == '':
        return None
    fn = str(fn).strip().strip('"').strip("'")
    p = Path(fn)
    # if fn already a path or absolute
    if p.exists():
        return p
    # try relative to input/docs
    cand = DOCS_DIR / p.name
    if cand.exists():
        return cand
    # try with .pdf
    cand2 = DOCS_DIR / (p.name + '.pdf')
    if cand2.exists():
        return cand2
    # try lower-case
    for ext in ('.pdf','.PDF','.png','.jpg','.jpeg'):
        cand3 = DOCS_DIR / (p.name + ext)
        if cand3.exists():
            return cand3
    return None


def call_local_llm_for_score(text: str, url: str = "http://127.0.0.1:1234/api/v1/chat", model: str = "google/gemma-4-e4b", timeout: int = 30):
    """Call local LM endpoint and request a strict JSON reply: {"score":720} or {"score":null}.

    Returns (score_or_None, candidates_list, raw_reply_text).
    """
    if not text:
        return None, [], ''
    try:
        import requests
    except Exception:
        raise

    system_prompt = (
        'You are a strict extractor. Reply with a single JSON object exactly like {"score":720} '
        'or {"score":null}. Do not add any other text, explanation, or surrounding markup.'
    )
    prompt = (
        "Extract a single CTOS / credit score (integer 300-850) from the text below. If none, return {\"score\":null}.\n\nText:\n"
        + text[:4000]
    )
    payload = {
        "model": model,
        "system_prompt": system_prompt,
        "input": prompt,
    }
    resp = requests.post(url, json=payload, timeout=timeout)
    # do not raise here; we'll capture status and body for debugging

    # collect candidate strings from common LM response shapes
    candidates = []
    raw_text = ''
    try:
        j = resp.json()
    except Exception:
        raw_text = resp.text
        j = None

    if isinstance(j, dict):
        out = j.get('output')
        if isinstance(out, list):
            for item in out:
                if isinstance(item, dict):
                    c = item.get('content')
                    if c is not None:
                        candidates.append(c if isinstance(c, str) else str(c))
        # older/alternate fields
        for key in ('result', 'text', 'reply', 'message'):
            v = j.get(key)
            if isinstance(v, str) and v:
                candidates.append(v)
        # fallback to full JSON string
        if not candidates:
            try:
                candidates.append(json.dumps(j, ensure_ascii=False))
            except Exception:
                pass
    else:
        candidates.append(resp.text)

    raw_text = '\n'.join(candidates)

    # include HTTP status for debugging clarity
    try:
        status = f"HTTP {resp.status_code}"
    except Exception:
        status = "HTTP ?"
    raw_text = f"{status}: {raw_text}"

    # try parsing JSON candidates first
    for cand in candidates:
        try:
            parsed = json.loads(cand)
            if isinstance(parsed, dict) and 'score' in parsed:
                sc = parsed['score']
                if sc is None:
                    return None, [], cand
                try:
                    v = int(sc)
                    if 300 <= v <= 850:
                        return v, [v], cand
                except Exception:
                    continue
        except Exception:
            continue

    # fallback: look for any 3-digit number in the reply text
    m = re.search(r"\\b(\\d{3})\\b", raw_text)
    if m:
        v = int(m.group(1))
        if 300 <= v <= 850:
            return v, [v], raw_text
    return None, [], raw_text


def main():
    if INPUT_CSV.exists():
        df = pd.read_csv(INPUT_CSV, dtype=object, low_memory=False)
    elif FALLBACK_CSV.exists():
        df = pd.read_csv(FALLBACK_CSV, dtype=object, low_memory=False)
    else:
        print('No input CSV found (prefer input/submissions_with_ctos.csv; fallback input/submissions.csv or output/flattened_submissions.csv)')
        return 2

    ctos_col = find_ctos_column(df.columns)
    if not ctos_col:
        print('No CTOS column found in input. Columns:', df.columns.tolist())
        return 3

    results = []
    llm_debug = []
    total = 0
    found = 0
    missing_file = 0
    no_score = 0

    for idx, row in df.iterrows():
        total += 1
        raw = row.get(ctos_col, '')
        if pd.isna(raw) or str(raw).strip() == '':
            results.append({'row_index': idx, 'name': row.get('name',''), 'ctos_file': '', 'ctos_score': '', 'raw_matches': ''})
            continue
        # if multiple files separated by ; or , take first
        first = re.split(r"[;,]", str(raw))[0].strip()
        located = locate_file(first)
        if not located:
            missing_file += 1
            results.append({'row_index': idx, 'name': row.get('name',''), 'ctos_file': first, 'ctos_score': '', 'raw_matches': ''})
            continue
        score = None
        matches = []
        text = ''
        # try PyMuPDF first (fast, good for selectable PDFs)
        tried = []
        try:
            text = extract_text_pdf_mupdf(located)
            tried.append('mupdf')
        except Exception:
            text = ''
        # then pdfplumber
        if not text:
            try:
                text = extract_text_pdf_pdfplumber(located)
                tried.append('pdfplumber')
            except Exception:
                text = ''
        # then PyPDF2
        if not text:
            try:
                text = extract_text_pdf_pypdf2(located)
                tried.append('pypdf2')
            except Exception:
                text = ''
        # if still empty and file might be image-based, try image OCR
        if not text:
            try:
                text = extract_text_image_ocr(located)
            except Exception:
                text = ''
        # if we got any text, write a small sample for debugging
        samples_dir = OUTPUT.parent / 'ctos_text_samples'
        try:
            samples_dir.mkdir(parents=True, exist_ok=True)
            if text:
                sample_path = samples_dir / f"{idx}_{(located.name if located else first)}.txt"
                # sanitize filename
                sample_path = Path(str(sample_path).replace('/', '_').replace('\\', '_'))
                with sample_path.open('w', encoding='utf-8') as sf:
                    sf.write(text[:4000])
        except Exception:
            pass
        score, matches = extract_score_from_text(text)
        # if the extracted text explicitly says there's no score, skip further attempts
        no_score_marker = False
        try:
            if text and text_has_no_score_marker(text):
                no_score_marker = True
        except Exception:
            no_score_marker = False
        # If no score found from text, try OCR on PDF pages and then optionally call local LLM
        if score is None:
            # try PDF OCR render (if pdf)
            try:
                if located.suffix.lower() == '.pdf':
                    ocr_text = extract_text_pdf_ocr(located)
                    if ocr_text and not text:
                        text = ocr_text
                        score, matches = extract_score_from_text(text)
            except Exception:
                pass

        # If still no score, call local LLM if available (skip if explicit no-score marker)
        llm_reply = ''
        if score is None and not no_score_marker:
            # allow skipping local LLM calls via env var for offline or long runs
            import os
            if os.environ.get('SKIP_LOCAL_LLM') in ("1", "true", "True"):
                llm_reply = 'SKIPPED'
            else:
                try:
                    print(f'Calling LM for row {idx}, file {located.name if located else first}')
                    llm_score, llm_candidates, llm_reply = call_local_llm_for_score(text)
                    print(f'LM reply length: {len(llm_reply) if llm_reply else 0}')
                    if llm_score is not None:
                        score = llm_score
                        matches = llm_candidates
                except Exception as e:
                    # capture exception message for debugging
                    try:
                        llm_reply = f"ERROR: {type(e).__name__}: {e}"
                    except Exception:
                        llm_reply = "ERROR: unknown"
        if score is not None:
            found += 1
        else:
            no_score += 1
        results.append({'row_index': idx, 'name': row.get('name',''), 'ctos_file': located.name if located else first, 'ctos_score': score if score is not None else '', 'raw_matches': '|'.join(str(m) for m in matches), 'llm_reply': llm_reply if 'llm_reply' in locals() else '', 'tried_methods': '|'.join(tried), 'text_len': len(text) if text else 0, 'no_score_marker': bool(no_score_marker)})
        # capture LM reply for debugging (record attempt even if empty)
        if 'llm_reply' in locals():
            llm_debug.append({'row_index': idx, 'ctos_file': located.name if located else first, 'llm_reply': llm_reply, 'text_len': len(text) if text else 0, 'tried_methods': '|'.join(tried), 'no_score_marker': bool(no_score_marker)})

    # write CSV
    with OUTPUT.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=['row_index','name','ctos_file','ctos_score','raw_matches','llm_reply','tried_methods','text_len','no_score_marker'])
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    # write LLM debug replies
    debug_path = OUTPUT.parent / 'ctos_llm_debug.jsonl'
    if llm_debug:
        with debug_path.open('w', encoding='utf-8') as fh:
            for d in llm_debug:
                fh.write(json.dumps(d, ensure_ascii=False) + '\n')

    print(f'Processed {total} rows; scores found: {found}; missing files: {missing_file}; no score found in file: {no_score}')
    print('Wrote', OUTPUT)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

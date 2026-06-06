import pandas as pd
from pathlib import Path

INPUT = Path('input') / 'submissions.xls'
OUT = Path('output') / 'flattened_submissions.csv'
OUT.parent.mkdir(parents=True, exist_ok=True)

if not INPUT.exists():
    raise SystemExit(f"Input file not found: {INPUT}")

xls = pd.ExcelFile(INPUT)
frames = []
for sheet in xls.sheet_names:
    try:
        df = pd.read_excel(xls, sheet_name=sheet, dtype=object)
    except Exception:
        # fallback to generic read
        df = pd.read_excel(INPUT, sheet_name=sheet, dtype=object)
    if df is None:
        continue
    df = df.dropna(how='all')
    if df.empty:
        continue
    df = df.copy()
    df['source_sheet'] = sheet
    frames.append(df)

if not frames:
    raise SystemExit('No non-empty sheets found in workbook')

out = pd.concat(frames, ignore_index=True)
# normalize column names to trimmed strings
out.columns = [str(c).strip() for c in out.columns]
# Clean text fields: strip surrounding quotes/whitespace and convert empty strings to NA
def _clean_cell(v):
    if pd.isna(v):
        return pd.NA
    s = str(v).strip()
    # strip common surrounding quotes
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()
    # remove any stray leading/trailing quotes
    s = s.strip('"').strip("'").strip()
    return s if s != "" else pd.NA

for col in out.columns:
    if col == 'source_sheet':
        continue
    out[col] = out[col].apply(_clean_cell)

# drop rows where all columns except source_sheet are empty
non_source = [c for c in out.columns if c != 'source_sheet']
out = out.dropna(how='all', subset=non_source).reset_index(drop=True)

# If a name column exists, drop rows where the name is missing — treat these as empty rows.
name_col = None
for c in out.columns:
    if str(c).strip().lower() == 'name':
        name_col = c
        break
if name_col is None:
    for c in out.columns:
        if 'name' in str(c).strip().lower():
            name_col = c
            break
if name_col is not None:
    out = out.dropna(subset=[name_col]).reset_index(drop=True)

# Add calculated flags: is_approved, is_rejected, is_self_employed
def _find_col(containing: str):
    needle = containing.lower()
    for c in out.columns:
        if needle == str(c).strip().lower():
            return c
    for c in out.columns:
        if needle in str(c).strip().lower():
            return c
    return None

status_col = _find_col('status')
employment_col = _find_col('employment type') or _find_col('employment')

if status_col is not None:
    s = out[status_col].fillna('').astype(str).str.lower()
    out['is_approved'] = s.str.contains('approved')
    out['is_rejected'] = s.str.contains('reject') | s.str.contains('declin')
else:
    out['is_approved'] = False
    out['is_rejected'] = False

if employment_col is not None:
    e = out[employment_col].fillna('').astype(str).str.lower()
    out['is_self_employed'] = e.str.contains('self employ') | e.str.contains('self-employed') | e.str.contains('selfemploy')
else:
    out['is_self_employed'] = False

out.to_csv(OUT, index=False)
print(f'Wrote {len(out)} rows to {OUT} (dropped rows missing `{name_col}`)')

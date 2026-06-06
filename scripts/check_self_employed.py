import pandas as pd
from pathlib import Path
import re

INPUT_PRIMARY = Path('input') / 'submissions_with_ctos.csv'
INPUT_SECONDARY = Path('input') / 'submissions.csv'
FALLBACK = Path('output') / 'flattened_submissions.csv'

if INPUT_PRIMARY.exists():
    df = pd.read_csv(INPUT_PRIMARY, low_memory=False, dtype=object)
elif INPUT_SECONDARY.exists():
    df = pd.read_csv(INPUT_SECONDARY, low_memory=False, dtype=object)
elif FALLBACK.exists():
    df = pd.read_csv(FALLBACK, low_memory=False, dtype=object)
else:
    raise SystemExit('No input CSV found (prefer input/submissions_with_ctos.csv; fallback input/submissions.csv or output/flattened_submissions.csv)')

# normalize columns
df.columns = [re.sub(r"\s+", " ", str(c).strip().lower()) for c in df.columns]

# find relevant columns
status_col = None
for c in df.columns:
    if c in ('status','approval_label'):
        status_col = c
        break

employment_col = None
for c in df.columns:
    if 'employ' in c:
        employment_col = c
        break

import re

# compute from raw columns using regex word boundaries to avoid substring collisions
if employment_col:
    e = df[employment_col].fillna('').astype(str)
    df['is_self_employed'] = e.str.contains(r"\bself\b", flags=re.IGNORECASE, regex=True) | e.str.contains(r"self-?employ", flags=re.IGNORECASE, regex=True)
else:
    df['is_self_employed'] = False

if status_col:
    s = df[status_col].fillna('').astype(str)
    df['is_approved'] = s.str.contains(r"\bapprove\w*\b", flags=re.IGNORECASE, regex=True)
    df['is_rejected'] = s.str.contains(r"\breject\w*\b", flags=re.IGNORECASE, regex=True) | s.str.contains(r"\bdeclin\w*\b", flags=re.IGNORECASE, regex=True)
else:
    df['is_approved'] = False
    df['is_rejected'] = False

# ensure booleans
df['is_self_employed'] = df['is_self_employed'].fillna(False).astype(bool)
df['is_approved'] = df['is_approved'].fillna(False).astype(bool)
df['is_rejected'] = df['is_rejected'].fillna(False).astype(bool)

se = df[df['is_self_employed']]
total_se = len(se)
approved_se = se['is_approved'].sum()
rejected_se = se['is_rejected'].sum()
print(f"self_employed_total,{total_se}")
print(f"self_employed_approved,{approved_se}")
print(f"self_employed_rejected,{rejected_se}")
if total_se>0:
    print(f"approval_rate,{approved_se/total_se:.3f}")
else:
    print("approval_rate,0.0")

# show top 5 approved self-employed rows
if approved_se>0:
    approved_rows = se[se['is_approved']].head(5)
    cols = approved_rows.columns.tolist()
    print('\n--- sample approved self-employed rows (top 5) ---')
    print(approved_rows.head(5).to_csv(index=False))

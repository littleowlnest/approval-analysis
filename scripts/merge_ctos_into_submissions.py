from pathlib import Path
import pandas as pd

SUB = Path('input') / 'submissions.csv'
CTOS = Path('output') / 'ctos_scores.csv'
OUT = Path('input') / 'submissions_with_ctos.csv'

if not SUB.exists():
    print('Missing', SUB)
    raise SystemExit(2)
if not CTOS.exists():
    print('Missing', CTOS)
    raise SystemExit(3)

s = pd.read_csv(SUB, dtype=object, low_memory=False)
c = pd.read_csv(CTOS, dtype=object, low_memory=False)

# ensure row_index available
if 'row_index' not in c.columns:
    print('ctos file missing row_index column')
    raise SystemExit(4)

c['row_index'] = pd.to_numeric(c['row_index'], errors='coerce')
# merge on row index
s = s.reset_index().rename(columns={'index':'row_index'})
merged = s.merge(c[['row_index','ctos_score']], on='row_index', how='left')
# if ctos_score exists, write as numeric where possible
merged['ctos_score'] = pd.to_numeric(merged['ctos_score'], errors='coerce')
merged.to_csv(OUT, index=False)
print('Wrote', OUT)
print('Total submissions:', len(s))
print('CTOS scores found:', merged['ctos_score'].notna().sum())

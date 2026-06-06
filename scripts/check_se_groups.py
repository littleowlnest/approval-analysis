import pandas as pd, re
from pathlib import Path
INPUT_PRIMARY = Path('input') / 'submissions_with_ctos.csv'
INPUT_SECONDARY = Path('input') / 'submissions.csv'
FALLBACK = Path('output') / 'flattened_submissions.csv'
if INPUT_PRIMARY.exists():
    path = INPUT_PRIMARY
elif INPUT_SECONDARY.exists():
    path = INPUT_SECONDARY
else:
    path = FALLBACK

if not path.exists():
    raise SystemExit('No input CSV found (prefer input/submissions_with_ctos.csv; fallback input/submissions.csv or output/flattened_submissions.csv)')

df = pd.read_csv(path, low_memory=False, dtype=object)
# normalize cols
cols=[re.sub(r"\s+"," ",str(c).strip().lower()) for c in df.columns]
df.columns=cols
# flags
emp_col=None
for c in df.columns:
    if 'employ' in c:
        emp_col=c; break
status_col=None
for c in df.columns:
    if c in ('status','approval_label'): status_col=c; break
if emp_col:
    e=df[emp_col].fillna('').astype(str)
    df['is_self_employed']=e.str.contains(r"\bself\b", regex=True, case=False) | e.str.contains(r"self-?employ", regex=True, case=False)
else:
    df['is_self_employed']=False
if status_col:
    s=df[status_col].fillna('').astype(str)
    df['is_approved']=s.str.contains(r"\bapprove\w*\b", regex=True, case=False)
else:
    df['is_approved']=False
# profile cols
profile_cols=[c for c in ['age_band','income_band','education','employment type','occupation','purpose of finance','purpose'] if c in df.columns]
# fill na
for c in profile_cols:
    df[c]=df[c].fillna('Unknown').astype(str)
se=df[df['is_self_employed']==True].copy()
print('total_self_employed',len(se))
print('approved_self_employed',int(se['is_approved'].sum()))
if se.empty:
    raise SystemExit('no se')
agg=se.groupby(profile_cols, dropna=False).agg(applications=('is_approved','size'), approvals=('is_approved','sum')).reset_index()
agg['applications']=agg['applications'].astype(int)
agg_sorted=agg.sort_values('applications', ascending=False)
print('groups_total',len(agg_sorted))
print('groups_with_support>=3', int((agg_sorted['applications']>=3).sum()))
print('\nTop groups:')
print(agg_sorted.head(20).to_csv(index=False))

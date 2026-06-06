import pandas as pd
from pathlib import Path
import numpy as np
import re

# Prefer the richer submissions file when present (contains CTOs)
INPUT_CSV_PRIMARY = Path('input') / 'submissions_with_ctos.csv'
INPUT_CSV_SECONDARY = Path('input') / 'submissions.csv'
FALLBACK = Path('output') / 'flattened_submissions.csv'
if INPUT_CSV_PRIMARY.exists():
    INPUT_CSV = INPUT_CSV_PRIMARY
else:
    INPUT_CSV = INPUT_CSV_SECONDARY

OUT_MD = Path('output') / 'profile_answers.md'
OUT_CSV = Path('output') / 'profile_answers.csv'
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

# Read source CSV
if INPUT_CSV.exists():
    df = pd.read_csv(INPUT_CSV, low_memory=False, dtype=object)
elif FALLBACK.exists():
    df = pd.read_csv(FALLBACK, low_memory=False, dtype=object)
else:
    raise SystemExit('No input CSV found (prefer input/submissions_with_ctos.csv; fallback input/submissions.csv or output/flattened_submissions.csv)')

# normalize column names
orig_columns = list(df.columns)
col_map = {c: re.sub(r"\s+", " ", str(c).strip().lower()) for c in orig_columns}
df = df.rename(columns=col_map)

# helper to pick column
def pick(*cands):
    for c in cands:
        if c in df.columns:
            return c
    return None

status_col = pick('status','approval_label')
approved_amount_col = pick('approved amount','approved_amount_numeric','approved_amount')
employment_col = pick('employment type','employment')
name_col = pick('name')
age_col = pick('age')
age_band_col = pick('age_band','age band')
income_col = pick('gross income','income','monthly income')
income_band_col = pick('income_band','income band')
edu_col = pick('education')
occupation_col = pick('occupation')
purpose_col = pick('purpose of finance','purpose')

# ensure flags
if 'is_approved' not in df.columns:
    if status_col:
        s = df[status_col].fillna('').astype(str).str.lower()
        df['is_approved'] = s.str.contains('approved')
        df['is_rejected'] = s.str.contains('reject') | s.str.contains('declin')
    else:
        df['is_approved'] = False
        df['is_rejected'] = False

if 'is_self_employed' not in df.columns:
    if employment_col:
        e = df[employment_col].fillna('').astype(str).str.lower()
        df['is_self_employed'] = e.str.contains('self employ') | e.str.contains('self-employed')
    else:
        df['is_self_employed'] = False

    # coerce flag columns to boolean in case they are strings in the CSV
    def _to_bool_series(s):
        return s.astype(str).str.lower().isin(['true','1','yes','approved'])

    df['is_approved'] = _to_bool_series(df['is_approved'])
    df['is_rejected'] = _to_bool_series(df['is_rejected'])
    df['is_self_employed'] = _to_bool_series(df['is_self_employed'])

# parse approved amount numeric
if approved_amount_col and approved_amount_col in df.columns:
    s = df[approved_amount_col].astype(str).str.replace(r"[RM,\s]", "", regex=True).str.replace(r"[^0-9.\-]","",regex=True)
    df['approved_amount_numeric'] = pd.to_numeric(s, errors='coerce')
else:
    df['approved_amount_numeric'] = pd.NA

# create age_band and income_band if not present
if age_band_col is None and age_col in df.columns:
    df['age_numeric'] = pd.to_numeric(df[age_col], errors='coerce')
    bins = [0,25,30,40,50,999]
    labels = ['18-25','26-30','31-40','41-50','51+']
    df['age_band'] = pd.cut(df['age_numeric'], bins=bins, labels=labels, include_lowest=True)

if income_band_col is None and income_col in df.columns:
    s = df[income_col].astype(str).str.replace(r"[RM,\s]", "", regex=True)
    df['gross_income_numeric'] = pd.to_numeric(s, errors='coerce')
    bins = [0,3000,5000,8000,12000,99999999]
    labels = ['RM0-3k','RM3k-5k','RM5k-8k','RM8k-12k','RM12k+']
    df['income_band'] = pd.cut(df['gross_income_numeric'], bins=bins, labels=labels, include_lowest=True)

profile_cols = []
for c in ['age_band','income_band','education','employment type','occupation','purpose of finance','purpose']:
    if c in df.columns:
        profile_cols.append(c)

if not profile_cols:
    raise SystemExit('No profile columns found to group by')

# normalize NaN and empty
for c in profile_cols:
    # convert categorical to object first to avoid category assignment errors
    df[c] = df[c].astype(object)
    df[c] = df[c].fillna('Unknown').astype(str)

# define name_profile
EDU_SHORT = {'secondary':'SPM','diploma':'Diploma','degree':'Degree','bachelor':'Degree','master':'Master\'s','phd':'PhD'}
EMP_SHORT = {'private employee':'Private','government employee':'Gov\'t','self employ':'Self-Employed','self-employed':'Self-Employed'}
PURPOSE_SHORT = {'home improvement':'Home Improvement','education':'Education Financing','personal':'Personal','business':'Business','vehicle':'Vehicle','medical':'Medical'}

def short(mapping, v):
    if pd.isna(v):
        return ''
    return mapping.get(str(v).strip().lower(), str(v).strip())

def name_profile_row(row):
    parts = []
    if 'age_band' in row and row['age_band'] and row['age_band']!='nan':
        parts.append(f"Age {row['age_band']}")
    if 'income_band' in row and row['income_band'] and row['income_band']!='nan':
        parts.append(f"Income {row['income_band']}")
    edu = short(EDU_SHORT, row.get('education',''))
    if edu:
        parts.append(edu)
    emp = short(EMP_SHORT, row.get('employment type', ''))
    if emp:
        parts.append(emp)
    occ = row.get('occupation','')
    if occ and occ.lower() not in ('unknown','nan','none'):
        parts.append(occ[:40])
    purpose = short(PURPOSE_SHORT, row.get('purpose of finance', row.get('purpose','')))
    if purpose:
        parts.append(purpose)
    return ' | '.join(parts) if parts else 'Unknown Profile'

# group and aggregate
agg = df.groupby(profile_cols, dropna=False).agg(
    applications=('is_approved','size'),
    approvals=('is_approved','sum'),
    avg_approved_amount=('approved_amount_numeric', 'mean')
).reset_index()
# ensure numeric types
agg['applications'] = pd.to_numeric(agg['applications'], errors='coerce').fillna(0).astype(int)
agg['approvals'] = pd.to_numeric(agg['approvals'], errors='coerce').fillna(0).astype(int)
agg['approval_rate'] = agg['approvals'] / agg['applications'].replace({0:1})
agg['rejections'] = agg['applications'] - agg['approvals']

# filter by min support
MIN_SUPPORT = 3
agg_filtered = agg[agg['applications'] >= MIN_SUPPORT].copy()

# add profile_name
agg_filtered['profile_name'] = agg_filtered.apply(name_profile_row, axis=1)

# helpers to pick top
def pick_top(df_in, condition=None, top=3, sort_by=('approval_rate','applications','approvals')):
    d = df_in.copy()
    if condition is not None:
        d = d.query(condition)
    if d.empty:
        return pd.DataFrame()
    d = d.sort_values(list(sort_by), ascending=[False, False, False])
    return d.head(top)

# Q1 most likely to get approved
q1 = pick_top(agg_filtered, None, top=1)
# Q2 top 3
q2 = pick_top(agg_filtered, None, top=3)
# Q3 approval above 30k
q3 = pick_top(agg_filtered, 'avg_approved_amount >= 30000', top=1)
# Q4 approval below 30k
q4 = pick_top(agg_filtered, 'avg_approved_amount < 30000', top=3)
# Q5 self-employed profiles
se_df = df[df['is_self_employed'] == True].copy()
if not se_df.empty:
    se_agg_all = se_df.groupby(profile_cols, dropna=False).agg(
        applications=('is_approved','size'),
        approvals=('is_approved','sum'),
        avg_approved_amount=('approved_amount_numeric','mean')
    ).reset_index()
    # ensure numeric types
    se_agg_all['applications'] = pd.to_numeric(se_agg_all['applications'], errors='coerce').fillna(0).astype(int)
    se_agg_all['approvals'] = pd.to_numeric(se_agg_all['approvals'], errors='coerce').fillna(0).astype(int)
    se_agg_all['approval_rate'] = se_agg_all['approvals'] / se_agg_all['applications'].replace({0:1})
    se_agg_all['rejections'] = se_agg_all['applications'] - se_agg_all['approvals']

    # apply MIN_SUPPORT; if that yields nothing, fall back to top groups without the support filter
    se_agg = se_agg_all[se_agg_all['applications'] >= MIN_SUPPORT].copy()
    if se_agg.empty:
        se_agg = se_agg_all.sort_values(['approval_rate','applications','approvals'], ascending=[False, False, False]).head(5).copy()

    se_agg['profile_name'] = se_agg.apply(name_profile_row, axis=1)
    q5 = pick_top(se_agg, None, top=1)
else:
    q5 = pd.DataFrame()

# Q6 ideal profile for RM67k
t6 = agg_filtered.copy()
# distance measure
t6['distance_to_67000'] = (t6['avg_approved_amount'].fillna(np.inf) - 67000).abs()
t6 = t6.sort_values(by=['approval_rate','distance_to_67000','applications'], ascending=[False, True, False])
q6 = t6.head(3)

# Q7 most rejection
q7 = agg_filtered.sort_values(by=['rejections','applications'], ascending=[False, False]).head(3)

# write outputs
with OUT_MD.open('w', encoding='utf-8') as f:
    f.write('# Profile Answers\n\n')
    f.write(f'Rows used: {len(df)}\n\n')
    def write_section(title, dfw):
        f.write(f'## {title}\n')
        if dfw is None or dfw.empty:
            f.write('No results\n\n')
            return
        for i, row in dfw.iterrows():
            f.write(f"- {row['profile_name']} — applications: {int(row['applications'])}, approvals: {int(row['approvals'])}, approval_rate: {row['approval_rate']:.1%}, avg_amount: {row.get('avg_approved_amount',pd.NA)}\n")
        f.write('\n')

    write_section('1) Most likely to get approved', q1)
    write_section('2) Top 3 client profiles with the highest approval', q2)
    write_section('3) Most likely to get approval above RM30,000', q3)
    write_section('4) Most likely to get approval below RM30,000', q4)
    write_section("5) Self-employed client profile most likely to get approved", q5)
    write_section('6) Ideal client profile for approvals at RM67,000', q6)
    write_section('7) Client profile with the most rejection', q7)

# Save summary CSV
agg_filtered.sort_values(['approval_rate','applications'], ascending=[False,False]).to_csv(OUT_CSV, index=False)
print('Wrote', OUT_MD, 'and', OUT_CSV)

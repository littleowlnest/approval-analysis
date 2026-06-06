import pandas as pd
from pathlib import Path

OUT = Path('output')
OUT.mkdir(exist_ok=True)
CSV = Path('output/scored_records.csv')
if not CSV.exists():
    raise SystemExit('Missing output/scored_records.csv')

pd.options.display.float_format = '{:.3f}'.format

df = pd.read_csv(CSV, low_memory=False)
cols = list(df.columns)
print('Columns found:', cols)

# helper to pick a column from candidates
def pick(cands):
    for c in cands:
        if c in df.columns:
            return c
    return None

col_age_band = pick(['age_band','age band','age band '])
col_income_band = pick(['income_band','income band','income_band '])
col_occupation = pick(['occupation','occupaction','occupaction','occupaction '])
col_approval = pick(['approval_label','approved','status','approval_label '])
col_approved_amount = pick(['approved_amount_numeric','approved amount','approved_amount_numeric '])

print('Mapped columns: age_band=%s, income_band=%s, occupation=%s, approval=%s, approved_amount=%s' % (
    col_age_band, col_income_band, col_occupation, col_approval, col_approved_amount))

# normalize and create boolean approved column
if col_approval:
    df['approved_flag'] = pd.to_numeric(df[col_approval], errors='coerce')
    df['approved_flag'] = df['approved_flag'].fillna(0).astype(int)
else:
    # fallback: try status text
    df['approved_flag'] = df.get('status','').str.lower().eq('approved').astype(int)

# fill missing categorical columns
for c in [col_age_band, col_income_band, col_occupation]:
    if c:
        df[c] = df[c].fillna('Unknown').astype(str)

# Full profile columns as used in report
profile_cols = []
for name in ['age','age_band','education','employment sector','employment type','income_band','occupation','purpose of finance']:
    # try to find column with similar name
    if name in df.columns:
        profile_cols.append(name)
    else:
        # try a few common variants
        alt = name.replace(' ','_')
        if alt in df.columns:
            profile_cols.append(alt)
        else:
            # try shorter tokens
            tokens = name.split()
            for t in tokens:
                if t in df.columns and t not in profile_cols:
                    profile_cols.append(t)

print('Profile columns detected for full grouping:', profile_cols)

# compute full-profile groups
fg = df.copy()
for c in profile_cols:
    fg[c] = fg[c].fillna('Unknown').astype(str)

full_groups = fg.groupby(profile_cols, dropna=False).agg(
    applications=('approved_flag','size'),
    approvals=('approved_flag','sum'),
    approval_rate=('approved_flag',lambda x: x.sum()/max(1,x.size)),
)
full_groups = full_groups.sort_values(['applications','approval_rate'], ascending=[False, False])
full_groups.to_csv(OUT / 'full_profile_groups.csv')
print('\nFull-profile groups: total groups=%d, median group size=%.2f, groups with <5 apps=%d' % (
    len(full_groups), full_groups['applications'].median(), (full_groups['applications']<5).sum()))

# Broader single-column stats
single_cols = [col_age_band, col_income_band, col_occupation]
single_cols = [c for c in single_cols if c]
reports = {}
for c in single_cols:
    g = df.groupby(c, dropna=False).agg(
        applications=('approved_flag','size'),
        approvals=('approved_flag','sum'),
        approval_rate=('approved_flag',lambda x: x.sum()/max(1,x.size)),
    ).sort_values('applications', ascending=False)
    g.to_csv(OUT / f'agg_by_{c}.csv')
    reports[c] = g
    print(f'\nTop groups for {c}:')
    print(g.head(10))

# Pairwise combos (top by applications)
from itertools import combinations
for a,b in combinations(single_cols,2):
    g = df.groupby([a,b], dropna=False).agg(applications=('approved_flag','size'), approvals=('approved_flag','sum'))
    g = g[g['applications']>=5].sort_values('applications', ascending=False).reset_index()
    if not g.empty:
        g.to_csv(OUT / f'agg_by_{a}__{b}.csv')
        print(f'\nSaved pairwise agg for {a} and {b}, groups>=5: {len(g)} rows')

# Occupation-level top stats
if col_occupation:
    occ = df.groupby(col_occupation).agg(applications=('approved_flag','size'), approvals=('approved_flag','sum'), approval_rate=('approved_flag',lambda x: x.sum()/max(1,x.size))).sort_values('applications', ascending=False)
    occ.to_csv(OUT / 'agg_by_occupation.csv')
    print('\nTop occupations:')
    print(occ.head(20))

print('\nWrote CSVs to output/. Files:')
for p in sorted(OUT.iterdir()):
    print('-', p.name)

print('\nDone')

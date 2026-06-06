"""Diagnostic: self-employed groupings

Usage:
    python scripts/diag_self_employed.py --input input/submissions_with_ctos.csv

Produces console output showing counts and top self-employed groupings.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import pandas as pd

sys.path.insert(0, "src")
from approval_analysis.pipeline import _make_bands_for_rebuild, _flag_to_binary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="input/submissions_with_ctos.csv")
    args = ap.parse_args()

    p = Path(args.input)
    if not p.exists():
        print("missing input file", p)
        return 1

    df = pd.read_csv(p, low_memory=False)
    # normalize columns to lowercase keys as pipeline does
    df.columns = [str(c).strip().lower() for c in df.columns]
    # apply rebuild bands
    rdf = _make_bands_for_rebuild(df)
    # detect self-employed rows
    if "is_self_employed" in rdf.columns:
        se_mask = rdf["is_self_employed"].apply(_flag_to_binary) == 1.0
        se = rdf[se_mask].copy()
    else:
        mask = pd.Series(False, index=rdf.index)
        for c in ("employment type", "occupation", "employment sector"):
            if c in rdf.columns:
                mask = mask | rdf[c].astype(str).str.lower().str.contains("self")
        se = rdf[mask].copy()

    print("total rows:", len(rdf))
    print("self-employed rows:", len(se))
    # show top groupings by age, education, purpose/emp type
    cols = [
        c
        for c in (
            "age",
            "age_band",
            "education",
            "employment type",
            "purpose of finance",
            "occupation",
        )
        if c in se.columns
    ]
    print("se grouping cols:", cols)
    if len(se) > 0:
        g = (
            se.groupby(cols, dropna=False)
            .agg(
                applications=("approval_label", "size"),
                approvals=("approval_label", lambda s: s.dropna().astype(int).sum()),
                avg_approved_amount=("approved_amount_numeric", "mean"),
            )
            .reset_index()
        )
        g["approval_rate"] = g["approvals"] / g["applications"].replace({0: 1})
        g = g.sort_values(["applications", "approval_rate"], ascending=[False, False])
        print("\nTop self-employed groups (head 20):")
        print(g.head(20).to_string(index=False))
        # show any with applications>=3
        print("\nGroups with applications>=3:")
        print(g[g["applications"] >= 3].to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

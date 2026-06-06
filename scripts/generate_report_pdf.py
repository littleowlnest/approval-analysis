"""Generate a formatted PDF from analysis outputs.

Usage:
    python scripts/generate_report_pdf.py --output-dir output

This script uses reportlab to build a PDF containing:
- Title and generation timestamp
- Main report narrative (from analysis_report.md)
- Tables for `profile_summary.csv`, `profile_summary_rebuild.csv`, and `pairwise_stats.csv` (top rows)

If `reportlab` isn't installed, install it in your virtualenv: `pip install reportlab`.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import textwrap

import pandas as pd

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
except Exception as exc:
    raise SystemExit("reportlab is required. Install with 'pip install reportlab' and retry.\n" + str(exc))


def read_markdown(path: Path) -> list[str]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    # Collapse long lines for PDF readability
    wrapped = []
    for line in lines:
        if line.strip().startswith("#") or line.strip().startswith("- "):
            wrapped.append(line)
        else:
            for chunk in textwrap.wrap(line, width=120):
                wrapped.append(chunk)
    return wrapped


def df_to_table_data(df: pd.DataFrame, max_rows: int | None = 15) -> list[list[str]]:
    if df is None:
        return []
    if max_rows is None:
        df2 = df.copy()
    else:
        df2 = df.head(max_rows).copy()
    # replace None/NaN with 'Unknown'
    df2 = df2.replace({None: "Unknown", float('nan'): "Unknown"}).where(pd.notna(df2), "Unknown")
    cols = list(df2.columns)
    rows = [cols]
    for _, r in df2.iterrows():
        row = []
        for c in cols:
            v = r[c]
            s = "" if pd.isna(v) else str(v)
            if len(s) > 200:
                s = s[:197] + "..."
            row.append(s)
        rows.append(row)
    return rows


def add_table(flow, df: pd.DataFrame, title: str):
    # Keep the heading and the table together to avoid the header being separated
    block = []
    # Use Titillium headings if registered, otherwise default Heading3
    gs = getSampleStyleSheet()
    body_font = "TitilliumWeb" if "TitilliumWeb" in pdfmetrics.getRegisteredFontNames() else "Helvetica"
    bold_font = "TitilliumWeb-Bold" if "TitilliumWeb-Bold" in pdfmetrics.getRegisteredFontNames() else "Helvetica-Bold"
    heading_style = ParagraphStyle("tbl_heading", parent=gs.get("Heading3"), fontName=bold_font)
    block.append(Paragraph(title, heading_style))
    block.append(Spacer(1, 4))
    if df is None or df.empty:
        block.append(Paragraph("No data.", getSampleStyleSheet()["BodyText"]))
        block.append(Spacer(1, 8))
        flow.append(KeepTogether(block))
        return
    # If the table is very wide, trim to a reasonable set of columns to avoid extreme wrapping
    preferred = [
        "profile",
        "profile_rank",
        "applications",
        "approvals",
        "approval_rate",
        "avg_approved_amount",
        "avg_approved_amount_RM",
        "ctos_band",
        "ctos_score",
        "name",
        "status",
        "predicted_approval_probability",
        "predicted_approved_amount",
    ]

    df_to_render = df.copy()
    if df_to_render.shape[1] > 12:
        cols = [c for c in preferred if c in df_to_render.columns]
        if not cols:
            cols = list(df_to_render.columns[:12])
        df_to_render = df_to_render.loc[:, cols]

    # limit rows to avoid extremely tall single tables
    max_rows_allowed = 50
    data_rows = df_to_table_data(df_to_render, max_rows=max_rows_allowed)
    if not data_rows:
        flow.append(Paragraph("No data.", getSampleStyleSheet()["BodyText"]))
        flow.append(Spacer(1, 8))
        return
    # build Paragraph cells to enable wrapping
    styles = getSampleStyleSheet()
    td_style = ParagraphStyle("td", parent=styles.get("BodyText"), fontName=body_font, fontSize=8, leading=10)
    hdr_style = ParagraphStyle("th", parent=styles.get("BodyText"), fontName=bold_font, fontSize=9, leading=11)
    data = []
    for i, row in enumerate(data_rows):
        new_row = []
        for j, cell in enumerate(row):
            text = cell if cell not in (None, "") else "Unknown"
            if i == 0:
                new_row.append(Paragraph(str(text), hdr_style))
            else:
                new_row.append(Paragraph(str(text), td_style))
        data.append(new_row)
    # compute column widths to fit page
    page_width = A4[0]
    margin = 20 * mm
    avail_width = page_width - margin * 2
    ncols = len(data[0])
    # If there's a `profile` column, make it double width and shrink others proportionally
    try:
        headers = [str(x).strip().lower() for x in data_rows[0]] if data_rows and data_rows[0] else []
    except Exception:
        headers = []
    if "profile" in headers:
        profile_idx = headers.index("profile")
        # allocate units: each normal column = 1 unit, profile = 2 units
        total_units = ncols + 1
        unit = avail_width / total_units
        col_widths = [unit * 2 if i == profile_idx else unit for i in range(ncols)]
    else:
        col_width = avail_width / max(1, ncols)
        col_widths = [col_width] * ncols
    tbl = Table(data, colWidths=col_widths, repeatRows=1, hAlign="LEFT")
    style = TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dce6f1")),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ])
    # Right-align numeric columns for better readability
    try:
        # prefer dtype-based detection
        numeric_cols = list(df_to_render.select_dtypes(include=["number"]).columns)
    except Exception:
        numeric_cols = []
    # fallback: common numeric name hints
    hints = ("applications", "approvals", "rejections", "amount", "avg", "score", "rank", "probability", "ctos_score")
    for i, col in enumerate(df_to_render.columns):
        name = str(col).lower()
        if col in numeric_cols or any(h in name for h in hints):
            # align header+cells to right for this column
            style.add("ALIGN", (i, 0), (i, -1), "RIGHT")
    tbl.setStyle(style)
    block.append(tbl)
    block.append(Spacer(1, 8))
    flow.append(KeepTogether(block))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="output")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    md_path = out / "analysis_report.md"

    # Prefer structured JSON if available (do NOT parse the markdown when JSON exists)
    results_json = out / "analysis_results.json"
    results = None
    if results_json.exists():
        try:
            import json

            results = json.loads(results_json.read_text(encoding="utf-8"))
        except Exception:
            results = None
    else:
        # only read the markdown narrative if JSON is not available
        narrative = read_markdown(md_path)

    # load key tables if present
    def load_csv(name):
        p = out / name
        if p.exists():
            try:
                return pd.read_csv(p)
            except Exception:
                return None
        return None

    profile_summary = load_csv("profile_summary.csv")
    profile_rebuild = load_csv("profile_summary_rebuild.csv")
    pairwise = load_csv("pairwise_stats.csv")
    ctos = None
    # try to extract CTOS table from pairwise or rebuild if present
    for candidate in ("profile_summary_rebuild.csv", "pairwise_stats.csv"):
        p = out / candidate
        if p.exists():
            try:
                dfc = pd.read_csv(p)
                if "ctos_band" in dfc.columns:
                    ctos = dfc[[c for c in dfc.columns if c.startswith("ctos")] + ["applications", "approvals", "approval_rate"]].head(20)
                    break
            except Exception:
                pass

    pdf_path = out / "analysis_report.pdf"
    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4, rightMargin=20*mm, leftMargin=20*mm, topMargin=20*mm, bottomMargin=20*mm)
    flow = []

    styles = getSampleStyleSheet()
    # Register custom Titillium fonts from project `src/fonts`
    try:
        fonts_dir = Path(__file__).resolve().parent.parent / "src" / "fonts"
        regular_path = fonts_dir / "TitilliumWeb-Regular.ttf"
        bold_path = fonts_dir / "TitilliumWeb-Bold.ttf"
        # register if files exist
        if regular_path.exists():
            pdfmetrics.registerFont(TTFont("TitilliumWeb", str(regular_path)))
        if bold_path.exists():
            pdfmetrics.registerFont(TTFont("TitilliumWeb-Bold", str(bold_path)))
    except Exception:
        # If font registration fails, fall back to built-ins
        pass

    # Build styles using Titillium if available and update the sample stylesheet headings
    body_font = "TitilliumWeb" if "TitilliumWeb" in pdfmetrics.getRegisteredFontNames() else styles.get("BodyText").fontName
    bold_font = "TitilliumWeb-Bold" if "TitilliumWeb-Bold" in pdfmetrics.getRegisteredFontNames() else styles.get("Title").fontName
    # override Heading styles to use bold Titillium by mutating existing styles
    for h in ("Heading1", "Heading2", "Heading3", "Heading4"):
        parent = styles.get(h)
        if parent is not None:
            parent.fontName = bold_font
            # keep existing size/leading
    # set BodyText to use Titillium regular by mutating existing style
    bt = styles.get("BodyText")
    if bt is not None:
        bt.fontName = body_font

    title_style = ParagraphStyle("Title", parent=styles.get("Title"), fontName=bold_font, fontSize=18, leading=22)
    centered_style = ParagraphStyle("Centered", parent=styles.get("BodyText"), fontName=body_font, fontSize=10, leading=12, alignment=1)
    normal = ParagraphStyle("Body", parent=styles.get("BodyText"), fontName=body_font, fontSize=10, leading=12)
    normal.spaceAfter = 6

    flow.append(Paragraph("Approval Analysis Report", title_style))
    flow.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", centered_style))
    flow.append(Spacer(1, 6))

    # If structured JSON is available, build PDF sections from it (no MD parsing)
    """
    if results is not None:
        flow.append(Paragraph(f"Rows analyzed: {results.get('rows_analyzed')}", normal))
        pcs = results.get("profile_columns") or []
        if pcs:
            flow.append(Paragraph(f"Profile columns: {', '.join(pcs)}", normal))
        ms = results.get("model_summary") or {}
        if ms:
            flow.append(Paragraph(f"Model summary: {ms}", normal))
        flow.append(Spacer(1, 8))
    else:
        # Narrative: include first ~400 lines from markdown as a fallback
        if narrative:
            flow.append(Paragraph("Report (excerpt)", styles["Heading2"]))
            excerpt = "\n".join(narrative[:400])
            for line in excerpt.splitlines():
                if line.startswith("#"):
                    level = line.count("#")
                    text = line.lstrip('#').strip()
                    style = styles.get("Heading%d" % min(4, max(1, level)), styles["Heading4"]) if True else styles["Heading4"]
                    flow.append(Paragraph(text, style))
                elif line.startswith("- "):
                    flow.append(Paragraph(line, normal))
                else:
                    flow.append(Paragraph(line, normal))
            flow.append(Spacer(1, 8))
    """
    # Tables
    # Insert a dedicated Profile Answers page from structured JSON if available (do not use MD)
    if results is not None and results.get("answers"):
        #flow.append(Paragraph("Profile Answers", styles["Heading2"]))
        #flow.append(Spacer(1, 6))
        answers = results.get("answers", {})
        # For each answer category, render a heading and a table if rows exist
        for key in ("most_likely_approved", "top_3_highest_approval", "likely_approved_above_30k", "likely_approved_below_30k", "self_employed_best", "self_employed_stats", "pairwise_self_employed", "ideal_for_67000", "most_rejected"):
            val = answers.get(key)
            if not val:
                continue
            title = key.replace("_", " ").title()
            try:
                df_ans = pd.DataFrame(val)
            except Exception:
                df_ans = None

            # If the answer is tabular, ensure there is a human-readable `profile` column
            if df_ans is not None and not df_ans.empty:
                # Special-case: pairwise records for self-employed are a list of pair blocks
                if key == "pairwise_self_employed":
                    try:
                        if isinstance(val, list):
                            for pair_rec in val:
                                pair_label = pair_rec.get("pair") or "Pair"
                                rows = pair_rec.get("rows") or []
                                df_pair = pd.DataFrame(rows) if rows else None
                                if df_pair is not None and not df_pair.empty:
                                    if "approval_rate" in df_pair.columns:
                                        df_pair["approval_rate"] = df_pair["approval_rate"].apply(lambda v: f"{float(v)*100:.2f}%" if pd.notna(v) else "Unknown")
                                    add_table(flow, df_pair, f"Pairwise (Self-employed): {pair_label}")
                                    flow.append(Spacer(1, 6))
                            continue
                    except Exception:
                        # fall through to the regular handling on error
                        pass
                if "profile" not in df_ans.columns:
                    # Use a shorter set of profile properties for the self_employed_best section
                    if key == "self_employed_best":
                        profile_fields = [
                            "age_band",
                            "income_band",
                            "ctos_band",
                            "occupation",
                            "employment type",
                            "purpose of finance",
                        ]
                    else:
                        profile_fields = [
                            "age_band",
                            "income_band",
                            "ctos_band",
                            "occupation",
                            "education",
                            "employment sector",
                            "employment type",
                            "purpose of finance",
                        ]
                    present = [c for c in profile_fields if c in df_ans.columns]
                    if present:
                        def _make_profile(r):
                            parts = []
                            for pc in present:
                                v = r.get(pc)
                                try:
                                    if pd.isna(v):
                                        continue
                                except Exception:
                                    pass
                                s = str(v).strip()
                                if not s or s.lower() in ("nan", "none"):
                                    continue
                                parts.append(f"{pc}={s}")
                            return " | ".join(parts) if parts else ""

                        df_ans["profile"] = df_ans.apply(_make_profile, axis=1)
                        # move profile to the front for display
                        cols = ["profile"] + [c for c in df_ans.columns if c != "profile"]
                        df_ans = df_ans.loc[:, cols]

                # format columns: approval_rate -> percentage, avg_approved_amount -> whole number
                if "approval_rate" in df_ans.columns:
                    df_ans["approval_rate"] = df_ans["approval_rate"].apply(lambda v: f"{float(v)*100:.2f}%" if pd.notna(v) else "Unknown")
                if "avg_approved_amount" in df_ans.columns:
                    df_ans["avg_approved_amount"] = df_ans["avg_approved_amount"].apply(lambda v: f"{int(round(v)):,}" if pd.notna(v) else "Unknown")

                # render table with title (avoid double-heading)
                add_table(flow, df_ans, title)
                flow.append(Spacer(1, 8))
                continue

            # At this point df_ans is None or empty
            # maybe it's a stats dict (e.g., self_employed_stats)
            if isinstance(val, dict):
                # render self_employed_stats as a one-row table for clarity
                if key == "self_employed_stats":
                    try:
                        df_stats = pd.DataFrame([val])
                        # format approval_rate (fraction) as percentage
                        if "approval_rate" in df_stats.columns:
                            df_stats["approval_rate"] = df_stats["approval_rate"].apply(lambda v: f"{float(v)*100:.2f}%" if pd.notna(v) else "Unknown")
                        if "self_percent" in df_stats.columns:
                            df_stats["self_percent"] = df_stats["self_percent"].apply(lambda v: f"{float(v):.2f}%" if pd.notna(v) else "Unknown")
                        add_table(flow, df_stats, "Self-employed Stats")
                        flow.append(Spacer(1, 6))
                        continue
                    except Exception:
                        # fallback to key:value lines on error
                        pass

                # Fallback: render heading then key:value lines
                flow.append(Paragraph(title, styles["Heading3"]))
                flow.append(Spacer(1, 4))
                for k, v in val.items():
                    flow.append(Paragraph(f"{k}: {v}", styles["BodyText"]))
                flow.append(Spacer(1, 6))
                continue

            # If non-dict and no table, show heading with 'No data.'
            flow.append(Paragraph(title, styles["Heading3"]))
            flow.append(Spacer(1, 4))
            flow.append(Paragraph("No data.", getSampleStyleSheet()["BodyText"]))
            flow.append(Spacer(1, 6))
            continue
        flow.append(PageBreak())

    if results is not None:
        # Render from structured JSON
        def df_from_list(key):
            v = results.get(key)
            if not v:
                return None
            try:
                return pd.DataFrame(v)
            except Exception:
                return None

        add_table(flow, df_from_list("broader_profiles"), "Broader Profiles")
        add_table(flow, df_from_list("ctos_summary"), "CTOS Summary")
        # attribute breakdowns
        attr = results.get("attribute_breakdowns", {}) or {}
        for k, rows in attr.items():
            add_table(flow, pd.DataFrame(rows) if rows else None, f"Attribute breakdown: {k}")

        # pairwise
        for p in results.get("pairwise", []) or []:
            rows = p.get("rows")
            add_table(flow, pd.DataFrame(rows) if rows else None, f"Pairwise: {p.get('pair')}")

        # answers / combo profiles
        add_table(flow, df_from_list("combo_profiles"), "Combo Profiles (top groups)")
        #add_table(flow, df_from_list("profile_summary"), "Profile Summary (full)")
        # scored rows (trimmed)
        #add_table(flow, df_from_list("scored_rows"), "Scored Rows (excerpt)")
    else:
        add_table(flow, profile_summary, "Profile Summary (top rows)")
        add_table(flow, profile_rebuild, "Profile Summary (rebuild combos, top rows)")
        add_table(flow, pairwise, "Pairwise Stats (top rows)")
        if ctos is not None:
            add_table(flow, ctos, "CTOS Bands (top rows)")

    doc.build(flow)
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, mean_absolute_error, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# No openpyxl monkeypatching: rely on installed openpyxl package.


STATUS_ALIASES = {
    "approved": 1,
    "approve": 1,
    "yes": 1,
    "y": 1,
    "true": 1,
    "1": 1,
    "rejected": 0,
    "reject": 0,
    "declined": 0,
    "no": 0,
    "n": 0,
    "false": 0,
    "0": 0,
}

ID_CANDIDATES = {
    "no", "date", "date of applied", "name", "i/c", "ic", "contact",
    "e-mail address", "email address", "email address", "email",
    "status", "approved amount",
    "ctos & doc", "ctos",
    "commitment",
    "remarks", "staff", "agent",
    "source_sheet",
}

PROFILE_CANDIDATES = [
    # derived from IC
    "age",
    # confirmed header columns
    "gross income",
    "employment type",
    "education",
    "purpose of finance",
    "employment sector",
    "occupation",
    # generic aliases
    "income",
    "monthly income",
    "net income",
    "salary",
    "credit score",
    "cibil",
    "industry",
    "employment status",
    "self-employed",
    "self employed",
    "state",
    "marital status",
    "tenure",
    "loan purpose",
]

MONTH_NAME_PATTERN = re.compile(r"^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[-_ ]?\d{2}$", re.IGNORECASE)

# Set to a set of sheet names to restrict analysis, or None to include all month sheets.
# {"may-26","jun-26","jul-26"}.
ACTIVE_SHEETS: set[str] | None = None

AGE_BINS = [0, 25, 30, 40, 50, np.inf]
AGE_BIN_LABELS = ["18-25", "26-30", "31-40", "41-50", "51+"]

INCOME_BINS = [0, 3000, 5000, 8000, 12000, np.inf]
INCOME_BIN_LABELS = ["RM0-3k", "RM3k-5k", "RM5k-8k", "RM8k-12k", "RM12k+"]

MIN_PROFILE_SUPPORT = 3
MIN_CORE_APPLICATION_FIELDS = 3


@dataclass
class AnalysisResult:
    message: str
    report_path: Path | None = None


@dataclass
class ModelBundle:
    model: Any | None
    features: list[str]
    metrics: dict[str, Any]


def run_analysis(input_path: Path | None, input_dir: Path, output_dir: Path) -> AnalysisResult:
    workbook = resolve_input_file(input_path=input_path, input_dir=input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Accept either an Excel workbook (multiple month sheets) or a CSV (single flattened source)
    if workbook.suffix.lower() == ".csv":
        df = pd.read_csv(workbook, low_memory=False)
        # drop fully-empty rows and attach a source_sheet marker
        df = df.dropna(how="all")
        if not df.empty:
            df = df.copy()
            df["source_sheet"] = workbook.name
            sheets = [df]
        else:
            sheets = []
    else:
        sheets = load_workbook_sheets(workbook)
    if not sheets:
        raise ValueError(f"No usable sheets were found in {workbook.name}.")

    raw = pd.concat(sheets, ignore_index=True)
    normalized = normalize_columns(raw)
    enriched = derive_ic_features(normalized)
    enriched = drop_empty_application_rows(enriched)

    if "status" not in enriched.columns:
        raise ValueError("The workbook must contain a Status column or an equivalent label column.")

    if "approved amount" not in enriched.columns:
        raise ValueError("The workbook must contain an Approved Amount column for amount-band analysis.")

    enriched = enriched.copy()
    # Determine approval_label using explicit flag columns when present.
    # Priority: `is_approved` (only used to mark approvals), `is_rejected` (only used to mark rejections),
    # fallback to legacy `status` parsing for unlabeled rows.
    enriched["approval_label"] = np.nan
    if "is_approved" in enriched.columns:
        enriched["approval_label"] = enriched["is_approved"].apply(_flag_to_binary)
    if "is_rejected" in enriched.columns:
        rej = enriched["is_rejected"].apply(_flag_to_binary)
        enriched.loc[rej == 1.0, "approval_label"] = 0.0
    if "status" in enriched.columns:
        still_na = enriched["approval_label"].isna()
        if still_na.any():
            enriched.loc[still_na, "approval_label"] = enriched.loc[still_na, "status"].map(normalize_status)

    enriched["approved_amount_numeric"] = parse_money_series(enriched["approved amount"])
    enriched = add_profile_bins(enriched)

    profile_columns = detect_profile_columns(enriched)
    if not profile_columns:
        raise ValueError(
            "No usable profile columns were found. Add fields such as Age, Income, Credit Score, Occupation, Education, or Industry."
        )

    summary = build_profile_summary(enriched, profile_columns)
    classifier_bundle = train_approval_model(enriched, profile_columns)
    regressor_bundle = train_amount_model(enriched, profile_columns)

    scored = score_records(enriched, profile_columns, classifier_bundle, regressor_bundle)
    broader = compute_broader_profiles(scored)
    report = build_report(enriched, summary, scored, profile_columns, classifier_bundle, regressor_bundle, broader)

    # produce rebuild-style profile answers and pairwise stats before writing JSON
    try:
        _build_profiles_and_answers(enriched, output_dir)
    except Exception:
        # don't fail the main pipeline for rebuild helper errors
        pass

    # write a structured JSON representation of key analysis results for downstream rendering
    try:
        write_structured_json(
            output_dir / "analysis_results.json",
            frame=enriched,
            summary=summary,
            scored=scored,
            profile_columns=profile_columns,
            classifier_bundle=classifier_bundle,
            regressor_bundle=regressor_bundle,
            broader_profiles=broader,
        )
    except Exception:
        # don't fail analysis if JSON write fails
        pass

    summary_path = output_dir / "profile_summary.csv"
    report_path = output_dir / "analysis_report.md"
    metrics_path = output_dir / "model_metrics.json"
    scored_path = output_dir / "scored_records.csv"

    summary.to_csv(summary_path, index=False)
    # If a standalone profile answers file exists, try to run generator and integrate it
    answers_path = output_dir / "profile_answers.md"
    try:
        # If the rebuild-style answers are generated by the integrated pipeline, prefer those
        rebuild_answers = output_dir / "profile_answers_rebuild.md"
        if not answers_path.exists() and Path("scripts/answer_profiles.py").exists() and not rebuild_answers.exists():
            import runpy

            try:
                runpy.run_path(str(Path("scripts/answer_profiles.py").resolve()), run_name="__main__")
            except Exception:
                # don't fail the main analysis if the helper script errors
                pass
        # Only append the older answers file if we did not produce rebuilt answers in this run
        if answers_path.exists() and not rebuild_answers.exists():
            appended = answers_path.read_text(encoding="utf-8")
            # insert the profile answers into the main report under the "## Answers" header if present
            if "## Answers" in report:
                before, sep, after = report.partition("## Answers")
                # place the appended answers immediately after the Answers header
                new_report = before + sep + "\n\n" + "## Profile Answers (from flattened source)\n\n" + appended + "\n\n" + after
            else:
                new_report = report + "\n\n" + "## Profile Answers (from flattened source)\n\n" + appended
            report_path.write_text(new_report, encoding="utf-8")
        else:
            report_path.write_text(report, encoding="utf-8")
    except Exception:
        # fallback: write the original report if anything goes wrong
        report_path.write_text(report, encoding="utf-8")
    write_metrics(metrics_path, classifier_bundle, regressor_bundle, profile_columns, enriched)
    scored.to_csv(scored_path, index=False)
    # write broader profiles CSV
    broader_path = output_dir / "top_broader_profiles.csv"
    broader.to_csv(broader_path, index=False)

    # also produce rebuild-style profile answers and pairwise stats
    try:
        _build_profiles_and_answers(enriched, output_dir)
    except Exception:
        # don't fail the main pipeline for rebuild helper errors
        pass

    return AnalysisResult(
        message=f"Analysis complete for {workbook.name} across {len(sheets)} sheet(s). Outputs written to {output_dir}.",
        report_path=report_path,
    )


def resolve_input_file(input_path: Path | None, input_dir: Path) -> Path:
    if input_path is not None:
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        return input_path

    # prefer the enriched CSV (with CTOS) if available, then the standard submissions.csv
    primary_csv = input_dir / "submissions_with_ctos.csv"
    if primary_csv.exists():
        return primary_csv

    # prefer a cleaned CSV if available
    preferred_csv = input_dir / "submissions.csv"
    if preferred_csv.exists():
        return preferred_csv

    # next prefer .xls files (legacy)
    preferred_xls = input_dir / "submissions.xls"
    if preferred_xls.exists():
        return preferred_xls

    candidates = sorted(
        [path for path in input_dir.glob("*.xls") if not path.name.startswith("~$")],
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]

    raise FileNotFoundError(
        f"No input files found in {input_dir} (prefer submissions_with_ctos.csv; fallback submissions.csv or .xls)"
    )


def load_workbook(path: Path) -> pd.DataFrame:
    return pd.read_excel(path)


def load_workbook_sheets(path: Path) -> list[pd.DataFrame]:
    workbook = pd.ExcelFile(path)
    sheets: list[pd.DataFrame] = []

    for sheet_name in workbook.sheet_names:
        normalised_name = sheet_name.strip().lower()

        if not is_expected_month_sheet_name(sheet_name):
            continue

        if ACTIVE_SHEETS is not None and normalised_name not in ACTIVE_SHEETS:
            continue

        sheet = pd.read_excel(workbook, sheet_name=sheet_name)
        if sheet.empty:
            break

        data_rows = sheet.dropna(how="all")
        if data_rows.empty:
            break

        cleaned = data_rows.copy()
        cleaned["source_sheet"] = sheet_name
        sheets.append(cleaned)

    if not sheets:
        fallback = pd.read_excel(path)
        if not fallback.empty:
            fallback = fallback.dropna(how="all")
            if not fallback.empty:
                fallback["source_sheet"] = workbook.sheet_names[0] if workbook.sheet_names else "Sheet1"
                sheets.append(fallback)

    return sheets


def is_expected_month_sheet_name(sheet_name: str) -> bool:
    normalized = str(sheet_name).strip()
    return bool(MONTH_NAME_PATTERN.match(normalized))


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = {}
    for column in frame.columns:
        key = re.sub(r"\s+", " ", str(column).strip().lower())
        renamed[column] = key
    return frame.rename(columns=renamed)


def derive_ic_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    ic_column = None
    for candidate in ("i/c", "ic"):
        if candidate in result.columns:
            ic_column = candidate
            break

    if ic_column is None:
        return result

    ic_values = (
        result[ic_column]
        .fillna("")
        .astype(str)
        .str.replace(r"\D", "", regex=True)
        .fillna("")
    )
    birth_dates = []
    today = datetime.today()

    for raw in ic_values:
        # pandas 2.x can still return pd.NA through str accessors; normalise defensively
        value = "" if (raw is None or (isinstance(raw, float) and pd.isna(raw))) else str(raw)
        if len(value) >= 12:
            try:
                yy = int(value[0:2])
                mm = int(value[2:4])
                dd = int(value[4:6])
                if not (1 <= mm <= 12 and 1 <= dd <= 31):
                    raise ValueError("out-of-range date components")
                century = 2000 if yy <= int(str(today.year)[2:]) else 1900
                year = century + yy
                birth_dates.append(pd.Timestamp(year=year, month=mm, day=dd))
            except (ValueError, OverflowError):
                birth_dates.append(pd.NaT)
        else:
            birth_dates.append(pd.NaT)

    if "date_of_birth" not in result.columns:
        result["date_of_birth"] = birth_dates
    if "age" not in result.columns:
        result["age"] = [
            today.year - date.year - ((today.month, today.day) < (date.month, date.day))
            if pd.notna(date)
            else np.nan
            for date in birth_dates
        ]
    return result


def normalize_status(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value in {0, 1}:
            return float(value)
    key = str(value).strip().lower()
    if key in STATUS_ALIASES:
        return float(STATUS_ALIASES[key])
    if any(token in key for token in ("approve", "success", "pass", "ok")):
        return 1.0
    if any(token in key for token in ("reject", "declin", "fail", "deny")):
        return 0.0
    return np.nan


def _flag_to_binary(value: Any) -> float:
    """Normalize boolean-like flag fields (is_approved, is_rejected, is_self_employed).

    Returns 1.0 for truthy/affirmative values, 0.0 for explicit negatives, or np.nan for unknown.
    """
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value in {0, 1}:
            return float(value)
    key = str(value).strip().lower()
    if key in {"1", "true", "yes", "y", "t"}:
        return 1.0
    if key in {"0", "false", "no", "n", "f"}:
        return 0.0
    return np.nan


def parse_money_series(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace(r"[RM,\s]", "", regex=True)
        .str.replace(r"[^0-9.\-]", "", regex=True)
        .replace({"": np.nan, "nan": np.nan, "None": np.nan})
    )
    return pd.to_numeric(cleaned, errors="coerce")


def add_profile_bins(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()

    if "age" in result.columns and "age_band" not in result.columns:
        age_values = pd.to_numeric(result["age"], errors="coerce")
        result["age_band"] = pd.cut(
            age_values,
            bins=AGE_BINS,
            labels=AGE_BIN_LABELS,
            include_lowest=True,
        )

    if "gross income" in result.columns and "income_band" not in result.columns:
        income_values = parse_money_series(result["gross income"])
        result["income_band"] = pd.cut(
            income_values,
            bins=INCOME_BINS,
            labels=INCOME_BIN_LABELS,
            include_lowest=True,
        )

    return result


def drop_empty_application_rows(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    core_columns = [
        column
        for column in (
            "name",
            "contact",
            "e-mail address",
            "email address",
            "email",
            "status",
            "approved amount",
            "gross income",
            "employment type",
            "education",
            "purpose of finance",
            "employment sector",
            "occupation",
        )
        if column in result.columns
    ]

    if not core_columns:
        return result

    non_empty_count = pd.Series(0, index=result.index, dtype="int64")
    for column in core_columns:
        values = result[column]
        if pd.api.types.is_numeric_dtype(values):
            non_empty_count += values.notna().astype(int)
            continue
        text = values.astype(str).str.strip().str.lower()
        non_empty_count += (values.notna() & ~text.isin({"", "nan", "none"})).astype(int)

    return result.loc[non_empty_count >= MIN_CORE_APPLICATION_FIELDS].copy()


def detect_profile_columns(frame: pd.DataFrame) -> list[str]:
    candidates = []
    for column in frame.columns:
        if column in {"status", "approved amount", "approval_label", "approved_amount_numeric"}:
            continue
        if column in ID_CANDIDATES:
            continue
        if column == "age" and "age_band" in frame.columns:
            continue
        if column == "gross income" and "income_band" in frame.columns:
            continue
        normalized = column.strip().lower()
        if normalized in PROFILE_CANDIDATES:
            candidates.append(column)
            continue
        if normalized.startswith("age") or normalized.startswith("income"):
            candidates.append(column)
            continue
        if any(token in normalized for token in ("credit", "occupation", "education", "industry", "employment", "self-employed", "self employed", "marital", "state", "purpose", "income", "salary", "score")):
            candidates.append(column)
    if "age" in frame.columns and "age" not in candidates:
        candidates.append("age")
    return sorted(dict.fromkeys(candidates))


def build_profile_summary(frame: pd.DataFrame, profile_columns: list[str]) -> pd.DataFrame:
    group_cols = profile_columns.copy()
    working = frame.copy()
    working["approved_flag"] = working["approval_label"]

    working = working.dropna(subset=["approved_flag"])
    if not group_cols:
        raise ValueError("No profile columns available after preprocessing.")

    summary = (
        working.groupby(group_cols, dropna=False)
        .agg(
            applications=("approved_flag", "size"),
            approvals=("approved_flag", "sum"),
            approval_rate=("approved_flag", "mean"),
            avg_approved_amount=("approved_amount_numeric", "mean"),
            median_approved_amount=("approved_amount_numeric", "median"),
        )
        .reset_index()
    )

    summary["rejections"] = summary["applications"] - summary["approvals"]
    summary = summary[summary["applications"] >= MIN_PROFILE_SUPPORT].copy()
    summary = summary.sort_values(
        by=["approval_rate", "applications", "approvals"],
        ascending=[False, False, False],
    )
    summary.insert(0, "profile_rank", np.arange(1, len(summary) + 1))
    return summary


def build_preprocessor(frame: pd.DataFrame, feature_columns: list[str]) -> ColumnTransformer:
    numeric_features = [column for column in feature_columns if pd.api.types.is_numeric_dtype(frame[column])]
    categorical_features = [column for column in feature_columns if column not in numeric_features]
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                numeric_features,
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore")),
                ]),
                categorical_features,
            ),
        ],
        remainder="drop",
    )


def train_approval_model(frame: pd.DataFrame, feature_columns: list[str]) -> ModelBundle:
    usable = frame.dropna(subset=["approval_label"]).copy()
    if usable.empty or usable["approval_label"].nunique() < 2:
        return ModelBundle(model=None, features=feature_columns, metrics={"status": "insufficient_labels"})

    X = usable[feature_columns]
    y = usable["approval_label"].astype(int)
    preprocessor = build_preprocessor(usable, feature_columns)
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=250, random_state=42, class_weight="balanced")),
    ])

    min_class_count = int(y.value_counts().min())
    if len(usable) < 6 or min_class_count < 2:
        model.fit(X, y)
        training_score = float(model.score(X, y))
        return ModelBundle(
            model=model,
            features=feature_columns,
            metrics={
                "status": "trained_on_full_data",
                "accuracy": training_score,
                "train_rows": int(len(y)),
            },
        )

    test_size = max(2, int(round(len(usable) * 0.25)))
    if test_size >= len(usable):
        test_size = max(1, len(usable) - 2)

    if len(usable) - test_size < 2 or test_size < 2:
        model.fit(X, y)
        training_score = float(model.score(X, y))
        return ModelBundle(
            model=model,
            features=feature_columns,
            metrics={
                "status": "trained_on_full_data",
                "accuracy": training_score,
                "train_rows": int(len(y)),
            },
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        "status": "trained",
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "test_rows": int(len(y_test)),
    }
    if y_prob is not None and len(np.unique(y_test)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_test, y_prob))
    return ModelBundle(model=model, features=feature_columns, metrics=metrics)


def train_amount_model(frame: pd.DataFrame, feature_columns: list[str]) -> ModelBundle:
    usable = frame.dropna(subset=["approved_amount_numeric"]).copy()
    usable = usable[usable["approved_amount_numeric"] > 0]
    if usable.empty:
        return ModelBundle(model=None, features=feature_columns, metrics={"status": "insufficient_amounts"})

    X = usable[feature_columns]
    y = usable["approved_amount_numeric"]
    preprocessor = build_preprocessor(usable, feature_columns)
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=250, random_state=42)),
    ])

    if len(usable) < 8:
        return ModelBundle(model=None, features=feature_columns, metrics={"status": "insufficient_rows_for_training", "usable_rows": int(len(usable))})

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return ModelBundle(
        model=model,
        features=feature_columns,
        metrics={
            "status": "trained",
            "mae": float(mean_absolute_error(y_test, predictions)),
            "test_rows": int(len(y_test)),
        },
    )


def score_records(frame: pd.DataFrame, feature_columns: list[str], classifier_bundle: ModelBundle, regressor_bundle: ModelBundle) -> pd.DataFrame:
    scored = frame.copy()
    if classifier_bundle.model is not None:
        scored["predicted_approval_probability"] = classifier_bundle.model.predict_proba(scored[feature_columns])[:, 1]
    else:
        scored["predicted_approval_probability"] = np.nan

    if regressor_bundle.model is not None:
        scored["predicted_approved_amount"] = regressor_bundle.model.predict(scored[feature_columns])
    else:
        scored["predicted_approved_amount"] = np.nan

    scored["predicted_amount_band"] = pd.Series(pd.NA, index=scored.index, dtype="object")
    amount_mask = scored["predicted_approved_amount"].notna()
    scored.loc[amount_mask, "predicted_amount_band"] = np.where(
        scored.loc[amount_mask, "predicted_approved_amount"].ge(30000),
        "above_30000",
        "below_30000",
    )
    return scored


def compute_broader_profiles(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute a top-N broader profile list using available simpler attributes.

    Strategy: prefer pairwise labels (age_band|income_band, age_band|occupation, income_band|occupation),
    fallback to single attribute.
    """
    df = frame.copy()
    def _val(x):
        return "" if pd.isna(x) else str(x).strip()

    def broader_label(row):
        ab = _val(row.get("age_band"))
        ib = _val(row.get("income_band"))
        occ = _val(row.get("occupation"))
        if ab and ab.lower() not in ("", "nan", "none", "unknown"):
            if ib and ib.lower() not in ("", "nan", "none", "unknown"):
                return f"{ab} | {ib}"
            if occ and occ.lower() not in ("", "nan", "none", "unknown"):
                return f"{ab} | {occ}"
            return ab
        if ib and ib.lower() not in ("", "nan", "none", "unknown"):
            if occ and occ.lower() not in ("", "nan", "none", "unknown"):
                return f"{ib} | {occ}"
            return ib
        if occ and occ.lower() not in ("", "nan", "none", "unknown"):
            return occ
        return "Unknown"

    df["broader_profile"] = df.apply(broader_label, axis=1)
    agg = (
        df.dropna(subset=["approval_label"])
        .groupby("broader_profile", dropna=False)
        .agg(applications=("approval_label", "size"), approvals=("approval_label", "sum"), avg_approved_amount=("approved_amount_numeric", "mean"))
        .reset_index()
    )
    agg["approval_rate"] = agg["approvals"] / agg["applications"].replace({0: 1})
    agg = agg[agg["applications"] >= MIN_PROFILE_SUPPORT].sort_values(by=["applications", "approval_rate"], ascending=[False, False])
    return agg.head(10).copy()


def compute_combo_profiles(df: pd.DataFrame, profile_cols: list[str], min_support: int = MIN_PROFILE_SUPPORT, combo_size: int = 3) -> pd.DataFrame:
    """Aggregate profiles using any combination of `combo_size` attributes (default 3).

    Returns a DataFrame with columns: profile, attributes, applications, approvals, approval_rate, avg_approved_amount, rejections.
    """
    cols = [c for c in profile_cols if c in df.columns]
    if not cols:
        return pd.DataFrame(columns=["profile", "attributes", "applications", "approvals", "approval_rate", "avg_approved_amount", "rejections"]) 

    from itertools import combinations
    k = min(combo_size, len(cols))
    rows = []
    for combo in combinations(cols, k):
        g = (
            df.dropna(subset=["approval_label"]) 
            .groupby(list(combo), dropna=False)
            .agg(applications=("approval_label", "size"), approvals=("approval_label", lambda s: int(s.dropna().astype(int).sum())), avg_approved_amount=("approved_amount_numeric", "mean"))
            .reset_index()
        )
        if g.empty:
            continue
        g["rejections"] = g["applications"] - g["approvals"]
        g["approval_rate"] = g.apply(lambda r: float(r["approvals"]) / r["applications"] if r["applications"] > 0 else np.nan, axis=1)
        def _valid_val(v):
            try:
                if pd.isna(v):
                    return False
            except Exception:
                pass
            s = str(v).strip().lower()
            return s not in ("", "nan", "none")

        for _, r in g.iterrows():
            # skip combos containing missing/NaN values to avoid 'nan' profiles
            vals = [r.get(kcol) for kcol in combo]
            if not all(_valid_val(v) for v in vals):
                continue
            profile_parts = []
            for kcol in combo:
                v = r.get(kcol)
                profile_parts.append(f"{kcol}={v}")
            rows.append({
                "profile": " | ".join(profile_parts),
                "attributes": ",".join(combo),
                "applications": int(r["applications"]),
                "approvals": int(r["approvals"]),
                "avg_approved_amount": float(r["avg_approved_amount"]) if pd.notna(r["avg_approved_amount"]) else np.nan,
                "rejections": int(r["rejections"]),
                "approval_rate": float(r["approval_rate"]) if pd.notna(r["approval_rate"]) else np.nan,
            })

    grouped = pd.DataFrame(rows)
    if grouped.empty:
        return grouped
    # prefer groups with at least one approval; then rank by approval_rate desc, applications desc
    grouped = grouped[grouped["applications"] >= min_support].copy()
    if grouped.empty:
        grouped = pd.DataFrame(rows)
    grouped = grouped.sort_values(["approval_rate", "applications"], ascending=[False, False]).reset_index(drop=True)
    return grouped


def _profile_text(row: pd.Series, profile_columns: list[str]) -> str:
    parts = []
    for column in profile_columns:
        value = row.get(column)
        if pd.notna(value):
            parts.append(f"{column}={value}")
    return " | ".join(parts)


_EDU_SHORT = {
    "secondary": "SPM",
    "diploma": "Diploma",
    "degree": "Degree",
    "bachelor": "Degree",
    "master": "Master's",
    "masters": "Master's",
    "phd": "PhD",
    "doctorate": "PhD",
}

_EMP_SHORT = {
    "private employee": "Private",
    "government employee": "Gov't",
    "self employ": "Self-Employed",
    "self-employed": "Self-Employed",
    "self employed": "Self-Employed",
    "contract": "Contract",
}

_PURPOSE_SHORT = {
    "home improvement": "Home Improvement",
    "education": "Education Financing",
    "personal": "Personal",
    "business": "Business",
    "vehicle": "Vehicle",
    "medical": "Medical",
}


def _age_band(age: Any) -> str:
    try:
        a = int(float(age))
    except (TypeError, ValueError):
        return ""
    if a < 26:
        return "Young (18-25)"
    if a < 31:
        return "Early Career (26-30)"
    if a < 41:
        return "Mid-Career (31-40)"
    if a < 51:
        return "Established (41-50)"
    return "Senior (51+)"


def _short(mapping: dict[str, str], value: Any, fallback: str = "") -> str:
    if pd.isna(value):
        return fallback
    return mapping.get(str(value).strip().lower(), str(value).strip().title())


def name_profile(row: pd.Series) -> str:
    """Return a human-readable profile label for a summary row."""
    parts: list[str] = []

    age_band = row.get("age_band")
    age_part = f"Age {age_band}" if pd.notna(age_band) else _age_band(row.get("age"))
    if age_part:
        parts.append(age_part)

    income_band = row.get("income_band")
    if pd.notna(income_band):
        parts.append(f"Income {income_band}")

    # Exclude gender from profile label (low coverage / suppressed)

    edu = _short(_EDU_SHORT, row.get("education"))
    if edu:
        parts.append(edu)

    emp = _short(_EMP_SHORT, row.get("employment type"))
    if emp:
        parts.append(emp)

    sector = str(row.get("employment sector", "")).strip().strip('"').strip()
    if sector and sector.lower() not in ("", "nan", "none"):
        # Shorten very long sector names to the key noun
        sector_clean = re.sub(r"[,;]\.+", "", sector)
        sector_clean = sector_clean[:40].strip()
        parts.append(sector_clean)

    if pd.isna(income_band):
        income = row.get("gross income")
        if pd.notna(income):
            try:
                parts.append(f"RM{int(float(income)):,}/mo")
            except (TypeError, ValueError):
                pass

    purpose = _short(_PURPOSE_SHORT, row.get("purpose of finance"))
    if purpose:
        parts.append(purpose)

    return " | ".join(parts) if parts else "Unknown Profile"


def dataframe_to_markdown(frame: pd.DataFrame, max_rows: int = 25) -> str:
    if frame.empty:
        return "| |\n| --- |"

    display_frame = frame.head(max_rows).copy().astype(object).fillna("")
    columns = [str(column) for column in display_frame.columns]

    def escape(value: Any) -> str:
        s = str(value)
        # remove internal newlines that break markdown tables
        s = s.replace("\r", " ").replace("\n", " ")
        # collapse multiple spaces
        s = re.sub(r"\s+", " ", s).strip()
        # escape pipe characters for markdown table cells
        return s.replace("|", "\\|")

    lines = ["| " + " | ".join(columns) + " |"]
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for _, row in display_frame.iterrows():
        lines.append("| " + " | ".join(escape(row[column]) for column in display_frame.columns) + " |")
    return "\n".join(lines)


def _sanitize_df_for_md(df: pd.DataFrame) -> pd.DataFrame:
    """Sanitize string columns to avoid newlines and unescaped pipes in Markdown table cells."""
    res = df.copy()
    for col in res.columns:
        if res[col].dtype == object or res[col].dtype.name == "string":
            res[col] = (
                res[col]
                .fillna("")
                .astype(str)
                .str.replace(r"[\r\n]+", " ", regex=True)
                .str.replace("|", "\\|")
                .str.replace(r"\s+", " ", regex=True)
                .str.strip()
            )
    return res


def build_report(
    frame: pd.DataFrame,
    summary: pd.DataFrame,
    scored: pd.DataFrame,
    profile_columns: list[str],
    classifier_bundle: ModelBundle,
    regressor_bundle: ModelBundle,
    broader_profiles: pd.DataFrame | None = None,
) -> str:
    def top_rows(source: pd.DataFrame, limit: int = 3) -> pd.DataFrame:
        return source.head(limit).copy()

    # Use combo-based profiles (any combination of 3 attributes) for Answers
    combo_profiles = compute_combo_profiles(frame, profile_columns, min_support=MIN_PROFILE_SUPPORT, combo_size=3)
    approved_only = combo_profiles[combo_profiles["approvals"] > 0].copy()
    high_amount = combo_profiles[combo_profiles["avg_approved_amount"].ge(30000)].copy()
    low_amount = combo_profiles[combo_profiles["avg_approved_amount"].lt(30000)].copy()
    _self_emp_values = {"self-employed", "self employed", "self employ", "yes", "y", "true", "1"}

    # Compute self-employed groups from the raw frame (not the already MIN_PROFILE_SUPPORT-filtered summary)
    def _row_is_self_employed(r: pd.Series) -> bool:
        # Prefer explicit `is_self_employed` flag if present
        if "is_self_employed" in r.index:
            val = r.get("is_self_employed")
            flag = _flag_to_binary(val)
            if flag == 1.0:
                return True
        for column in profile_columns:
            if column not in r:
                continue
            v = r.get(column)
            if pd.isna(v):
                continue
            s = str(v).strip().lower()
            if re.search(r"\bself\b", s) or "self employ" in s or "self-employed" in s:
                return True
        return False

    se_df = frame[frame.apply(_row_is_self_employed, axis=1)].copy()
    if not se_df.empty:
        # compute self-employed combos using same combo logic
        se_combo = compute_combo_profiles(se_df, profile_columns, min_support=MIN_PROFILE_SUPPORT, combo_size=3)
        se_agg_all = se_combo.rename(columns={"profile": "profile_label", "avg_approved_amount": "avg_approved_amount"})
        # coerce numeric types and compute rates
        if not se_agg_all.empty:
            se_agg_all["applications"] = pd.to_numeric(se_agg_all["applications"], errors="coerce").fillna(0).astype(int)
            se_agg_all["approvals"] = pd.to_numeric(se_agg_all["approvals"], errors="coerce").fillna(0).astype(int)
            se_agg_all["approval_rate"] = se_agg_all["approvals"] / se_agg_all["applications"].replace({0: 1})
            se_agg_all["rejections"] = se_agg_all["applications"] - se_agg_all["approvals"]

        # apply MIN_PROFILE_SUPPORT; fallback to top groups if none meet support
        se_agg = se_agg_all.copy()
        if se_agg.empty:
            self_employed = pd.DataFrame(columns=["profile_label", "applications", "approvals", "approval_rate", "avg_approved_amount"])
        else:
            self_employed = se_agg.head(3).copy()
    else:
        self_employed = summary.iloc[0:0].copy()

    target_67000 = summary.copy()
    target_67000["distance_to_67000"] = (target_67000["avg_approved_amount"].fillna(np.inf) - 67000).abs()
    target_67000 = target_67000.sort_values(
        by=["approval_rate", "distance_to_67000", "applications"],
        ascending=[False, True, False],
    )

    rejection_rank = summary.sort_values(by=["approval_rate", "rejections", "applications"], ascending=[True, False, False])

    lines = [
        "# Approval Analysis Report",
        "",
        f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"Rows analyzed: {len(frame)}",
        f"Profile columns used: {', '.join(profile_columns)}",
        "",
        "## Model Summary",
        f"- Approval model: {classifier_bundle.metrics}",
        f"- Amount model: {regressor_bundle.metrics}",
        "",
        # Insert broader-profile summary near the top for visibility
    ]

    if broader_profiles is not None and not broader_profiles.empty:
        lines.extend([
            "## Broader Profile Rankings (Top 10)",
            "",
            dataframe_to_markdown(
                _sanitize_df_for_md(
                    broader_profiles.loc[:, ["broader_profile", "applications", "approvals", "approval_rate", "avg_approved_amount"]]
                    .rename(columns={"broader_profile": "profile", "avg_approved_amount": "avg_approved_amount_RM"})
                ),
                max_rows=10,
            ),
            "",
        ])

    # CTOS summary (if CTOS data is available)
    if "ctos_score" in frame.columns or "ctos_numeric" in frame.columns:
        cdf = frame.copy()
        if "ctos_score" in cdf.columns:
            cdf["ctos_numeric"] = pd.to_numeric(cdf["ctos_score"], errors="coerce")
        if "ctos_band" not in cdf.columns:
            try:
                cdf["ctos_band"] = pd.cut(cdf["ctos_numeric"], bins=CTOS_BINS, labels=CTOS_LABELS, include_lowest=True)
            except Exception:
                cdf["ctos_band"] = pd.Series([np.nan] * len(cdf))

        ctos_agg = (
            cdf.dropna(subset=["approval_label"]) 
            .groupby("ctos_band", dropna=False)
            .agg(applications=("approval_label", "size"), approvals=("approval_label", "sum"))
            .assign(approval_rate=lambda df: df["approvals"] / df["applications"].replace({0: 1}))
            .sort_values(["approval_rate", "applications"], ascending=[False, False])
            .reset_index()
        )
        lines.extend([
            "## CTOS Summary",
            "",
            dataframe_to_markdown(_sanitize_df_for_md(ctos_agg.rename(columns={"ctos_band": "ctos_band", "applications": "applications", "approvals": "approvals", "approval_rate": "approval_rate"})), max_rows=10),
            "",
        ])

    simple_cols = [c for c in ("age_band", "income_band", "occupation") if c in frame.columns]
    if simple_cols:
        lines.append("## Attribute breakdowns (single and pairwise, min support = %d)" % MIN_PROFILE_SUPPORT)
        for col in simple_cols:
            lines.append(f"### By {col}")
            g = (
                frame.dropna(subset=["approval_label"]) 
                .groupby(col, dropna=False)
                .agg(applications=("approval_label", "size"), approvals=("approval_label", "sum"))
                .assign(approval_rate=lambda df: df["approvals"] / df["applications"]) 
            )
            g = g.sort_values(["applications"], ascending=False)
            if g.empty:
                lines.append("No data for this attribute.")
                lines.append("")
                continue
            for idx, row in g.head(8).iterrows():
                apps = int(row["applications"])
                approved = int(row["approvals"])
                rate = row["approval_rate"]
                lines.append(f"- {idx}: {apps} apps, {approved} approvals, rate {rate:.1%}")
            lines.append("")

        from itertools import combinations
        lines.append("### Pairwise attribute top groups")
        for a, b in combinations(simple_cols, 2):
            lines.append(f"**By {a} + {b}**")
            g = (
                frame.dropna(subset=["approval_label"]) 
                .groupby([a, b], dropna=False)
                .agg(applications=("approval_label", "size"), approvals=("approval_label", "sum"))
                .assign(approval_rate=lambda df: df["approvals"] / df["applications"]) 
            )
            # Rank pairwise groups by approval rate first, then by applications
            g = g[g["applications"] >= MIN_PROFILE_SUPPORT].assign(approval_rate=lambda df: df["approvals"] / df["applications"]).reset_index()
            g = g.sort_values(["approval_rate", "applications"], ascending=[False, False]).reset_index(drop=True)
            if g.empty:
                lines.append("No robust pairwise groups for these attributes.")
                lines.append("")
                continue
            for _, row in g.head(10).iterrows():
                apps = int(row["applications"])
                approved = int(row["approvals"])
                rate = row["approval_rate"]
                a_val = row[a]
                b_val = row[b]
                lines.append(f"- {a_val} | {b_val}: {apps} apps, {approved} approvals, rate {rate:.1%}")
            lines.append("")

    lines.extend([
        "## Answers",
    ])

    sections = [
        ("1) Most likely to get approved", approved_only),
        ("2) Top 3 client profiles with the highest approval", approved_only),
        ("3) Most likely to get approval above RM30,000", high_amount),
        ("4) Most likely to get approval below RM30,000", low_amount),
        ("5) Self-employed client profile most likely to get approved", self_employed),
        ("6) Ideal client profile for approvals at RM67,000", combo_profiles),
        ("7) Client profile with the most rejection", combo_profiles.sort_values(["rejections","applications"], ascending=[False,False])),
    ]

    for title, table in sections:
        lines.extend([f"### {title}"])
        if table.empty:
            lines.append("No ranked profile could be derived from the available data.")
            lines.append("")
            continue

        # Render the top N rows as a markdown table; use up to 10 rows by default
        sel = table.head(10).copy()
        # Normalize column names for display
        if "profile" not in sel.columns and "profile_label" in sel.columns:
            sel = sel.rename(columns={"profile_label": "profile"})
        display_cols = [c for c in ("profile", "applications", "approvals", "approval_rate", "avg_approved_amount") if c in sel.columns]
        if not display_cols:
            # fallback: stringify rows
            for i, (_, row) in enumerate(sel.iterrows(), start=1):
                lines.append(str(row.to_dict()))
            lines.append("")
            continue

        # sanitize any string columns (remove newlines, escape pipes) to prevent broken markdown table rows
        sel_md = _sanitize_df_for_md(sel.loc[:, display_cols].reset_index(drop=True))
        lines.append(dataframe_to_markdown(sel_md, max_rows=99999))
        lines.append("")

    # Broader groupings are shown at the top of the report.
    lines.append("(Broader profile breakdowns are shown near the top of this report.)")

    # Simple aggregate stats section
    lines.append("## Simple Stats")
    total_apps = len(frame)
    total_approvals = int(frame.get("approval_label", pd.Series(0, index=frame.index)).fillna(0).astype(int).sum())
    lines.append(f"- Total applications: {total_apps}")
    lines.append(f"- Total approvals: {total_approvals}")

    # income / occupation summaries (top 5)

    if "income_band" in frame.columns:
        g = (
            frame.groupby("income_band", dropna=False)
            .agg(applications=("approval_label", "size"), approvals=("approval_label", "sum"))
            .assign(approval_rate=lambda df: df["approvals"] / df["applications"]) 
            .sort_values("applications", ascending=False)
        )
        lines.append("- Income band breakdown (top rows):")
        for idx, row in g.head(8).iterrows():
            lines.append(f"  - {idx}: {int(row['applications'])} apps, {int(row['approvals'])} approvals, rate {row['approval_rate']:.1%}")

    if "occupation" in frame.columns:
        g = (
            frame.groupby("occupation", dropna=False)
            .agg(applications=("approval_label", "size"), approvals=("approval_label", "sum"))
            .assign(approval_rate=lambda df: df["approvals"] / df["applications"]) 
            .sort_values("applications", ascending=False)
        )
        lines.append("- Occupation breakdown (top rows):")
        for idx, row in g.head(10).iterrows():
            lines.append(f"  - {idx}: {int(row['applications'])} apps, {int(row['approvals'])} approvals, rate {row['approval_rate']:.1%}")
    lines.append("")

    lines.extend(
        [
            "## Ranking Table",
            dataframe_to_markdown(
                _sanitize_df_for_md(
                    summary.assign(profile_name=summary.apply(name_profile, axis=1))
                    .loc[:, ["profile_rank", "profile_name", "applications", "approvals", "approval_rate",
                             "avg_approved_amount", "median_approved_amount", "rejections"]]
                ),
                max_rows=25,
            ),
            "",
            "## Scored Rows",
            dataframe_to_markdown(_sanitize_df_for_md(scored), max_rows=20),
            "",
        ]
    )
    return "\n".join(lines)


def write_metrics(path: Path, classifier_bundle: ModelBundle, regressor_bundle: ModelBundle, profile_columns: list[str], frame: pd.DataFrame) -> None:
    payload = {
        "generated_at": datetime.now().isoformat(),
        "rows": int(len(frame)),
        "profile_columns": profile_columns,
        "approval_model": classifier_bundle.metrics,
        "amount_model": regressor_bundle.metrics,
    }
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def write_structured_json(path: Path, *, frame: pd.DataFrame, summary: pd.DataFrame, scored: pd.DataFrame, profile_columns: list[str], classifier_bundle: ModelBundle, regressor_bundle: ModelBundle, broader_profiles: pd.DataFrame | None = None) -> None:
    data: dict[str, Any] = {}
    data["generated_at"] = datetime.now().isoformat()
    data["rows_analyzed"] = int(len(frame))
    data["profile_columns"] = profile_columns
    data["model_summary"] = {"approval_model": classifier_bundle.metrics, "amount_model": regressor_bundle.metrics}

    # broader profiles
    if broader_profiles is not None and not broader_profiles.empty:
        bp = broader_profiles.copy()
        # format avg amounts as whole numbers and approval_rate as percentage strings
        if "avg_approved_amount" in bp.columns:
            bp["avg_approved_amount"] = bp["avg_approved_amount"].apply(lambda v: int(round(v)) if pd.notna(v) else None)
        if "approval_rate" in bp.columns:
            bp["approval_rate"] = bp["approval_rate"].apply(lambda v: f"{float(v)*100:.2f}%" if pd.notna(v) else None)
        data["broader_profiles"] = bp.replace({np.nan: None}).to_dict(orient="records")
    else:
        data["broader_profiles"] = []

    # summary and ranking
    # Format numeric fields for display
    ps = summary.copy()
    if "avg_approved_amount" in ps.columns:
        ps["avg_approved_amount"] = ps["avg_approved_amount"].apply(lambda v: int(round(v)) if pd.notna(v) else None)
    if "median_approved_amount" in ps.columns:
        ps["median_approved_amount"] = ps["median_approved_amount"].apply(lambda v: int(round(v)) if pd.notna(v) else None)
    if "approval_rate" in ps.columns:
        ps["approval_rate"] = ps["approval_rate"].apply(lambda v: f"{float(v)*100:.2f}%" if pd.notna(v) else None)
    data["profile_summary"] = ps.replace({np.nan: None}).to_dict(orient="records")

    # top scored rows (trim to first 200 for JSON size)
    try:
        data["scored_rows"] = scored.head(200).replace({np.nan: None}).to_dict(orient="records")
    except Exception:
        data["scored_rows"] = []

    # CTOS summary if present
    if "ctos_score" in frame.columns or "ctos_numeric" in frame.columns:
        cdf = frame.copy()
        if "ctos_score" in cdf.columns:
            cdf["ctos_numeric"] = pd.to_numeric(cdf["ctos_score"], errors="coerce")
        try:
            cdf["ctos_band"] = pd.cut(cdf.get("ctos_numeric", pd.Series([np.nan]*len(cdf))), bins=CTOS_BINS, labels=CTOS_LABELS, include_lowest=True)
        except Exception:
            cdf["ctos_band"] = pd.Series([np.nan] * len(cdf))
        ctos_agg = (
            cdf.dropna(subset=["approval_label"]) 
            .groupby("ctos_band", dropna=False)
            .agg(applications=("approval_label", "size"), approvals=("approval_label", "sum"))
            .assign(approval_rate=lambda df: df["approvals"] / df["applications"].replace({0: 1}))
            .reset_index()
        )
        ca = ctos_agg.copy()
        if "approval_rate" in ca.columns:
            ca["approval_rate"] = ca["approval_rate"].apply(lambda v: f"{float(v)*100:.2f}%" if pd.notna(v) else None)
        data["ctos_summary"] = ca.replace({np.nan: None}).to_dict(orient="records")
    else:
        data["ctos_summary"] = []

    # attribute breakdowns
    attrs = [c for c in ("age_band", "income_band", "occupation") if c in frame.columns]
    attr_breakdown: dict[str, Any] = {}
    for col in attrs:
        g = (
            frame.dropna(subset=["approval_label"]) 
            .groupby(col, dropna=False)
            .agg(applications=("approval_label", "size"), approvals=("approval_label", "sum"))
            .assign(approval_rate=lambda df: df["approvals"] / df["applications"]) 
            .reset_index()
        )
        if "approval_rate" in g.columns:
            g["approval_rate"] = g["approval_rate"].apply(lambda v: f"{float(v)*100:.2f}%" if pd.notna(v) else None)
        attr_breakdown[col] = g.replace({np.nan: None}).to_dict(orient="records")
    data["attribute_breakdowns"] = attr_breakdown

    # pairwise stats
    from itertools import combinations
    pairwise_records = []
    for a, b in combinations(attrs, 2):
        g = (
            frame.dropna(subset=["approval_label"]) 
            .groupby([a, b], dropna=False)
            .agg(applications=("approval_label", "size"), approvals=("approval_label", "sum"))
            .assign(approval_rate=lambda df: df["approvals"] / df["applications"]) 
            .reset_index()
        )
        if g.empty:
            continue
        g = g[g["applications"] >= MIN_PROFILE_SUPPORT].sort_values(["approval_rate", "applications"], ascending=[False, False]).reset_index(drop=True)
        # format approval_rate in pairwise rows
        if "approval_rate" in g.columns:
            g["approval_rate"] = g["approval_rate"].apply(lambda v: f"{float(v)*100:.2f}%" if pd.notna(v) else None)
        pairwise_records.append({"pair": f"{a} | {b}", "rows": g.replace({np.nan: None}).to_dict(orient="records")})
    data["pairwise"] = pairwise_records

    # combo-based answers (reuse compute_combo_profiles)
    combo = compute_combo_profiles(frame, profile_columns, min_support=MIN_PROFILE_SUPPORT, combo_size=3)
    cp = combo.copy()
    if "avg_approved_amount" in cp.columns:
        cp["avg_approved_amount"] = cp["avg_approved_amount"].apply(lambda v: int(round(v)) if pd.notna(v) else None)
    if "approval_rate" in cp.columns:
        cp["approval_rate"] = cp["approval_rate"].apply(lambda v: f"{float(v)*100:.2f}%" if pd.notna(v) else None)
    data["combo_profiles"] = cp.replace({np.nan: None}).to_dict(orient="records")

    # write to path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

    # include rebuilt answers if available
    try:
        answers_json = path.parent / "profile_answers_rebuild.json"
        if answers_json.exists():
            ans = json.loads(answers_json.read_text(encoding="utf-8"))
            data["answers"] = ans
            # Do not promote fields into top-level to avoid duplication.
            path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        else:
            # fallback: include the md if present
            answers_md = path.parent / "profile_answers_rebuild.md"
            if answers_md.exists():
                data["answers_md"] = answers_md.read_text(encoding="utf-8")
                path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    except Exception:
        # ignore failures when attaching answers
        pass


# --- Rebuild-style profile answers (imported from scripts/rebuild_profiles_and_answers.py) ---
def _parse_money_scalar(s: Any) -> float:
    try:
        if pd.isna(s):
            return np.nan
    except Exception:
        pass
    t = re.sub(r"[RM,\s]", "", str(s))
    t = re.sub(r"[^0-9.\-]", "", t)
    try:
        return float(t) if t not in ("", "nan") else np.nan
    except Exception:
        return np.nan


CTOS_BINS = [0, 550, 650, 700, 750, 9999]
CTOS_LABELS = ["<550", "550-649", "650-699", "700-749", "750+"]


def _make_bands_for_rebuild(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # age band
    if "age" in df.columns:
        df["age_numeric"] = pd.to_numeric(df["age"], errors="coerce")
        df["age_band"] = pd.cut(df["age_numeric"], bins=[0,25,30,40,50,np.inf], labels=["18-25","26-30","31-40","41-50","51+"], include_lowest=True)

    # income band
    for col in ("gross income", "income", "monthly income"):
        if col in df.columns:
            df["income_numeric"] = df[col].apply(_parse_money_scalar)
            break
    if "income_numeric" in df.columns:
        df["income_band"] = pd.cut(df["income_numeric"], bins=[0,3000,5000,8000,12000,np.inf], labels=["RM0-3k","RM3k-5k","RM5k-8k","RM8k-12k","RM12k+"], include_lowest=True)

    # ctos
    if "ctos_score" in df.columns:
        df["ctos_numeric"] = pd.to_numeric(df["ctos_score"], errors="coerce")
        df["ctos_band"] = pd.cut(df["ctos_numeric"], bins=CTOS_BINS, labels=CTOS_LABELS, include_lowest=True)

    # approved amount
    if "approved amount" in df.columns:
        df["approved_amount_numeric"] = df["approved amount"].apply(_parse_money_scalar)
    elif "approved_amount_numeric" in df.columns:
        df["approved_amount_numeric"] = pd.to_numeric(df["approved_amount_numeric"], errors="coerce")
    else:
        df["approved_amount_numeric"] = np.nan

    df["approved_above_30k"] = df["approved_amount_numeric"].gt(30000)

    # approval label
    def _norm_status(v: Any):
        if pd.isna(v):
            return np.nan
        s = str(v).strip().lower()
        if s in ("approved","approve","yes","y","true","1"):
            return 1
        if s in ("rejected","reject","declined","no","n","false","0"):
            return 0
        return np.nan

    # Prefer explicit flags: `is_approved` to mark approvals, `is_rejected` to mark rejections.
    df["approval_label"] = np.nan
    if "is_approved" in df.columns:
        df["approval_label"] = df["is_approved"].apply(_flag_to_binary)
    if "is_rejected" in df.columns:
        rej_flag = df["is_rejected"].apply(_flag_to_binary)
        df.loc[rej_flag == 1.0, "approval_label"] = 0
    # Fallback to legacy status strings for any rows still unlabeled
    if df["approval_label"].isna().any():
        for col in ("status", "approval", "approved"):
            if col in df.columns:
                still = df["approval_label"].isna()
                df.loc[still, "approval_label"] = df.loc[still, col].apply(_norm_status)
                break

    return df


def _build_profiles_and_answers(frame: pd.DataFrame, output_dir: Path, min_support: int = 3) -> None:
    df = frame.copy()
    # ensure lowercase columns (normalized earlier in pipeline)
    df.columns = [str(c).strip().lower() for c in df.columns]
    df = _make_bands_for_rebuild(df)

    # Include employment-related fields and explicit self-employed flag so
    # self-employed analysis can be grouped and reported. Do NOT include
    # `approved_above_30k`/`approved_below_30k` as profile attributes.
    profile_cols = [
        c
        for c in (
            "age",
            "age_band",
            "income_band",
            "ctos_band",
            "occupation",
            "education",
            "employment sector",
            "employment type",
            "purpose of finance",
        )
        if c in df.columns
    ]

    # Build grouped summaries across any combination of 3 attributes (or fewer if not available)
    from itertools import combinations
    combo_size = min(3, len(profile_cols))
    rows = []
    for combo in combinations(profile_cols, combo_size):
        g = (
            df.groupby(list(combo), dropna=False)
            .agg(applications=("approval_label","size"), approvals=("approval_label", lambda s: s.dropna().astype(int).sum()), avg_approved_amount=("approved_amount_numeric","mean"))
            .reset_index()
        )
        if g.empty:
            continue
        g["rejections"] = g["applications"] - g["approvals"]
        g["approval_rate"] = g.apply(lambda r: float(r["approvals"]) / r["applications"] if r["applications"]>0 else np.nan, axis=1)
        # produce a compact profile label and record attribute names
        for _, r in g.iterrows():
            profile_parts = []
            for k in combo:
                v = r.get(k)
                profile_parts.append(f"{k}={v}")
            rows.append({
                "profile": " | ".join(profile_parts),
                #"attributes": ",".join(combo),
                "applications": int(r["applications"]),
                "approvals": int(r["approvals"]),
                "avg_approved_amount": float(r["avg_approved_amount"]) if pd.notna(r["avg_approved_amount"]) else np.nan,
                "rejections": int(r["rejections"]),
                "approval_rate": float(r["approval_rate"]) if pd.notna(r["approval_rate"]) else np.nan,
            })
    grouped = pd.DataFrame(rows)
    if grouped.empty:
        grouped = pd.DataFrame(columns=["profile","attributes","applications","approvals","avg_approved_amount","rejections","approval_rate"])
    grouped.to_csv(output_dir / "profile_summary_rebuild.csv", index=False)

    def _top_by(gdf, n=1):
        sel = gdf[gdf["applications"] >= min_support].copy()
        if sel.empty:
            sel = gdf.copy()
        # prefer groups that have at least one approval
        with_approvals = sel[sel["approvals"] > 0]
        if not with_approvals.empty:
            return with_approvals.sort_values(["approval_rate","applications"], ascending=[False,False]).head(n)
        # fallback: return best by approval_rate (may be zero)
        return sel.sort_values(["approval_rate","applications"], ascending=[False,False]).head(n)

    out = {}
    out["most_likely_approved"] = _top_by(grouped,1).to_dict(orient="records")
    out["top_3_highest_approval"] = _top_by(grouped,3).to_dict(orient="records")
    above = grouped[grouped["avg_approved_amount"] > 30000] if "avg_approved_amount" in grouped.columns else grouped.iloc[0:0]
    out["likely_approved_above_30k"] = _top_by(above,1).to_dict(orient="records")
    below = grouped[grouped["avg_approved_amount"] <= 30000] if "avg_approved_amount" in grouped.columns else grouped.iloc[0:0]
    out["likely_approved_below_30k"] = _top_by(below,1).to_dict(orient="records")

    # self-employed detection: prefer explicit `is_self_employed` flag if available
    if "is_self_employed" in df.columns:
        se_mask = df["is_self_employed"].apply(_flag_to_binary) == 1.0
        se = df[se_mask].copy()
    else:
        mask = pd.Series(False, index=df.index)
        for c in ("employment type","occupation","employment sector"):
            if c in df.columns:
                mask = mask | df[c].astype(str).str.lower().str.contains("self")
        se = df[mask].copy()
    if se.empty:
        out["self_employed_best"] = None
        out["self_employed_stats"] = {"self_percent": 0.0, "approval_rate": None, "applications": 0, "approvals": 0, "rejections": 0}
    else:
        total = len(df)
        total_se = len(se)
        approvals_count = int(se["approval_label"].dropna().astype(int).sum()) if se["approval_label"].dropna().size>0 else 0
        rejections_count = int(total_se - approvals_count)
        out["self_employed_stats"] = {
            "self_percent": len(se) / total * 100 if total > 0 else 0,
            "approval_rate": se["approval_label"].dropna().astype(int).mean() if se["approval_label"].dropna().size > 0 else None,
            "applications": total_se,
            "approvals": approvals_count,
            "rejections": rejections_count,
        }
        # Use a reduced set of profile columns for self-employed grouping so groups have higher support
        se_profile_cols = [
            c
            for c in (
                "age_band",
                "income_band",
                "ctos_band",
                "occupation",
                "employment type",
                "purpose of finance",
            )
            if c in se.columns
        ]
        if se_profile_cols:
            gse = (
                se.groupby([c for c in se_profile_cols if c in se.columns], dropna=False)
                .agg(applications=("approval_label", "size"), approvals=("approval_label", lambda s: s.dropna().astype(int).sum()), avg_approved_amount=("approved_amount_numeric", "mean"))
                .reset_index()
            )
        else:
            gse = (
                se.groupby([c for c in profile_cols if c in se.columns], dropna=False)
                .agg(applications=("approval_label", "size"), approvals=("approval_label", lambda s: s.dropna().astype(int).sum()), avg_approved_amount=("approved_amount_numeric", "mean"))
                .reset_index()
            )
        # compute approval_rate and rejections for self-employed groups
        if not gse.empty:
            gse["approval_rate"] = gse["approvals"] / gse["applications"].replace({0: 1})
            gse["rejections"] = gse["applications"] - gse["approvals"]

        if gse.empty:
            out["self_employed_best"] = None
        else:
            # Require at least `se_min_support` applications for robust selection (user-requested minimum = 1)
            se_min_support = 1
            # 1) prefer groups with at least one approval AND at least se_min_support applications
            with_approvals_and_min = gse[(gse["approvals"] > 0) & (gse["applications"] >= se_min_support)].copy()
            if not with_approvals_and_min.empty:
                best_df = with_approvals_and_min.sort_values(["approval_rate", "applications"], ascending=[False, False]).head(1)
                # create a compact human-readable `profile` field instead of exporting all columns
                row = best_df.iloc[0]
                profile_keys = [c for c in se_profile_cols if c in best_df.columns]
                parts = []
                for pc in profile_keys:
                    try:
                        v = row.get(pc)
                    except Exception:
                        v = None
                    if pd.isna(v):
                        continue
                    s = str(v).strip()
                    if not s or s.lower() in ("nan", "none"):
                        continue
                    parts.append(f"{pc}={s}")
                profile_text = " | ".join(parts) if parts else ""
                rec = {
                    "profile": profile_text,
                    "applications": int(row["applications"]),
                    "approvals": int(row["approvals"]),
                    "approval_rate": float(row["approval_rate"]) if pd.notna(row.get("approval_rate")) else None,
                    "avg_approved_amount": float(row["avg_approved_amount"]) if pd.notna(row.get("avg_approved_amount")) else None,
                    "rejections": int(row.get("rejections") if pd.notna(row.get("rejections")) else (int(row["applications"] - row["approvals"]))),
                }
                out["self_employed_best"] = [rec]
            else:
                # No group meets both the approval and minimum-application requirements.
                # Do not fall back to groups without approvals to avoid misleading recommendations.
                out["self_employed_best"] = None

    out["ideal_for_67000"] = grouped.assign(dist=(grouped["avg_approved_amount"].fillna(1e9)-67000).abs()).sort_values(["approval_rate","dist","applications"], ascending=[False,True,False]).head(1).to_dict(orient="records")
    out["most_rejected"] = grouped.sort_values(["rejections","applications"], ascending=[False,False]).head(1).to_dict(orient="records")

    # pairwise
    from itertools import combinations
    records = []
    for a,b in combinations([c for c in profile_cols if c in df.columns],2):
        g = (df.dropna(subset=["approval_label"]).groupby([a,b], dropna=False).agg(applications=("approval_label","size"), approvals=("approval_label", lambda s: s.dropna().astype(int).sum())).reset_index())
        if g.empty:
            continue
        g["approval_rate"] = g.apply(lambda r: float(r["approvals"]) / r["applications"] if r["applications"]>0 else np.nan, axis=1)
        # Rank pairwise groups primarily by approval rate (desc), then by applications (desc)
        g = g[g["applications"]>=min_support].sort_values(["approval_rate","applications"], ascending=[False,False]).head(10)
        for _,row in g.iterrows():
            records.append({"pair":f"{a} | {b}", a:row[a], b:row[b], "applications":int(row["applications"]), "approvals":int(row["approvals"]), "approval_rate":row["approval_rate"]})
    pairwise_df = pd.DataFrame(records)
    if not pairwise_df.empty:
        pairwise_df = pairwise_df.sort_values(["approval_rate", "applications"], ascending=[False, False]).reset_index(drop=True)
    pairwise_df.to_csv(output_dir / "pairwise_stats.csv", index=False)

    # Pairwise stats restricted to self-employed rows
    try:
        se_records = []
        # determine which profile columns to use for pairwise among self-employed
        se_pair_cols = [c for c in (se_profile_cols if 'se_profile_cols' in locals() else profile_cols) if c in se.columns]
        if se_pair_cols and not se.empty:
            for a, b in combinations(se_pair_cols, 2):
                g = (se.dropna(subset=["approval_label"]).groupby([a, b], dropna=False).agg(applications=("approval_label", "size"), approvals=("approval_label", lambda s: s.dropna().astype(int).sum())).reset_index())
                if g.empty:
                    continue
                g["approval_rate"] = g.apply(lambda r: float(r["approvals"]) / r["applications"] if r["applications"] > 0 else np.nan, axis=1)
                g = g[g["applications"] >= min_support].sort_values(["approval_rate", "applications"], ascending=[False, False]).reset_index(drop=True)
                rows = []
                for _, row in g.iterrows():
                    rows.append({
                        a: (None if pd.isna(row[a]) else row[a]),
                        b: (None if pd.isna(row[b]) else row[b]),
                        "applications": int(row["applications"]),
                        "approvals": int(row["approvals"]),
                        "approval_rate": (None if pd.isna(row.get("approval_rate")) else float(row.get("approval_rate")))
                    })
                if rows:
                    se_records.append({"pair": f"{a} | {b}", "rows": rows})
        # write CSV for convenience
        if se_records:
            # flatten to csv rows for output file
            flat_rows = []
            for rec in se_records:
                pair_label = rec.get("pair")
                for r in rec.get("rows", []):
                    flat = {"pair": pair_label}
                    flat.update(r)
                    flat_rows.append(flat)
            pd.DataFrame(flat_rows).to_csv(output_dir / "pairwise_stats_self_employed.csv", index=False)
        # attach to rebuild JSON output
        out["pairwise_self_employed"] = se_records
    except Exception:
        # non-fatal: continue without self-employed pairwise
        out["pairwise_self_employed"] = []

    # write md + json
    md = ["# Rebuilt Profile Answers", ""]

    def _append_table(title: str, df_sel: pd.DataFrame | list | None):
        md.append(f"## {title}")
        if df_sel is None:
            md.append("No data available.")
            md.append("")
            return
        if isinstance(df_sel, list):
            try:
                df_sel = pd.DataFrame(df_sel)
            except Exception:
                md.append("No tabular data.")
                md.append("")
                return
        if isinstance(df_sel, pd.DataFrame) and not df_sel.empty:
            if "profile" not in df_sel.columns and "profile_label" in df_sel.columns:
                df_sel = df_sel.rename(columns={"profile_label": "profile"})
            display_cols = [c for c in ("profile", "applications", "approvals", "approval_rate", "avg_approved_amount") if c in df_sel.columns]
            if not display_cols:
                # fallback to JSON-ish lines
                for _, r in df_sel.iterrows():
                    md.append(str(r.to_dict()))
                md.append("")
                return
            md.append(dataframe_to_markdown(_sanitize_df_for_md(df_sel.loc[:, display_cols].reset_index(drop=True)), max_rows=10))
        else:
            md.append("No data available.")
        md.append("")

    _append_table("1) Most likely to get approved", out.get("most_likely_approved"))
    _append_table("2) Top 3 client profiles with the highest approval", out.get("top_3_highest_approval"))
    _append_table("3) Most likely to get approval ABOVE RM30,000", out.get("likely_approved_above_30k"))
    _append_table("4) Most likely to get approval BELOW RM30,000", out.get("likely_approved_below_30k"))

    # Self-employed section (may be a single record or stats)
    if out.get("self_employed_best"):
        _append_table("5) Self-employed client profile most likely to get approved", out.get("self_employed_best"))
    else:
        stats = out.get("self_employed_stats", {})
        md.append("## 5) Self-employed client profile most likely to get approved")
        md.append(f"Self-employed share: {stats.get('self_percent',0):.1f}%; approval_rate: {stats.get('approval_rate')}")
        md.append("")

    _append_table("6) Ideal client profile for approvals at RM67,000", out.get("ideal_for_67000"))
    _append_table("7) Client profile with the most rejection", out.get("most_rejected"))

    md.append("## Pairwise stats (top pairs written to pairwise_stats.csv)")

    (output_dir / "profile_answers_rebuild.md").write_text("\n".join(md), encoding="utf-8")
    # Ensure JSON is valid by converting NaN/NA values to null (None) before writing.
    def _clean(v):
        if isinstance(v, dict):
            return {k: _clean(val) for k, val in v.items()}
        if isinstance(v, list):
            return [_clean(i) for i in v]
        try:
            if pd.isna(v):
                return None
        except Exception:
            pass
        return v

    safe_out = _clean(out)
    (output_dir / "profile_answers_rebuild.json").write_text(json.dumps(safe_out, indent=2), encoding="utf-8")


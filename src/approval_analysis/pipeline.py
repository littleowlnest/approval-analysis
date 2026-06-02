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

ID_CANDIDATES = {"no", "date of applied", "name", "i/c", "ic", "contact", "e-mail address", "email address", "email", "status", "approved amount", "ctos & doc", "commend", "agent"}

PROFILE_CANDIDATES = [
    "age",
    "gender",
    "income",
    "monthly income",
    "net income",
    "salary",
    "credit score",
    "cibil",
    "occupation",
    "education",
    "industry",
    "employment type",
    "employment status",
    "self-employed",
    "self employed",
    "state",
    "marital status",
    "tenure",
    "loan purpose",
]

MONTH_NAME_PATTERN = re.compile(r"^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[-_ ]?\d{2}$", re.IGNORECASE)


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

    sheets = load_workbook_sheets(workbook)
    if not sheets:
        raise ValueError(f"No usable sheets were found in {workbook.name}.")

    raw = pd.concat(sheets, ignore_index=True)
    normalized = normalize_columns(raw)
    enriched = derive_ic_features(normalized)

    if "status" not in enriched.columns:
        raise ValueError("The workbook must contain a Status column or an equivalent label column.")

    if "approved amount" not in enriched.columns:
        raise ValueError("The workbook must contain an Approved Amount column for amount-band analysis.")

    enriched = enriched.copy()
    enriched["approval_label"] = enriched["status"].map(normalize_status)
    enriched["approved_amount_numeric"] = parse_money_series(enriched["approved amount"])

    profile_columns = detect_profile_columns(enriched)
    if not profile_columns:
        raise ValueError(
            "No usable profile columns were found. Add fields such as Age, Gender, Income, Credit Score, Occupation, Education, or Industry."
        )

    summary = build_profile_summary(enriched, profile_columns)
    classifier_bundle = train_approval_model(enriched, profile_columns)
    regressor_bundle = train_amount_model(enriched, profile_columns)

    scored = score_records(enriched, profile_columns, classifier_bundle, regressor_bundle)
    report = build_report(enriched, summary, scored, profile_columns, classifier_bundle, regressor_bundle)

    summary_path = output_dir / "profile_summary.csv"
    report_path = output_dir / "analysis_report.md"
    metrics_path = output_dir / "model_metrics.json"
    scored_path = output_dir / "scored_records.csv"

    summary.to_csv(summary_path, index=False)
    report_path.write_text(report, encoding="utf-8")
    write_metrics(metrics_path, classifier_bundle, regressor_bundle, profile_columns, enriched)
    scored.to_csv(scored_path, index=False)

    return AnalysisResult(
        message=f"Analysis complete for {workbook.name} across {len(sheets)} sheet(s). Outputs written to {output_dir}.",
        report_path=report_path,
    )


def resolve_input_file(input_path: Path | None, input_dir: Path) -> Path:
    if input_path is not None:
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        return input_path

    preferred = input_dir / "submissions.xlsx"
    if preferred.exists():
        return preferred

    candidates = sorted(
        [path for path in input_dir.glob("*.xlsx") if not path.name.startswith("~$")],
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No .xlsx files found in {input_dir}")
    return candidates[0]


def load_workbook(path: Path) -> pd.DataFrame:
    return pd.read_excel(path)


def load_workbook_sheets(path: Path) -> list[pd.DataFrame]:
    workbook = pd.ExcelFile(path)
    sheets: list[pd.DataFrame] = []

    for sheet_name in workbook.sheet_names:
        if not is_expected_month_sheet_name(sheet_name):
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

    ic_values = result[ic_column].astype(str).str.replace(r"\D", "", regex=True)
    birth_dates = []
    genders = []
    today = datetime.today()

    for value in ic_values:
        if len(value) >= 12:
            yy = int(value[0:2])
            mm = int(value[2:4])
            dd = int(value[4:6])
            century = 2000 if yy <= int(str(today.year)[2:]) else 1900
            year = century + yy
            try:
                birth_dates.append(pd.Timestamp(year=year, month=mm, day=dd))
            except ValueError:
                birth_dates.append(pd.NaT)
            genders.append("Male" if int(value[-1]) % 2 == 1 else "Female")
        else:
            birth_dates.append(pd.NaT)
            genders.append(np.nan)

    if "date_of_birth" not in result.columns:
        result["date_of_birth"] = birth_dates
    if "age" not in result.columns:
        result["age"] = [
            today.year - date.year - ((today.month, today.day) < (date.month, date.day))
            if pd.notna(date)
            else np.nan
            for date in birth_dates
        ]
    if "gender" not in result.columns:
        result["gender"] = genders
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


def parse_money_series(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace(r"[RM,\s]", "", regex=True)
        .str.replace(r"[^0-9.\-]", "", regex=True)
        .replace({"": np.nan, "nan": np.nan, "None": np.nan})
    )
    return pd.to_numeric(cleaned, errors="coerce")


def detect_profile_columns(frame: pd.DataFrame) -> list[str]:
    candidates = []
    for column in frame.columns:
        if column in {"status", "approved amount", "approval_label", "approved_amount_numeric", "source_sheet"}:
            continue
        if column in ID_CANDIDATES:
            continue
        normalized = column.strip().lower()
        if normalized in PROFILE_CANDIDATES:
            candidates.append(column)
            continue
        if normalized.startswith("age") or normalized.startswith("gender") or normalized.startswith("income"):
            candidates.append(column)
            continue
        if any(token in normalized for token in ("credit", "occupation", "education", "industry", "employment", "self-employed", "self employed", "marital", "state", "purpose", "income", "salary", "score")):
            candidates.append(column)
    if "age" in frame.columns and "age" not in candidates:
        candidates.append("age")
    if "gender" in frame.columns and "gender" not in candidates:
        candidates.append("gender")
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


def _profile_text(row: pd.Series, profile_columns: list[str]) -> str:
    parts = []
    for column in profile_columns:
        value = row.get(column)
        if pd.notna(value):
            parts.append(f"{column}={value}")
    return " | ".join(parts)


def dataframe_to_markdown(frame: pd.DataFrame, max_rows: int = 25) -> str:
    if frame.empty:
        return "| |\n| --- |"

    display_frame = frame.head(max_rows).copy().fillna("")
    columns = [str(column) for column in display_frame.columns]

    def escape(value: Any) -> str:
        return str(value).replace("|", "\\|")

    lines = ["| " + " | ".join(columns) + " |"]
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for _, row in display_frame.iterrows():
        lines.append("| " + " | ".join(escape(row[column]) for column in display_frame.columns) + " |")
    return "\n".join(lines)


def build_report(
    frame: pd.DataFrame,
    summary: pd.DataFrame,
    scored: pd.DataFrame,
    profile_columns: list[str],
    classifier_bundle: ModelBundle,
    regressor_bundle: ModelBundle,
) -> str:
    def top_rows(source: pd.DataFrame, limit: int = 3) -> pd.DataFrame:
        return source.head(limit).copy()

    approved_only = summary[summary["approvals"] > 0].copy()
    high_amount = summary[summary["avg_approved_amount"].ge(30000)].copy()
    low_amount = summary[summary["avg_approved_amount"].lt(30000)].copy()
    self_employed = summary[
        summary.apply(
            lambda row: any(
                str(row.get(column, "")).strip().lower() in {"self-employed", "self employed", "yes", "y", "true", "1"}
                for column in profile_columns
            ),
            axis=1,
        )
    ].copy()

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
        "## Answers",
    ]

    sections = [
        ("1) Most likely to get approved", approved_only),
        ("2) Top 3 client profiles with the highest approval", approved_only),
        ("3) Most likely to get approval above RM30,000", high_amount),
        ("4) Most likely to get approval below RM30,000", low_amount),
        ("5) Self-employed client profile most likely to get approved", self_employed),
        ("6) Ideal client profile for approvals at RM67,000", target_67000),
        ("7) Client profile with the most rejection", rejection_rank),
    ]

    for title, table in sections:
        lines.extend([f"### {title}"])
        if table.empty:
            lines.append("No ranked profile could be derived from the available data.")
            lines.append("")
            continue

        selected = top_rows(table, 3)
        for _, row in selected.iterrows():
            profile = _profile_text(row, profile_columns)
            lines.append(
                f"- {profile} | approval_rate={row.get('approval_rate', np.nan):.3f} | applications={int(row.get('applications', 0))} | approvals={int(row.get('approvals', 0))} | avg_approved_amount={row.get('avg_approved_amount', np.nan):.2f}"
            )
        lines.append("")

    lines.extend(
        [
            "## Ranking Table",
            dataframe_to_markdown(summary, max_rows=25),
            "",
            "## Scored Rows",
            dataframe_to_markdown(scored, max_rows=20),
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

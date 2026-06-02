# Approval Analysis Project

Drop an `.xlsx` file into `input/` and run the analysis script to generate ranked client profiles, approval/rejection summaries, and amount-band views.

## What it does

- Reads an Excel workbook from `input/`, preferring `input/submissions.xlsx` when present
- Validates the expected loan/application columns
- Uses any available profile fields such as age, gender, income, credit score, occupation, education, industry, and self-employment
- Derives age and gender from Malaysian IC numbers when possible
- Trains a simple approval classifier when a usable `Status` label exists
- Trains an amount regressor when `Approved Amount` values exist
- Writes CSV and Markdown outputs into `output/`
- Processes multiple sheets in workbook order
- Expects monthly sheet names like `Jan-26`, `Feb-26`, or `Mar 26`
- Stops processing when it reaches the first sheet with no data rows

## Folder Layout

- `input/` - place source `.xlsx` files here
- `output/` - generated results are written here
- `src/approval_analysis/` - analysis code

## Setup

1. Create and activate the virtual environment:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

2. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

## Run

Place your spreadsheet in `input/`, then run:

```powershell
python run_analysis.py
```

Or pass a specific file:

```powershell
python run_analysis.py --input input\your_file.xlsx
```

## Outputs

The script writes these files into `output/`:

- `analysis_report.md` - human-readable summary of the ranked profile cases
- `profile_summary.csv` - ranked profile table
- `model_metrics.json` - model quality and data coverage details
- `scored_records.csv` - row-level scoring output when a model can be trained

## Notes

- The spreadsheet may contain only the administrative columns you listed, but the analysis is stronger when it also includes profile fields such as age, gender, income, credit score, occupation, education, industry, and employment type.
- If profile fields are missing, the script still uses any available columns and tries to derive age and gender from `I/C` where possible.
- `Status` should indicate approval/rejection in a consistent way, such as `Approved`, `Rejected`, `Yes`, `No`, `1`, or `0`.
- If the workbook contains multiple sheets, each sheet should use a month-year name and include its own header row.

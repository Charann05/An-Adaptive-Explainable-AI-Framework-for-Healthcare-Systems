from pathlib import Path
import pandas as pd
from . import constants
from .. import cli_display

DATASET_EXTENSIONS = {
    ".csv",
    ".xls",
    ".xlsx"
}

TARGET_COLUMN_NAMES = {
    "target",
    "label",
    "class",
    "output",
    "outcome",
    "response",
    "y",
    "diagnosis",
    "disease",
    "prediction"
}

def scan_dataset(MODEL_FOLDER, evidences, file_info):
    cli_display.print_status("Scanning dataset files")
    for file in file_info:
        extension = file["extension"].lower()
        if extension not in DATASET_EXTENSIONS:
            continue

        file_path = Path(MODEL_FOLDER) / file["name"]
        print("Dataset found:", file_path)

        has_header = file.get("has_header", has_header_heuristic(file_path, extension))
        if not has_header:
            print("Dataset has no header. Skipping column evidence.")
            return

        dataframe = read_dataset(file_path, extension, has_header=True)

        if dataframe is None or dataframe.empty:
            if dataframe is not None and dataframe.empty:
                print("Dataset is empty:", file_path)
            continue

        columns = clean_columns(dataframe.columns)
        if not columns:
            continue

        target_column = detect_target_column(columns)
        feature_names = [column for column in columns if column != target_column]

        evidences["dataset_columns"].append({
            "value": columns,
            "source": "dataset",
            "weight": constants.SOURCE_WEIGHTS["dataset"]
        })

        if target_column:
            evidences["target_column"].append({
                "value": target_column,
                "source": "dataset",
                "weight": constants.SOURCE_WEIGHTS["dataset"]
            })

        if feature_names:
            evidences["feature_names"].append({
                "value": feature_names,
                "source": "dataset",
                "weight": constants.SOURCE_WEIGHTS["dataset"]
            })
            evidences["num_inputs"].append({
                "value": len(feature_names),
                "source": "dataset",
                "weight": constants.SOURCE_WEIGHTS["dataset"]
            })

        return

def has_header_heuristic(file_path, extension):
    try:
        if extension == ".csv":
            df_sample = pd.read_csv(file_path, nrows=2, header=None)
        else:
            df_sample = pd.read_excel(file_path, nrows=2, header=None)

        if df_sample.empty:
            return True

        first_row = df_sample.iloc[0]
        numeric_count = sum(pd.to_numeric(first_row, errors="coerce").notna())
        return (numeric_count / len(first_row)) < 0.5
    except Exception:
        return True

def read_dataset(file_path, extension, has_header=True):
    header_arg = 0 if has_header else None
    try:
        if extension == ".csv":
            return pd.read_csv(file_path, header=header_arg)
        if extension in {".xls", ".xlsx"}:
            return pd.read_excel(file_path, header=header_arg)
    except Exception as e:
        print(f"Error reading dataset {file_path}: {e}")
    return None

def clean_columns(columns):
    cleaned_columns = []
    for column in columns:
        if isinstance(column, str):
            column = column.strip()
            if not column:
                continue
        if column not in cleaned_columns:
            cleaned_columns.append(column)
    return cleaned_columns

def detect_target_column(columns):
    for column in columns:
        if isinstance(column, str):
            normalized = (
                column
                .strip()
                .lower()
                .replace(" ", "_")
                .replace("-", "_")
            )
            if normalized in TARGET_COLUMN_NAMES:
                return column
    return None
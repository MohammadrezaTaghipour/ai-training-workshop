"""UI-free table loading and preparation (Session 9: Pandas, missingno).

Callers pass bytes, paths, or file objects. Functions return new frames
and do not write files.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import BinaryIO, Literal

import pandas as pd

FillStrategy = Literal["mean", "median", "mode"]
MergeHow = Literal["inner", "left", "right", "outer"]
FilterOp = Literal["contains", "equals", "at least", "at most"]

SUPPORTED_EXTENSIONS = ("csv", "tsv", "txt")
MAX_UPLOAD_MB = 20


class TableError(ValueError):
    """A table could not be read or transformed."""


def upload_allowed(filename: str) -> bool:
    """CSV, TSV, TXT, or a delimited file with no extension (SMSSpamCollection)."""
    suffix = Path(filename).suffix.lower().lstrip(".")
    return suffix in SUPPORTED_EXTENSIONS or suffix == ""


def suggest_read_options(filename: str, sample: bytes) -> tuple[str, bool]:
    """Pick a delimiter label and header flag from the file name and a sample.

    Session 9 reads SMSSpamCollection with a tab delimiter and ``header=None``.
    That file has no extension, so the name alone cannot tell us the format.
    """
    suffix = Path(filename).suffix.lower()
    if suffix == ".csv":
        return "Comma", True
    if suffix == ".tsv":
        return "Tab", True

    text = sample[:16384].decode("utf-8", errors="replace")
    lines = [line for line in text.splitlines() if line.strip()][:30]
    if not lines:
        return "Comma", True

    tab_hits = sum("\t" in line for line in lines)
    if tab_hits >= max(1, len(lines) // 2):
        # Extensionless course files such as SMSSpamCollection have no header.
        header = suffix != ""
        return "Tab", header

    semi_hits = sum(line.count(";") > line.count(",") for line in lines)
    if semi_hits >= max(1, len(lines) // 2):
        return "Semicolon", True
    return "Comma", True


def read_table(
    source: str | Path | bytes | BinaryIO,
    *,
    delimiter: str,
    header: bool,
    encoding: str = "utf-8",
) -> pd.DataFrame:
    """Read a delimited text table the way Session 9 used ``read_csv``."""
    if isinstance(source, bytes):
        source = BytesIO(source)
    header_row: int | None = 0 if header else None
    try:
        frame = pd.read_csv(
            source,
            sep=delimiter,
            header=header_row,
            encoding=encoding,
        )
    except UnicodeDecodeError as exc:
        raise TableError(
            f"Could not read the file as {encoding}. Try another encoding."
        ) from exc
    except pd.errors.EmptyDataError as exc:
        raise TableError("The file is empty.") from exc
    except pd.errors.ParserError as exc:
        raise TableError(
            "Could not parse the file. Check the delimiter and whether the first row is a header."
        ) from exc
    except OSError as exc:
        raise TableError(f"Could not read the file: {exc}") from exc

    if frame.shape[1] == 0:
        raise TableError("The file has no columns.")
    if not header:
        frame.columns = [f"column_{index}" for index in range(frame.shape[1])]
    return frame


def overview(frame: pd.DataFrame) -> dict[str, int]:
    """Row, column, missing-cell, and numeric-column counts."""
    return {
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "missing_cells": int(frame.isna().sum().sum()),
        "numeric_columns": int(frame.select_dtypes(include="number").shape[1]),
    }


def column_profile(series: pd.Series) -> dict[str, object]:
    """Session 9 Series summaries for one column."""
    missing = int(series.isna().sum())
    profile: dict[str, object] = {
        "dtype": str(series.dtype),
        "count": int(series.count()),
        "missing": missing,
        "unique": int(series.nunique(dropna=True)),
    }
    modes = series.mode(dropna=True)
    profile["mode"] = None if modes.empty else modes.iloc[0]
    if pd.api.types.is_numeric_dtype(series):
        profile["min"] = series.min()
        profile["max"] = series.max()
        profile["mean"] = series.mean()
        profile["median"] = series.median()
        profile["sum"] = series.sum()
    return profile


def schema_table(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per column: dtype, non-null count, missing count."""
    return pd.DataFrame(
        {
            "column": frame.columns.astype(str),
            "dtype": frame.dtypes.astype(str).to_numpy(),
            "non_null": frame.notna().sum().to_numpy(),
            "missing": frame.isna().sum().to_numpy(),
        }
    )


def drop_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if not columns:
        raise TableError("Choose at least one column to drop.")
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise TableError(f"Unknown columns: {', '.join(missing)}")
    if len(columns) >= frame.shape[1]:
        raise TableError("Keep at least one column.")
    return frame.drop(columns=columns)


def drop_missing_rows(
    frame: pd.DataFrame,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Drop rows with missing values. ``columns`` limits the check (``dropna`` subset)."""
    if columns:
        missing = [name for name in columns if name not in frame.columns]
        if missing:
            raise TableError(f"Unknown columns: {', '.join(missing)}")
        return frame.dropna(subset=columns)
    return frame.dropna()


def fill_missing(
    frame: pd.DataFrame,
    column: str,
    strategy: FillStrategy,
) -> pd.DataFrame:
    """Fill empty cells in one column with mean, median, or mode."""
    if column not in frame.columns:
        raise TableError(f"Unknown column: {column}")
    series = frame[column]
    if strategy in {"mean", "median"} and not pd.api.types.is_numeric_dtype(series):
        raise TableError(f"{strategy.capitalize()} fill needs a numeric column.")
    if strategy == "mean":
        value = series.mean()
    elif strategy == "median":
        value = series.median()
    else:
        modes = series.mode(dropna=True)
        if modes.empty:
            raise TableError(f"Column {column} has no value to use as the mode.")
        value = modes.iloc[0]
    if pd.isna(value):
        raise TableError(f"Column {column} has no values to compute {strategy}.")
    result = frame.copy()
    result[column] = series.fillna(value)
    return result


def sort_frame(frame: pd.DataFrame, column: str, *, ascending: bool) -> pd.DataFrame:
    if column not in frame.columns:
        raise TableError(f"Unknown column: {column}")
    return frame.sort_values(column, ascending=ascending, kind="mergesort")


def filter_rows(
    frame: pd.DataFrame,
    column: str,
    op: FilterOp,
    raw_value: str,
) -> pd.DataFrame:
    """Keep rows that match one comparison. Text match is case-insensitive."""
    if column not in frame.columns:
        raise TableError(f"Unknown column: {column}")
    text = raw_value.strip()
    if text == "":
        raise TableError("Enter a value to filter on.")
    series = frame[column]
    if op == "contains":
        mask = series.astype("string").str.contains(text, case=False, na=False)
        return frame.loc[mask]
    if op == "equals":
        if pd.api.types.is_numeric_dtype(series):
            number = _as_number(text)
            mask = series == number
        else:
            mask = series.astype("string").str.casefold() == text.casefold()
        return frame.loc[mask]
    number = _as_number(text)
    if not pd.api.types.is_numeric_dtype(series):
        raise TableError("At least / at most need a numeric column.")
    if op == "at least":
        return frame.loc[series >= number]
    return frame.loc[series <= number]


def concat_tables(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """Stack two tables and reset the index (Session 9 ``concat``)."""
    return pd.concat([left, right], ignore_index=True)


def merge_tables(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    on: str,
    how: MergeHow,
) -> pd.DataFrame:
    """Join two tables on a shared column name."""
    if on not in left.columns or on not in right.columns:
        raise TableError(f"Both tables need a column named {on}.")
    return pd.merge(left, right, how=how, on=on, suffixes=("", "_right"))


def correlation(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = frame.select_dtypes(include="number")
    if numeric.shape[1] < 2:
        raise TableError("Correlation needs at least two numeric columns.")
    return numeric.corr(numeric_only=True)


def missingness_figure(frame: pd.DataFrame):
    """Missing-value matrix from missingno. Returns a Matplotlib figure."""
    import matplotlib.pyplot as plt
    import missingno as msno

    fig, ax = plt.subplots(figsize=(10, 4))
    msno.matrix(frame, ax=ax, sparkline=False)
    fig.tight_layout()
    return fig


def correlation_figure(frame: pd.DataFrame):
    """Seaborn heatmap of the numeric correlation matrix."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    matrix = correlation(frame)
    fig, ax = plt.subplots(figsize=(8, 6))
    annotate = matrix.shape[1] <= 12
    sns.heatmap(matrix, ax=ax, annot=annotate, cmap="Blues", fmt=".2f")
    fig.tight_layout()
    return fig


def to_csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8")


def _as_number(text: str) -> float:
    try:
        return float(text)
    except ValueError as exc:
        raise TableError(f"{text!r} is not a number.") from exc

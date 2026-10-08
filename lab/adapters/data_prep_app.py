"""Streamlit data-prep studio for Session 9 table tools.

Run from the repo root:

    uv run streamlit run lab/adapters/data_prep_app.py

Uploads stay in the browser session. Nothing is written to disk.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from core.tabular.tables import (
    MAX_UPLOAD_MB,
    TableError,
    column_profile,
    concat_tables,
    correlation,
    correlation_figure,
    drop_columns,
    drop_missing_rows,
    fill_missing,
    filter_rows,
    merge_tables,
    missingness_figure,
    overview,
    read_table,
    schema_table,
    sort_frame,
    suggest_read_options,
    to_csv_bytes,
    upload_allowed,
)

_SAMPLE = (
    Path(__file__).resolve().parents[2]
    / "sessions"
    / "session9"
    / "datasets"
    / "titanic.csv"
)
_DELIMITERS = {
    "Comma": ",",
    "Tab": "\t",
    "Semicolon": ";",
    "Pipe": "|",
}
_ENCODINGS = ("utf-8", "utf-8-sig", "latin-1")


def _init_state() -> None:
    st.session_state.setdefault("frame", None)
    st.session_state.setdefault("source_name", None)
    st.session_state.setdefault("pending_bytes", None)
    st.session_state.setdefault("parsed_settings", None)
    st.session_state.setdefault("applied_upload", None)
    st.session_state.setdefault("load_error", None)
    st.session_state.setdefault("delimiter_label", "Comma")
    st.session_state.setdefault("header_row", True)
    st.session_state.setdefault("encoding_name", "utf-8")
    st.session_state.setdefault("second_delimiter", "Comma")
    st.session_state.setdefault("second_header", True)


def _remember_upload(upload) -> None:
    """Store a new upload and choose delimiter/header the way Session 9 did."""
    raw = upload.getvalue()
    if not upload_allowed(upload.name):
        st.session_state.load_error = (
            f"{upload.name} is not a supported table. Use CSV, TSV, TXT, "
            "or a delimited file with no extension."
        )
        st.session_state.pending_bytes = None
        st.session_state.frame = None
        return
    label, header = suggest_read_options(upload.name, raw)
    st.session_state.delimiter_label = label
    st.session_state.header_row = header
    st.session_state.encoding_name = "utf-8"
    st.session_state.applied_upload = f"{upload.name}:{upload.size}"
    st.session_state.pending_bytes = raw
    st.session_state.source_name = upload.name
    st.session_state.parsed_settings = None
    st.session_state.load_error = None


def _parse_pending() -> None:
    raw = st.session_state.get("pending_bytes")
    label = st.session_state.get("delimiter_label") or "Comma"
    header = bool(st.session_state.get("header_row", True))
    encoding = st.session_state.get("encoding_name") or "utf-8"
    settings = (label, header, encoding)
    if raw is None or settings == st.session_state.get("parsed_settings"):
        return
    try:
        frame = read_table(
            raw,
            delimiter=_DELIMITERS[label],
            header=header,
            encoding=encoding,
        )
    except TableError as exc:
        st.session_state.load_error = str(exc)
        return
    st.session_state.frame = frame
    st.session_state.parsed_settings = settings
    st.session_state.load_error = None


def _apply(action, success: str) -> None:
    frame = st.session_state.frame
    if frame is None:
        return
    try:
        st.session_state.frame = action(frame)
    except TableError as exc:
        st.session_state.notice_error = str(exc)
    else:
        st.session_state.notice = success
    st.rerun()


def _sidebar() -> None:
    with st.sidebar:
        st.header("Data")
        st.caption(
            f"CSV, TSV, TXT, or a delimited file with no extension. "
            f"Up to {MAX_UPLOAD_MB} MB. The table stays in this browser session."
        )
        upload = st.file_uploader(
            "Upload a table",
            max_upload_size=MAX_UPLOAD_MB,
            help="SMSSpamCollection is tab-separated and has no header. The app detects that.",
        )
        if st.button("Load Titanic sample", icon=":material/school:", width="stretch"):
            try:
                frame = read_table(_SAMPLE, delimiter=",", header=True, encoding="utf-8")
            except TableError as exc:
                st.session_state.load_error = str(exc)
            else:
                st.session_state.frame = frame
                st.session_state.source_name = "titanic.csv (sample)"
                st.session_state.pending_bytes = None
                st.session_state.parsed_settings = ("sample",)
                # Match the current upload stamp so this click is not replaced
                # by the file already sitting in the uploader.
                if upload is not None:
                    st.session_state.applied_upload = f"{upload.name}:{upload.size}"
                st.session_state.load_error = None
                st.session_state.notice = "Loaded the Titanic sample."
            st.rerun()

        if upload is not None:
            stamp = f"{upload.name}:{upload.size}"
            if stamp != st.session_state.get("applied_upload"):
                _remember_upload(upload)

        st.segmented_control(
            "Delimiter",
            list(_DELIMITERS),
            required=True,
            key="delimiter_label",
            width="stretch",
        )
        st.checkbox("First row is the header", key="header_row")
        st.selectbox("Encoding", _ENCODINGS, key="encoding_name")
        _parse_pending()
        if st.session_state.load_error:
            st.error(st.session_state.load_error)


def _metrics(frame: pd.DataFrame) -> None:
    stats = overview(frame)
    with st.container(horizontal=True):
        st.metric("Rows", f"{stats['rows']:,}", border=True)
        st.metric("Columns", f"{stats['columns']:,}", border=True)
        st.metric("Missing cells", f"{stats['missing_cells']:,}", border=True)
        st.metric("Numeric columns", f"{stats['numeric_columns']:,}", border=True)


def _preview(frame: pd.DataFrame) -> None:
    view = st.segmented_control(
        "Rows",
        ["First", "Last"],
        default="First",
        required=True,
        key="preview_edge",
    )
    count = st.slider("How many rows", min_value=5, max_value=100, value=10, key="preview_count")
    shown = frame.head(count) if view == "First" else frame.tail(count)
    st.dataframe(shown, hide_index=True)
    with st.container(border=True):
        st.subheader("Columns")
        st.dataframe(schema_table(frame), hide_index=True)
    numeric = frame.select_dtypes(include="number")
    if not numeric.empty:
        with st.container(border=True):
            st.subheader("Numeric summary")
            st.dataframe(numeric.describe().T, hide_index=False)


def _column_tab(frame: pd.DataFrame) -> None:
    column = st.selectbox("Column", list(frame.columns), key="profile_column")
    series = frame[column]
    profile = column_profile(series)
    with st.container(horizontal=True):
        st.metric("Non-null", f"{profile['count']:,}", border=True)
        st.metric("Missing", f"{profile['missing']:,}", border=True)
        st.metric("Unique", f"{profile['unique']:,}", border=True)
    rows = [
        {"stat": key, "value": "" if value is None else str(value)}
        for key, value in profile.items()
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    counts = series.astype("string").fillna("(missing)").value_counts().head(15)
    chart = counts.rename_axis("value").reset_index(name="count")
    st.bar_chart(chart, x="value", y="count")
    ascending = st.toggle("Sort ascending", value=True, key="sort_ascending")
    if st.button("Sort table by this column", icon=":material/sort:"):
        _apply(
            lambda current: sort_frame(current, column, ascending=ascending),
            f"Sorted by {column}.",
        )


def _missing_tab(frame: pd.DataFrame) -> None:
    missing = (
        frame.isna()
        .sum()
        .rename_axis("column")
        .reset_index(name="missing")
        .sort_values("missing", ascending=False)
    )
    st.bar_chart(missing, x="column", y="missing")
    if int(missing["missing"].sum()) == 0:
        st.info("This table has no missing values.")
        return
    figure = missingness_figure(frame)
    st.pyplot(figure)
    plt.close(figure)


def _clean_tab(frame: pd.DataFrame) -> None:
    st.subheader("Drop columns")
    drop_names = st.multiselect(
        "Columns to drop",
        list(frame.columns),
        key="drop_names",
    )
    if st.button("Drop columns", icon=":material/delete:"):
        _apply(lambda current: drop_columns(current, drop_names), "Dropped columns.")

    st.subheader("Missing rows")
    subset = st.multiselect(
        "Only consider these columns",
        list(frame.columns),
        key="dropna_subset",
        help="Leave empty to drop a row when any cell is missing.",
    )
    if st.button("Drop rows with missing values"):
        columns = subset or None
        _apply(
            lambda current: drop_missing_rows(current, columns),
            "Dropped rows with missing values.",
        )

    st.subheader("Fill missing values")
    fill_column = st.selectbox("Column to fill", list(frame.columns), key="fill_column")
    strategy = st.segmented_control(
        "Fill with",
        ["mean", "median", "mode"],
        default="median",
        required=True,
        key="fill_strategy",
    )
    if st.button("Fill missing values", icon=":material/healing:"):
        _apply(
            lambda current: fill_missing(current, fill_column, strategy),
            f"Filled {fill_column} with the {strategy}.",
        )

    st.subheader("Keep matching rows")
    with st.form("row_filter"):
        filter_column = st.selectbox("Column", list(frame.columns), key="filter_column")
        filter_op = st.selectbox(
            "Keep rows where the value",
            ["contains", "equals", "at least", "at most"],
            key="filter_op",
        )
        filter_value = st.text_input("Value", key="filter_value")
        submitted = st.form_submit_button("Apply filter")
    if submitted:
        _apply(
            lambda current: filter_rows(current, filter_column, filter_op, filter_value),
            "Filtered rows.",
        )


def _combine_tab(frame: pd.DataFrame) -> None:
    st.caption("Upload a second table and stack it (concat) or join it (merge).")
    other_file = st.file_uploader(
        "Second table",
        max_upload_size=MAX_UPLOAD_MB,
        key="second_upload",
        help="CSV, TSV, TXT, or a delimited file with no extension.",
    )
    if other_file is not None:
        second_stamp = f"{other_file.name}:{other_file.size}"
        if second_stamp != st.session_state.get("applied_second"):
            if upload_allowed(other_file.name):
                label, header = suggest_read_options(other_file.name, other_file.getvalue())
                st.session_state.second_delimiter = label
                st.session_state.second_header = header
                st.session_state.second_error = None
            else:
                st.session_state.second_error = (
                    f"{other_file.name} is not a supported table."
                )
            st.session_state.applied_second = second_stamp
    other_delim = st.segmented_control(
        "Second delimiter",
        list(_DELIMITERS),
        required=True,
        key="second_delimiter",
    )
    other_header = st.checkbox("Second file has a header", key="second_header")
    if st.session_state.get("second_error"):
        st.error(st.session_state.second_error)
    how = st.segmented_control(
        "Combine",
        ["concat", "inner", "left", "right", "outer"],
        default="concat",
        required=True,
        key="combine_how",
    )
    shared = list(frame.columns)
    on_column = st.selectbox(
        "Merge on",
        shared,
        key="merge_on",
        help="Used for inner, left, right, and outer. Both tables need this column name.",
    )
    if st.button("Combine tables", icon=":material/join_inner:"):
        if other_file is None:
            st.session_state.notice_error = "Upload a second table first."
            st.rerun()
        try:
            other = read_table(
                other_file.getvalue(),
                delimiter=_DELIMITERS[other_delim],
                header=other_header,
                encoding="utf-8",
            )
            if how == "concat":
                result = concat_tables(frame, other)
                message = "Stacked the two tables."
            else:
                result = merge_tables(frame, other, on=on_column, how=how)
                message = f"Merged with {how} join on {on_column}."
        except TableError as exc:
            st.session_state.notice_error = str(exc)
        else:
            st.session_state.frame = result
            st.session_state.notice = message
        st.rerun()


def _correlation_tab(frame: pd.DataFrame) -> None:
    try:
        matrix = correlation(frame)
        figure = correlation_figure(frame)
    except TableError as exc:
        st.info(str(exc))
        return
    st.pyplot(figure)
    plt.close(figure)
    st.dataframe(matrix)


def _download(frame: pd.DataFrame) -> None:
    stem = Path(st.session_state.source_name or "table").stem
    st.download_button(
        "Download CSV",
        data=to_csv_bytes(frame),
        file_name=f"{stem}_cleaned.csv",
        mime="text/csv",
        icon=":material/download:",
    )


def main() -> None:
    st.set_page_config(
        page_title="Data prep",
        page_icon=":material/table_view:",
        layout="wide",
    )
    _init_state()
    _sidebar()

    st.title("Data prep")
    st.caption(
        "Inspect a table, fill or drop missing values, combine two files, "
        "and download the result. Built from the Session 9 Pandas tools."
    )
    error = st.session_state.pop("notice_error", None)
    notice = st.session_state.pop("notice", None)
    if error:
        st.error(error)
    elif notice:
        st.success(notice)

    frame = st.session_state.frame
    if frame is None:
        st.info(
            "Upload a CSV, TSV, TXT, or extensionless delimited file in the sidebar, "
            "or load the Titanic sample to try the tools."
        )
        return

    st.caption(f"Working table: {st.session_state.source_name}")
    _metrics(frame)
    preview, column, missing, clean, combine, relate = st.tabs(
        [
            "Preview",
            "Column",
            "Missing values",
            "Clean",
            "Combine",
            "Correlation",
        ],
        on_change="rerun",
        key="tool_tabs",
        default="Preview",
    )
    with preview:
        if preview.open:
            _preview(frame)
    with column:
        if column.open:
            _column_tab(frame)
    with missing:
        if missing.open:
            _missing_tab(frame)
    with clean:
        if clean.open:
            _clean_tab(frame)
    with combine:
        if combine.open:
            _combine_tab(frame)
    with relate:
        if relate.open:
            _correlation_tab(frame)
    _download(frame)


main()

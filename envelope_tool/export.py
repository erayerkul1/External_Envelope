"""Export envelope and governing-subcase results to Excel or CSV."""

import os
from typing import Dict

import pandas as pd


def export_excel(
    data: Dict[str, pd.DataFrame],
    output_path: str,
    sheet_suffix: str = "",
) -> None:
    """Write each result type to a separate sheet in an Excel workbook."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for result_type, df in data.items():
            # Excel sheet names are limited to 31 chars
            sheet_name = (result_type + sheet_suffix)[:31]
            df.to_excel(writer, sheet_name=sheet_name, index=False)


def export_csv(
    data: Dict[str, pd.DataFrame],
    output_path: str,
    suffix: str = "",
) -> None:
    """Write each result type to a separate CSV file.

    Files are named  <base>_<result_type><suffix>.csv  next to output_path.
    """
    base = os.path.splitext(output_path)[0]
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    for result_type, df in data.items():
        out = f"{base}_{result_type}{suffix}.csv"
        df.to_csv(out, index=False)


def export_results(
    envelope_max: Dict[str, pd.DataFrame],
    envelope_min: Dict[str, pd.DataFrame],
    output_path: str,
    fmt: str = "xlsx",
    governing: Dict[str, pd.DataFrame] | None = None,
    governing_path: str | None = None,
) -> None:
    """Export envelope and optionally governing-subcase data.

    Parameters
    ----------
    envelope_max / envelope_min : dicts of DataFrames per result type
    output_path : path for the primary envelope output file
    fmt : 'xlsx' or 'csv'
    governing : optional governing-subcase report dict
    governing_path : output path for governing report (required if governing given)
    """
    # Merge max and min into a single dict with labelled sheets/files
    if fmt == "xlsx":
        # Combine max and min into one workbook: separate sheets
        combined: Dict[str, pd.DataFrame] = {}
        for rt, df in envelope_max.items():
            combined[f"{rt}_MAX"] = df
        for rt, df in envelope_min.items():
            combined[f"{rt}_MIN"] = df
        export_excel(combined, output_path)

        if governing and governing_path:
            export_excel(governing, governing_path)
    else:
        export_csv(envelope_max, output_path, suffix="_max")
        export_csv(envelope_min, output_path, suffix="_min")

        if governing and governing_path:
            export_csv(governing, governing_path, suffix="_governing")

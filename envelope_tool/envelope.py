"""Compute envelope (element-wise max/min) across all subcases."""

import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd


# Columns that identify an element/node rather than a numeric result
_ID_COLS = {"element_id", "node_id", "nid", "eid", "ElementID", "NodeID"}


def _id_column(df: pd.DataFrame) -> str | None:
    """Return the first ID column found in the DataFrame, or None."""
    for col in df.columns:
        if col.lower() in {c.lower() for c in _ID_COLS}:
            return col
    # Fallback: use the first column if it looks like integer IDs
    if not df.empty and pd.api.types.is_integer_dtype(df.iloc[:, 0]):
        return df.columns[0]
    return None


def _numeric_cols(df: pd.DataFrame, id_col: str) -> list:
    return [c for c in df.columns if c != id_col and pd.api.types.is_numeric_dtype(df[c])]


def compute_envelope(
    all_results: Dict[Tuple[str, int], Dict[str, pd.DataFrame]],
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """Compute per-element max and min envelope across all (file, subcase) pairs.

    Parameters
    ----------
    all_results : dict
        {(filepath, subcase_id): {result_type: DataFrame}}

    Returns
    -------
    envelope_max : dict  {result_type: DataFrame}
        For each element: max value of each numeric column, plus
        'governing_file' and 'governing_subcase' columns.
    envelope_min : dict  {result_type: DataFrame}
        Same but for min values.
    """
    # Collect all result types present
    all_types: set = set()
    for type_dict in all_results.values():
        all_types.update(type_dict.keys())

    envelope_max: Dict[str, pd.DataFrame] = {}
    envelope_min: Dict[str, pd.DataFrame] = {}

    for result_type in sorted(all_types):
        frames = []
        for (filepath, subcase_id), type_dict in all_results.items():
            if result_type not in type_dict:
                continue
            df = type_dict[result_type].copy()
            df["__file__"] = filepath
            df["__subcase__"] = subcase_id
            frames.append(df)

        if not frames:
            continue

        combined = pd.concat(frames, ignore_index=True)
        id_col = _id_column(combined)
        if id_col is None:
            continue

        num_cols = _numeric_cols(combined, id_col)
        if not num_cols:
            continue

        max_rows = []
        min_rows = []

        for elem_id, group in combined.groupby(id_col):
            max_row = {id_col: elem_id}
            min_row = {id_col: elem_id}

            for col in num_cols:
                col_vals = group[col]
                max_idx = col_vals.idxmax()
                min_idx = col_vals.idxmin()
                max_row[col] = col_vals[max_idx]
                min_row[col] = col_vals[min_idx]

            # For governing: use the subcase that produced the overall
            # largest absolute value (Von Mises-like single representative)
            abs_sums = group[num_cols].abs().sum(axis=1)
            max_gov_idx = abs_sums.idxmax()
            min_gov_idx = abs_sums.idxmin()

            max_row["governing_file"] = group.loc[max_gov_idx, "__file__"]
            max_row["governing_subcase"] = group.loc[max_gov_idx, "__subcase__"]
            min_row["governing_file"] = group.loc[min_gov_idx, "__file__"]
            min_row["governing_subcase"] = group.loc[min_gov_idx, "__subcase__"]

            max_rows.append(max_row)
            min_rows.append(min_row)

        envelope_max[result_type] = pd.DataFrame(max_rows)
        envelope_min[result_type] = pd.DataFrame(min_rows)

    return envelope_max, envelope_min


def build_governing_report(
    all_results: Dict[Tuple[str, int], Dict[str, pd.DataFrame]],
    envelope_max: Dict[str, pd.DataFrame],
    envelope_min: Dict[str, pd.DataFrame],
) -> Dict[str, pd.DataFrame]:
    """Build a detailed governing-subcase report per result type.

    For every element and every numeric column, records the governing
    (file, subcase) for both the maximum and minimum value.
    """
    all_types = set(envelope_max.keys()) | set(envelope_min.keys())
    report: Dict[str, pd.DataFrame] = {}

    for result_type in sorted(all_types):
        frames = []
        for (filepath, subcase_id), type_dict in all_results.items():
            if result_type not in type_dict:
                continue
            df = type_dict[result_type].copy()
            df["__file__"] = filepath
            df["__subcase__"] = subcase_id
            frames.append(df)

        if not frames:
            continue

        combined = pd.concat(frames, ignore_index=True)
        id_col = _id_column(combined)
        if id_col is None:
            continue

        num_cols = _numeric_cols(combined, id_col)
        if not num_cols:
            continue

        rows = []
        for elem_id, group in combined.groupby(id_col):
            for col in num_cols:
                col_vals = group[col]
                max_idx = col_vals.idxmax()
                min_idx = col_vals.idxmin()
                rows.append(
                    {
                        id_col: elem_id,
                        "result_column": col,
                        "max_value": col_vals[max_idx],
                        "max_governing_file": os.path.basename(group.loc[max_idx, "__file__"]),
                        "max_governing_subcase": group.loc[max_idx, "__subcase__"],
                        "min_value": col_vals[min_idx],
                        "min_governing_file": os.path.basename(group.loc[min_idx, "__file__"]),
                        "min_governing_subcase": group.loc[min_idx, "__subcase__"],
                    }
                )

        report[result_type] = pd.DataFrame(rows)

    return report

#!/usr/bin/env python3
"""External Envelope Tool

Reads multiple Nastran .op2 or .h5 files (each may contain multiple subcases),
computes element/node-wise max and min envelope across all subcases, and
exports the results to Excel or CSV.

Usage examples
--------------
python envelope.py results1.op2 results2.h5 --export-envelope envelope.xlsx

python envelope.py a.op2 b.op2 --result-types displacements plate_stress \\
    --export-envelope envelope.xlsx --export-governing governing.xlsx

python envelope.py a.op2 b.op2 --format csv --export-envelope envelope.csv \\
    --export-governing governing.csv
"""

import argparse
import os
import sys
from typing import Dict, Tuple

import pandas as pd
from pyNastran.op2.op2 import OP2

# ---------------------------------------------------------------------------
# Result type definitions
# ---------------------------------------------------------------------------

RESULT_ATTRIBUTES: Dict[str, Tuple[str, ...]] = {
    "displacements":  ("displacements",),
    "velocities":     ("velocities",),
    "accelerations":  ("accelerations",),
    "spc_forces":     ("spc_forces",),
    "mpc_forces":     ("mpc_forces",),
    "plate_stress":   ("cquad4_stress", "ctria3_stress", "cquad8_stress", "ctria6_stress"),
    "plate_strain":   ("cquad4_strain", "ctria3_strain", "cquad8_strain", "ctria6_strain"),
    "solid_stress":   ("chexa_stress", "ctetra_stress", "cpenta_stress"),
    "solid_strain":   ("chexa_strain", "ctetra_strain", "cpenta_strain"),
    "bar_stress":     ("cbar_stress",),
    "bar_strain":     ("cbar_strain",),
    "beam_stress":    ("cbeam_stress",),
    "beam_strain":    ("cbeam_strain",),
    "rod_stress":     ("crod_stress", "conrod_stress"),
    "rod_strain":     ("crod_strain", "conrod_strain"),
    "plate_force":    ("cquad4_force", "ctria3_force", "cquad8_force", "ctria6_force"),
    "bar_force":      ("cbar_force",),
    "beam_force":     ("cbeam_force",),
}

_ID_COLS = {"element_id", "node_id", "nid", "eid", "elementid", "nodeid"}

# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def read_files(
    filepaths: list,
    requested_types: list | None = None,
) -> Dict[Tuple[str, int], Dict[str, pd.DataFrame]]:
    """Read one or more .op2 / .h5 files.

    Returns {(filepath, subcase_id): {result_type: DataFrame}}
    """
    types_to_read = requested_types or list(RESULT_ATTRIBUTES.keys())
    combined: Dict[Tuple[str, int], Dict[str, pd.DataFrame]] = {}

    for filepath in filepaths:
        ext = os.path.splitext(filepath)[1].lower()
        if ext not in (".op2", ".h5", ".hdf5"):
            raise ValueError(f"Unsupported file extension: {ext}  (expected .op2 or .h5/.hdf5)")

        model = OP2(debug=False, log=None)
        model.IS_TESTING = False

        if ext == ".op2":
            model.read_op2(filepath, combine=True)
        else:
            model.load_hdf5_filename(filepath)

        for result_type in types_to_read:
            for attr in RESULT_ATTRIBUTES[result_type]:
                result_dict = getattr(model, attr, {})
                if not result_dict:
                    continue
                for subcase_id, result_obj in result_dict.items():
                    try:
                        df = result_obj.to_dataframe()
                    except Exception:
                        continue
                    if df.empty:
                        continue
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = ["_".join(str(c) for c in col).strip("_") for col in df.columns]
                    df = df.reset_index()
                    key = (filepath, subcase_id)
                    if key not in combined:
                        combined[key] = {}
                    if result_type in combined[key]:
                        combined[key][result_type] = pd.concat(
                            [combined[key][result_type], df], ignore_index=True
                        )
                    else:
                        combined[key][result_type] = df

    return combined

# ---------------------------------------------------------------------------
# Envelope computation
# ---------------------------------------------------------------------------

def _id_column(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        if col.lower() in _ID_COLS:
            return col
    if not df.empty and pd.api.types.is_integer_dtype(df.iloc[:, 0]):
        return df.columns[0]
    return None


def _numeric_cols(df: pd.DataFrame, id_col: str) -> list:
    return [c for c in df.columns if c != id_col and pd.api.types.is_numeric_dtype(df[c])]


def compute_envelope(
    all_results: Dict[Tuple[str, int], Dict[str, pd.DataFrame]],
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """Return (envelope_max, envelope_min) dicts keyed by result_type."""
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

        max_rows, min_rows = [], []

        for elem_id, group in combined.groupby(id_col):
            max_row = {id_col: elem_id}
            min_row = {id_col: elem_id}

            for col in num_cols:
                vals = group[col]
                max_row[col] = vals[vals.idxmax()]
                min_row[col] = vals[vals.idxmin()]

            abs_sums = group[num_cols].abs().sum(axis=1)
            max_row["governing_file"]     = group.loc[abs_sums.idxmax(), "__file__"]
            max_row["governing_subcase"]  = group.loc[abs_sums.idxmax(), "__subcase__"]
            min_row["governing_file"]     = group.loc[abs_sums.idxmin(), "__file__"]
            min_row["governing_subcase"]  = group.loc[abs_sums.idxmin(), "__subcase__"]

            max_rows.append(max_row)
            min_rows.append(min_row)

        envelope_max[result_type] = pd.DataFrame(max_rows)
        envelope_min[result_type] = pd.DataFrame(min_rows)

    return envelope_max, envelope_min


def build_governing_report(
    all_results: Dict[Tuple[str, int], Dict[str, pd.DataFrame]],
) -> Dict[str, pd.DataFrame]:
    """For every element × result column, record which file/subcase governs max and min."""
    all_types: set = set()
    for type_dict in all_results.values():
        all_types.update(type_dict.keys())

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
                vals = group[col]
                max_idx = vals.idxmax()
                min_idx = vals.idxmin()
                rows.append({
                    id_col:                   elem_id,
                    "result_column":          col,
                    "max_value":              vals[max_idx],
                    "max_governing_file":     os.path.basename(group.loc[max_idx, "__file__"]),
                    "max_governing_subcase":  group.loc[max_idx, "__subcase__"],
                    "min_value":              vals[min_idx],
                    "min_governing_file":     os.path.basename(group.loc[min_idx, "__file__"]),
                    "min_governing_subcase":  group.loc[min_idx, "__subcase__"],
                })

        report[result_type] = pd.DataFrame(rows)

    return report

# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def _write_excel(data: Dict[str, pd.DataFrame], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet, df in data.items():
            df.to_excel(writer, sheet_name=sheet[:31], index=False)


def _write_csv(data: Dict[str, pd.DataFrame], base_path: str, suffix: str = "") -> None:
    base = os.path.splitext(base_path)[0]
    os.makedirs(os.path.dirname(os.path.abspath(base_path)), exist_ok=True)
    for key, df in data.items():
        df.to_csv(f"{base}_{key}{suffix}.csv", index=False)


def export_results(
    envelope_max: Dict[str, pd.DataFrame],
    envelope_min: Dict[str, pd.DataFrame],
    output_path: str,
    fmt: str = "xlsx",
    governing: Dict[str, pd.DataFrame] | None = None,
    governing_path: str | None = None,
) -> None:
    if fmt == "xlsx":
        combined = {f"{rt}_MAX": df for rt, df in envelope_max.items()}
        combined.update({f"{rt}_MIN": df for rt, df in envelope_min.items()})
        _write_excel(combined, output_path)
        if governing and governing_path:
            _write_excel(governing, governing_path)
    else:
        _write_csv(envelope_max, output_path, "_max")
        _write_csv(envelope_min, output_path, "_min")
        if governing and governing_path:
            _write_csv(governing, governing_path, "_governing")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Create an envelope (worst-case max/min) from multiple Nastran "
            ".op2 or .h5 analysis output files."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "input_files", nargs="+", metavar="FILE",
        help=".op2 or .h5/.hdf5 Nastran result files (one or more)",
    )
    parser.add_argument(
        "--result-types", nargs="+", metavar="TYPE", default=None,
        help=(
            "Result types to include. Default: all available. "
            f"Choices: {', '.join(sorted(RESULT_ATTRIBUTES.keys()))}"
        ),
    )
    parser.add_argument(
        "--export-envelope", metavar="PATH", required=True,
        help="Output file for envelope results (.xlsx or .csv)",
    )
    parser.add_argument(
        "--export-governing", metavar="PATH", default=None,
        help=(
            "Optional output file for the governing-subcase report (.xlsx or .csv). "
            "Shows which file/subcase governs each element for each result column."
        ),
    )
    parser.add_argument(
        "--format", choices=["xlsx", "csv"], default="xlsx",
        help="Output format (default: xlsx)",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.result_types:
        invalid = [t for t in args.result_types if t not in RESULT_ATTRIBUTES]
        if invalid:
            print(
                f"ERROR: Unknown result type(s): {invalid}\n"
                f"Available: {sorted(RESULT_ATTRIBUTES.keys())}",
                file=sys.stderr,
            )
            sys.exit(1)

    print(f"Reading {len(args.input_files)} file(s)...")
    try:
        all_results = read_files(args.input_files, requested_types=args.result_types)
    except Exception as exc:
        print(f"ERROR while reading files: {exc}", file=sys.stderr)
        sys.exit(1)

    if not all_results:
        print(
            "No results found in the provided files. "
            "Check that the files contain the requested result types.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Found {len(all_results)} subcase(s) across all files:")
    for (filepath, subcase_id), type_dict in all_results.items():
        print(f"  {filepath}  subcase {subcase_id}: {list(type_dict.keys())}")

    print("\nComputing envelope...")
    envelope_max, envelope_min = compute_envelope(all_results)

    governing = None
    if args.export_governing:
        print("Building governing-subcase report...")
        governing = build_governing_report(all_results)

    print(f"\nExporting envelope → {args.export_envelope}")
    export_results(
        envelope_max=envelope_max,
        envelope_min=envelope_min,
        output_path=args.export_envelope,
        fmt=args.format,
        governing=governing,
        governing_path=args.export_governing,
    )

    if args.export_governing:
        print(f"Exporting governing report → {args.export_governing}")

    print("Done.")


if __name__ == "__main__":
    main()

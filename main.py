#!/usr/bin/env python3
"""External Envelope Tool

Reads multiple Nastran .op2 or .h5 files (each may contain multiple subcases),
computes element/node-wise max and min envelope across all subcases, and
exports the results to Excel or CSV.

Usage examples
--------------
# Basic: envelope all result types from two files
python main.py results1.op2 results2.h5 --export-envelope envelope.xlsx

# Specific result types only
python main.py a.op2 b.op2 --result-types displacements plate_stress \\
    --export-envelope envelope.xlsx

# Also export which subcase governs each element
python main.py a.op2 b.op2 \\
    --export-envelope envelope.xlsx \\
    --export-governing governing.xlsx

# CSV output
python main.py a.op2 b.op2 --format csv --export-envelope envelope.csv \\
    --export-governing governing.csv
"""

import argparse
import sys

from envelope_tool.reader import read_files, RESULT_ATTRIBUTES
from envelope_tool.envelope import compute_envelope, build_governing_report
from envelope_tool.export import export_results


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
        "input_files",
        nargs="+",
        metavar="FILE",
        help=".op2 or .h5/.hdf5 Nastran result files (one or more)",
    )
    parser.add_argument(
        "--result-types",
        nargs="+",
        metavar="TYPE",
        default=None,
        help=(
            "Result types to include. Default: all available. "
            f"Choices: {', '.join(sorted(RESULT_ATTRIBUTES.keys()))}"
        ),
    )
    parser.add_argument(
        "--export-envelope",
        metavar="PATH",
        required=True,
        help="Output file for the envelope results (.xlsx or .csv)",
    )
    parser.add_argument(
        "--export-governing",
        metavar="PATH",
        default=None,
        help=(
            "Optional output file for the governing-subcase report "
            "(.xlsx or .csv). Shows which file/subcase governs each "
            "element for each result column."
        ),
    )
    parser.add_argument(
        "--format",
        choices=["xlsx", "csv"],
        default="xlsx",
        help="Output format (default: xlsx)",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # Validate result type choices
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

    subcases_found = [(f, sc) for (f, sc) in all_results]
    print(f"Found {len(subcases_found)} subcase(s) across all files:")
    for filepath, subcase_id in subcases_found:
        types = list(all_results[(filepath, subcase_id)].keys())
        print(f"  {filepath}  subcase {subcase_id}: {types}")

    print("\nComputing envelope...")
    envelope_max, envelope_min = compute_envelope(all_results)

    governing = None
    if args.export_governing:
        print("Building governing-subcase report...")
        governing = build_governing_report(all_results, envelope_max, envelope_min)

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

    print("\nDone.")


if __name__ == "__main__":
    main()

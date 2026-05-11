"""Read Nastran .op2 and .h5 files and extract result DataFrames per subcase."""

import os
from typing import Dict, Tuple

import pandas as pd
from pyNastran.op2.op2 import OP2

# Map of user-facing result type keys to (op2_attribute, description) tuples.
# Each attribute is a dict keyed by subcase_id on the OP2 model object.
RESULT_ATTRIBUTES: Dict[str, Tuple[str, ...]] = {
    "displacements": ("displacements",),
    "velocities": ("velocities",),
    "accelerations": ("accelerations",),
    "spc_forces": ("spc_forces",),
    "mpc_forces": ("mpc_forces",),
    "plate_stress": ("cquad4_stress", "ctria3_stress", "cquad8_stress", "ctria6_stress"),
    "plate_strain": ("cquad4_strain", "ctria3_strain", "cquad8_strain", "ctria6_strain"),
    "solid_stress": ("chexa_stress", "ctetra_stress", "cpenta_stress"),
    "solid_strain": ("chexa_strain", "ctetra_strain", "cpenta_strain"),
    "bar_stress": ("cbar_stress",),
    "bar_strain": ("cbar_strain",),
    "beam_stress": ("cbeam_stress",),
    "beam_strain": ("cbeam_strain",),
    "rod_stress": ("crod_stress", "conrod_stress"),
    "rod_strain": ("crod_strain", "conrod_strain"),
    "plate_force": ("cquad4_force", "ctria3_force", "cquad8_force", "ctria6_force"),
    "bar_force": ("cbar_force",),
    "beam_force": ("cbeam_force",),
}


def _available_result_types() -> list:
    return list(RESULT_ATTRIBUTES.keys())


def read_file(
    filepath: str,
    requested_types: list | None = None,
) -> Dict[Tuple[str, int], Dict[str, pd.DataFrame]]:
    """Read a single .op2 or .h5 file.

    Returns
    -------
    dict keyed by (filepath, subcase_id) → {result_type: DataFrame}
    """
    ext = os.path.splitext(filepath)[1].lower()
    if ext not in (".op2", ".h5", ".hdf5"):
        raise ValueError(f"Unsupported file extension: {ext}  (expected .op2 or .h5/.hdf5)")

    model = OP2(debug=False, log=None)
    model.IS_TESTING = False

    if ext == ".op2":
        model.read_op2(filepath, combine=True)
    else:
        model.load_hdf5_filename(filepath)

    types_to_read = requested_types if requested_types else _available_result_types()

    results: Dict[Tuple[str, int], Dict[str, pd.DataFrame]] = {}

    for result_type in types_to_read:
        if result_type not in RESULT_ATTRIBUTES:
            raise ValueError(
                f"Unknown result type '{result_type}'. "
                f"Available: {_available_result_types()}"
            )
        attrs = RESULT_ATTRIBUTES[result_type]
        for attr in attrs:
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
                # Flatten MultiIndex columns if present
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = ["_".join(str(c) for c in col).strip("_") for col in df.columns]
                # Reset index so element/node IDs become a regular column
                df = df.reset_index()
                key = (filepath, subcase_id)
                if key not in results:
                    results[key] = {}
                # If multiple element types map to same result_type, concat
                if result_type in results[key]:
                    existing = results[key][result_type]
                    results[key][result_type] = pd.concat(
                        [existing, df], ignore_index=True
                    )
                else:
                    results[key][result_type] = df

    return results


def read_files(
    filepaths: list,
    requested_types: list | None = None,
) -> Dict[Tuple[str, int], Dict[str, pd.DataFrame]]:
    """Read multiple files and merge their results into a single dict."""
    combined: Dict[Tuple[str, int], Dict[str, pd.DataFrame]] = {}
    for fp in filepaths:
        file_results = read_file(fp, requested_types)
        combined.update(file_results)
    return combined

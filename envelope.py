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
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from typing import Dict, Tuple

# Heavy dependencies are imported lazily inside functions so that a missing
# package shows a proper error dialog instead of silently crashing on
# double-click. Top-level names are set to None and resolved on first use.
try:
    import pandas as pd
    from pyNastran.op2.op2 import OP2
    _DEPS_OK = True
    _DEPS_ERR = ""
except ImportError as _e:
    pd = None  # type: ignore
    OP2 = None  # type: ignore
    _DEPS_OK = False
    _DEPS_ERR = (
        f"Gerekli kütüphane(ler) yüklü değil:\n{_e}\n\n"
        "Lütfen terminalde şunu çalıştırın:\n"
        "  pip install -r requirements.txt"
    )

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
    requested_types: list = None,
) -> Dict[Tuple[str, int], Dict]:
    """Read one or more .op2 / .h5 files.

    Returns {(filepath, subcase_id): {result_type: DataFrame}}
    """
    if not _DEPS_OK:
        raise RuntimeError(_DEPS_ERR)
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


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

class LoadExtractionApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("External Envelope Tool")
        self.root.resizable(True, True)

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        pad = {"padx": 6, "pady": 4}

        # ── Input files ────────────────────────────────────────────────
        frm_files = ttk.LabelFrame(self.root, text="Giriş Dosyaları (.op2 / .h5)")
        frm_files.pack(fill="both", expand=False, **pad)

        self.lb_files = tk.Listbox(frm_files, selectmode=tk.EXTENDED, height=5, width=70)
        self.lb_files.pack(side="left", fill="both", expand=True, padx=(4, 0), pady=4)

        sb_files = ttk.Scrollbar(frm_files, orient="vertical", command=self.lb_files.yview)
        sb_files.pack(side="left", fill="y", pady=4)
        self.lb_files.configure(yscrollcommand=sb_files.set)

        frm_file_btns = ttk.Frame(frm_files)
        frm_file_btns.pack(side="left", padx=4, pady=4, anchor="n")
        ttk.Button(frm_file_btns, text="Ekle",   width=10, command=self._add_files).pack(pady=2)
        ttk.Button(frm_file_btns, text="Kaldır", width=10, command=self._remove_files).pack(pady=2)

        # ── Result types ───────────────────────────────────────────────
        frm_types = ttk.LabelFrame(self.root, text="Result Tipleri")
        frm_types.pack(fill="both", expand=False, **pad)

        self._type_vars: Dict[str, tk.BooleanVar] = {}
        grid_frame = ttk.Frame(frm_types)
        grid_frame.pack(fill="x", padx=4, pady=2)

        cols = 4
        for i, rt in enumerate(sorted(RESULT_ATTRIBUTES.keys())):
            var = tk.BooleanVar(value=True)
            self._type_vars[rt] = var
            ttk.Checkbutton(grid_frame, text=rt, variable=var).grid(
                row=i // cols, column=i % cols, sticky="w", padx=4, pady=1
            )

        frm_type_btns = ttk.Frame(frm_types)
        frm_type_btns.pack(anchor="w", padx=4, pady=(0, 4))
        ttk.Button(frm_type_btns, text="Tümünü Seç",    command=lambda: self._set_all_types(True)).pack(side="left", padx=2)
        ttk.Button(frm_type_btns, text="Tümünü Kaldır", command=lambda: self._set_all_types(False)).pack(side="left", padx=2)

        # ── Envelope output ────────────────────────────────────────────
        frm_out = ttk.LabelFrame(self.root, text="Envelope Çıktısı")
        frm_out.pack(fill="x", **pad)

        frm_env_path = ttk.Frame(frm_out)
        frm_env_path.pack(fill="x", padx=4, pady=2)
        ttk.Label(frm_env_path, text="Dosya:").pack(side="left")
        self.sv_envelope = tk.StringVar()
        ttk.Entry(frm_env_path, textvariable=self.sv_envelope, width=55).pack(side="left", padx=4)
        ttk.Button(frm_env_path, text="Gözat", command=self._browse_envelope).pack(side="left")

        frm_fmt = ttk.Frame(frm_out)
        frm_fmt.pack(anchor="w", padx=4, pady=(0, 4))
        ttk.Label(frm_fmt, text="Format:").pack(side="left")
        self.sv_format = tk.StringVar(value="xlsx")
        ttk.Radiobutton(frm_fmt, text="xlsx", variable=self.sv_format, value="xlsx").pack(side="left", padx=4)
        ttk.Radiobutton(frm_fmt, text="csv",  variable=self.sv_format, value="csv").pack(side="left")

        # ── Governing report ───────────────────────────────────────────
        frm_gov = ttk.LabelFrame(self.root, text="Governing Subcase Raporu (opsiyonel)")
        frm_gov.pack(fill="x", **pad)

        self.bv_governing = tk.BooleanVar(value=False)
        ttk.Checkbutton(frm_gov, text="Governing raporu oluştur",
                        variable=self.bv_governing,
                        command=self._toggle_governing).pack(anchor="w", padx=4, pady=2)

        frm_gov_path = ttk.Frame(frm_gov)
        frm_gov_path.pack(fill="x", padx=4, pady=(0, 4))
        ttk.Label(frm_gov_path, text="Dosya:").pack(side="left")
        self.sv_governing = tk.StringVar()
        self.ent_governing = ttk.Entry(frm_gov_path, textvariable=self.sv_governing, width=55, state="disabled")
        self.ent_governing.pack(side="left", padx=4)
        self.btn_gov_browse = ttk.Button(frm_gov_path, text="Gözat",
                                         command=self._browse_governing, state="disabled")
        self.btn_gov_browse.pack(side="left")

        # ── Run button ─────────────────────────────────────────────────
        self.btn_run = ttk.Button(self.root, text="Envelope Hesapla",
                                  command=self._run, style="Accent.TButton")
        self.btn_run.pack(fill="x", padx=8, pady=6)

        # ── Log ────────────────────────────────────────────────────────
        frm_log = ttk.LabelFrame(self.root, text="Log")
        frm_log.pack(fill="both", expand=True, **pad)

        self.log = scrolledtext.ScrolledText(frm_log, height=10, state="disabled",
                                              wrap="word", font=("Courier", 9))
        self.log.pack(fill="both", expand=True, padx=4, pady=4)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _log(self, msg: str):
        """Append a line to the log widget (thread-safe)."""
        def _append():
            self.log.configure(state="normal")
            self.log.insert("end", msg + "\n")
            self.log.see("end")
            self.log.configure(state="disabled")
        self.root.after(0, _append)

    def _add_files(self):
        paths = filedialog.askopenfilenames(
            title="Nastran dosyaları seç",
            filetypes=[("Nastran files", "*.op2 *.h5 *.hdf5"), ("All files", "*.*")],
        )
        existing = list(self.lb_files.get(0, "end"))
        for p in paths:
            if p not in existing:
                self.lb_files.insert("end", p)

    def _remove_files(self):
        for idx in reversed(self.lb_files.curselection()):
            self.lb_files.delete(idx)

    def _set_all_types(self, state: bool):
        for var in self._type_vars.values():
            var.set(state)

    def _browse_envelope(self):
        fmt = self.sv_format.get()
        ext = ".xlsx" if fmt == "xlsx" else ".csv"
        path = filedialog.asksaveasfilename(
            title="Envelope çıktı dosyası",
            defaultextension=ext,
            filetypes=[("Excel", "*.xlsx"), ("CSV", "*.csv"), ("All", "*.*")],
        )
        if path:
            self.sv_envelope.set(path)

    def _browse_governing(self):
        fmt = self.sv_format.get()
        ext = ".xlsx" if fmt == "xlsx" else ".csv"
        path = filedialog.asksaveasfilename(
            title="Governing rapor dosyası",
            defaultextension=ext,
            filetypes=[("Excel", "*.xlsx"), ("CSV", "*.csv"), ("All", "*.*")],
        )
        if path:
            self.sv_governing.set(path)

    def _toggle_governing(self):
        state = "normal" if self.bv_governing.get() else "disabled"
        self.ent_governing.configure(state=state)
        self.btn_gov_browse.configure(state=state)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def _run(self):
        input_files = list(self.lb_files.get(0, "end"))
        if not input_files:
            messagebox.showwarning("Eksik giriş", "Lütfen en az bir dosya ekleyin.")
            return

        selected_types = [rt for rt, var in self._type_vars.items() if var.get()]
        if not selected_types:
            messagebox.showwarning("Eksik seçim", "Lütfen en az bir result tipi seçin.")
            return

        envelope_path = self.sv_envelope.get().strip()
        if not envelope_path:
            messagebox.showwarning("Eksik yol", "Envelope çıktı dosyası belirtin.")
            return

        governing_path = None
        if self.bv_governing.get():
            governing_path = self.sv_governing.get().strip()
            if not governing_path:
                messagebox.showwarning("Eksik yol", "Governing rapor dosyası belirtin.")
                return

        fmt = self.sv_format.get()

        # Clear log
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

        self.btn_run.configure(state="disabled")
        threading.Thread(
            target=self._worker,
            args=(input_files, selected_types, envelope_path, governing_path, fmt),
            daemon=True,
        ).start()

    def _worker(self, input_files, selected_types, envelope_path, governing_path, fmt):
        try:
            self._log(f"{len(input_files)} dosya okunuyor...")
            all_results = read_files(input_files, requested_types=selected_types)

            if not all_results:
                self.root.after(0, lambda: messagebox.showerror(
                    "Sonuç yok",
                    "Dosyalarda seçilen result tipleri bulunamadı."
                ))
                return

            self._log(f"{len(all_results)} subcase bulundu:")
            for (fp, sc), type_dict in all_results.items():
                self._log(f"  {os.path.basename(fp)}  subcase {sc}: {list(type_dict.keys())}")

            self._log("\nEnvelope hesaplanıyor...")
            envelope_max, envelope_min = compute_envelope(all_results)

            governing = None
            if governing_path:
                self._log("Governing raporu oluşturuluyor...")
                governing = build_governing_report(all_results)

            self._log(f"\nDosyaya yazılıyor → {envelope_path}")
            export_results(
                envelope_max=envelope_max,
                envelope_min=envelope_min,
                output_path=envelope_path,
                fmt=fmt,
                governing=governing,
                governing_path=governing_path,
            )
            if governing_path:
                self._log(f"Governing raporu → {governing_path}")

            self._log("\nTamamlandı.")
            self.root.after(0, lambda: messagebox.showinfo("Tamamlandı", "Envelope başarıyla oluşturuldu."))

        except Exception as exc:
            self._log(f"\nHATA: {exc}")
            self.root.after(0, lambda: messagebox.showerror("Hata", str(exc)))
        finally:
            self.root.after(0, lambda: self.btn_run.configure(state="normal"))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # CLI mode: let missing deps surface as a normal error message
        if not _DEPS_OK:
            print(f"HATA: {_DEPS_ERR}", file=sys.stderr)
            sys.exit(1)
        main()
    else:
        # GUI mode: show a dialog if deps are missing, then exit gracefully
        root = tk.Tk()
        if not _DEPS_OK:
            root.withdraw()
            messagebox.showerror("Eksik Kütüphane", _DEPS_ERR)
            sys.exit(1)
        app = LoadExtractionApp(root)
        root.mainloop()

#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import threading
import traceback
import tkinter as tk
from copy import deepcopy
from tkinter import ttk, filedialog, messagebox, scrolledtext

try:
    import numpy as np
    import pandas as pd
    from pyNastran.op2.op2 import OP2
    _DEPS_OK = True
    _DEPS_ERR = ""
except ImportError as _e:
    np = pd = OP2 = None  # type: ignore
    _DEPS_OK = False
    _DEPS_ERR = (
        f"Gerekli kütüphane(ler) yüklü değil:\n{_e}\n\n"
        "Lütfen terminalde şunu çalıştırın:\n"
        "  pip install -r requirements.txt"
    )

# ─── Result type → pyNastran attribute(s) ───────────────────────────────────

RESULT_ATTRIBUTES: dict[str, tuple[str, ...]] = {
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

CHECK_ON  = "☑"
CHECK_OFF = "☐"

# ─── Backend ────────────────────────────────────────────────────────────────

def _load_model(filepath: str) -> OP2:
    model = OP2(debug=False, log=None)
    model.IS_TESTING = False
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".op2":
        model.read_op2(filepath, combine=True)
    else:
        model.load_hdf5_filename(filepath)
    return model


def discover_results(filepaths: list[str]) -> dict:
    """Scan files, return what result types are present.

    Returns
    -------
    {result_type: {"subcases": [(filepath, sc_id), ...], "n_files": int}}
    """
    discovered: dict = {}
    for filepath in filepaths:
        model = _load_model(filepath)
        for rt, attrs in RESULT_ATTRIBUTES.items():
            for attr in attrs:
                rd = getattr(model, attr, {})
                if not rd:
                    continue
                entry = discovered.setdefault(rt, {"subcases": [], "_files": set()})
                for sc_id in rd:
                    entry["subcases"].append((filepath, sc_id))
                entry["_files"].add(filepath)
    for rt in discovered:
        discovered[rt]["n_files"] = len(discovered[rt].pop("_files"))
    return discovered


def collect_raw(filepaths: list[str], selected_types: list[str]) -> dict:
    """Read files and collect raw pyNastran result objects.

    Returns
    -------
    {attr: [(filepath, sc_id, result_obj), ...]}
    """
    raw: dict = {}
    for filepath in filepaths:
        model = _load_model(filepath)
        for rt in selected_types:
            for attr in RESULT_ATTRIBUTES.get(rt, ()):
                rd = getattr(model, attr, {})
                if not rd:
                    continue
                for sc_id, result_obj in rd.items():
                    raw.setdefault(attr, []).append((filepath, sc_id, result_obj))
    return raw


def _get_ids(result_obj) -> list:
    """Extract element or node IDs from a result object."""
    try:
        if hasattr(result_obj, "node_gridtype"):
            return result_obj.node_gridtype[:, 0].tolist()
        if hasattr(result_obj, "element_node"):
            return result_obj.element_node[:, 0].tolist()
        if hasattr(result_obj, "element"):
            return result_obj.element.tolist()
    except Exception:
        pass
    return list(range(result_obj.data.shape[1]))


def compute_envelope(raw: dict) -> tuple:
    """Compute element-wise max and min across all subcases.

    Returns
    -------
    env_max   : {attr: ndarray (nelems, ncomp)}
    env_min   : {attr: ndarray (nelems, ncomp)}
    governing : {attr: {"max": [(file, sc), ...], "min": [(file, sc), ...]}}
                 indexed by element position
    templates : {attr: result_obj}   — used as template for output writing
    elem_ids  : {attr: [int, ...]}
    """
    env_max: dict = {}
    env_min: dict = {}
    governing: dict = {}
    templates: dict = {}
    elem_ids: dict = {}

    for attr, entries in raw.items():
        if not entries:
            continue

        # result_obj.data shape: (ntimes, nelems, ncomp)
        # For static analysis ntimes == 1; use last time step for others.
        arrays = [e[2].data[-1] for e in entries]  # list of (nelems, ncomp)

        try:
            stacked = np.stack(arrays, axis=0)  # (n_sc, nelems, ncomp)
        except ValueError:
            continue  # inconsistent shapes — skip

        env_max[attr] = stacked.max(axis=0)
        env_min[attr] = stacked.min(axis=0)

        # Governing: which subcase yields max / min absolute sum per element
        abs_sum     = np.abs(stacked).sum(axis=2)   # (n_sc, nelems)
        gov_max_idx = abs_sum.argmax(axis=0)         # (nelems,)
        gov_min_idx = abs_sum.argmin(axis=0)

        governing[attr] = {
            "max": [(entries[i][0], entries[i][1]) for i in gov_max_idx],
            "min": [(entries[i][0], entries[i][1]) for i in gov_min_idx],
        }
        templates[attr] = entries[0][2]
        elem_ids[attr]  = _get_ids(entries[0][2])

    return env_max, env_min, governing, templates, elem_ids


def write_output(env_max: dict, env_min: dict, templates: dict,
                 output_path: str, fmt: str) -> None:
    """Write envelope as .op2 or .h5.

    Subcase 1 = MAX envelope, Subcase 2 = MIN envelope.
    Uses deepcopy of template result objects so the original data is preserved.
    """
    out_model = OP2(debug=False, log=None)
    out_model.IS_TESTING = False

    for attr, max_arr in env_max.items():
        min_arr  = env_min[attr]
        template = templates[attr]

        res_max = deepcopy(template)
        res_max.data = max_arr[np.newaxis, :, :]   # (1, nelems, ncomp)

        res_min = deepcopy(template)
        res_min.data = min_arr[np.newaxis, :, :]

        try:
            res_max.isubcase = 1
            res_min.isubcase = 2
        except Exception:
            pass

        setattr(out_model, attr, {1: res_max, 2: res_min})

    if fmt == "op2":
        out_model.write_op2(output_path)
    else:
        out_model.export_hdf5(output_path)


def build_governing_df(governing: dict, elem_ids: dict) -> pd.DataFrame:
    rows = []
    for attr, gov in governing.items():
        ids = elem_ids.get(attr, list(range(len(gov["max"]))))
        for i, eid in enumerate(ids):
            max_file, max_sc = gov["max"][i]
            min_file, min_sc = gov["min"][i]
            rows.append({
                "Result_Type":       attr,
                "Element/Node_ID":   int(eid),
                "Max_Gov_File":      os.path.basename(max_file),
                "Max_Gov_Subcase":   max_sc,
                "Min_Gov_File":      os.path.basename(min_file),
                "Min_Gov_Subcase":   min_sc,
            })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def export_governing_excel(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Tümü", index=False)
        for rt in df["Result_Type"].unique():
            df[df["Result_Type"] == rt].to_excel(
                writer, sheet_name=rt[:31], index=False
            )

# ─── GUI ────────────────────────────────────────────────────────────────────

class LoadExtractionApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("External Envelope Tool")
        self.root.resizable(True, True)

        self._discovered: dict = {}
        self._check_states: dict = {}   # {treeview_iid: bool}

        self._build_ui()

        if not _DEPS_OK:
            messagebox.showerror("Eksik Kütüphane", _DEPS_ERR)
            self.btn_scan.configure(state="disabled")
            self.btn_run.configure(state="disabled")

    # ── UI construction ─────────────────────────────────────────────────────

    def _build_ui(self):
        pad = {"padx": 6, "pady": 4}

        # ── Giriş dosyaları ──────────────────────────────────────────
        frm_files = ttk.LabelFrame(self.root, text="Giriş Dosyaları (.op2 / .h5)")
        frm_files.pack(fill="both", expand=False, **pad)

        self.lb_files = tk.Listbox(frm_files, selectmode=tk.EXTENDED,
                                    height=4, width=72)
        self.lb_files.pack(side="left", fill="both", expand=True,
                           padx=(4, 0), pady=4)
        sb = ttk.Scrollbar(frm_files, orient="vertical",
                            command=self.lb_files.yview)
        sb.pack(side="left", fill="y", pady=4)
        self.lb_files.configure(yscrollcommand=sb.set)

        frm_fb = ttk.Frame(frm_files)
        frm_fb.pack(side="left", padx=4, pady=4, anchor="n")
        ttk.Button(frm_fb, text="Ekle",   width=14,
                   command=self._add_files).pack(pady=2)
        ttk.Button(frm_fb, text="Kaldır", width=14,
                   command=self._remove_files).pack(pady=2)
        self.btn_scan = ttk.Button(frm_fb, text="Dosyaları Tara", width=14,
                                    command=self._scan_files)
        self.btn_scan.pack(pady=(10, 2))

        # ── Bulunan result tipleri ────────────────────────────────────
        frm_tree = ttk.LabelFrame(
            self.root,
            text="Bulunan Result Tipleri  —  seçmek/kaldırmak için satıra tıkla"
        )
        frm_tree.pack(fill="both", expand=True, **pad)

        cols = ("n_sc", "n_files")
        self.tv_results = ttk.Treeview(frm_tree, columns=cols,
                                        show="tree headings", height=7)
        self.tv_results.heading("#0",      text="Result Tipi")
        self.tv_results.heading("n_sc",    text="Subcase Sayısı")
        self.tv_results.heading("n_files", text="Dosya Sayısı")
        self.tv_results.column("#0",      width=270)
        self.tv_results.column("n_sc",    width=110, anchor="center")
        self.tv_results.column("n_files", width=100, anchor="center")
        self.tv_results.pack(side="left", fill="both", expand=True,
                              padx=(4, 0), pady=4)
        sb2 = ttk.Scrollbar(frm_tree, orient="vertical",
                              command=self.tv_results.yview)
        sb2.pack(side="left", fill="y", pady=4)
        self.tv_results.configure(yscrollcommand=sb2.set)
        self.tv_results.bind("<Button-1>", self._toggle_check)

        frm_tb = ttk.Frame(frm_tree)
        frm_tb.pack(side="left", padx=4, pady=4, anchor="n")
        ttk.Button(frm_tb, text="Tümünü Seç",    width=14,
                   command=lambda: self._set_all(True)).pack(pady=2)
        ttk.Button(frm_tb, text="Tümünü Kaldır", width=14,
                   command=lambda: self._set_all(False)).pack(pady=2)

        # ── Çıktı ────────────────────────────────────────────────────
        frm_out = ttk.LabelFrame(self.root, text="Çıktı Dosyası")
        frm_out.pack(fill="x", **pad)

        frm_op = ttk.Frame(frm_out)
        frm_op.pack(fill="x", padx=4, pady=2)
        ttk.Label(frm_op, text="Dosya:").pack(side="left")
        self.sv_output = tk.StringVar()
        ttk.Entry(frm_op, textvariable=self.sv_output, width=55).pack(
            side="left", padx=4)
        ttk.Button(frm_op, text="Gözat",
                   command=self._browse_output).pack(side="left")

        frm_fmt = ttk.Frame(frm_out)
        frm_fmt.pack(anchor="w", padx=4, pady=(0, 4))
        ttk.Label(frm_fmt, text="Format:").pack(side="left")
        self.sv_fmt = tk.StringVar(value="op2")
        ttk.Radiobutton(frm_fmt, text=".op2", variable=self.sv_fmt,
                         value="op2").pack(side="left", padx=4)
        ttk.Radiobutton(frm_fmt, text=".h5",  variable=self.sv_fmt,
                         value="h5").pack(side="left")

        # ── Excel governing raporu ────────────────────────────────────
        frm_xl = ttk.LabelFrame(self.root,
                                  text="Excel Governing Raporu (opsiyonel)")
        frm_xl.pack(fill="x", **pad)

        self.bv_excel = tk.BooleanVar(value=False)
        ttk.Checkbutton(frm_xl, text="Excel raporu oluştur",
                         variable=self.bv_excel,
                         command=self._toggle_excel).pack(anchor="w",
                                                           padx=4, pady=2)
        frm_xlp = ttk.Frame(frm_xl)
        frm_xlp.pack(fill="x", padx=4, pady=(0, 4))
        ttk.Label(frm_xlp, text="Dosya:").pack(side="left")
        self.sv_excel = tk.StringVar()
        self.ent_excel = ttk.Entry(frm_xlp, textvariable=self.sv_excel,
                                    width=55, state="disabled")
        self.ent_excel.pack(side="left", padx=4)
        self.btn_xl_browse = ttk.Button(frm_xlp, text="Gözat",
                                         command=self._browse_excel,
                                         state="disabled")
        self.btn_xl_browse.pack(side="left")

        # ── Hesapla ──────────────────────────────────────────────────
        self.btn_run = ttk.Button(self.root, text="Envelope Hesapla",
                                   command=self._run)
        self.btn_run.pack(fill="x", padx=8, pady=6)

        # ── Governing tablosu ─────────────────────────────────────────
        frm_gov = ttk.LabelFrame(self.root, text="Governing Subcase Tablosu")
        frm_gov.pack(fill="both", expand=True, **pad)

        gov_cols = ("result", "id", "max_file", "max_sc", "min_file", "min_sc")
        self.tv_gov = ttk.Treeview(frm_gov, columns=gov_cols,
                                    show="headings", height=7)
        headers = {
            "result":   "Result Tipi",
            "id":       "Elem/Node ID",
            "max_file": "Max Dosya",
            "max_sc":   "Max Subcase",
            "min_file": "Min Dosya",
            "min_sc":   "Min Subcase",
        }
        for col, hdr in headers.items():
            self.tv_gov.heading(col, text=hdr)
            self.tv_gov.column(col, width=130, anchor="center")
        self.tv_gov.pack(side="left", fill="both", expand=True,
                          padx=(4, 0), pady=4)
        sb3 = ttk.Scrollbar(frm_gov, orient="vertical",
                              command=self.tv_gov.yview)
        sb3.pack(side="left", fill="y", pady=4)
        self.tv_gov.configure(yscrollcommand=sb3.set)

        # ── Log ──────────────────────────────────────────────────────
        frm_log = ttk.LabelFrame(self.root, text="Log")
        frm_log.pack(fill="both", expand=False, **pad)
        self.log_text = scrolledtext.ScrolledText(
            frm_log, height=5, state="disabled",
            wrap="word", font=("Courier", 9))
        self.log_text.pack(fill="both", expand=True, padx=4, pady=4)

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _log(self, msg: str):
        def _append():
            self.log_text.config(state="normal")
            self.log_text.insert("end", msg + "\n")
            self.log_text.see("end")
            self.log_text.config(state="disabled")
        self.root.after(0, _append)

    def _clear_log(self):
        self.log_text.config(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.config(state="disabled")

    def _set_buttons(self, state: str):
        self.root.after(0, lambda: self.btn_scan.configure(state=state))
        self.root.after(0, lambda: self.btn_run.configure(state=state))

    # ── File helpers ────────────────────────────────────────────────────────

    def _add_files(self):
        paths = filedialog.askopenfilenames(
            title="Nastran dosyaları seç",
            filetypes=[("Nastran", "*.op2 *.h5 *.hdf5"), ("Tümü", "*.*")],
        )
        existing = list(self.lb_files.get(0, "end"))
        for p in paths:
            if p not in existing:
                self.lb_files.insert("end", p)

    def _remove_files(self):
        for idx in reversed(self.lb_files.curselection()):
            self.lb_files.delete(idx)

    def _browse_output(self):
        fmt = self.sv_fmt.get()
        ext = ".op2" if fmt == "op2" else ".h5"
        path = filedialog.asksaveasfilename(
            title="Envelope çıktı dosyası",
            defaultextension=ext,
            filetypes=[("OP2", "*.op2"), ("HDF5", "*.h5"), ("Tümü", "*.*")],
        )
        if path:
            self.sv_output.set(path)

    def _browse_excel(self):
        path = filedialog.asksaveasfilename(
            title="Excel raporu",
            defaultextension=".xlsx",
            filetypes=[("Excel", "*.xlsx")],
        )
        if path:
            self.sv_excel.set(path)

    def _toggle_excel(self):
        state = "normal" if self.bv_excel.get() else "disabled"
        self.ent_excel.configure(state=state)
        self.btn_xl_browse.configure(state=state)

    # ── Checkable treeview ──────────────────────────────────────────────────

    def _toggle_check(self, event):
        iid = self.tv_results.identify_row(event.y)
        if not iid:
            return
        new_state = not self._check_states.get(iid, True)
        self._check_states[iid] = new_state
        raw_text = self.tv_results.item(iid, "text").lstrip(
            f"{CHECK_ON}{CHECK_OFF} "
        )
        self.tv_results.item(
            iid, text=f"{CHECK_ON if new_state else CHECK_OFF} {raw_text}"
        )

    def _set_all(self, state: bool):
        for iid in self.tv_results.get_children():
            self._check_states[iid] = state
            raw_text = self.tv_results.item(iid, "text").lstrip(
                f"{CHECK_ON}{CHECK_OFF} "
            )
            self.tv_results.item(
                iid, text=f"{CHECK_ON if state else CHECK_OFF} {raw_text}"
            )

    # ── Scan ────────────────────────────────────────────────────────────────

    def _scan_files(self):
        files = list(self.lb_files.get(0, "end"))
        if not files:
            messagebox.showwarning("Eksik", "Önce dosya ekleyin.")
            return
        self._clear_log()
        self._set_buttons("disabled")
        threading.Thread(target=self._worker_scan, args=(files,),
                          daemon=True).start()

    def _worker_scan(self, files: list[str]):
        try:
            self._log(f"{len(files)} dosya taranıyor...")
            discovered = discover_results(files)
            self._discovered = discovered
            self.root.after(0, lambda: self._populate_results_tree(discovered))
            self._log(f"{len(discovered)} result tipi bulundu.")
        except Exception:
            err = traceback.format_exc()
            self._log(f"HATA:\n{err}")
            self.root.after(0, lambda: messagebox.showerror("Tarama Hatası", err))
        finally:
            self._set_buttons("normal")

    def _populate_results_tree(self, discovered: dict):
        for iid in self.tv_results.get_children():
            self.tv_results.delete(iid)
        self._check_states.clear()
        for rt, info in sorted(discovered.items()):
            iid = self.tv_results.insert(
                "", "end",
                text=f"{CHECK_ON} {rt}",
                values=(len(info["subcases"]), info["n_files"]),
            )
            self._check_states[iid] = True

    # ── Run ─────────────────────────────────────────────────────────────────

    def _run(self):
        files = list(self.lb_files.get(0, "end"))
        if not files:
            messagebox.showwarning("Eksik", "Dosya ekleyin.")
            return

        selected_types = [
            self.tv_results.item(iid, "text").lstrip(f"{CHECK_ON}{CHECK_OFF} ")
            for iid, checked in self._check_states.items()
            if checked
        ]
        if not selected_types:
            messagebox.showwarning("Eksik", "En az bir result tipi seçin.")
            return

        output_path = self.sv_output.get().strip()
        if not output_path:
            messagebox.showwarning("Eksik", "Çıktı dosyası yolu belirtin.")
            return

        excel_path = None
        if self.bv_excel.get():
            excel_path = self.sv_excel.get().strip()
            if not excel_path:
                messagebox.showwarning("Eksik", "Excel dosya yolu belirtin.")
                return

        fmt = self.sv_fmt.get()
        self._clear_log()
        self._set_buttons("disabled")
        threading.Thread(
            target=self._worker_run,
            args=(files, selected_types, output_path, fmt, excel_path),
            daemon=True,
        ).start()

    def _worker_run(self, files, selected_types, output_path, fmt, excel_path):
        try:
            self._log(f"Seçilen tipler: {', '.join(selected_types)}")
            self._log(f"{len(files)} dosyadan veri okunuyor...")
            raw = collect_raw(files, selected_types)

            if not raw:
                self.root.after(0, lambda: messagebox.showerror(
                    "Sonuç Yok",
                    "Seçilen result tipleri dosyalarda bulunamadı."
                ))
                return

            self._log(f"Bulunan attribute'lar: {list(raw.keys())}")
            self._log("Envelope hesaplanıyor...")
            env_max, env_min, governing, templates, elem_ids = compute_envelope(raw)

            self._log(f"Çıktı yazılıyor → {output_path}")
            write_output(env_max, env_min, templates, output_path, fmt)

            self._log("Governing tablosu oluşturuluyor...")
            gov_df = build_governing_df(governing, elem_ids)
            self.root.after(0, lambda: self._populate_governing_table(gov_df))

            if excel_path and not gov_df.empty:
                self._log(f"Excel raporu → {excel_path}")
                export_governing_excel(gov_df, excel_path)

            self._log("\nTamamlandı.")
            self.root.after(0, lambda: messagebox.showinfo(
                "Tamamlandı",
                f"Envelope dosyası oluşturuldu:\n{output_path}"
            ))

        except Exception:
            err = traceback.format_exc()
            self._log(f"HATA:\n{err}")
            self.root.after(0, lambda: messagebox.showerror("Hata", err))
        finally:
            self._set_buttons("normal")

    def _populate_governing_table(self, df: pd.DataFrame):
        for iid in self.tv_gov.get_children():
            self.tv_gov.delete(iid)
        if df.empty:
            return
        for _, row in df.head(2000).iterrows():  # cap at 2000 rows for perf
            self.tv_gov.insert("", "end", values=(
                row["Result_Type"],
                row["Element/Node_ID"],
                row["Max_Gov_File"],
                row["Max_Gov_Subcase"],
                row["Min_Gov_File"],
                row["Min_Gov_Subcase"],
            ))


# ─── Entry point ────────────────────────────────────────────────────────────

if __name__ == '__main__':
    _log_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "envelope_error.txt"
    )
    try:
        root = tk.Tk()
        app = LoadExtractionApp(root)
        root.mainloop()
    except Exception:
        err = traceback.format_exc()
        with open(_log_path, "w", encoding="utf-8") as _f:
            _f.write(err)
        try:
            messagebox.showerror("Başlatma Hatası", err)
        except Exception:
            pass

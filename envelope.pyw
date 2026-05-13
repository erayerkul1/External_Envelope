#!/usr/bin/env python3
from __future__ import annotations

import os
import struct as _struct
import sys
import threading
import traceback
import tkinter as tk
from copy import deepcopy
from tkinter import ttk, filedialog, messagebox, scrolledtext

try:
    import numpy as np
    import pandas as pd
    from pyNastran.op2.op2 import OP2, read_op2
    # np.float was removed in NumPy 1.20; old openpyxl versions still reference it.
    if not hasattr(np, 'float'):
        np.float = float   # type: ignore[attr-defined]
    _DEPS_OK = True
    _DEPS_ERR = ""
except ImportError as _e:
    np = pd = OP2 = read_op2 = None  # type: ignore
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
    "eigenvectors":   ("eigenvectors",),
    "temperatures":   ("temperatures",),
    "spc_forces":     ("spc_forces",),
    "mpc_forces":     ("mpc_forces",),
    "gpforce":        ("grid_point_forces",),
    "oload":          ("load_vectors", "load_vectors_v", "force_vectors"),
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
    "rod_force":      ("crod_force", "conrod_force", "ctube_force"),
    "spring_force":   ("celas1_force", "celas2_force", "celas3_force", "celas4_force"),
    "bush_force":     ("cbush_force",),
    "gap_force":      ("cgap_force",),
}

CHECK_ON  = "☑"
CHECK_OFF = "☐"


class _ProgressTimer:
    """Logs elapsed seconds every 5 s while a blocking operation runs."""

    def __init__(self, log_fn, interval: int = 30):
        self._log = log_fn
        self._interval = interval
        self._stop_evt = threading.Event()
        self._t0 = 0.0

    def start(self):
        import time
        self._t0 = time.time()
        self._stop_evt.clear()
        threading.Thread(target=self._run, daemon=True).start()

    def stop(self):
        self._stop_evt.set()

    def _run(self):
        import time
        while not self._stop_evt.wait(timeout=self._interval):
            elapsed = int(time.time() - self._t0)
            self._log(f"  ... hâlâ okunuyor ({elapsed}s geçti, lütfen bekleyin)")


# ─── Backend ────────────────────────────────────────────────────────────────

def _load_model(filepath: str, log_fn=None, load_geometry: bool = False):
    ext = os.path.splitext(filepath)[1].lower()

    try:
        size_mb = os.path.getsize(filepath) / 1024 / 1024
        if log_fn:
            log_fn(f"  Dosya boyutu: {size_mb:.1f} MB — okunuyor, lütfen bekleyin...")
    except OSError:
        pass

    if ext == ".op2" and load_geometry:
        # read_op2() returns an OP2Geom which includes GEOM1/GEOM2/EPT/MPT
        # tables. write_op2() then writes those tables so HyperView can
        # validate and attach results.
        model = read_op2(filepath, load_geometry=True,
                         combine=True, log=None, debug=False)
    elif ext == ".op2":
        model = OP2(debug=False, log=None)
        model.IS_TESTING = False
        model.read_op2(filepath, combine=True)
    else:
        model = OP2(debug=False, log=None)
        model.IS_TESTING = False
        model.load_hdf5_filename(filepath)
    return model


def discover_results(filepaths: list[str], log_fn=None) -> dict:
    """Scan files, return what result types are present.

    Returns
    -------
    {result_type: {"subcases": [(filepath, sc_id), ...], "n_files": int}}
    """
    discovered: dict = {}
    for i, filepath in enumerate(filepaths):
        if log_fn:
            log_fn(f"[{i+1}/{len(filepaths)}] Taranıyor: {os.path.basename(filepath)}")
        model = _load_model(filepath, log_fn=log_fn)
        if log_fn:
            log_fn(f"  Dosya okundu. Result tipleri aranıyor...")
        for rt, attrs in RESULT_ATTRIBUTES.items():
            for attr in attrs:
                rd = getattr(model, attr, {})
                if not rd:
                    continue
                entry = discovered.setdefault(rt, {"subcases": [], "_files": set()})
                for sc_id in rd:
                    # store as (filepath, sc_id, attr) to allow dedup by sc_id
                    entry["subcases"].append((filepath, sc_id, attr))
                entry["_files"].add(filepath)
    for rt in discovered:
        discovered[rt]["n_files"] = len(discovered[rt].pop("_files"))
        # unique subcase IDs (deduplicated across multiple element type attrs)
        seen = set()
        unique = []
        for item in discovered[rt]["subcases"]:
            key = (item[0], item[1])   # (filepath, sc_id)
            if key not in seen:
                seen.add(key)
                unique.append(item)
        discovered[rt]["subcases"] = unique
    return discovered


def collect_raw(filepaths: list[str], selected_types: list[str],
                log_fn=None) -> dict:
    """Read files and collect raw pyNastran result objects.

    Returns
    -------
    {attr: [(filepath, sc_id, result_obj), ...]}
    """
    raw: dict = {}
    nastran_fmt: str = 'msc'
    first_model = None
    for i, filepath in enumerate(filepaths):
        if log_fn:
            log_fn(f"[{i+1}/{len(filepaths)}] Okunuyor: {os.path.basename(filepath)}")
        if first_model is None:
            # Load geometry tables on the first file so write_op2 includes
            # GEOM1/GEOM2/EPT/MPT — required for HyperView to attach results.
            model = _load_model(filepath, log_fn=log_fn, load_geometry=True)
            first_model = model
            if log_fn:
                log_fn("  Geometry tablolar yüklendi (HyperView uyumu için).")
        else:
            model = _load_model(filepath, log_fn=log_fn, load_geometry=False)
        fmt = getattr(model, 'nastran_format', None)
        if fmt in ('msc', 'nx', 'optistruct'):
            nastran_fmt = fmt
        if log_fn:
            log_fn(f"  Dosya okundu (nastran_format={nastran_fmt}). Seçili result tipleri yükleniyor...")
        for rt in selected_types:
            for attr in RESULT_ATTRIBUTES.get(rt, ()):
                rd = getattr(model, attr, {})
                if not rd:
                    continue
                for sc_id, result_obj in rd.items():
                    raw.setdefault(attr, []).append((filepath, sc_id, result_obj))
        if log_fn:
            n = sum(len(v) for v in raw.values())
            log_fn(f"  Toplam {n} subcase yüklendi.")
    return raw, nastran_fmt, first_model


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


def write_msc_displacement_op2(
        node_ids, gridtypes, disp_data, output_path,
        isubcase=1, lsdvmns=1,
        approach_code=12, table_code=1,
        num_wide=8, random_code=0, thermal=0, format_code=1,
        title='', subtitle='', label='',
        date=None):
    """Write a HyperView-compatible MSC Nastran binary OP2 with OUGV1 displacements.

    Uses raw struct.pack to produce the exact Fortran-record byte format that
    MSC Nastran itself writes (and that HyperView's OP2 reader expects).

    pyNastran's write_op2 writes multiple result tables together (displacement,
    gpforce, spc_forces, load_vectors, …). At least one of those extra tables
    appears in a format that confuses HyperView's parser, causing it to report
    "No results/supported Result Blocks found".  Writing ONLY the single OUGV1
    displacement table avoids this problem entirely.

    approach_code, table_code, num_wide, random_code, thermal, format_code
    should be extracted from the original displacement result template so that
    HyperView recognises the analysis type correctly.
    """
    from datetime import datetime as _dt
    if date is None:
        t = _dt.today()
        date = (t.month, t.day, t.year)
    month, day, year = date
    dyear = year - 2000

    node_ids  = np.asarray(node_ids,  dtype=np.int32)
    gridtypes = np.asarray(gridtypes, dtype=np.int32)
    disp_data = np.asarray(disp_data, dtype=np.float32)
    if disp_data.ndim == 1:
        disp_data = disp_data.reshape(-1, 6)
    disp_data = disp_data[:, :6]
    nnodes = len(node_ids)

    device_code = 2   # Plot

    # Per-node data: [node*10+device, gridtype, t1..t6] written as float32 bytes
    dev_ids    = (node_ids * 10 + device_code).astype(np.int32)
    ids_gt     = np.column_stack([dev_ids, gridtypes])   # (nnodes, 2) int32
    ids_gt_f32 = ids_gt.view(np.float32)                  # same bytes, float32 view
    row_data   = np.hstack([ids_gt_f32, disp_data])       # (nnodes, 8) float32
    data_bytes = row_data.tobytes()
    ntotal     = nnodes * 8   # number of float32 words

    P = _struct.pack

    def rec(fmt, *vals):
        """Fortran record: [length][data][length]"""
        d = P('<' + fmt, *vals)
        n = len(d)
        return P('<i', n) + d + P('<i', n)

    def mk(v):
        """Single-word marker record: [4][v][4]"""
        return P('<iii', 4, v, 4)

    def _pad128(s):
        b = s[:128].encode('ascii', errors='replace')
        return b.ljust(128, b' ')[:128]

    # Table3 payload: 50 int32 words + 3 × 128-byte strings = 584 bytes
    # Layout matches pyNastran _write_table_3 exactly (words 1-50):
    #   1:approach_code  2:table_code  3:0           4:isubcase     5:lsdvmns
    #   6:field6=0       7:field7=0    8:random_code 9:format_code  10:num_wide
    #  11:oCode=0       12:acoustic=0 13-15:0
    #  16-20:0
    #  21-22:0          23:thermal    24-33:0
    #  34-46:0
    #  47-50:0
    ti = [
        approach_code, table_code, 0, isubcase, lsdvmns,   # words 1-5
        0, 0, random_code, format_code, num_wide,           # words 6-10
        0, 0, 0, 0, 0,                                      # words 11-15
        0, 0, 0, 0, 0,                                      # words 16-20
        0, 0, thermal, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,       # words 21-33
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,             # words 34-46
        0, 0, 0, 0,                                         # words 47-50
    ]   # total: 50 integers
    assert len(ti) == 50
    t3_bytes = P('<50i', *ti) + _pad128(title) + _pad128(subtitle) + _pad128(label)
    assert len(t3_bytes) == 584

    tape_code = b'NASTRAN FORT TAPE ID CODE - '   # exactly 28 bytes
    assert len(tape_code) == 28

    with open(output_path, 'wb') as f:
        # ── File header (PARAM,POST,-1 / MSC format) ────────────────────
        f.write(mk(3))                                   # marker = 3
        f.write(rec('3i', day, month, dyear))            # date record
        f.write(mk(7))                                   # marker = 7
        f.write(rec('28s', tape_code))                   # tape code string
        f.write(mk(2))                                   # marker = 2
        f.write(rec('8s', b'XXXXXXXX'))                  # Nastran version
        f.write(mk(-1))                                  # end-header marker
        f.write(mk(0))

        # ── OUGV1 result table header ────────────────────────────────────
        # Table-name block: marker=2 then 8-char name
        f.write(P('<iii', 4, 2, 4))
        f.write(P('<i8si', 8, b'OUGV1   ', 8))

        f.write(mk(-1))
        f.write(mk(7))
        f.write(rec('7i', 102, 0, 0, 0, 512, 0, 0))     # table info block

        f.write(mk(-2))
        f.write(mk(1))
        f.write(mk(0))

        f.write(mk(7))
        f.write(rec('8siiiii', b'OUG1    ', month, day, dyear, 0, 1))

        # ── TABLE3 — subcase descriptor (itable = -3) ────────────────────
        itable = -3
        f.write(mk(itable)); f.write(mk(1)); f.write(mk(0)); f.write(mk(146))
        f.write(P('<i', 584) + t3_bytes + P('<i', 584))

        # ── Data record (itable = -4) ────────────────────────────────────
        itable -= 1   # -4
        f.write(mk(itable)); f.write(mk(1)); f.write(mk(0)); f.write(mk(ntotal))
        rec_len = ntotal * 4
        f.write(P('<i', rec_len) + data_bytes + P('<i', rec_len))

        # ── End-of-subcase markers (itable = -5) ────────────────────────
        itable -= 1   # -5
        f.write(mk(itable)); f.write(mk(1)); f.write(mk(0))

        # ── End-of-table + end-of-file ───────────────────────────────────
        f.write(mk(0))   # close result table
        f.write(mk(0))   # close file


def write_output(env_max: dict, env_min: dict, templates: dict,
                 output_path: str, fmt: str,
                 nastran_format: str = 'msc',
                 out_model=None) -> list:
    """Write displacement envelope as OP2 or write all results to HDF5.

    For fmt='op2':
        Writes displacement (OUGV1) using raw struct.pack to produce the exact
        MSC Nastran binary format that HyperView can read.  pyNastran's
        write_op2 produces a subtly different format that HyperView rejects.
        Returns a list of attribute names NOT written to OP2 — caller should
        export those to Excel.

    For fmt='h5':
        Writes all result types via pyNastran export_hdf5.  Returns [].
    """
    if fmt == "op2":
        disp_attrs = set(RESULT_ATTRIBUTES.get("displacements", ()))
        written = []

        for attr, max_arr in env_max.items():
            if attr not in disp_attrs:
                continue
            template = templates[attr]
            if not hasattr(template, "node_gridtype"):
                continue

            node_ids  = template.node_gridtype[:, 0]
            gridtypes = template.node_gridtype[:, 1]

            # Preserve original subcase/lsdvmns so HyperView matches the BDF.
            isubcase = int(getattr(template, "isubcase", 1) or 1)
            lsdvmns  = isubcase
            if hasattr(template, "lsdvmns") and len(template.lsdvmns) > 0:
                lsdvmns = int(template.lsdvmns[0])
                isubcase = lsdvmns

            # Extract TABLE3 codes from the original result so HyperView
            # recognises the result type correctly.
            approach_code = int(getattr(template, "approach_code", 12) or 12)
            table_code    = int(getattr(template, "table_code",    1)  or 1)
            num_wide      = int(getattr(template, "num_wide",      8)  or 8)
            random_code   = int(getattr(template, "random_code",   0))
            thermal       = int(getattr(template, "thermal",       0))
            format_code   = int(getattr(template, "format_code",   1)  or 1)

            title    = getattr(template, "title",    "") or ""
            subtitle = getattr(template, "subtitle", "") or ""
            label    = getattr(template, "label",    "") or ""

            write_msc_displacement_op2(
                node_ids, gridtypes, max_arr, output_path,
                isubcase=isubcase, lsdvmns=lsdvmns,
                approach_code=approach_code, table_code=table_code,
                num_wide=num_wide, random_code=random_code,
                thermal=thermal, format_code=format_code,
                title=title, subtitle=subtitle, label=label,
            )
            written.append(attr)
            break   # only one displacement attribute → OP2

        if not written:
            raise RuntimeError(
                "Displacement (OUGV1) verisi bulunamadı.\n"
                "OP2 çıktısı için seçili tipler arasında 'displacements' olmalı."
            )

        not_written = [a for a in env_max if a not in written]
        return not_written

    else:   # h5
        if out_model is None:
            raise RuntimeError(
                "write_output: h5 için temel model gerekli (out_model=None)"
            )
        for attrs in RESULT_ATTRIBUTES.values():
            for attr in attrs:
                if getattr(out_model, attr, {}):
                    setattr(out_model, attr, {})

        for attr, max_arr in env_max.items():
            template = templates[attr]
            res_max = deepcopy(template)
            res_max.data = max_arr[np.newaxis, :, :]
            try:
                res_max.isubcase = 1
                if hasattr(res_max, "lsdvmns"):
                    res_max.lsdvmns = np.array([1], dtype=res_max.lsdvmns.dtype)
                if hasattr(res_max, "dts"):
                    res_max.dts = np.array([0.0], dtype=res_max.dts.dtype)
            except Exception:
                pass
            setattr(out_model, attr, {1: res_max})

        out_model.export_hdf5(output_path)
        return []


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


def export_envelope_excel(
        env_max: dict, env_min: dict,
        elem_ids: dict, governing: dict,
        path: str, templates: dict = None) -> None:
    """Write envelope values and governing info to Excel.

    One sheet per result attribute (max and min columns), plus a 'governing'
    sheet showing which subcase/file drives each element.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for attr, max_arr in sorted(env_max.items()):
            min_arr = env_min.get(attr)
            ids     = elem_ids.get(attr, list(range(max_arr.shape[0])))

            # Try to get meaningful column header names from the template
            hdrs: list = []
            if templates and attr in templates:
                raw_hdrs = getattr(templates[attr], "headers", None)
                if raw_hdrs:
                    hdrs = list(raw_hdrs)

            n_comp = max_arr.shape[1]
            if len(hdrs) != n_comp:
                hdrs = [f"comp{i+1}" for i in range(n_comp)]

            row: dict = {"ID": ids}
            for j, h in enumerate(hdrs):
                row[f"max_{h}"] = max_arr[:, j]
            if min_arr is not None:
                for j, h in enumerate(hdrs):
                    row[f"min_{h}"] = min_arr[:, j]

            pd.DataFrame(row).to_excel(
                writer, sheet_name=attr[:31], index=False
            )

        # Governing sheet
        gov_df = build_governing_df(governing, elem_ids)
        if not gov_df.empty:
            gov_df.to_excel(writer, sheet_name="governing"[:31], index=False)

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

        # ── Excel çıktısı ─────────────────────────────────────────────
        frm_xl = ttk.LabelFrame(
            self.root,
            text="Excel Çıktısı  (.op2 modunda kuvvet/gerilme tipleri otomatik eklenir)")
        frm_xl.pack(fill="x", **pad)

        self.bv_excel = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            frm_xl,
            text="Excel yolu belirt  (belirtilmezse OP2 yanına otomatik kaydedilir)",
            variable=self.bv_excel,
            command=self._toggle_excel).pack(anchor="w", padx=4, pady=2)
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
        timer = _ProgressTimer(self._log)
        try:
            self._log(f"{len(files)} dosya taranıyor...")
            timer.start()
            discovered = discover_results(files, log_fn=self._log)
            timer.stop()
            self._discovered = discovered
            self.root.after(0, lambda: self._populate_results_tree(discovered))
            self._log(f"{len(discovered)} result tipi bulundu.")
        except Exception:
            timer.stop()
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
        timer = _ProgressTimer(self._log)
        try:
            self._log(f"Seçilen tipler: {', '.join(selected_types)}")
            self._log(f"{len(files)} dosyadan veri okunuyor...")
            timer.start()
            raw, nastran_fmt, first_model = collect_raw(
                files, selected_types, log_fn=self._log)
            timer.stop()

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
            not_in_op2 = write_output(
                env_max, env_min, templates, output_path, fmt,
                nastran_format=nastran_fmt, out_model=first_model)

            # For OP2 format: non-displacement results go to Excel automatically.
            # If the user specified an Excel path, use that; otherwise auto-name it.
            if fmt == "op2" and not_in_op2:
                xl_target = excel_path or (
                    os.path.splitext(output_path)[0] + "_results.xlsx")
                self._log(
                    f"Displacement dışı result tipleri Excel'e aktarılıyor "
                    f"({', '.join(not_in_op2)}) → {xl_target}"
                )
                extra_max = {a: env_max[a] for a in not_in_op2 if a in env_max}
                extra_min = {a: env_min[a] for a in not_in_op2 if a in env_min}
                extra_ids = {a: elem_ids[a] for a in not_in_op2 if a in elem_ids}
                extra_gov = {a: governing[a] for a in not_in_op2 if a in governing}
                export_envelope_excel(extra_max, extra_min, extra_ids,
                                      extra_gov, xl_target, templates=templates)
                finish_msg = (
                    f"OP2 (displacement):\n  {output_path}\n\n"
                    f"Excel (diğer result tipleri + governing):\n  {xl_target}"
                )
            elif fmt != "op2" and excel_path:
                # H5 mode: write governing-only Excel if user requested it
                gov_df = build_governing_df(governing, elem_ids)
                self._log(f"Excel raporu → {excel_path}")
                export_governing_excel(gov_df, excel_path)
                finish_msg = (
                    f"H5 dosyası:\n  {output_path}\n\n"
                    f"Excel raporu:\n  {excel_path}"
                )
            else:
                finish_msg = f"Envelope dosyası oluşturuldu:\n{output_path}"

            self._log("Governing tablosu oluşturuluyor...")
            gov_df = build_governing_df(governing, elem_ids)
            self.root.after(0, lambda: self._populate_governing_table(gov_df))

            self._log("\nTamamlandı.")
            self.root.after(0, lambda: messagebox.showinfo(
                "Tamamlandı", finish_msg))

        except Exception:
            timer.stop()
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

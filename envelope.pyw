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


def collect_and_envelope(filepaths, selected_types, log_fn=None):
    """Single-pass streaming envelope — O(1) peak memory vs O(N) for collect_raw+compute_envelope.

    Returns: (env_max, env_min, governing, templates, elem_ids, nastran_fmt, first_model)
    """
    import time as _time
    env_max = {}; env_min = {}
    abs_sum_max = {}; abs_sum_min = {}
    gov_file_max = {}; gov_sc_max = {}
    gov_file_min = {}; gov_sc_min = {}
    templates = {}; elem_ids = {}
    nastran_fmt = 'msc'; first_model = None
    t0 = _time.time()

    for i, filepath in enumerate(filepaths):
        if log_fn:
            log_fn(f"[{i+1}/{len(filepaths)}] {os.path.basename(filepath)}  "
                   f"({int(_time.time()-t0)}s geçti)")
        load_geom = (first_model is None)
        model = _load_model(filepath, log_fn=log_fn, load_geometry=load_geom)
        if first_model is None:
            first_model = model
        fmt = getattr(model, 'nastran_format', None)
        if fmt in ('msc', 'nx', 'optistruct'):
            nastran_fmt = fmt

        for rt in selected_types:
            for attr in RESULT_ATTRIBUTES.get(rt, ()):
                rd = getattr(model, attr, {})
                if not rd:
                    continue
                for sc_id, result_obj in rd.items():
                    try:
                        arr = result_obj.data[-1].astype(np.float32, copy=False)
                    except Exception:
                        continue
                    abs_s = np.abs(arr).sum(axis=1)   # (nelems,)

                    if attr not in env_max:
                        env_max[attr]      = arr.copy()
                        env_min[attr]      = arr.copy()
                        abs_sum_max[attr]  = abs_s.copy()
                        abs_sum_min[attr]  = abs_s.copy()
                        n = len(abs_s)
                        gov_file_max[attr] = np.full(n, filepath, dtype=object)
                        gov_sc_max[attr]   = np.full(n, sc_id,   dtype=object)
                        gov_file_min[attr] = np.full(n, filepath, dtype=object)
                        gov_sc_min[attr]   = np.full(n, sc_id,   dtype=object)
                        templates[attr]    = result_obj
                        elem_ids[attr]     = _get_ids(result_obj)
                    else:
                        if arr.shape != env_max[attr].shape:
                            continue
                        np.maximum(env_max[attr], arr, out=env_max[attr])
                        np.minimum(env_min[attr], arr, out=env_min[attr])
                        mask = abs_s > abs_sum_max[attr]
                        if mask.any():
                            abs_sum_max[attr][mask]  = abs_s[mask]
                            gov_file_max[attr][mask] = filepath
                            gov_sc_max[attr][mask]   = sc_id
                        mask = abs_s < abs_sum_min[attr]
                        if mask.any():
                            abs_sum_min[attr][mask]  = abs_s[mask]
                            gov_file_min[attr][mask] = filepath
                            gov_sc_min[attr][mask]   = sc_id

    governing = {
        attr: {
            "max": list(zip(gov_file_max[attr], gov_sc_max[attr])),
            "min": list(zip(gov_file_min[attr], gov_sc_min[attr])),
        }
        for attr in env_max
    }
    if log_fn:
        log_fn(f"Envelope tamamlandı. ({int(_time.time()-t0)}s)")
    return env_max, env_min, governing, templates, elem_ids, nastran_fmt, first_model


# ─── Raw MSC Nastran binary OP2 writer ─────────────────────────────────────
# Attribute sets used to classify results for OP2 table type selection.
# Nodal results share the OUGV1 TABLE3 layout (word 3 = 0, word 8 = random_code).
# Element results use the OES/OEF TABLE3 layout (word 3 = element_type, word 8 = load_set).

_OP2_NODAL_ATTRS = frozenset([
    'displacements', 'velocities', 'accelerations', 'eigenvectors', 'temperatures',
    'spc_forces', 'mpc_forces',
])
# Centroid plate stress/strain: element_node.shape=(2*N,2), data.shape=(2*N,8), num_wide=17
_OP2_PLATE_STRESS_ATTRS = frozenset([
    'cquad4_stress', 'ctria3_stress', 'cquadr_stress', 'ctriar_stress',
    'cquad4_strain', 'ctria3_strain', 'cquadr_strain', 'ctriar_strain',
])
# Plate forces: element.shape=(N,), data.shape=(N,8), num_wide=9
_OP2_PLATE_FORCE_ATTRS = frozenset([
    'cquad4_force', 'ctria3_force',
])
# Centroid element types supported by _prep_plate_stress_data
_CENTROID_ELEMENT_TYPES = frozenset([33, 74, 227, 228])


def _mk(v: int) -> bytes:
    """Single-word Fortran marker: [4, v, 4]"""
    return _struct.pack('<iii', 4, v, 4)


def _frec(fmt: str, *vals) -> bytes:
    """Fortran record: [n][payload][n]"""
    d = _struct.pack('<' + fmt, *vals)
    n = len(d)
    return _struct.pack('<i', n) + d + _struct.pack('<i', n)


def _pad128(s: str) -> bytes:
    b = s[:128].encode('ascii', errors='replace')
    return b.ljust(128, b' ')[:128]


def _op2_file_header(month: int, day: int, dyear: int) -> bytes:
    """MSC Nastran OP2 file header (PARAM,POST,-1 format)."""
    tape = b'NASTRAN FORT TAPE ID CODE - '
    out  = _mk(3)
    out += _frec('3i', day, month, dyear)
    out += _mk(7)
    out += _frec('28s', tape)
    out += _mk(2)
    out += _frec('8s', b'XXXXXXXX')
    out += _mk(-1)
    out += _mk(0)
    return out


def _op2_table_header_block(table_name: str, subtable_name: bytes,
                             month: int, day: int, dyear: int) -> bytes:
    """Write one result-table's opening: table-name block + date record."""
    tname8 = ('%-8s' % table_name[:8]).encode('ascii')
    sub8   = b'%-8s' % subtable_name[:8]
    out  = _struct.pack('<iii',  4, 2, 4)
    out += _struct.pack('<i8si', 8, tname8, 8)
    out += _struct.pack('<6i',   4, -1, 4, 4, 7, 4)
    out += _struct.pack('<9i',   28, 102, 0, 0, 0, 512, 0, 0, 28)
    out += _struct.pack('<9i',   4, -2, 4, 4, 1, 4, 4, 0, 4)
    out += _struct.pack('<iii',  4, 7, 4)
    out += _struct.pack('<i', 28) + sub8
    out += _struct.pack('<iiiii', month, day, dyear, 0, 1)
    out += _struct.pack('<i', 28)
    return out


def _op2_table3_block(t3_words: list, title: str, subtitle: str, label: str) -> bytes:
    """Write TABLE3 descriptor (itable=-3): 50 int32 words + 3×128-byte strings."""
    assert len(t3_words) == 50
    payload = _struct.pack('<50i', *t3_words) + _pad128(title) + _pad128(subtitle) + _pad128(label)
    assert len(payload) == 584
    out  = _mk(-3); out += _mk(1); out += _mk(0); out += _mk(146)
    out += _struct.pack('<i', 584) + payload + _struct.pack('<i', 584)
    return out


def _op2_data_and_end(data_bytes: bytes, ntotal: int) -> bytes:
    """Write data record (itable=-4) then end-of-subcase markers (itable=-5)."""
    rec_len = ntotal * 4
    assert len(data_bytes) == rec_len, f'{len(data_bytes)} != {rec_len}'
    out  = _mk(-4); out += _mk(1); out += _mk(0); out += _mk(ntotal)
    out += _struct.pack('<i', rec_len) + data_bytes + _struct.pack('<i', rec_len)
    out += _mk(-5); out += _mk(1); out += _mk(0)
    return out


def _t3_ougv(approach_code, table_code, isubcase, lsdvmns,
             random_code, format_code, num_wide, thermal) -> list:
    """50-word TABLE3 for OUGV1/OQG1/OQMG1 (nodal). Word 3=0, word 8=random_code."""
    return [
        approach_code, table_code, 0, isubcase, lsdvmns,
        0, 0, random_code, format_code, num_wide,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, thermal, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0,
    ]


def _t3_oes(approach_code, table_code, element_type, isubcase, lsdvmns,
            load_set, format_code, num_wide, s_code, thermal) -> list:
    """50-word TABLE3 for OES1X/OEF1X (element). Word 3=element_type, word 8=load_set."""
    return [
        approach_code, table_code, element_type, isubcase, lsdvmns,
        0, 0, load_set, format_code, num_wide,
        s_code, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, thermal, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0,
    ]


def _get_lsdvmns(template) -> int:
    """Return lsdvmns[0] from template, falling back to isubcase."""
    isubcase = int(getattr(template, 'isubcase', 1) or 1)
    lsv = getattr(template, 'lsdvmns', None)
    if lsv is not None and len(lsv) > 0:
        return int(lsv[0])
    return isubcase


def _prep_nodal_data(template, env_data):
    """(data_bytes, ntotal) for OUGV1-style nodal results (disp/vel/accel/spc/mpc)."""
    if not hasattr(template, 'node_gridtype'):
        return None
    node_ids  = template.node_gridtype[:, 0].astype(np.int32)
    gridtypes = template.node_gridtype[:, 1].astype(np.int32)
    data = np.asarray(env_data, dtype=np.float32)
    if data.ndim == 1:
        data = data.reshape(-1, 6)
    data = data[:, :6]
    if len(data) != len(node_ids):
        return None
    dev = int(getattr(template, 'device_code', 2) or 2)
    dev_ids    = (node_ids * 10 + dev).astype(np.int32)
    ids_gt     = np.column_stack([dev_ids, gridtypes]).view(np.float32)
    row_data   = np.hstack([ids_gt, data])
    return row_data.tobytes(), len(node_ids) * 8


def _prep_plate_stress_data(template, env_data):
    """(data_bytes, ntotal) for OES1X centroid plate stress/strain.
    element_node.shape=(2*N,2), env_data.shape=(2*N,8), num_wide=17.
    Each element row: [eid_device_as_f32, 16 float32 values].
    """
    if not hasattr(template, 'element_node'):
        return None
    etype = getattr(template, 'element_type', None)
    if etype not in _CENTROID_ELEMENT_TYPES:
        return None
    eids        = template.element_node[:, 0].astype(np.int32)
    nelements   = len(np.unique(eids))
    if len(eids) != 2 * nelements:
        return None
    dev         = int(getattr(template, 'device_code', 2) or 2)
    eid_device  = (eids[::2] * 10 + dev).astype(np.int32)
    data_f32    = np.asarray(env_data, dtype=np.float32)
    if data_f32.shape != (2 * nelements, 8):
        return None
    data2       = data_f32.reshape(nelements, 16)
    out         = np.empty((nelements, 17), dtype=np.float32)
    out[:, 0]   = eid_device.view(np.float32)
    out[:, 1:]  = data2
    return out.tobytes(), nelements * 17


def _prep_plate_force_data(template, env_data):
    """(data_bytes, ntotal) for OEF1X plate forces (CQUAD4/CTRIA3).
    element.shape=(N,), env_data.shape=(N,8), num_wide=9.
    Each element row: [eid_device_as_f32, 8 float32 values].
    """
    if not hasattr(template, 'element'):
        return None
    eids        = template.element.astype(np.int32)
    nelements   = len(eids)
    dev         = int(getattr(template, 'device_code', 2) or 2)
    eid_device  = (eids * 10 + dev).astype(np.int32)
    data_f32    = np.asarray(env_data, dtype=np.float32)
    if data_f32.shape != (nelements, 8):
        return None
    out         = np.empty((nelements, 9), dtype=np.float32)
    out[:, 0]   = eid_device.view(np.float32)
    out[:, 1:]  = data_f32
    return out.tobytes(), nelements * 9


def _build_nodal_spec(attr, template, env_data):
    """Table spec dict for OUGV1/OQG1/OQMG1 nodal result."""
    prep = _prep_nodal_data(template, env_data)
    if prep is None:
        return None
    data_bytes, ntotal = prep
    lsdvmns     = _get_lsdvmns(template)
    approach    = int(getattr(template, 'approach_code', 12) or 12)
    tc          = int(getattr(template, 'table_code', 1) or 1)
    rand        = int(getattr(template, 'random_code', 0))
    thermal     = int(getattr(template, 'thermal', 0))
    fmt_code    = int(getattr(template, 'format_code', 1) or 1)
    tname = getattr(template, 'table_name', None) or (
        'OQG1' if attr == 'spc_forces' else
        'OQMG1' if attr == 'mpc_forces' else 'OUGV1')
    sub = getattr(template, 'subtable_name', None) or (
        b'OQG1    ' if attr == 'spc_forces' else
        b'OQMG1   ' if attr == 'mpc_forces' else b'OUG1    ')
    if isinstance(sub, str):
        sub = sub.encode('ascii')
    return {
        'table_name':    tname,
        'subtable_name': sub,
        'table3_words':  _t3_ougv(approach, tc, lsdvmns, lsdvmns, rand, fmt_code, 8, thermal),
        'data_bytes':    data_bytes,
        'ntotal':        ntotal,
        'title':    getattr(template, 'title',    '') or '',
        'subtitle': getattr(template, 'subtitle', '') or '',
        'label':    getattr(template, 'label',    '') or '',
    }


def _build_plate_stress_spec(attr, template, env_data):
    """Table spec dict for OES1X centroid plate stress/strain."""
    prep = _prep_plate_stress_data(template, env_data)
    if prep is None:
        return None
    data_bytes, ntotal = prep
    lsdvmns  = _get_lsdvmns(template)
    approach = int(getattr(template, 'approach_code', 12) or 12)
    tc       = int(getattr(template, 'table_code', 5) or 5)
    etype    = int(getattr(template, 'element_type', 33) or 33)
    ls       = int(getattr(template, 'load_set', 1) or 1)
    s_code   = int(getattr(template, 's_code', 0))
    thermal  = int(getattr(template, 'thermal', 0))
    fmt_code = int(getattr(template, 'format_code', 1) or 1)
    tname = getattr(template, 'table_name', None) or 'OES1X'
    sub   = getattr(template, 'subtable_name', None) or b'OES1    '
    if isinstance(sub, str):
        sub = sub.encode('ascii')
    return {
        'table_name':    tname,
        'subtable_name': sub,
        'table3_words':  _t3_oes(approach, tc, etype, lsdvmns, lsdvmns, ls, fmt_code, 17, s_code, thermal),
        'data_bytes':    data_bytes,
        'ntotal':        ntotal,
        'title':    getattr(template, 'title',    '') or '',
        'subtitle': getattr(template, 'subtitle', '') or '',
        'label':    getattr(template, 'label',    '') or '',
    }


def _build_plate_force_spec(attr, template, env_data):
    """Table spec dict for OEF1X plate forces (CQUAD4/CTRIA3)."""
    prep = _prep_plate_force_data(template, env_data)
    if prep is None:
        return None
    data_bytes, ntotal = prep
    lsdvmns  = _get_lsdvmns(template)
    approach = int(getattr(template, 'approach_code', 12) or 12)
    tc       = int(getattr(template, 'table_code', 4) or 4)
    etype    = int(getattr(template, 'element_type', 33) or 33)
    ls       = int(getattr(template, 'load_set', 1) or 1)
    thermal  = int(getattr(template, 'thermal', 0))
    fmt_code = int(getattr(template, 'format_code', 1) or 1)
    tname = getattr(template, 'table_name', None) or 'OEF1X'
    sub   = getattr(template, 'subtable_name', None) or b'OEF1    '
    if isinstance(sub, str):
        sub = sub.encode('ascii')
    return {
        'table_name':    tname,
        'subtable_name': sub,
        'table3_words':  _t3_oes(approach, tc, etype, lsdvmns, lsdvmns, ls, fmt_code, 9, 0, thermal),
        'data_bytes':    data_bytes,
        'ntotal':        ntotal,
        'title':    getattr(template, 'title',    '') or '',
        'subtitle': getattr(template, 'subtitle', '') or '',
        'label':    getattr(template, 'label',    '') or '',
    }


_OP2_RESULT_TABLE_NAMES = frozenset([
    b'OUGV1   ', b'OUG1    ', b'OES1X   ', b'OES1    ', b'OES1C   ',
    b'OEF1X   ', b'OEF1    ', b'OQG1    ', b'OQMG1   ', b'OGPFB1  ',
    b'OLOAD1  ', b'OUGV1PAT', b'OSTR1X  ', b'OEFIT   ', b'ONRGY1  ',
    b'ONRGY2  ', b'OUGATO1 ', b'OUGCRM1 ', b'OUGPSD1 ', b'OUGRMS1 ',
])


def _extract_op2_geom_prefix(filepath: str) -> bytes:
    """Return raw bytes of all non-result tables (geometry prefix) from an OP2.

    Scans for [4][-1][4] (table-start marker) followed by a record whose
    content contains a known result table name.  Everything before the first
    such marker is the geometry prefix (NASTRAN header + GEOM1/GEOM2/EPT/MPT
    tables written by Nastran, in their original binary format).
    Returns b'' if parsing fails or no geometry prefix is found.
    """
    try:
        with open(filepath, 'rb') as fh:
            data = fh.read()
    except OSError:
        return b''

    marker = _struct.pack('<3i', 4, -1, 4)   # little-endian [4][-1][4]
    pos = 0
    while True:
        idx = data.find(marker, pos)
        if idx == -1:
            break
        rec_start = idx + 12             # skip 12-byte marker → Fortran record
        if rec_start + 8 > len(data):
            break
        rec_len = _struct.unpack_from('<i', data, rec_start)[0]
        if 0 < rec_len <= len(data) - rec_start - 8:
            content = data[rec_start + 4: rec_start + 4 + rec_len]
            for name in _OP2_RESULT_TABLE_NAMES:
                if name in content:
                    return data[:idx]    # geometry prefix found
        pos = idx + 1

    return b''


def write_msc_op2(table_specs: list, output_path: str, date=None,
                  geom_prefix: bytes = b'') -> None:
    """Write multiple result tables to a single HyperView-compatible MSC Nastran OP2.

    Uses raw struct.pack so only the requested tables appear in the file —
    pyNastran's write_op2 emits extra tables (GPFORCE, OLOAD, …) that confuse
    HyperView's parser and cause "No results/supported Result Blocks found".

    If geom_prefix is provided (raw bytes extracted from the source OP2 via
    _extract_op2_geom_prefix), it is written first instead of a synthetic
    NASTRAN file header, producing a standalone OP2 with embedded geometry.

    Each spec in table_specs:
        table_name   : str    8-char table name  (e.g. 'OUGV1', 'OES1X')
        subtable_name: bytes  8-char subtable     (e.g. b'OUG1    ')
        table3_words : list[int]  50 TABLE3 int32 words
        data_bytes   : bytes  packed float32 result data
        ntotal       : int    number of float32 words in data_bytes
        title/subtitle/label : str  (optional)
    """
    from datetime import datetime as _dt
    if date is None:
        t = _dt.today()
        date = (t.month, t.day, t.year)
    month, day, year = date
    dyear = year - 2000

    with open(output_path, 'wb') as f:
        if geom_prefix:
            f.write(geom_prefix)          # original NASTRAN header + GEOM tables
        else:
            f.write(_op2_file_header(month, day, dyear))
        for spec in table_specs:
            f.write(_op2_table_header_block(
                spec['table_name'], spec['subtable_name'], month, day, dyear))
            f.write(_op2_table3_block(
                spec['table3_words'],
                spec.get('title', ''), spec.get('subtitle', ''), spec.get('label', '')))
            f.write(_op2_data_and_end(spec['data_bytes'], spec['ntotal']))
            f.write(_mk(0))   # close this table
        f.write(_mk(0))       # EOF


def write_output(env_max: dict, env_min: dict, templates: dict,
                 output_path: str, fmt: str,
                 nastran_format: str = 'msc',
                 out_model=None) -> list:
    """Write envelope results to OP2 (all supported types) or HDF5.

    For fmt='op2':
        Always uses the raw struct.pack writer (HyperView-compatible).
        If out_model has geometry and its source path is known, raw GEOM
        table bytes from the input OP2 are prepended via
        _extract_op2_geom_prefix(), producing a standalone OP2 that
        HyperView can open without a BDF.
        Returns a list of attribute names NOT written to OP2.

    For fmt='h5':
        Writes all result types via pyNastran export_hdf5. Returns [].
    """
    if fmt == "op2":
        has_geom = out_model is not None and bool(getattr(out_model, "nodes", {}))
        geom_prefix = b''
        if has_geom:
            src = getattr(out_model, 'op2_filename', None)
            if src:
                geom_prefix = _extract_op2_geom_prefix(src)

        table_specs = []
        written: set = set()

        for attr, max_arr in env_max.items():
            template = templates.get(attr)
            if template is None:
                continue
            try:
                if attr in _OP2_NODAL_ATTRS:
                    spec = _build_nodal_spec(attr, template, max_arr)
                elif attr in _OP2_PLATE_STRESS_ATTRS:
                    spec = _build_plate_stress_spec(attr, template, max_arr)
                elif attr in _OP2_PLATE_FORCE_ATTRS:
                    spec = _build_plate_force_spec(attr, template, max_arr)
                else:
                    spec = None
            except Exception:
                spec = None
            if spec is not None:
                table_specs.append(spec)
                written.add(attr)

        if not table_specs:
            raise RuntimeError(
                "OP2 çıktısı için desteklenen result tipi bulunamadı.\n"
                "Seçili tipler arasında displacements, spc_forces, plate_stress "
                "veya plate_force olmalı."
            )

        write_msc_op2(table_specs, output_path, geom_prefix=geom_prefix)
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
            self._log(f"{len(files)} dosyadan veri okunuyor ve envelope hesaplanıyor (streaming)...")
            timer.start()
            (env_max, env_min, governing, templates, elem_ids,
             nastran_fmt, first_model) = collect_and_envelope(
                files, selected_types, log_fn=self._log)
            timer.stop()

            if not env_max:
                self.root.after(0, lambda: messagebox.showerror(
                    "Sonuç Yok",
                    "Seçilen result tipleri dosyalarda bulunamadı."
                ))
                return

            self._log(f"Bulunan attribute'lar: {list(env_max.keys())}")

            self._log(f"Çıktı yazılıyor → {output_path}")
            not_in_op2 = write_output(
                env_max, env_min, templates, output_path, fmt,
                nastran_format=nastran_fmt, out_model=first_model)

            # For OP2 format: unsupported result types (solid stress, rod/bar/beam
            # stress, etc.) go to Excel automatically alongside all governing info.
            # If the user specified an Excel path, use that; otherwise auto-name it.
            if fmt == "op2" and not_in_op2:
                xl_target = excel_path or (
                    os.path.splitext(output_path)[0] + "_results.xlsx")
                self._log(
                    f"OP2'ye yazılamayan tipler Excel'e aktarılıyor "
                    f"({', '.join(not_in_op2)}) → {xl_target}"
                )
                extra_max = {a: env_max[a] for a in not_in_op2 if a in env_max}
                extra_min = {a: env_min[a] for a in not_in_op2 if a in env_min}
                extra_ids = {a: elem_ids[a] for a in not_in_op2 if a in elem_ids}
                extra_gov = {a: governing[a] for a in not_in_op2 if a in governing}
                export_envelope_excel(extra_max, extra_min, extra_ids,
                                      extra_gov, xl_target, templates=templates)
                finish_msg = (
                    f"OP2 (displacement, plate_stress, plate_force, spc/mpc forces):\n"
                    f"  {output_path}\n\n"
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

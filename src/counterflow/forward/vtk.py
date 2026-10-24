"""Small VTKHDF 2.8 snapshot writers, independently readable by ParaView.

Snapshots rather than one append-only temporal file keep the short benchmark
robust.  The companion .vtkhdf.series file exposes the physical times.
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np


def _root(handle: h5py.File, vtk_type: str) -> h5py.Group:
    root = handle.create_group("VTKHDF")
    root.attrs["Type"] = np.bytes_(vtk_type)
    root.attrs["Version"] = np.array([2, 8], dtype=np.int32)
    return root


def _periodic_points(field: np.ndarray) -> np.ndarray:
    # For a single-z-slice ImageData VTKHDF expects 2 spatial axes, plus an
    # optional component axis.  VTK x varies fastest; solver is (ny,nx[,3]).
    a = np.concatenate((field, field[:, :1, ...]), axis=1)
    a = np.concatenate((a, a[:1, ...]), axis=0)
    return np.ascontiguousarray(a)


def write_field(path: Path, fields: dict, length: float) -> None:
    n = fields["omega_n"].shape[0]
    with h5py.File(path, "w") as h:
        root = _root(h, "ImageData")
        root.attrs["Direction"] = np.eye(3).reshape(-1)
        root.attrs["Origin"] = np.zeros(3)
        root.attrs["Spacing"] = np.array([length/n, length/n, 1.0])
        root.attrs["WholeExtent"] = np.array([0, n, 0, n, 0, 0], np.int32)
        pd = root.create_group("PointData")
        root.create_group("CellData")
        for name in ("vn", "vs", "F_ns"):
            v = fields[name]
            vector = np.zeros((n, n, 3), dtype=np.float64)
            vector[..., :2] = v
            pd.create_dataset(name, data=_periodic_points(vector), compression="gzip", compression_opts=1)
        for name in ("omega_n", "omega_s", "line_density"):
            pd.create_dataset(name, data=_periodic_points(fields[name]), compression="gzip", compression_opts=1)


def write_points(path: Path, pos: np.ndarray, data: dict[str, np.ndarray]) -> None:
    n = len(pos)
    points = np.zeros((n, 3), dtype=float)
    points[:, :2] = pos
    with h5py.File(path, "w") as h:
        root = _root(h, "PolyData")
        root.create_dataset("Points", data=points)
        root.create_dataset("NumberOfPoints", data=np.array([n], dtype="i8"))
        pd = root.create_group("PointData")
        root.create_group("CellData")
        for name, value in data.items():
            arr = np.asarray(value)
            if arr.ndim == 2 and arr.shape[1] == 2:
                padded = np.zeros((n, 3), dtype=arr.dtype)
                padded[:, :2] = arr
                arr = padded
            pd.create_dataset(name, data=arr)
        for kind in ("Vertices", "Lines", "Polygons", "Strips"):
            group = root.create_group(kind)
            if kind == "Vertices":
                group.create_dataset("Connectivity", data=np.arange(n, dtype="i8"))
                group.create_dataset("Offsets", data=np.arange(n+1, dtype="i8"))
                group.create_dataset("NumberOfCells", data=np.array([n], dtype="i8"))
                group.create_dataset("NumberOfConnectivityIds", data=np.array([n], dtype="i8"))
            else:
                group.create_dataset("Connectivity", data=np.empty(0, dtype="i8"))
                group.create_dataset("Offsets", data=np.array([0], dtype="i8"))
                group.create_dataset("NumberOfCells", data=np.array([0], dtype="i8"))
                group.create_dataset("NumberOfConnectivityIds", data=np.array([0], dtype="i8"))


def write_series(directory: Path, prefix: str, times: list[float]) -> None:
    series = {"file-series-version": "1.0", "files": [
        {"name": f"{prefix}_{i:04d}.vtkhdf", "time": float(t)}
        for i, t in enumerate(times)]}
    (directory/f"{prefix}.vtkhdf.series").write_text(json.dumps(series, indent=2), encoding="utf-8")

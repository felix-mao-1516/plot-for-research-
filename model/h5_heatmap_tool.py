#!/usr/bin/env python3
"""
h5_heatmap_tool.py
------------------
Quickly inspect an HDF5 (.h5) file, list datasets, and plot heatmaps from a chosen dataset.

Usage examples:
  1) List datasets:
     python h5_heatmap_tool.py --file your_file.h5 --list

  2) Plot a heatmap (auto vmin/vmax by percentiles):
     python h5_heatmap_tool.py --file your_file.h5 --dataset /path/to/dataset --out heatmap.png

  3) Plot a subset with strides to reduce size (e.g., every 4th row and 2nd column):
     python h5_heatmap_tool.py --file your_file.h5 --dataset /DAS/strain --row-slice ::4 --col-slice ::2 --out heatmap_subset.png

  4) Fix obvious outliers by clipping to percentiles (e.g., 1st to 99th):
     python h5_heatmap_tool.py --file your_file.h5 --dataset /DSS/RFS --vmin-p 1 --vmax-p 99 --out clipped.png
"""
import argparse
import ast
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def list_datasets(h5, prefix="/"):
    """Recursively list dataset paths with shape and dtype."""
    out = []
    def visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            out.append((f"/{name}", obj.shape, str(obj.dtype)))
    h5.visititems(visit)
    return out

def parse_slice(text):
    """
    Parse a python slice notation string like ':', '::5', '10:100:2', '5:' into a slice object.
    If text is None, return slice(None).
    """
    if text is None or text.strip() == "":
        return slice(None)
    # safely parse using ast to avoid eval
    parts = text.split(":")
    parts = [p.strip() for p in parts]
    # Allow forms like ":", "5:", ":10", "5:10", "::2", "5:10:2"
    def to_int_or_none(s):
        return None if s == "" else int(s)
    if len(parts) == 1:
        # single index, e.g., '10' (treated as start only -> [10])
        idx = int(parts[0])
        return slice(idx, idx+1, 1)
    elif len(parts) == 2:
        start = to_int_or_none(parts[0])
        stop  = to_int_or_none(parts[1])
        return slice(start, stop, None)
    elif len(parts) == 3:
        start = to_int_or_none(parts[0])
        stop  = to_int_or_none(parts[1])
        step  = to_int_or_none(parts[2])
        return slice(start, stop, step)
    else:
        raise ValueError(f"Cannot parse slice: {text}")

def robust_limits(arr, vmin_p=2.0, vmax_p=98.0):
    """Compute percentile-based vmin/vmax ignoring NaNs/infs."""
    finite = np.isfinite(arr)
    if not finite.any():
        return None, None
    vmin = np.nanpercentile(arr[finite], vmin_p) if vmin_p is not None else None
    vmax = np.nanpercentile(arr[finite], vmax_p) if vmax_p is not None else None
    if vmin is not None and vmax is not None and vmin >= vmax:
        # Fallback to min/max if percentiles are degenerate
        vmin, vmax = np.nanmin(arr[finite]), np.nanmax(arr[finite])
    return vmin, vmax

def plot_heatmap(arr, out_path=None, vmin_p=2.0, vmax_p=98.0, title=None, interpolation="nearest", extent=None, xlabel=None, ylabel=None):
    """
    Plot a 2D heatmap using matplotlib.
    - Does not specify colors explicitly per instruction; uses matplotlib defaults.
    - interpolation: 'nearest' (default) to avoid smoothing away sparse points; choose 'none' to see raw pixels.
    - extent: optional [xmin, xmax, ymin, ymax] to label axes with physical units (e.g., time, depth).
    """
    arr = np.array(arr, dtype=np.float32)
    # Replace inf with NaN, and keep NaNs (will show as gaps)
    arr[~np.isfinite(arr)] = np.nan

    vmin, vmax = robust_limits(arr, vmin_p, vmax_p)
    plt.figure()
    im = plt.imshow(arr, aspect="auto", origin="lower", interpolation=interpolation,
                    vmin=vmin, vmax=vmax, extent=extent)
    plt.colorbar(im)
    if title:
        plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    if out_path:
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"Saved heatmap to: {out_path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="List and plot heatmaps from HDF5 (.h5) files")
    parser.add_argument("--file", required=True, help="Path to .h5 file")
    parser.add_argument("--list", action="store_true", help="List datasets and exit")

    parser.add_argument("--dataset", help="Full dataset path inside HDF5 (e.g., /DAS/strain)")
    parser.add_argument("--row-slice", help="Row slice like ':', '::4', '100:200', '100:200:2'")
    parser.add_argument("--col-slice", help="Column slice like ':', '::2', '0:5000'")
    parser.add_argument("--vmin-p", type=float, default=2.0, help="Lower percentile for color scale (e.g., 1, 2); set None to auto")
    parser.add_argument("--vmax-p", type=float, default=98.0, help="Upper percentile for color scale (e.g., 98, 99); set None to auto")
    parser.add_argument("--interpolation", default="nearest", help="Matplotlib interpolation (e.g., 'nearest', 'none')")
    parser.add_argument("--out", help="Path to save figure (PNG). If not set, will display window.")
    parser.add_argument("--extent", help="Axis extent as [xmin, xmax, ymin, ymax], e.g. '[0,60,0,12000]'")
    parser.add_argument("--xlabel", help="X-axis label")
    parser.add_argument("--ylabel", help="Y-axis label")

    args = parser.parse_args()

    h5_path = Path(args.file)
    if not h5_path.exists():
        raise FileNotFoundError(f"File not found: {h5_path}")

    with h5py.File(h5_path, "r") as h5:
        if args.list or not args.dataset:
            items = list_datasets(h5)
            if not items:
                print("No datasets found.")
            else:
                print("Datasets in file:")
                for p, shape, dtype in items:
                    print(f"  {p}  shape={shape}  dtype={dtype}")
            if args.list and not args.dataset:
                return

        if not args.dataset:
            return

        if args.dataset not in h5:
            # try absolute path
            if args.dataset.startswith("/"):
                ds_path = args.dataset
            else:
                ds_path = "/" + args.dataset
            if ds_path not in h5:
                raise KeyError(f"Dataset not found: {args.dataset}")
            args.dataset = ds_path

        ds = h5[args.dataset]
        if ds.ndim == 1:
            # make it a 2D array with one row for plotting
            data = ds[...][None, :]
        elif ds.ndim >= 2:
            # Assume [rows, cols, ...]; take first two dims
            rsl = parse_slice(args.row_slice)
            csl = parse_slice(args.col_slice)
            # Build a slice tuple to handle 2+ dims gracefully
            slicer = [rsl, csl] + [slice(0,1)]*(ds.ndim-2)
            data = ds[tuple(slicer)]
            # If we pulled a singleton 3rd dim, squeeze it
            data = np.squeeze(data)
        else:
            raise ValueError(f"Unsupported dataset ndim={ds.ndim}")

        # Optional axis extent
        extent = None
        if args.extent:
            try:
                extent = ast.literal_eval(args.extent)
            except Exception as e:
                raise ValueError(f"Failed to parse extent: {e}")

        title = f"{h5_path.name}: {args.dataset}  shape={data.shape}"
        plot_heatmap(
            data,
            out_path=args.out,
            vmin_p=None if str(args.vmin_p).lower()=="none" else args.vmin_p,
            vmax_p=None if str(args.vmax_p).lower()=="none" else args.vmax_p,
            title=title,
            interpolation=args.interpolation,
            extent=extent,
            xlabel=args.xlabel,
            ylabel=args.ylabel,
        )

if __name__ == "__main__":
    main()

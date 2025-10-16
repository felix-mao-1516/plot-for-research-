
# -*- coding: utf-8 -*-
"""
plot_style.py
--------------
Core plotting style helpers (Times New Roman + STIX math) and
a utility to plot depth profiles (strain vs depth) within a *real depth range*
for one or multiple times, using dstrain with shape (Nt, Nz).

- Your original styling code is preserved as the foundation.
- New functions added:
    * _to_numpy, _as_1d
    * decode_stamps_to_datetimeindex
    * _normalize_times_to_row_indices
    * plot_depth_profiles_in_window

Author: You + ChatGPT
Date: 2025-10-16
"""

from __future__ import annotations

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from cycler import cycler
import pandas as pd
from typing import Optional, Sequence, Union, Tuple, List

# ----------------------
# Global default fonts
# ----------------------
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['mathtext.fontset'] = 'stix'


def make_standard_figure(
    figsize=(6, 12),         # width, height (inches)
    grid=False,              # show grid (1 pt)
    color_cycle=('k', 'r', 'b', 'g', 'm', 'c', 'y'),
    legend=True,             # place legend (upper-right, equal margins)
    legend_border_pad=0.5,   # padding to axes box (in font-size units)
    nbins=7,                 # aim for 5–7 major ticks (min_n_ticks=5)
    set_rc_globally=True,    # update rcParams globally (recommended)
    return_legend_placer=False # optionally return a legend-placer function
):
    """
    Create a standardized single-axes figure and return fig, ax
    (and optionally a legend placer).

    Usage:
        fig, ax = make_standard_figure(grid=True)
        ax.plot(x, y, label='Case A')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Pressure (MPa)')
        plt.show()
    """
    # ---------- Global defaults ----------
    if set_rc_globally:
        mpl.rcParams.update({
            # Fonts & sizes
            'font.family': 'Times New Roman',
            'axes.labelsize': 20,
            'axes.labelweight': 'bold',
            'xtick.labelsize': 20,
            'ytick.labelsize': 20,
            'legend.fontsize': 20,

            # Lines & color cycle
            'lines.linewidth': 2.25,
            'axes.prop_cycle': cycler('color', list(color_cycle)),

            # Tick directions & thickness
            'xtick.direction': 'out',
            'ytick.direction': 'out',
            'xtick.major.size': 6,
            'ytick.major.size': 6,
            'xtick.major.width': 2.0,
            'ytick.major.width': 2.0,

            # Axes spines
            'axes.linewidth': 2.0,
        })

    # ---------- Figure & axes ----------
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    # Hide top/right spines (outer box), bold bottom/left
    # (Fix: set_visible should take booleans)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2.0)
    ax.spines['left'].set_linewidth(2.0)

    # Keep only bottom/left ticks, outward
    ax.tick_params(bottom=True, left=True, top=False, right=False, direction='out')

    # Control major tick count ~5–7
    ax.xaxis.set_major_locator(MaxNLocator(nbins=nbins, min_n_ticks=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=nbins, min_n_ticks=5))

    # Optional grid
    if grid:
        ax.grid(True, linewidth=1.0)

    # Legend placer (equal margins to top-right, anchored at (1,1))
    def _place_legend(**kwargs):
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            lgd = ax.legend(
                loc='upper right',
                bbox_to_anchor=(1, 1),
                bbox_transform=ax.transAxes,
                borderaxespad=legend_border_pad,
                **kwargs
            )
            return lgd
        return None

    # Place legend immediately (if labels exist already)
    if legend is True:
        _place_legend()

    elif legend == 'auto':
        # Place legend the first time something with a label is drawn
        def auto_once():
            if auto_once.done:
                return
            if _place_legend() is not None:
                auto_once.done = True
        auto_once.done = False

        # Wrap common draw methods
        def _wrap(method):
            def inner(*args, **kwargs):
                out = method(*args, **kwargs)
                auto_once()
                return out
            return inner

        ax.plot         = _wrap(ax.plot)
        ax.scatter      = _wrap(ax.scatter)
        ax.step         = _wrap(ax.step)
        ax.bar          = _wrap(ax.bar)
        ax.barh         = _wrap(ax.barh)
        ax.fill_between = _wrap(ax.fill_between)

    if return_legend_placer:
        return fig, ax, _place_legend
    return fig, ax


def save_figure(fig=None, path='figure.png', dpi=300, transparent=False, tight=True, pad=0.05):
    """
    Save current or specified figure to path. Format determined by extension.
    """
    if fig is None:
        fig = plt.gcf()
    kwargs = {'dpi': dpi, 'transparent': transparent}
    if tight:
        kwargs.update({'bbox_inches': 'tight', 'pad_inches': pad})
    fig.savefig(path, **kwargs)


# ========================= NEW: depth profile utilities =========================

def _to_numpy(a) -> np.ndarray:
    """Convert h5py.Dataset / pandas DataFrame/Series / numpy to np.ndarray."""
    if hasattr(a, "values"):  # pandas -> numpy
        a = a.values
    return np.asarray(a)


def _as_1d(a) -> np.ndarray:
    """Return a flattened 1D numpy array."""
    arr = _to_numpy(a)
    return arr.reshape(-1)


def decode_stamps_to_datetimeindex(stamps: Union[Sequence[str], Sequence[bytes], np.ndarray, pd.Series],
                                   fmt: str = '%m/%d/%Y %H:%M:%S.%f') -> pd.DatetimeIndex:
    """
    Decode a 1D 'stamps' array (bytes/str) into a pandas.DatetimeIndex.
    """
    arr = _to_numpy(stamps)
    if arr.dtype.kind in ("S", "O"):  # bytes/object -> decode to str if needed
        try:
            arr = np.char.decode(arr, "utf-8")
        except Exception:
            arr = arr.astype(str)

    ts = pd.to_datetime(arr, format=fmt, errors="coerce")
    return pd.DatetimeIndex(ts)


def _normalize_times_to_row_indices(times: Sequence[Union[int, str, pd.Timestamp]],
                                    nrows: int,
                                    stamps_index: Optional[pd.DatetimeIndex] = None) -> List[int]:
    """
    Normalize user-provided 'times' (row indices or timestamps) to row indices.
    """
    if times is None:
        raise ValueError("Provide 'times' as a list of row indices or timestamps.")

    # Accept single value
    if not isinstance(times, (list, tuple, np.ndarray, pd.Index)):
        times = [times]

    arr_obj = np.asarray(times, dtype=object)
    # If all are integers -> treat as explicit row indices
    if np.all([isinstance(x, (int, np.integer)) for x in arr_obj]):
        idx = arr_obj.astype(int)
        if (idx < 0).any() or (idx >= nrows).any():
            raise IndexError("Some row indices out of bounds.")
        return idx.tolist()

    # Otherwise interpret as timestamps
    if stamps_index is None:
        raise ValueError("Timestamp-like 'times' given but 'stamps_index' is None.")

    targets = pd.to_datetime(times)
    pos = stamps_index.get_indexer(targets, method="nearest")
    if (pos < 0).any():
        bad = [times[i] for i in np.where(pos < 0)[0]]
        raise ValueError(f"Cannot match these times to a valid row: {bad}")
    return pos.tolist()


def plot_depth_profiles_in_window(
    dstrain,
    depth,
    depth_min: float,
    depth_max: float,
    times: Sequence[Union[int, str, pd.Timestamp]],
    *,
    stamps_index: Optional[pd.DatetimeIndex] = None,
    depth_unit: str = "ft",
    strain_unit: str = "(strain)",
    invert_depth_axis: bool = True,
    figsize: Tuple[float, float] = (6, 12),
    grid: bool = False,
    ax: Optional[plt.Axes] = None,
    color_cycle=('k', 'r', 'b', 'g', 'm', 'c', 'y'),
    legend: Union[bool, str] = True,
    title: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot strain-vs-depth profiles inside a real depth window [depth_min, depth_max]
    for one or multiple times (each time -> one curve).

    Parameters
    ----------
    dstrain : array-like (Nt, Nz)
        Strain array with time along rows and depth along columns.
    depth : array-like (Nz,) or (Nz,1)
        Real depth for each column (same order as dstrain columns).
    depth_min, depth_max : float
        Real depth window bounds (same units as 'depth').
    times : list[int | str | pandas.Timestamp]
        Moments to extract; if ints -> row indices; if str/Timestamp -> nearest stamp.
    stamps_index : pandas.DatetimeIndex | None
        Needed when using time strings / Timestamps.
    depth_unit, strain_unit : str
        Axis labels units.
    invert_depth_axis : bool
        If True, shallow at top (井口在上).
    figsize, grid, color_cycle, legend :
        Passed to make_standard_figure if ax is None.
    ax : matplotlib.axes.Axes | None
        Existing axes to draw on; if None, a new (fig, ax) is created.
    title : str | None
        Custom title; default shows depth window.

    Returns
    -------
    fig, ax : Figure and Axes with the plotted profiles.
    """
    D = _as_1d(depth)                 # (Nz,)
    X = _to_numpy(dstrain)            # (Nt, Nz)
    Nt, Nz = X.shape

    # Swap bounds if needed
    if depth_min > depth_max:
        depth_min, depth_max = depth_max, depth_min

    # Column selection by real depth
    mask = (D >= depth_min) & (D <= depth_max)
    if not np.any(mask):
        raise ValueError("No depth samples fall inside the requested window.")
    col_idx = np.where(mask)[0]
    Dwin = D[col_idx]

    # Normalize 'times' -> row indices
    row_idx = _normalize_times_to_row_indices(times, Nt, stamps_index)

    # Prepare figure/axes
    created = False
    if ax is None:
        fig, ax = make_standard_figure(
            figsize=figsize, grid=grid, color_cycle=color_cycle, legend=legend
        )
        created = True
    else:
        fig = ax.figure

    # Plot each profile
    for r in row_idx:
        prof = X[r, col_idx]  # h5py.Dataset slicing-friendly

        if stamps_index is not None:
            base_label = pd.to_datetime(stamps_index[r]).strftime("%Y-%m-%d %H:%M:%S")
        else:
            base_label = f"row {r}"

        ax.plot(prof, Dwin, label=base_label)

    # Labels & title
    ax.set_xlabel(f"Strain {strain_unit}")
    ax.set_ylabel(f"Depth [{depth_unit}]")
    if title is None:
        title = f"Depth profiles: [{depth_min}, {depth_max}] {depth_unit}"
    ax.set_title(title)

    # Depth orientation
    if invert_depth_axis:
        ax.invert_yaxis()

    if created:
        plt.show()

    return fig, ax


# --------------------------
# Optional quick self-test
# --------------------------
if __name__ == "__main__":
    # Synthetic smoke test
    Nt, Nz = 120, 600
    z = np.linspace(0, 10000, Nz)
    t = np.arange(Nt)
    data = np.sin((z[None, :] / 900) + (t[:, None] / 10.0))

    # Create fake stamps (string) to demonstrate timestamp-based selection
    stamps = pd.date_range("2022-03-14 20:56:49", periods=Nt, freq="90S").strftime("%m/%d/%Y %H:%M:%S.000000")
    stamps_index = decode_stamps_to_datetimeindex(stamps)

    # Example 1: using row indices
    fig, ax = plot_depth_profiles_in_window(
        data, z, depth_min=2000, depth_max=3000, times=[5, 30, 90],
        depth_unit="ft", strain_unit="(a.u.)", grid=True
    )

    # Example 2: using time strings (nearest match via stamps_index)
    fig, ax = plot_depth_profiles_in_window(
        data, z, depth_min=6000, depth_max=7000,
        times=["2022-03-14 21:05:00", "2022-03-14 21:35:00"],
        stamps_index=stamps_index, depth_unit="ft", strain_unit="(a.u.)",
        grid=True
    )

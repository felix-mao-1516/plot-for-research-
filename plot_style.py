
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
import matplotlib.dates as mdates

# ----------------------
# Global default fonts
# ----------------------
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['mathtext.fontset'] = 'stix'


def make_standard_figure(
    figsize=(6, 12),         # width, height (inches)
    grid=False,              # show grid (1 pt)
    color_cycle=( 'r', 'b', 'g', 'm', 'c', 'y', 'k'),
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
    ax.spines['top'].set_visible(2.0) #False
    ax.spines['right'].set_visible(2.0)
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
    invert_depth_axis: bool = False,
    figsize: Tuple[float, float] = (6, 12),
    grid: bool = False,
    ax: Optional[plt.Axes] = None,
    color_cycle=( 'b', 'g', 'm', 'c', 'y'),#'k', 'r',
    legend: Union[bool, str] = True,
    title: Optional[str] = None,
    show=False,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot *depth on x-axis* vs *strain on y-axis* profiles inside a real depth window
    [depth_min, depth_max] for one or multiple times (each time -> one curve).

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
        If True, invert the *x-axis* (depth), so shallow appears on the left.
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

    # Plot each profile: x = depth, y = strain
    for r in row_idx:
        prof = X[r, col_idx]  # h5py.Dataset slicing-friendly

        if stamps_index is not None:
            base_label = pd.to_datetime(stamps_index[r]).strftime("%Y-%m-%d %H:%M:%S")
        else:
            base_label = f"row {r}"

        ax.plot(Dwin, prof, label=base_label)

    # Labels & title
    ax.set_xlabel(f"Depth [{depth_unit}]")
    ax.set_ylabel(f"Strain {strain_unit}")
    if title == "auto":
        title = f"Depth profiles: [{depth_min}, {depth_max}] {depth_unit}"
    ax.set_title(title)

    # Depth orientation (now along x-axis)
    if invert_depth_axis:
        ax.invert_xaxis()
    # === 新增：自动刷新图例，确保收集到全部曲线 ===
    if legend in (True, 'auto'):
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, loc='best')

    if created and show:
        plt.show()

    return fig, ax

def plot_dual_axis_depth_profile(
    dstrain_left, depth_left, times_left,
    dstrain_right, depth_right, times_right,
    *,
    depth_window_left: Tuple[float, float],
    depth_window_right: Tuple[float, float],
    stamps_index_left: Optional[pd.DatetimeIndex] = None,
    stamps_index_right: Optional[pd.DatetimeIndex] = None,
    depth_unit: str = "ft",
    strain_unit_left: str = "(strain)",
    strain_unit_right: str = "(strain)",
    strain_scale_left: float = 1.0,
    strain_scale_right: float = 1.0,
    invert_left_y: bool = False,
    invert_right_y: bool = False,
    xlim: Optional[Tuple[float, float]] = None,
    figsize: Tuple[float, float] = (12, 6),
    grid: bool = True,
    color_cycle=('r', 'b', 'g', 'm', 'c', 'y'),
    labels: Tuple[str, str] = ("Series A", "Series B"),
    title: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes, plt.Axes]:
    fig, ax = make_standard_figure(figsize=figsize, grid=grid, color_cycle=color_cycle, legend=False)
    ax_r = ax.twinx()

    def _next_color(the_ax):
        # 先试专用方法（兼容大多数版本）
        if hasattr(the_ax, "_get_lines") and hasattr(the_ax._get_lines, "get_next_color"):
            return the_ax._get_lines.get_next_color()
        # 兜底：从 rcParams 里取颜色表，按当前已画线条数轮询
        colors = plt.rcParams.get('axes.prop_cycle', cycler(color=['C0'])).by_key().get('color', ['C0'])
        return colors[len(the_ax.get_lines()) % len(colors)]
    
    # === 方案A：固定左右两条线使用不同颜色 ===
    colors = mpl.rcParams.get('axes.prop_cycle', cycler(color=['C0', 'C1'])).by_key().get('color', ['C0', 'C1'])
    left_color  = colors[0]
    right_color = colors[1 if len(colors) > 1 else 0]

    # LEFT
    D1 = _as_1d(depth_left)
    X1 = _to_numpy(dstrain_left)
    Nt1, Nz1 = X1.shape
    dmin1, dmax1 = depth_window_left
    if dmin1 > dmax1:
        dmin1, dmax1 = dmax1, dmin1
    mask1 = (D1 >= dmin1) & (D1 <= dmax1)
    if not np.any(mask1):
        raise ValueError("Left series: no depth samples in the requested window.")
    col1 = np.where(mask1)[0]
    D1w = D1[col1]
    rows1 = _normalize_times_to_row_indices(times_left, Nt1, stamps_index_left)

    # color_left = _next_color(ax)
    for r in rows1:
        prof = X1[r, col1] * float(strain_scale_left)
        if stamps_index_left is not None:
            tlabel = pd.to_datetime(stamps_index_left[r]).strftime("%Y-%m-%d %H:%M:%S")
        else:
            tlabel = f"row {r}"
        ax.plot(D1w, prof, label=f"{labels[0]} @ {tlabel}", color=left_color)

    ax.set_xlabel(f"Depth [{depth_unit}]")
    ax.set_ylabel(f"Strain {strain_unit_left}", color=left_color)
    ax.tick_params(axis='y', labelcolor=left_color)

    # RIGHT
    D2 = _as_1d(depth_right)
    X2 = _to_numpy(dstrain_right)
    Nt2, Nz2 = X2.shape
    dmin2, dmax2 = depth_window_right
    if dmin2 > dmax2:
        dmin2, dmax2 = dmax2, dmin2
    mask2 = (D2 >= dmin2) & (D2 <= dmax2)
    if not np.any(mask2):
        raise ValueError("Right series: no depth samples in the requested window.")
    col2 = np.where(mask2)[0]
    D2w = D2[col2]
    rows2 = _normalize_times_to_row_indices(times_right, Nt2, stamps_index_right)

    # color_right = _next_color(ax_r)
    for r in rows2:
        prof = X2[r, col2] * float(strain_scale_right)
        if stamps_index_right is not None:
            tlabel = pd.to_datetime(stamps_index_right[r]).strftime("%Y-%m-%d %H:%M:%S")
        else:
            tlabel = f"row {r}"
        ax_r.plot(D2w, prof, label=f"{labels[1]} @ {tlabel}", color=right_color, linestyle='--')

    ax_r.set_ylabel(f"Strain {strain_unit_right}", color=right_color)
    ax_r.tick_params(axis='y', labelcolor=right_color)

    # X limits
    if xlim is None:
        xmin = min(dmin1, dmin2)
        xmax = max(dmax1, dmax2)
        xlim = (xmin, xmax)
    ax.set_xlim(*xlim)

    # Y inversions
    if invert_left_y:
        ax.invert_yaxis()
    if invert_right_y:
        ax_r.invert_yaxis()

    # Title
    if title  =="auto":
        title = "Depth profiles with dual strain axes"
    ax.set_title(title)

    # Combined legend
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax_r.get_legend_handles_labels()
    if h1 or h2:
        ax.legend(h1 + h2, l1 + l2, loc='upper right')

    return fig, ax, ax_r


# ========================= NEW: strain vs time utilities =========================
def _nearest_rows_for_window(time_window, stamps_index, nrows: int):
    """
    把时间窗口转为行号 slice。
    time_window:
      - None: 全部
      - (int_start, int_end): 行号（含端点）
      - (str/datetime, str/datetime): 时间戳（含端点，按“最近邻”匹配；允许重复/NaT）
    """
    if time_window is None:
        return slice(None)

    # 行号窗口
    if isinstance(time_window[0], (int, np.integer)):
        a, b = int(time_window[0]), int(time_window[1])
        if a > b:
            a, b = b, a
        a = max(0, a)
        b = min(nrows - 1, b)
        return slice(a, b + 1)

    # 时间戳窗口
    if stamps_index is None:
        raise ValueError("给了时间字符串窗口，但未提供 stamps_index。")

    # 稳健“最近邻”（允许 stamps_index 重复或包含 NaT）
    idx_ns = stamps_index.asi8  # int64 ns; NaT 为最小 int64
    valid_mask = idx_ns != np.iinfo(np.int64).min
    if not valid_mask.any():
        raise ValueError("stamps_index 全是 NaT，无法匹配时间。")

    valid_pos = np.where(valid_mask)[0]
    idx_ns_valid = idx_ns[valid_mask]

    t0 = np.int64(pd.Timestamp(time_window[0]).value)
    t1 = np.int64(pd.Timestamp(time_window[1]).value)
    if t0 > t1:
        t0, t1 = t1, t0

    i0 = int(valid_pos[np.argmin(np.abs(idx_ns_valid - t0))])
    i1 = int(valid_pos[np.argmin(np.abs(idx_ns_valid - t1))])
    if i0 > i1:
        i0, i1 = i1, i0
    return slice(i0, i1 + 1)


def plot_strain_vs_time_at_depths(
    dstrain, depth,
    *,
    stamps_index=None,                     # 若要横轴显示时间，建议传入 DatetimeIndex
    target_depths=None,                    # 例如 [12000, 15000]：按最近列或窗口聚合
    depth_tolerance=None,                  # 例如 25（表示 ±25 ft 聚合窗口）
    reducer="mean",                        # 'mean' 或 'median'（用于窗口/区间聚合）
    depth_windows=None,                    # 例如 [(11950,12050), (15950,16050)]：区间聚合
    time_window=None,                      # None 或 (start, end): 可填行号或时间字符串
    scale=1.0,                             # y 轴缩放（如 1e6 -> microstrain）
    depth_unit="ft", strain_unit="(microstrain)",
    time_format='%m/%d/%Y\n %H:%M:%S',
    time_tick_rotation=0,
    figsize=(12, 6), grid=True, legend='auto',
    legend_labels=None,                    # 可选：自定义图例文本列表（覆盖默认）
    ax=None, show=False                    # 跟你现有风格统一：默认不 plt.show()
):
    """
    在给定的一个/多个深度位置处，绘制 strain vs time。
    - 若提供 target_depths：
        * depth_tolerance is None/0 -> 每个深度取“最近列”
        * depth_tolerance>0         -> 对每个深度的 ±tol 窗口内列做聚合
    - 或者提供 depth_windows：对每个 (zmin,zmax) 区间聚合
    """
    D = _as_1d(depth)
    X = _to_numpy(dstrain)
    Nt, Nz = X.shape

    # 时间窗口 -> 行 slice
    rs = _nearest_rows_for_window(time_window, stamps_index, Nt)

    created = False
    if ax is None:
        # 复用你已有的统一风格
        fig, ax = make_standard_figure(figsize=figsize, grid=grid, legend=legend)
        created = True
    else:
        fig = ax.figure

    def _reduce_cols(Y2d):
        if Y2d.ndim == 1:
            return Y2d
        if reducer == "median":
            return np.nanmedian(Y2d, axis=1)
        return np.nanmean(Y2d, axis=1)

    # 1) 按 target_depths（点或小窗口）
    if target_depths is not None:
        for zt in list(target_depths):
            if not depth_tolerance:
                # 最近列
                j = int(np.argmin(np.abs(D - zt)))
                Y = X[rs, j] * float(scale)
                lbl = f"{zt:g} {depth_unit} (nearest {D[j]:.1f})"
            else:
                mask = np.abs(D - zt) <= float(depth_tolerance)
                cols = np.where(mask)[0]
                if cols.size == 0:
                    # 无列命中 -> 退化到最近列
                    j = int(np.argmin(np.abs(D - zt)))
                    Y = X[rs, j] * float(scale)
                    lbl = f"{zt:g}±{depth_tolerance:g} {depth_unit} (fallback {D[j]:.1f})"
                else:
                    Y = _reduce_cols(X[rs, :][:, cols]) * float(scale)
                    lbl = f"{zt:g}±{depth_tolerance:g} {depth_unit} ({reducer})"

            if stamps_index is not None:
                ax.plot(stamps_index[rs], Y, label=lbl)
            else:
                ax.plot(np.arange(Nt)[rs], Y, label=lbl)

    # 2) 按 depth_windows（区间聚合）
    if depth_windows is not None:
        for (zmin, zmax) in depth_windows:
            if zmin > zmax:
                zmin, zmax = zmax, zmin
            mask = (D >= zmin) & (D <= zmax)
            cols = np.where(mask)[0]
            if cols.size == 0:
                continue
            Y = _reduce_cols(X[rs, :][:, cols]) * float(scale)
            lbl = f"[{zmin:g}, {zmax:g}] {depth_unit} ({reducer})"
            if stamps_index is not None:
                ax.plot(stamps_index[rs], Y, label=lbl)
            else:
                ax.plot(np.arange(Nt)[rs], Y, label=lbl)

    # 轴标签
    ax.set_ylabel(r'Strain change, $\mu\varepsilon$')
    ax.set_xlabel("Time" if stamps_index is not None else "Time (row index)")

    if stamps_index is not None and time_format:
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter(time_format))
        ha = 'center' if time_tick_rotation % 360 == 0 else 'right'
        for lab in ax.get_xticklabels():
            lab.set_rotation(time_tick_rotation)
            lab.set_rotation_mode('anchor')   # 锚点固定在刻度位置
            lab.set_horizontalalignment(ha)


    # 自动刷新图例（含 legend='auto' 的情况）
    if legend in (True, 'auto'):
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            if legend_labels is not None:
                ax.legend(handles, legend_labels, loc='best')
            else:
                ax.legend(handles, labels, loc='best')

    if created and show:
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

if __name__ == "__main__":
    # Quick synthetic test
    Nt, Nz = 120, 600
    z  = np.linspace(10000, 20000, Nz)
    z2 = np.linspace(11000, 18000, Nz)
    t = np.arange(Nt)
    data1 = np.sin((z[None, :]  / 900) + (t[:, None] / 10.0))
    data2 = 0.5*np.cos((z2[None, :] / 700) + (t[:, None] / 11.0))

    fig, ax, axr = plot_dual_axis_depth_profile(
        dstrain_left=data1, depth_left=z,  times_left=[5],
        dstrain_right=data2, depth_right=z2, times_right=[30],
        depth_window_left=(12000, 18000),
        depth_window_right=(12000, 16000),
        depth_unit="ft",
        strain_unit_left="(a.u.)",
        strain_unit_right="(a.u.)",
        xlim=(12000, 18000),
        invert_left_y=False,
        invert_right_y=False,
        labels=("Left series", "Right series"),
        figsize=(12,6)
    )
    plt.show()

# coding=utf-8
# description: 根据测深（MD）区间，从 dstrain(channel, time) 中抽取/重采样子集以便按 MD 作图

import numpy as np
import datetime
import pandas as pd

#========================depth section========================#
def _ensure_ascending(x, y_along_axis0):
    """确保 x 递增；如递减则翻转 x 与 y 的 axis=0。"""
    x = np.asarray(x, float)
    y = np.asarray(y_along_axis0)
    if x[0] <= x[-1]:
        return x, y
    return x[::-1].copy(), y[::-1, :].copy()

def _normalize_ranges(ranges):
    """把 ranges 统一成 [(lo, hi), ...] 形式并排序；允许单个 float 当“点区间”用。"""
    out = []
    for r in ranges:
        if np.isscalar(r):
            lo = hi = float(r)
        else:
            a, b = float(r[0]), float(r[1])
            lo, hi = (a, b) if a <= b else (b, a)
        out.append((lo, hi))
    # 去重+排序
    out = sorted(set(out), key=lambda x: (x[0], x[1]))
    return out

def subset_dstrain_by_md_ranges(dstrain, MD, ranges,
                                method="interp",
                                keep_gaps=True,
                                gap_rows=2,
                                include_endpoints=True):
    """
    根据 MD 的若干区间，从 dstrain(channel, time) 中抽取/重采样子集以便按 MD 作图。
    从 dstrain(channel, time) 中按 MD 区间抽取子集；可在段间插 NaN 分隔，不在最后一段后插。

    参数
    ----
    dstrain : ndarray, shape (N_ch, N_t)
        原始 2D 数据（axis=0 是 channel/depth，axis=1 是 time）
    MD : ndarray, shape (N_ch,)
        每个 channel 对应的测深（ft），通常由你的线性公式算得
    ranges : list[tuple|float]
        MD 区间列表，如 [(12000, 12145), (16000, 16160)] 或 [12000, 13000]
    method : {"interp", "nearest_channel"}
        - "interp": 在 MD 轴上线性插值到新网格（推荐，边界精确）
        - "nearest_channel": 不插值，就近取整 channel（更快）
    keep_gaps : bool
        在各区间之间插入 NaN 行，便于瀑布图显示分隔
    gap_rows : int
        分隔带厚度（行数）
    include_endpoints : bool
        interp 模式下是否强制把每段区间的 lo/hi 作为新网格端点

    返回
    ----
    data_sub : ndarray, shape (M, N_t)
    md_sub   : ndarray, shape (M,)
    meta     : dict，包含各段在拼接后的行号范围、是否发生裁剪等信息
    """
    D, T = dstrain.shape
    MD, dstrain = _ensure_ascending(MD, dstrain)
    ranges_norm = _normalize_ranges(ranges)
    md_min, md_max = float(MD[0]), float(MD[-1])

    # 先构建所有“有效段”
    segments = []   # each: dict(md_seg=..., data_seg=..., orig=(lo,hi), used=(clo,chi), clipped=bool)

    ch_idx = np.arange(D, dtype=float)
    for (lo, hi) in ranges_norm:
        clo, chi = max(lo, md_min), min(hi, md_max)
        clipped = (clo != lo) or (chi != hi)
        if clo > chi:
            # 完全落在范围外，跳过
            continue

        if method == "nearest_channel":
            ch_lo = int(np.ceil(np.interp(clo, MD, ch_idx)))
            ch_hi = int(np.floor(np.interp(chi, MD, ch_idx)))
            if ch_lo > ch_hi:       # 可能极端情况下仍为空
                continue
            md_seg = MD[ch_lo:ch_hi+1]
            data_seg = dstrain[ch_lo:ch_hi+1, :]

        elif method == "interp":
            inside = (MD >= clo) & (MD <= chi)
            md_seg = MD[inside]
            if include_endpoints:
                md_seg = np.unique(np.concatenate([md_seg, [clo, chi]]))
            if md_seg.size == 0:
                continue
            data_seg = np.empty((md_seg.size, T), dtype=float)
            for j in range(T):
                data_seg[:, j] = np.interp(md_seg, MD, dstrain[:, j])

        else:
            raise ValueError("method 必须为 'interp' 或 'nearest_channel'")

        segments.append({
            "md_seg": md_seg,
            "data_seg": data_seg,
            "orig": (lo, hi),
            "used": (clo, chi),
            "clipped": clipped
        })

    if not segments:
        raise ValueError("所有 ranges 与可用 MD 范围都不相交；请检查取值。")

    # 再把段拼接；只在段与段之间插 NaN，不在最后一段后插
    seg_mds = []
    seg_datas = []
    spans = []   # 记录拼接后的行号范围
    cursor = 0
    for idx, seg in enumerate(segments):
        md_seg = seg["md_seg"]
        data_seg = seg["data_seg"]
        seg_mds.append(md_seg)
        seg_datas.append(data_seg)
        spans.append((cursor, cursor + md_seg.size))
        cursor += md_seg.size

        if keep_gaps and idx < len(segments) - 1:
            seg_mds.append(np.full((gap_rows,), np.nan))
            seg_datas.append(np.full((gap_rows, T), np.nan))
            spans.append((cursor, cursor + gap_rows))  # 这是分隔带的行号范围
            cursor += gap_rows

    md_sub = np.concatenate(seg_mds, axis=0)
    data_sub = np.concatenate(seg_datas, axis=0)

    meta = {
        "segments_row_spans": spans,
        "segments_meta": [(s["orig"], s["used"], s["clipped"]) for s in segments],
        "method": method,
        "kept_gaps": keep_gaps
    }
    return data_sub, md_sub, meta

#========================time section========================#

def _to_seconds_offsets(t_axis, start_time=None):
    """
    把时间轴规范为“相对秒”的 1D ndarray：
    - 若 t_axis 是数字(list/ndarray)，按秒理解直接返回
    - 若是 pandas.DatetimeIndex / numpy.datetime64 / datetime 列表，则需要提供 start_time，
      返回 (t - start_time).total_seconds()
    """
    t = np.asarray(t_axis)
    if np.issubdtype(t.dtype, np.number):
        return t.astype(float)
    # Datetime-like
    if start_time is None:
        raise ValueError("t_axis 为绝对时间时需要提供 start_time 以换算成相对秒。")
    st = pd.to_datetime(start_time)
    t_dt = pd.to_datetime(t_axis)
    return (t_dt - st).total_seconds().astype(float)

def _normalize_time_ranges(ranges, start_time=None):
    """
    ranges 中的元素可为：
      - (lo_sec, hi_sec) 的数字（相对秒）
      - (lo_dt,  hi_dt) 的 datetime（绝对时间）
      - 单个数/单个 datetime（“点区间”）
    统一归一化为相对秒、[lo, hi] 升序且去重。
    """
    out = []
    for r in ranges:
        if np.isscalar(r):
            lo = hi = r
        else:
            lo, hi = r
        # 转成相对秒
        if isinstance(lo, (datetime.datetime, np.datetime64, pd.Timestamp)):
            if start_time is None:
                raise ValueError("ranges 含 datetime，请提供 start_time。")
            lo = (pd.to_datetime(lo) - pd.to_datetime(start_time)).total_seconds()
        if isinstance(hi, (datetime.datetime, np.datetime64, pd.Timestamp)):
            if start_time is None:
                raise ValueError("ranges 含 datetime，请提供 start_time。")
            hi = (pd.to_datetime(hi) - pd.to_datetime(start_time)).total_seconds()
        lo, hi = float(lo), float(hi)
        if lo > hi: lo, hi = hi, lo
        out.append((lo, hi))
    # 去重+排序
    out = sorted(set(out), key=lambda x: (x[0], x[1]))
    return out

def subset_dstrain_by_time_ranges(dstrain, t_axis, ranges, *,
                                  start_time=None,
                                  method="interp",
                                  keep_gaps=True,
                                  gap_cols=2,
                                  include_endpoints=True):
    """
    沿时间轴切片：
      - method="nearest": 不插值，直接选落在区间内的原列
      - method="interp" : 在时间轴上做线性插值，把区间边界精确纳入
      - 支持多个不连续区间，并仅在“段与段之间”插 NaN 列作分隔（不会在两端插）
    返回:
      data_sub: (N_ch, M_t)
      t_sub   : (M_t,)  相对秒（配合原 start_time 使用）
      meta    : 一些段信息
    """
    # 统一成“相对秒”的时间轴
    t = _to_seconds_offsets(t_axis, start_time)
    t = t.astype(float)
    if t.ndim != 1:
        raise ValueError("t_axis 必须是一维时间轴")
    if dstrain.shape[1] != t.size:
        raise ValueError("dstrain 的列数与 t_axis 长度不一致")

    ranges_norm = _normalize_time_ranges(ranges, start_time=start_time)
    t_min, t_max = float(t[0]), float(t[-1])
    if t_min > t_max:  # 保险：若时间轴为降序，翻转
        t = t[::-1].copy()
        dstrain = dstrain[:, ::-1].copy()
        t_min, t_max = t[0], t[-1]

    seg_t_list, seg_data_list, spans = [], [], []
    cursor = 0

    for idx, (lo, hi) in enumerate(ranges_norm):
        clo, chi = max(lo, t_min), min(hi, t_max)  # 与可用范围求交
        if clo > chi:
            continue

        if method == "nearest":
            mask = (t >= clo) & (t <= chi)
            t_seg = t[mask]
            if t_seg.size == 0:
                continue
            data_seg = dstrain[:, mask]

        elif method == "interp":
            # 新时间网格：区间内已有点 + 可选端点
            inside = (t >= clo) & (t <= chi)
            t_seg = t[inside]
            if include_endpoints:
                t_seg = np.unique(np.concatenate([t_seg, [clo, chi]]))
            if t_seg.size == 0:
                continue
            # 沿时间轴对每个通道插值
            data_seg = np.empty((dstrain.shape[0], t_seg.size), dtype=float)
            for i in range(dstrain.shape[0]):
                data_seg[i, :] = np.interp(t_seg, t, dstrain[i, :])
        else:
            raise ValueError("method 仅支持 'nearest' 或 'interp'")

        seg_t_list.append(t_seg)
        seg_data_list.append(data_seg)
        spans.append((cursor, cursor + t_seg.size))
        cursor += t_seg.size

        # 段间插入 NaN 分隔列（不在最后一段后插）
        if keep_gaps and idx < len(ranges_norm) - 1:
            seg_t_list.append(np.linspace(t_seg[-1], t_seg[-1], gap_cols))  # 数值随便，占位即可
            seg_data_list.append(np.full((dstrain.shape[0], gap_cols), np.nan))
            spans.append((cursor, cursor + gap_cols))
            cursor += gap_cols

    if not seg_t_list:
        raise ValueError("所有时间区间与可用时间轴不相交；请检查 ranges。")

    t_sub = np.concatenate(seg_t_list, axis=0)
    data_sub = np.concatenate(seg_data_list, axis=1)

    meta = {
        "segments_col_spans": spans,
        "method": method,
        "kept_gaps": keep_gaps
    }
    return data_sub, t_sub, meta

import os
import glob
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# --------------------------
# 工具函数
# --------------------------
def _decode_stamps(stamps_raw):
    """把 stamps 数据集转成 DatetimeIndex。
    兼容 bytes/utf-8 字符串/np.datetime64/时间戳秒数 等常见格式。
    """
    arr = np.asarray(stamps_raw)

    # bytes 或对象数组里是 bytes
    if arr.dtype.kind in ("S", "O"):
        try:
            arr = np.char.decode(arr, "utf-8")
        except Exception:
            # 若里面掺杂非 bytes，逐个处理
            arr = np.array([x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x) for x in arr])

        # 常见格式： "03/14/2022 20:56:49.350706"
        try:
            ts = pd.to_datetime(arr, format="%m/%d/%Y %H:%M:%S.%f", errors="raise")
        except Exception:
            # 如果格式不完全一致，就让 pandas 自动识别
            ts = pd.to_datetime(arr, errors="coerce")
        if ts.isna().any():
            raise ValueError("有无法解析的时间戳，请检查 stamps 格式/例子。")
        return pd.DatetimeIndex(ts)

    # Unicode 字符串
    if arr.dtype.kind == "U":
        ts = pd.to_datetime(arr, errors="coerce")
        if ts.isna().any():
            raise ValueError("有无法解析的时间戳，请检查 stamps 格式/例子。")
        return pd.DatetimeIndex(ts)

    # np.datetime64
    if np.issubdtype(arr.dtype, np.datetime64):
        return pd.DatetimeIndex(arr)

    # 数字（可能是 epoch 秒）
    if np.issubdtype(arr.dtype, np.number):
        # 尝试当作秒
        return pd.to_datetime(arr, unit="s")

    raise TypeError(f"不支持的 stamps dtype: {arr.dtype}")


def _normalize_data_shape(data, depth_len):
    """把 data 规整成 (depth, time)；返回规整后的 data 以及时间轴长度。
    """
    d = np.asarray(data)
    if d.shape[0] == depth_len:
        # (depth, time)
        return d, d.shape[1]
    elif d.shape[1] == depth_len:
        # (time, depth) -> 转置
        d2 = d.T
        return d2, d2.shape[1]
    else:
        raise ValueError(f"data 的任一维度都不等于 depth_len={depth_len}，无法识别深度维。实际 shape={d.shape}")


def read_one_h5(path, data_key="/data", depth_key="/depth", stamps_key="/stamps"):
    """读取单个 H5：返回 (data_(depth,time), depth_(depth,), stamps_DTI_(time,))"""
    with h5py.File(path, "r") as f:
        data = f[data_key][...]          # 形状未知，稍后规整
        depth = f[depth_key][...].squeeze()
        stamps_raw = f[stamps_key][...]
    data_norm, T = _normalize_data_shape(data, depth_len=len(depth))
    stamps = _decode_stamps(stamps_raw)
    if len(stamps) != T:
        raise ValueError(f"{os.path.basename(path)}: stamps 长度({len(stamps)})与时间维({T})不一致。")
    return data_norm, depth, stamps


def sort_files_by_first_timestamp(filepaths, data_key="/data", depth_key="/depth", stamps_key="/stamps"):
    """按每个文件第一条时间戳排序；若失败则按文件名排序。"""
    items = []
    for p in filepaths:
        try:
            with h5py.File(p, "r") as f:
                sraw = f[stamps_key][...]
            ts = _decode_stamps(sraw)
            first = ts[0]
            items.append((p, first))
        except Exception:
            # 回退用文件名
            items.append((p, None))

    if all(t is not None for _, t in items):
        items.sort(key=lambda x: x[1])
    else:
        items.sort(key=lambda x: os.path.basename(x[0]))
    return [p for p, _ in items]


def check_depth_consistency(depth_list, atol=1e-6):
    """确保所有文件的 depth 一致；不一致直接抛错。"""
    base = depth_list[0]
    for i, d in enumerate(depth_list[1:], start=2):
        if len(d) != len(base) or not np.allclose(d, base, atol=atol, rtol=0):
            raise ValueError(f"第 {i} 个文件的 depth 与第 1 个不一致。请先统一深度网格。")


# --------------------------
# 方案 A：内存合并，直接画图
# --------------------------
def merge_in_memory(filepaths, data_key="/data", depth_key="/depth", stamps_key="/stamps"):
    """返回 dict: {'data': (depth, total_time), 'depth': (depth,), 'stamps': DatetimeIndex(total_time)}"""
    filepaths = sort_files_by_first_timestamp(filepaths, data_key, depth_key, stamps_key)

    data_blocks = []
    depth_list = []
    stamps_all = []

    for p in filepaths:
        d, z, t = read_one_h5(p, data_key, depth_key, stamps_key)
        data_blocks.append(d)     # (depth, time_i)
        depth_list.append(z)
        stamps_all.append(t)

    check_depth_consistency(depth_list)
    depth = depth_list[0]
    data_merged = np.concatenate(data_blocks, axis=1)  # 按时间拼
    stamps_merged = pd.DatetimeIndex(np.concatenate([np.asarray(s.values) for s in stamps_all]))
    return {"data": data_merged, "depth": depth, "stamps": stamps_merged}


# --------------------------
# 方案 B：流式写合并 HDF5
# --------------------------
def merge_to_h5(filepaths, out_h5,
                data_key="/data", depth_key="/depth", stamps_key="/stamps",
                compression="gzip", compression_opts=4):
    """把多个文件按时间拼成一个 H5；返回 (out_h5, total_time)"""
    filepaths = sort_files_by_first_timestamp(filepaths, data_key, depth_key, stamps_key)

    # 先扫一遍确定尺寸
    depths = []
    T_list = []
    for p in filepaths:
        with h5py.File(p, "r") as f:
            depth = f[depth_key][...].squeeze()
            data_shape = f[data_key].shape
        data_norm_shape, T = _normalize_data_shape(np.empty(data_shape), depth_len=len(depth))
        depths.append(depth)
        T_list.append(T)

    check_depth_consistency(depths)
    depth = depths[0]
    D = len(depth)
    T_total = int(np.sum(T_list))

    # 创建输出文件
    with h5py.File(out_h5, "w") as g:
        g.create_dataset(depth_key, data=depth)
        # 先创建空的 data、stamps（可压缩）
        dset = g.create_dataset(data_key, shape=(D, T_total), dtype="float32",
                                compression=compression, compression_opts=compression_opts)
        stset = g.create_dataset(stamps_key, shape=(T_total,), dtype=h5py.special_dtype(vlen=str))

        # 逐文件写入
        t0 = 0
        for p, Ti in zip(filepaths, T_list):
            data_i, depth_i, stamps_i = read_one_h5(p, data_key, depth_key, stamps_key)
            dset[:, t0:t0+Ti] = data_i.astype("float32")  # 节省体积
            # 写 stamps（以字符串形式写入，通用）
            stset[t0:t0+Ti] = np.array(stamps_i.strftime("%Y-%m-%d %H:%M:%S.%f").tolist(), dtype=object)
            t0 += Ti

    return out_h5, T_total


# --------------------------
# 画瀑布图（统一接口）
# --------------------------
def plot_waterfall(data, depth, stamps,
                   invert_depth=True,
                   cmap="bwr",
                   interpolation="antialiased",
                   title=None):
    """data:(depth, time)；depth:(depth,)；stamps: DatetimeIndex"""
    data = np.asarray(data)
    depth = np.asarray(depth)

    # x 轴用时间戳
    times_num = mdates.date2num(pd.DatetimeIndex(stamps).to_pydatetime())
    extent = (times_num[0], times_num[-1], depth.min(), depth.max())

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(data, cmap=cmap, aspect="auto", extent=extent, interpolation=interpolation)
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d\n%H:%M:%S"))
    fig.autofmt_xdate(rotation=0)

    if invert_depth:
        ax.set_ylim(depth.max(), depth.min())  # 井口在上：小深度在上

    ax.set_xlabel("Time")
    ax.set_ylabel("Depth (ft)")
    if title:
        ax.set_title(title)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("Strain (ε) or chosen unit")
    plt.tight_layout()
    plt.show()


# --------------------------
# Usage 示例
# --------------------------
if __name__ == "__main__":
    # 1) 找到一批 h5 文件（按你的路径修改）
    folder = r"./h5_batch"   # 改成你的目录
    files = sorted(glob.glob(os.path.join(folder, "*.h5")))

    # 2A) 方案A：内存合并+直接画图
    merged = merge_in_memory(files, data_key="/data", depth_key="/depth", stamps_key="/stamps")
    plot_waterfall(merged["data"], merged["depth"], merged["stamps"],
                   invert_depth=True, cmap="bwr",
                   title=f"Waterfall (N files={len(files)}, total T={merged['data'].shape[1]})")

    # 2B) 方案B：写出一个合并 H5（可选）
    # out_h5, Ttot = merge_to_h5(files, out_h5="./merged_all.h5",
    #                            data_key="/data", depth_key="/depth", stamps_key="/stamps")
    # print("Merged H5 saved:", out_h5, " total time points=", Ttot)
    # # 读回验证再画（可选）
    # with h5py.File(out_h5, "r") as f:
    #     data = f["/data"][...]
    #     depth = f["/depth"][...]
    #     stamps = _decode_stamps(f["/stamps"][...])
    # plot_waterfall(data, depth, stamps, invert_depth=True, title="Waterfall from merged H5")

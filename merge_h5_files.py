import os
import glob
import h5py
import numpy as np

# 源数据键（与你的实际文件结构一致）
DATA_KEY = "/Acquisition/Raw[0]/RawData"
TIME_KEY = "/Acquisition/Raw[0]/RawDataTime"

def _normalize_shape(arr, expected_depth, expected_T):
    """把 RawData 规整成 (expected_depth, expected_T)，
    如遇 (expected_T, expected_depth) 自动转置。"""
    if arr.shape == (expected_depth, expected_T):
        return arr
    if arr.shape == (expected_T, expected_depth):
        return arr.T
    raise ValueError(
        f"RawData 形状异常: {arr.shape}，既不是({expected_depth},{expected_T})也不是({expected_T},{expected_depth})"
    )

def _first_time_value(path):
    """读取某文件的第一个 RawDataTime 值，用于排序；失败时返回 None。"""
    try:
        with h5py.File(path, "r") as f:
            t = f[TIME_KEY][...]
            t = np.asarray(t).ravel()
            return int(t[0])
    except Exception:
        return None

def _sort_files_by_first_time(filepaths):
    """按每个文件的首个 RawDataTime 升序排序；如无法读取则退回按文件名排序。"""
    items = []
    all_ok = True
    for p in filepaths:
        v = _first_time_value(p)
        if v is None:
            all_ok = False
        items.append((p, v))
    if all_ok:
        items.sort(key=lambda x: x[1])
    else:
        items.sort(key=lambda x: os.path.basename(x[0]))
    return [p for p, _ in items]

def merge_preserve_time(filepaths, out_h5,
                        expected_depth=6100, expected_T_per_file=30,
                        sort_by_time=True, compression=None, compression_opts=None):
    """
    合并多个 HDF5：
      - /Acquisition/Raw[0]/RawData     -> (expected_depth, N*expected_T_per_file)
      - /Acquisition/Raw[0]/RawDataTime -> (N*expected_T_per_file,) 直接串接各文件原始 int 值（保持原样）
      - /Acquisition/FileNames          -> 源文件名（vlen utf-8）
      - /Acquisition/FileOffsets        -> 每个文件在时间维上的起止偏移（0,30,60,...）
    """
    assert filepaths, "没有找到任何输入文件。"

    files = _sort_files_by_first_time(filepaths) if sort_by_time else sorted(filepaths)

    # 读取首文件确定 dtype
    with h5py.File(files[0], "r") as f0:
        d0 = f0[DATA_KEY]
        t0 = f0[TIME_KEY]
        data_dtype = d0.dtype
        time_dtype = t0.dtype  # 通常是 int64
        shape0 = d0.shape

    # 统计文件数、总时间长度
    N = len(files)
    D = expected_depth
    T_each = expected_T_per_file
    T_total = N * T_each

    # 创建输出文件与分组
    with h5py.File(out_h5, "w") as g:
        acq_grp = g.require_group("Acquisition")
        raw_grp = acq_grp.require_group("Raw[0]")

        # 目标数据集
        dset = raw_grp.create_dataset(
            "RawData",
            shape=(D, T_total),
            dtype=data_dtype,
            compression=compression,
            compression_opts=compression_opts,
        )
        tset = raw_grp.create_dataset(
            "RawDataTime",
            shape=(T_total,),
            dtype=time_dtype,   # 用源文件的时间 dtype（通常为 <i8）
        )

        # 溯源信息
        str_dt = h5py.string_dtype(encoding="utf-8")
        acq_grp.create_dataset("FileNames",
                               data=np.array([os.path.basename(p) for p in files], dtype=object),
                               dtype=str_dt)
        acq_grp.create_dataset("FileOffsets",
                               data=np.arange(0, T_total + 1, T_each, dtype=np.int64))

        # 逐文件写入
        t0_off = 0
        last_tail_time = None
        for i, p in enumerate(files):
            with h5py.File(p, "r") as f:
                raw = f[DATA_KEY][...]
                tim = f[TIME_KEY][...]
            raw = _normalize_shape(raw, expected_depth=D, expected_T=T_each)
            tim = np.asarray(tim).ravel()

            if raw.shape[1] != T_each:
                raise ValueError(f"{os.path.basename(p)}: 时间维应为 {T_each}，实际 {raw.shape[1]}")
            if tim.shape[0] != T_each:
                raise ValueError(f"{os.path.basename(p)}: RawDataTime 长度应为 {T_each}，实际 {tim.shape[0]}")

            # 可选：连续性/单调性轻检查（发现异常仅打印提示，不阻塞）
            try:
                if not np.all(np.diff(tim) >= 0):
                    print(f"[警告] {os.path.basename(p)} 的 RawDataTime 非单调非降。")
                if last_tail_time is not None and tim[0] < last_tail_time:
                    print(f"[警告] 跨文件时间不递增：{os.path.basename(files[i-1])} -> {os.path.basename(p)}")
                last_tail_time = tim[-1]
            except Exception:
                pass

            # 写入数据
            dset[:, t0_off:t0_off + T_each] = raw
            tset[t0_off:t0_off + T_each] = tim.astype(time_dtype, copy=False)
            t0_off += T_each

    print(f"合并完成：{out_h5}")
    print(f"  {DATA_KEY} 形状: ({D}, {T_total})")
    print(f"  {TIME_KEY} 形状: ({T_total},)")
    print("  还写入了 /Acquisition/FileNames 与 /Acquisition/FileOffsets（便于追溯映射）")



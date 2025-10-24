# --- 分段线性平移函数：返回【平移后的深度】 ---

import numpy as np


def shift_depth_piecewise(depth,
                          z0=12000.0, z1=16000.0,
                          s0=145.0,  s1=160.0):
    """
    depth: 1D 深度数组（与 dstrain 的列一一对应）
    z0/z1: 起止深度（ft）
    s0/s1: 在 z0/z1 处的平移量（ft）
    规则：<=z0 用 s0；>=z1 用 s1；(z0,z1) 内线性插值
    """
    z = np.asarray(depth, dtype=float).reshape(-1)
    shift = np.empty_like(z)

    low  = (z <= z0)
    high = (z >= z1)
    mid  = ~(low | high)

    shift[low]  = s0
    shift[high] = s1
    shift[mid]  = s0 + (s1 - s0) * ( (z[mid] - z0) / (z1 - z0) )

    return z + shift  # ← 平移后的深度


def invert_shift_depth_piecewise(shifted_depth,
                                 z0=12000.0, z1=16000.0,
                                 s0=145.0,   s1=160.0):
    """
    给定【平移后的深度】(由 shift_depth_piecewise 返回)，求平移前的原始深度。

    参数
    ----
    shifted_depth : array-like
        平移后的深度数组（与 dstrain 的列一一对应）
    z0, z1 : float
        起止深度 (ft)，要求 z1 > z0
    s0, s1 : float
        在 z0/z1 处的平移量 (ft)

    反解规则
    --------
    令 a = (s1 - s0) / (z1 - z0)，t0 = z0 + s0，t1 = z1 + s1。
    - 若 z' ≤ t0 ：来自左段，原深度 z = z' - s0
    - 若 z' ≥ t1 ：来自右段，原深度 z = z' - s1
    - 否则（中段）：z = ( z' - s0 + a*z0 ) / (1 + a)

    注意
    ----
    仅当 1 + a > 0（中段单调递增）时，反解是良定的；若 1 + a <= 0，映射非单射，
    同一 z' 可能对应多个 z，将抛出异常以避免歧义。
    """
    shifted_depth = np.asarray(shifted_depth, dtype=float)
    orig_shape = shifted_depth.shape
    zs = shifted_depth.reshape(-1)

    if not (z1 > z0):
        raise ValueError("要求 z1 > z0。")

    a = (s1 - s0) / (z1 - z0)
    denom = 1.0 + a
    if not (denom > 0):
        raise ValueError("反解不唯一：需要 1 + (s1 - s0)/(z1 - z0) > 0。")

    t0 = z0 + s0
    t1 = z1 + s1  # 在 denom>0 时，t1 > t0

    z = np.empty_like(zs)

    low  = (zs <= t0)
    high = (zs >= t1)
    mid  = ~(low | high)

    z[low]  = zs[low]  - s0
    z[high] = zs[high] - s1
    z[mid]  = (zs[mid] - s0 + a * z0) / denom

    return z.reshape(orig_shape)
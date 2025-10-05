# plot_style.py
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from cycler import cycler

# 全局默认字体
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['mathtext.fontset'] = 'stix'


def make_standard_figure(
    figsize=(6, 12),         # 宽, 高（英寸）
    grid=False,              # 是否显示网格线（1 pt）
    color_cycle=('k', 'r', 'b', 'g', 'm', 'c', 'y'),
    legend=True,             # 是否放置图例（右上角等距）
    legend_border_pad=0.5,   # 图例与内框的等距 padding（单位：字体大小的倍数）
    nbins=7,                 # 主刻度最多分成的“箱数”，结合 min_n_ticks≈5 达到 5–7 个刻度
    set_rc_globally=True,    # 是否更新 rcParams（建议 True，确保“线宽/字体/颜色顺序”等默认生效）
    return_legend_placer=False # 如需对图例再精调，返回一个放置图例的函数
):
    """
    创建一张符合规范的单轴图，并返回 fig, ax(以及可选的 legend_placer)。
    使用范式：
        fig, ax = make_standard_figure(grid=True)
        ax.plot(x, y, label='Case A')  # 不用手动设置线宽/颜色，已按规范生效
        ax.set_xlabel('Time (s)')      # 轴标题会自动加粗、字号=20、字体=Times New Roman
        ax.set_ylabel('Pressure (MPa)')
        # 图例已自动放置(legend=True 时）；如需精调：
        # legend_placer()  # 若 return_legend_placer=True
        plt.show()
    """
    # ---------- 全局默认（推荐开启） ----------
    if set_rc_globally:
        mpl.rcParams.update({
            # 字体与字号
            'font.family': 'Times New Roman',
            'axes.labelsize': 20,
            'axes.labelweight': 'bold',   # 轴标题加粗
            'xtick.labelsize': 20,
            'ytick.labelsize': 20,
            'legend.fontsize': 20,

            # 线宽与颜色顺序
            'lines.linewidth': 2.25,      # 所有线条默认 2.25 pt
            'axes.prop_cycle': cycler('color', list(color_cycle)),

            # 刻度朝外 & 粗细
            'xtick.direction': 'out',
            'ytick.direction': 'out',
            'xtick.major.size': 6,
            'ytick.major.size': 6,
            'xtick.major.width': 2.0,
            'ytick.major.width': 2.0,

            # 坐标轴（内框）线宽
            'axes.linewidth': 2.0,
        })

    # ---------- 画布与坐标轴 ----------
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    # 去除“外框”：隐藏上、右脊线；下、左脊线加粗
    ax.spines['top'].set_visible(2.0)        #(False)
    ax.spines['right'].set_visible(2.0)
    ax.spines['bottom'].set_linewidth(2.0)
    ax.spines['left'].set_linewidth(2.0)

    # 刻度只保留下、左，朝外
    ax.tick_params(bottom=True, left=True, top=False, right=False, direction='out')

    # 控制主刻度数量在 5–7 之间
    ax.xaxis.set_major_locator(MaxNLocator(nbins=nbins, min_n_ticks=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=nbins, min_n_ticks=5))

    # 可选网格线
    if grid:
        ax.grid(True, linewidth=1.0)

    # 图例放置器（等距：对右与上等距，锚到(1,1)）
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

    # 默认立刻放置图例（如果用户已经给出了 label）
    if legend is True:
        _place_legend()  # 若此时无句柄，不会报错；后续你可再次手动调用 ax.legend()

    elif legend == 'auto':
        # 包一层：当第一次成功画出“带 label 的对象”后，立即放一次图例（只放一次）
        def auto_once():
            if auto_once.done:
                return
            if _place_legend() is not None:
                auto_once.done = True
        auto_once.done = False

        # 包装常见绘图方法：plot / scatter / bar / step / fill_between 等
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
    保存当前图或指定 fig 到 path。格式由扩展名决定:.png / .pdf / .svg ...
    """
    if fig is None:
        fig = plt.gcf()
    kwargs = {'dpi': dpi, 'transparent': transparent}
    if tight:
        kwargs.update({'bbox_inches': 'tight', 'pad_inches': pad})
    fig.savefig(path, **kwargs)
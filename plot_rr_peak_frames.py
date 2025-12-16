"""
绘制兴奋性RR最大峰值帧与抑制性RR最小峰值帧的精细分布
使用原始 segments 数据，结合抛物线插值得到子帧精度
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 导入 four_class 模块获取数据
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from four_class import (  # noqa: E402
    cfg,
    load_preprocessed_data_npz,
    load_data,
    filter_and_segment_data,
)


def load_segments():
    cache_file = os.path.join(cfg.data_path, "preprocessed_data_cache.npz")
    if os.path.exists(cache_file):
        segments, labels, neuron_pos_filtered = load_preprocessed_data_npz(cache_file)
        if segments is not None:
            return segments
        print("缓存加载失败，改为完整预处理...")
    else:
        print("未找到缓存，执行完整预处理...")

    neuron_data_orig, neuron_pos_orig, start_edges, stimulus_data = load_data()
    segments, labels, neuron_pos_filtered = filter_and_segment_data(
        neuron_data_orig, neuron_pos_orig, start_edges, stimulus_data, cfg
    )
    return segments


def load_rr_indices():
    rr_path = os.path.join(cfg.data_path, "rr_neuron_indices.csv")
    if not os.path.exists(rr_path):
        raise FileNotFoundError(f"未找到RR索引文件: {rr_path}")
    rr_df = pd.read_csv(rr_path)
    if "category" not in rr_df.columns or "neuron_index" not in rr_df.columns:
        raise ValueError("rr_neuron_indices.csv 缺少必要列 category 或 neuron_index")
    return rr_df


def parabolic_refine(trace, idx):
    if idx <= 0 or idx >= len(trace) - 1:
        return float(idx)
    ym1 = trace[idx - 1]
    y0 = trace[idx]
    yp1 = trace[idx + 1]
    denom = (ym1 - 2 * y0 + yp1)
    if denom == 0:
        return float(idx)
    offset = 0.5 * (ym1 - yp1) / denom
    return float(idx + offset)


def compute_peak_frames(segments, rr_df):
    results = []
    n_timepoints = segments.shape[2]

    for _, row in rr_df.iterrows():
        neuron_idx = int(row["neuron_index"])
        category = row["category"]
        trace = segments[:, neuron_idx, :].mean(axis=0)

        if category == "exc":
            frame_idx = int(np.argmax(trace))
            peak_value = trace[frame_idx]
        else:
            frame_idx = int(np.argmin(trace))
            peak_value = trace[frame_idx]

        refined_frame = parabolic_refine(trace, frame_idx)
        refined_frame = max(0.0, min(refined_frame, n_timepoints - 1))

        results.append(
            {
                "neuron_index": neuron_idx,
                "category": category,
                "peak_frame": refined_frame,
                "peak_value": peak_value,
            }
        )

    return pd.DataFrame(results), n_timepoints


def plot_peak_frames(result_df, n_timepoints, output_path, precision=1):
    """
    以指定小数精度绘制峰值帧分布直方图（无KDE）
    """
    sns.set(style="whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    step = 10 ** (-precision)
    categories = [
        ("exc", "兴奋性RR最大峰值帧分布", "#ff7f0e"),
        ("inh", "抑制性RR最小峰值帧分布", "#1f77b4"),
    ]

    for ax, (cat, title, color) in zip(axes, categories):
        subset = result_df[result_df["category"] == cat]["peak_frame"].to_numpy()
        if subset.size == 0:
            ax.set_visible(False)
            continue

        subset = np.round(subset, decimals=precision)
        counts = (
            pd.Series(subset)
            .value_counts()
            .sort_index()
        )

        ax.bar(
            counts.index,
            counts.values,
            width=step * 0.8,
            color=color,
            edgecolor="black",
            linewidth=0.8,
        )
        ax.set_title(title, fontsize=14)
        ax.set_ylabel("神经元数量（个）", fontsize=12)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

        x_min = max(0, counts.index.min() - step)
        x_max = min(n_timepoints - 1, counts.index.max() + step)
        ax.set_xlim(x_min, x_max)
        ax.set_xticks(np.round(np.arange(x_min, x_max + step, step), precision))

    axes[-1].set_xlabel(f"峰值帧（四舍五入至{precision}位小数）", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def main():
    print("=" * 80)
    print("加载数据并计算峰值帧（抛物线插值）")
    print("=" * 80)

    segments = load_segments()
    rr_df = load_rr_indices()

    result_df, n_timepoints = compute_peak_frames(segments, rr_df)

    exc_df = result_df[result_df["category"] == "exc"]
    inh_df = result_df[result_df["category"] == "inh"]

    print(f"\n兴奋性RR数量: {len(exc_df)}, 峰值帧范围: {exc_df['peak_frame'].min():.2f} - {exc_df['peak_frame'].max():.2f}")
    print(f"抑制性RR数量: {len(inh_df)}, 峰值帧范围: {inh_df['peak_frame'].min():.2f} - {inh_df['peak_frame'].max():.2f}")
    print(f"兴奋性RR峰值帧均值±标准差: {exc_df['peak_frame'].mean():.2f} ± {exc_df['peak_frame'].std():.2f}")
    print(f"抑制性RR峰值帧均值±标准差: {inh_df['peak_frame'].mean():.2f} ± {inh_df['peak_frame'].std():.2f}")

    plot_dir = os.path.join(cfg.data_path, "plot")
    os.makedirs(plot_dir, exist_ok=True)
    output_path = os.path.join(plot_dir, "rr_peak_frames_distribution.png")
    plot_peak_frames(result_df, n_timepoints, output_path)
    print(f"\n分布图已保存到: {output_path}")


if __name__ == "__main__":
    main()


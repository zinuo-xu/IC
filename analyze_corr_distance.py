import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_distance_pairs(distance_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(distance_csv)
    required = {"neuron_i", "neuron_j", "distance"}
    if not required.issubset(df.columns):
        raise ValueError(f"距离文件缺少列: {required - set(df.columns)}")
    return df.astype({"neuron_i": int, "neuron_j": int, "distance": float})


def load_corr_pairs(func_csv: Path) -> pd.DataFrame:
    matrix = pd.read_csv(func_csv, index_col=0)
    neuron_ids = matrix.index.astype(int).tolist()
    records = []
    for i in range(len(neuron_ids)):
        for j in range(i + 1, len(neuron_ids)):
            records.append(
                {
                    "neuron_i": neuron_ids[i],
                    "neuron_j": neuron_ids[j],
                    "corr": float(matrix.iat[i, j]),
                }
            )
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(
        description="统计高相关兴奋性 RR 神经元的物理距离分布"
    )
    parser.add_argument(
        "--func-csv",
        required=True,
        help="功能连接矩阵 CSV 路径（任意刺激条件）",
    )
    parser.add_argument(
        "--distance-csv",
        default="functional_connectivity/rr_exc_physical_distance_pairs.csv",
        help="距离 pair 列表 CSV，默认使用 compute_rr_physical_distance.py 的输出",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="相关系数筛选阈值（默认 0.8）",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="保存图像的路径；未指定时与功能连接 CSV 同目录同名添加后缀",
    )
    args = parser.parse_args()

    func_path = Path(args.func_csv).resolve()
    dist_path = Path(args.distance_csv).resolve()

    distance_df = load_distance_pairs(dist_path)
    corr_df = load_corr_pairs(func_path)

    merged = pd.merge(
        corr_df,
        distance_df,
        on=["neuron_i", "neuron_j"],
        how="inner",
    )

    filtered = merged[merged["corr"] >= args.threshold].copy()
    if filtered.empty:
        raise ValueError("没有找到满足阈值的神经元对，请降低 threshold。")

    stats = {
        "count": len(filtered),
        "min": filtered["distance"].min(),
        "median": filtered["distance"].median(),
        "max": filtered["distance"].max(),
    }
    print(
        f"共有 {stats['count']} 对神经元满足 corr >= {args.threshold}："
        f"最小 {stats['min']:.2f} μm，中位 {stats['median']:.2f} μm，最大 {stats['max']:.2f} μm"
    )

    sns.set_style("whitegrid")
    plt.figure(figsize=(7, 4))
    sns.histplot(filtered["distance"], bins=20, kde=True, color="#c44e52")
    plt.title(
        f"Distance Distribution (corr ≥ {args.threshold}, n={stats['count']})",
        fontsize=12,
    )
    plt.xlabel("Distance (μm)")
    plt.ylabel("Count")
    plt.tight_layout()

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = func_path.with_name(
            func_path.stem + f"_dist_corr_ge{args.threshold}.png"
        )
    plt.savefig(output_path, dpi=300)
    plt.close()

    filtered_path = output_path.with_suffix(".csv")
    filtered.to_csv(filtered_path, index=False)
    print(f"分布图保存至: {output_path}")
    print(f"满足阈值的 pair 详细列表写入: {filtered_path}")


if __name__ == "__main__":
    main()


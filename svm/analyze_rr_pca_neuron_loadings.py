"""
分析 RR 兴奋性神经元在 PCA 各主成分上的载荷：

- 使用与 `svm_multiclass_rr_pca.py` 相同的特征提取方式：
  - 只取兴奋性 RR 神经元
  - 使用第 13-14 帧的平均 dF/F 作为特征
- 对特征做一次标准化（mean=0, std=1）
- 在全部特征上做 PCA（保留所有主成分）
- 对每个主成分，将神经元按绝对载荷从大到小排序，导出 CSV
"""

import os
import sys
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# 确保可以从工程根目录导入 four_class.py 和 svm_multiclass_rr_pca.py
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CUR_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from four_class import cfg  # 仅用于拿到 data_path
from svm_multiclass_rr_pca import (
    load_rr_neurons_and_data,
    extract_multiclass_features,
)


def analyze_pca_neuron_loadings(
    frame_indices=(13, 14),
    random_state: int = 42,
    top_k_print: int = 10,
) -> pd.DataFrame:
    """
    做 PCA 并分析每个主成分对应的神经元载荷（从大到小排序）。

    参数
    ----
    frame_indices : tuple/list
        取平均的帧索引，默认 (13, 14)。
    random_state : int
        PCA 的随机种子。
    top_k_print : int
        在控制台打印每个主成分载荷最大的前 K 个神经元。

    返回
    ----
    df_long : DataFrame
        长表格式，每一行对应 (pc_index, neuron_index, loading, abs_loading, rank_in_pc)。
    """
    print("=" * 80)
    print("加载数据并提取 RR 兴奋性神经元特征用于 PCA 载荷分析")
    print("=" * 80)

    # 1. 加载 segments, labels, RR 兴奋性神经元索引
    segments, labels, exc_indices = load_rr_neurons_and_data()

    # 2. 提取特征 X（只用兴奋性 RR 神经元 + 指定帧的平均值），y 这里只是顺带不使用
    X, y = extract_multiclass_features(
        segments, labels, exc_indices, frame_indices=frame_indices
    )

    print("\n对特征做一次标准化（mean=0, std=1）用于 PCA...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("\n在全部标准化特征上做 PCA（保留所有主成分）...")
    # n_components=None 表示保留 min(n_samples, n_features) 个主成分
    pca = PCA(n_components=None, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)

    n_samples, n_features = X_scaled.shape
    n_components = pca.components_.shape[0]
    print(f"样本数: {n_samples}, 特征数(兴奋性 RR 神经元): {n_features}")
    print(f"保留主成分数: {n_components}")

    # pca.components_ 形状为 (n_components, n_features)
    components = pca.components_

    # 构造一个长表：每一行是一个 (pc, feature/neuron) 的组合
    records = []
    for pc_idx in range(n_components):
        comp = components[pc_idx]  # shape: (n_features,)
        abs_comp = np.abs(comp)

        # 排序索引：从绝对值最大的特征到最小
        sorted_idx = np.argsort(-abs_comp)  # 降序

        for rank, feat_pos in enumerate(sorted_idx, start=1):
            neuron_index = int(exc_indices[feat_pos])  # 对应到原始神经元索引
            loading = float(comp[feat_pos])
            abs_loading = float(abs_comp[feat_pos])

            records.append(
                {
                    "pc_index": pc_idx + 1,  # 从 1 开始编号更直观
                    "neuron_col_in_X": int(feat_pos),
                    "neuron_index_original": neuron_index,
                    "loading": loading,
                    "abs_loading": abs_loading,
                    "rank_in_pc": rank,
                }
            )

        # 控制台简单打印当前主成分的前 top_k 个神经元
        print("\n" + "-" * 80)
        print(f"PC{pc_idx + 1}: 绝对载荷最大的前 {top_k_print} 个兴奋性 RR 神经元")
        print("-" * 80)
        for k in range(min(top_k_print, n_features)):
            feat_pos = sorted_idx[k]
            neuron_index = int(exc_indices[feat_pos])
            loading = float(comp[feat_pos])
            abs_loading = float(abs_comp[feat_pos])
            print(
                f"Rank {k+1:2d}: neuron_col_in_X={feat_pos:3d}, "
                f"neuron_index_original={neuron_index:4d}, "
                f"loading={loading:+.4f}, |loading|={abs_loading:.4f}"
            )

    df_long = pd.DataFrame.from_records(records)

    # ---------------------------------------------------------------------
    # 额外：计算“每个神经元在整体 PCA 方差解释中的贡献率”
    # 思路：
    #   - 对标准化后的特征做 PCA，得到每个主成分的方差解释比例 w_k（explained_variance_ratio_）
    #   - 每个主成分的载荷向量为 a_kj（第 k 个 PC 在第 j 个神经元上的系数）
    #   - 对于神经元 j，定义其整体贡献率为：
    #         contrib_j ∝ Σ_k (w_k * a_kj^2)
    #     然后再在所有神经元上归一化，使 Σ_j contrib_j = 1
    #   - 这样既考虑了“神经元在重要主成分上的权重大不大”，又考虑了“该主成分本身解释了多少方差”。
    # ---------------------------------------------------------------------
    var_ratio = pca.explained_variance_ratio_  # shape: (n_components,)
    loadings = components  # shape: (n_components, n_features)

    # 先算每个主成分对每个神经元的加权贡献：w_k * a_kj^2
    # broadcast: (n_components, 1) * (n_components, n_features) -> (n_components, n_features)
    weighted_sq_loadings = (var_ratio[:, None] * (loadings ** 2))

    # 再对所有主成分求和，得到每个神经元的总贡献（尚未归一化）
    neuron_total_contrib = weighted_sq_loadings.sum(axis=0)  # shape: (n_features,)

    # 归一化为“贡献率”，保证所有神经元加起来为 1
    neuron_total_contrib = neuron_total_contrib / neuron_total_contrib.sum()

    neuron_contrib_records = []
    for feat_pos in range(n_features):
        neuron_index = int(exc_indices[feat_pos])
        neuron_contrib_records.append(
            {
                "neuron_col_in_X": int(feat_pos),
                "neuron_index_original": neuron_index,
                "pca_global_contribution": float(neuron_total_contrib[feat_pos]),
            }
        )

    df_neuron_contrib = pd.DataFrame.from_records(neuron_contrib_records)

    # 按贡献率从大到小排序，方便查看“最重要”的神经元
    df_neuron_contrib.sort_values(
        by="pca_global_contribution",
        ascending=False,
        inplace=True,
    )

    # 保存到 csv（所有 PC × 所有神经元的详细载荷）
    out_path = os.path.join(
        cfg.data_path,
        "rr_pca_neuron_loadings_all_components.csv",
    )
    df_long.to_csv(out_path, index=False, encoding="utf-8-sig")

    # 保存每个神经元的总体 PCA 贡献率
    contrib_out_path = os.path.join(
        cfg.data_path,
        "rr_pca_neuron_global_contribution.csv",
    )
    df_neuron_contrib.to_csv(contrib_out_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print(f"PCA 各主成分神经元载荷明细已保存到: {out_path}")
    print(f"每个神经元在整体 PCA 方差解释中的贡献率已保存到: {contrib_out_path}")
    print("=" * 80)

    return df_long


if __name__ == "__main__":
    analyze_pca_neuron_loadings(frame_indices=(13, 14), random_state=42, top_k_print=10)



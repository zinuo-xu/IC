"""
使用兴奋性 RR 神经元在第 13-14 帧的平均 dF/F 作为特征，
对 IC2、IC4、LC2、LC4 四类进行多分类 SVM。

特征维度较高（约 58 维），先对特征做标准化，再做 PCA 降维，保留能够解释 >=80% 方差的主成分。
"""

import os
import sys
import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    adjusted_rand_score,
    normalized_mutual_info_score,
)
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

# 确保可以从工程根目录导入 four_class.py
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CUR_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from four_class import cfg, load_preprocessed_data_npz, load_data, filter_and_segment_data


def load_rr_neurons_and_data():
    """
    加载 segments、labels 以及兴奋性 RR 神经元索引
    （逻辑拷贝自 svm_classification.py 中的同名函数，以避免导入冲突）。
    """
    print("=" * 80)
    print("加载数据和 RR 神经元索引（多分类）")
    print("=" * 80)

    # 1. 加载 segments 数据
    cache_file = os.path.join(cfg.data_path, "preprocessed_data_cache.npz")

    if os.path.exists(cache_file):
        segments, labels, neuron_pos_filtered = load_preprocessed_data_npz(cache_file)
        if segments is None:
            print("缓存加载失败，执行完整预处理...")
            neuron_data_orig, neuron_pos_orig, start_edges, stimulus_data = load_data()
            segments, labels, neuron_pos_filtered = filter_and_segment_data(
                neuron_data_orig, neuron_pos_orig, start_edges, stimulus_data, cfg
            )
    else:
        print("未找到缓存，执行完整预处理...")
        neuron_data_orig, neuron_pos_orig, start_edges, stimulus_data = load_data()
        segments, labels, neuron_pos_filtered = filter_and_segment_data(
            neuron_data_orig, neuron_pos_orig, start_edges, stimulus_data, cfg
        )

    print(f"数据加载完成: segments 形状 = {segments.shape}, labels 形状 = {labels.shape}")

    # 2. 读取兴奋性 RR 神经元索引
    rr_index_path = os.path.join(cfg.data_path, "rr_neuron_indices.csv")
    if not os.path.exists(rr_index_path):
        raise FileNotFoundError(f"未找到 RR 神经元索引文件: {rr_index_path}")

    rr_df = pd.read_csv(rr_index_path)
    exc_indices = rr_df[rr_df["category"] == "exc"]["neuron_index"].values

    print(f"兴奋性 RR 神经元数量: {len(exc_indices)}")

    return segments, labels, exc_indices


def extract_multiclass_features(segments,
                                labels,
                                exc_indices,
                                frame_indices=(13, 14)):
    """
    提取多分类特征：
    - 只使用兴奋性 RR 神经元
    - 取给定帧（默认 13、14 帧）的平均 dF/F 作为特征

    参数
    ----
    segments : ndarray, shape (n_trials, n_neurons, n_timepoints)
    labels   : ndarray, shape (n_trials,)
               原始标签 (1=IC2, 2=IC4, 3=LC2, 4=LC4)
    exc_indices : ndarray
        兴奋性 RR 神经元索引（在原始神经元维度上）
    frame_indices : tuple or list
        需要取平均的时间帧索引（Python 索引，从 0 开始）

    返回
    ----
    X : ndarray, shape (n_trials, n_exc_rr)
        每个 trial 的特征向量
    y : ndarray, shape (n_trials,)
        多分类标签（1/2/3/4）
    """
    frame_indices = np.atleast_1d(frame_indices).astype(int)
    frame_indices = np.unique(frame_indices)

    if np.any(frame_indices < 0):
        raise ValueError("帧索引必须为非负整数")

    max_idx = frame_indices.max()
    print("=" * 80)
    print(f"提取帧 {frame_indices.tolist()} 的平均特征 (多分类)...")
    print("=" * 80)

    if max_idx >= segments.shape[2]:
        raise ValueError(
            f"帧索引 {max_idx} 超出范围 (总帧数: {segments.shape[2]})"
        )

    # 只取兴奋性 RR 神经元，并在指定帧上取平均
    # 结果形状: (n_trials, n_exc_rr)
    X = segments[:, exc_indices, :][:, :, frame_indices].mean(axis=2)

    print(f"特征矩阵形状: {X.shape}")
    print(f"  样本数: {X.shape[0]}")
    print(f"  特征数（兴奋性 RR 神经元数）: {X.shape[1]}")

    # 多分类标签直接使用原始 labels: 1=IC2, 2=IC4, 3=LC2, 4=LC4
    y = labels.astype(int)

    # 统计各类别数量
    label_names = {
        1: "IC2",
        2: "IC4",
        3: "LC2",
        4: "LC4",
    }
    print("\n类别分布:")
    for k in sorted(label_names.keys()):
        cnt = np.sum(y == k)
        print(f"  {label_names[k]} (label={k}): {cnt} 个样本")

    return X, y


def train_and_evaluate_multiclass_svm(var_ratio=0.8,
                                      test_size=0.2,
                                      random_state=42):
    """
    训练并评估多分类 SVM（IC2 / IC4 / LC2 / LC4），特征为 RR 兴奋性神经元
    第 13-14 帧的平均激活值，先做标准化，再用 PCA 降维到解释 >=var_ratio 方差（默认 80%）。
    """
    # 1. 加载数据和 RR 神经元索引
    segments, labels, exc_indices = load_rr_neurons_and_data()

    # 2. 提取多分类特征（第 13、14 帧）
    X, y = extract_multiclass_features(
        segments, labels, exc_indices, frame_indices=(13, 14)
    )

    # 3. 在整个流程最开始做一次特征标准化
    print("\n" + "=" * 80)
    print("对原始 RR 特征做标准化（mean=0, std=1）")
    print("=" * 80)
    global_scaler = StandardScaler()
    X_scaled = global_scaler.fit_transform(X)

    # 4. 在标准化后的特征上划分训练/测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    print("\n数据划分:")
    print(f"  训练集: {X_train.shape[0]} 个样本")
    print(f"  测试集: {X_test.shape[0]} 个样本")
    print(f"  原始特征维度: {X_train.shape[1]}")

    # 5. 在标准化特征上做 PCA（解释 >= var_ratio 方差，默认 80%）
    print("\n开始 PCA 降维...")
    pca = PCA(n_components=var_ratio, random_state=random_state)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    n_components_ = pca.n_components_
    explained_ratio = pca.explained_variance_ratio_.sum()
    print(
        f"\nPCA 实际保留主成分数: {n_components_} "
        f"(累计解释方差比例: {explained_ratio*100:.2f}%)"
    )

    # 输出前 10 个主成分的方差解释比例（及其累计值）
    max_pc = min(10, len(pca.explained_variance_ratio_))
    print("\nExplained variance by first 10 PCs:")
    cum = 0.0
    for i in range(max_pc):
        var_i = pca.explained_variance_ratio_[i]
        cum += var_i
        print(f"  PC{i+1}: {var_i*100:.2f}% (cumulative: {cum*100:.2f}%)")

    # 6. 在 PCA 空间上训练线性 SVM（不再通过外部脚本或通用 Pipeline）
    print("\n开始训练多分类线性 SVM（在 PCA 空间）...")
    clf = SVC(kernel="linear", random_state=random_state)
    clf.fit(X_train_pca, y_train)
    print("训练完成。")

    # 定义类别顺序和名称（后面散点图、报告、混淆矩阵都会用到）
    label_order = [1, 2, 3, 4]
    target_names = ["IC2", "IC4", "LC2", "LC4"]

    # 7. 可视化 PCA 聚类效果（使用全部样本的前两个主成分）
    # 先用与训练时相同的 global_scaler 标准化全部样本，再做 PCA 投影
    X_scaled_for_pca = global_scaler.transform(X)
    X_pca_all = pca.transform(X_scaled_for_pca)
    pc1 = X_pca_all[:, 0]
    pc2 = X_pca_all[:, 1]

    plt.figure(figsize=(6, 5))
    colors = {1: "tab:blue", 2: "tab:orange", 3: "tab:green", 4: "tab:red"}
    for lab, name in zip(label_order, target_names):
        mask = (y == lab)
        plt.scatter(
            pc1[mask],
            pc2[mask],
            s=20,
            alpha=0.8,
            c=colors[lab],
            label=name,
            edgecolors="none",
        )
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA projection of RR features (IC2 / IC4 / LC2 / LC4)")
    plt.legend(title="Class", fontsize=8)
    plt.tight_layout()

    plot_dir = os.path.join(cfg.data_path, "plot")
    os.makedirs(plot_dir, exist_ok=True)
    scatter_path = os.path.join(plot_dir, "svm_pca_scatter_multiclass_rr.png")
    plt.savefig(scatter_path, dpi=300)
    plt.close()
    print(f"\nPCA 聚类散点图已保存到: {scatter_path}")

    # 7.1 在 PCA 空间上做 KMeans 聚类并可视化
    print("\nRunning KMeans clustering in PCA space (k=4)...")
    kmeans = KMeans(n_clusters=4, random_state=42, n_init= 50)
    cluster_labels = kmeans.fit_predict(X_pca_all)

    # 计算与真实标签的一致性指标
    ari = adjusted_rand_score(y, cluster_labels)
    nmi = normalized_mutual_info_score(y, cluster_labels)
    print(f"KMeans clustering ARI: {ari:.4f}, NMI: {nmi:.4f}")

    plt.figure(figsize=(6, 5))
    # 用 cluster id 上色
    cmap = plt.get_cmap("tab10")
    for cid in range(4):
        mask = (cluster_labels == cid)
        plt.scatter(
            pc1[mask],
            pc2[mask],
            s=20,
            alpha=0.7,
            color=cmap(cid),
            label=f"Cluster {cid}",
            edgecolors="none",
        )
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("KMeans clustering in PCA space (k=4)")
    plt.legend(title="Cluster", fontsize=8)
    plt.tight_layout()

    kmeans_pca_path = os.path.join(plot_dir, "kmeans_pca_clusters_multiclass_rr.png")
    plt.savefig(kmeans_pca_path, dpi=300)
    plt.close()
    print(f"KMeans 聚类（PCA 空间）散点图已保存到: {kmeans_pca_path}")

    # 8. 使用 t-SNE 可视化聚类效果
    print("\nRunning t-SNE for visualization (this may take some time)...")
    # 直接复用前面全局标准化后的特征
    X_scaled = global_scaler.transform(X)
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate=200,
        n_iter=1000,
        random_state=random_state,
        init="pca",
    )
    X_tsne = tsne.fit_transform(X_scaled)

    plt.figure(figsize=(6, 5))
    for lab, name in zip(label_order, target_names):
        mask = (y == lab)
        plt.scatter(
            X_tsne[mask, 0],
            X_tsne[mask, 1],
            s=20,
            alpha=0.8,
            c=colors[lab],
            label=name,
            edgecolors="none",
        )
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.title("t-SNE projection of RR features (IC2 / IC4 / LC2 / LC4)")
    plt.legend(title="Class", fontsize=8)
    plt.tight_layout()

    tsne_path = os.path.join(plot_dir, "tsne_scatter_multiclass_rr.png")
    plt.savefig(tsne_path, dpi=300)
    plt.close()
    print(f"t-SNE 聚类散点图已保存到: {tsne_path}")

    # 9. 评估（注意需要在 PCA 空间上做预测）
    y_train_pred = clf.predict(X_train_pca)
    y_test_pred = clf.predict(X_test_pca)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    print("\n准确率:")
    print(f"  训练集: {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"  测试集: {test_acc:.4f} ({test_acc*100:.2f}%)")

    print("\n测试集分类报告:")
    print(
        classification_report(
            y_test, y_test_pred, labels=label_order, target_names=target_names
        )
    )

    # 8. 混淆矩阵可视化
    cm = confusion_matrix(y_test, y_test_pred, labels=label_order)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
    )
    plt.title("SVM + PCA (>=80% variance) confusion matrix")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()

    fig_path = os.path.join(plot_dir, "svm_confusion_matrix_multiclass_rr_pca.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"\n混淆矩阵图已保存到: {fig_path}")

    # 8. 保存结果到 CSV
    results = {
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "pca_n_components": n_components_,
        "pca_explained_ratio": explained_ratio,
    }
    results_df = pd.DataFrame(
        list(results.items()), columns=["metric", "value"]
    )

    results_path = os.path.join(
        cfg.data_path, "svm_multiclass_rr_pca_results.csv"
    )
    results_df.to_csv(results_path, index=False, encoding="utf-8-sig")
    print(f"结果已保存到: {results_path}")

    return clf, results


if __name__ == "__main__":
    clf, results = train_and_evaluate_multiclass_svm(var_ratio=0.8)



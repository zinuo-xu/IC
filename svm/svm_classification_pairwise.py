"""
使用兴奋性RR神经元指定帧的平均dF/F值作为特征，进行配对二分类
1. IC2 vs LC2 (标签1 vs 标签3)
2. IC4 vs LC4 (标签2 vs 标签4)
"""
import numpy as np
import pandas as pd
import os
import sys
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

import matplotlib.pyplot as plt
import seaborn as sns

# 导入 four_class 模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from four_class import cfg, load_preprocessed_data_npz, load_data, filter_and_segment_data


# 可配置参数
USE_PCA = False
PCA_COMPONENTS = None  # None 表示保留全部
USE_L1_REGULARIZATION = False
SVM_C = None  # 设为None则使用模型默认C值

ENABLE_CV = True
CV_SPLITS = 5
CV_REPEATS = 3


def load_rr_neurons_and_data():
    """
    加载RR神经元索引和数据
    """
    print("=" * 80)
    print("加载数据和RR神经元索引")
    print("=" * 80)
    
    # 1. 加载segments数据
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
    
    print(f"数据加载完成: segments形状={segments.shape}, labels形状={labels.shape}")
    
    # 2. 读取兴奋性RR神经元索引
    rr_index_path = os.path.join(cfg.data_path, "rr_neuron_indices.csv")
    if not os.path.exists(rr_index_path):
        raise FileNotFoundError(f"未找到RR神经元索引文件: {rr_index_path}")
    
    rr_df = pd.read_csv(rr_index_path)
    exc_indices = rr_df[rr_df['category'] == 'exc']['neuron_index'].values
    
    print(f"兴奋性RR神经元数量: {len(exc_indices)}")
    
    return segments, labels, exc_indices


def extract_features(segments, labels, exc_indices, frame_indices=(13, 14), task='ic2_vs_lc2'):
    """
    提取指定帧兴奋性RR神经元的dF/F平均值作为特征，并根据任务类型过滤和标记
    
    参数:
    segments: (n_trials, n_neurons, n_timepoints)
    labels: (n_trials,) 原始标签 (1=IC2, 2=IC4, 3=LC2, 4=LC4)
    exc_indices: 兴奋性RR神经元的索引数组
    frame_indices: 需要提取并求均值的帧索引序列（默认13、14，即刺激开始后第1、2帧）
    task: 'ic2_vs_lc2' (标签1 vs 标签3) 或 'ic4_vs_lc4' (标签2 vs 标签4)
    
    返回:
    X: (n_trials, n_exc_neurons) 特征矩阵（已过滤）
    y: (n_trials,) 二分类标签 (1=IC类, -1=LC类)
    """
    frame_indices = np.atleast_1d(frame_indices).astype(int)
    frame_indices = np.unique(frame_indices)
    
    if np.any(frame_indices < 0):
        raise ValueError("帧索引必须为非负整数")
    
    max_idx = frame_indices.max()
    print(f"\n提取帧 {frame_indices.tolist()} 的平均特征...")
    
    if max_idx >= segments.shape[2]:
        raise ValueError(f"帧索引 {max_idx} 超出范围 (总帧数: {segments.shape[2]})")
    
    # 根据任务类型过滤数据
    labels = np.asarray(labels)
    if task == 'ic2_vs_lc2':
        # IC2 (1) vs LC2 (3)
        mask = (labels == 1) | (labels == 3)
        class_names = ['IC2', 'LC2']
    elif task == 'ic4_vs_lc4':
        # IC4 (2) vs LC4 (4)
        mask = (labels == 2) | (labels == 4)
        class_names = ['IC4', 'LC4']
    else:
        raise ValueError("task 必须是 'ic2_vs_lc2' 或 'ic4_vs_lc4'")
    
    # 过滤segments和labels
    segments_filtered = segments[mask, :, :]
    labels_filtered = labels[mask]
    
    # 提取指定帧并在时间维度上取平均
    X = segments_filtered[:, exc_indices, :][:, :, frame_indices].mean(axis=2)
    
    print(f"特征矩阵形状: {X.shape}")
    print(f"  样本数: {X.shape[0]}")
    print(f"  特征数（兴奋性RR神经元数）: {X.shape[1]}")
    
    # 重新标记：IC类 -> 1, LC类 -> -1
    y = np.ones(len(labels_filtered), dtype=int)
    if task == 'ic2_vs_lc2':
        y[labels_filtered == 3] = -1  # LC2标记为-1
    else:  # ic4_vs_lc4
        y[labels_filtered == 4] = -1  # LC4标记为-1
    
    # 统计各类别数量
    print(f"\n类别分布 ({task}):")
    ic_count = np.sum(y == 1)
    lc_count = np.sum(y == -1)
    print(f"  {class_names[0]}: {ic_count} 个样本")
    print(f"  {class_names[1]}: {lc_count} 个样本")
    
    return X, y, class_names


def build_classifier_pipeline(use_pca=False,
                              pca_components=None,
                              use_l1=False,
                              C=None,
                              random_state=42):
    """
    构建包含标准化、可选PCA和线性分类器的Pipeline
    """
    steps = [('scaler', StandardScaler())]
    
    if use_pca:
        steps.append(('pca', PCA(n_components=pca_components, random_state=random_state)))
    
    c_value = 1.0 if C is None else C
    
    if use_l1:
        classifier = LinearSVC(
            penalty='l1',
            dual=False,
            C=c_value,
            random_state=random_state,
            max_iter=10000
        )
    else:
        classifier = SVC(kernel='linear', C=c_value, random_state=random_state)
    
    steps.append(('clf', classifier))
    
    return Pipeline(steps)


def train_svm_classifier(
    X,
    y,
    test_size=0.2,
    random_state=42,
    use_pca=False,
    pca_components=None,
    use_l1=False,
    C=None
):
    """
    训练线性SVM分类器
    
    参数:
    X: 特征矩阵
    y: 标签
    test_size: 测试集比例
    random_state: 随机种子
    
    返回:
    clf: 训练好的Pipeline
    X_train, X_test, y_train, y_test: 划分后的数据
    """
    print("\n" + "=" * 80)
    print("训练线性SVM分类器")
    print("=" * 80)
    
    # 划分训练集和测试集（8:2）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"\n数据划分:")
    print(f"  训练集: {X_train.shape[0]} 个样本")
    print(f"  测试集: {X_test.shape[0]} 个样本")
    print(f"  特征维度: {X_train.shape[1]}")
    
    # 构建Pipeline并训练
    clf = build_classifier_pipeline(
        use_pca=use_pca,
        pca_components=pca_components,
        use_l1=use_l1,
        C=C,
        random_state=random_state
    )
    
    print("\n训练线性分类器...")
    clf.fit(X_train, y_train)
    print("训练完成")
    
    return clf, X_train, X_test, y_train, y_test


def evaluate_classifier(clf, X_train, X_test, y_train, y_test, class_names, task_name):
    """
    评估分类器性能
    """
    print("\n" + "=" * 80)
    print("评估分类器性能")
    print("=" * 80)
    
    # 预测
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
    
    # 计算准确率
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print(f"\n准确率:")
    print(f"  训练集: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    print(f"  测试集: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # 混淆矩阵（sklearn会按照标签值排序，-1在前，1在后）
    print(f"\n训练集混淆矩阵:")
    cm_train = confusion_matrix(y_train, y_train_pred, labels=[-1, 1])
    print(cm_train)
    print(f"  类别-1 ({class_names[1]}): {cm_train[0, 0]} 正确, {cm_train[0, 1]} 错误预测为{class_names[0]}")
    print(f"  类别1 ({class_names[0]}): {cm_train[1, 1]} 正确, {cm_train[1, 0]} 错误预测为{class_names[1]}")
    
    print(f"\n测试集混淆矩阵:")
    cm_test = confusion_matrix(y_test, y_test_pred, labels=[-1, 1])
    print(cm_test)
    print(f"  类别-1 ({class_names[1]}): {cm_test[0, 0]} 正确, {cm_test[0, 1]} 错误预测为{class_names[0]}")
    print(f"  类别1 ({class_names[0]}): {cm_test[1, 1]} 正确, {cm_test[1, 0]} 错误预测为{class_names[1]}")
    
    # 分类报告
    print(f"\n测试集分类报告:")
    print(classification_report(y_test, y_test_pred, 
                                labels=[-1, 1],
                                target_names=class_names))
    
    # 绘制混淆矩阵
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for ax, cm, title in zip(axes, [cm_train, cm_test], ['训练集', '测试集']):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=class_names,
                    yticklabels=class_names)
        ax.set_title(f'{title}混淆矩阵', fontsize=12)
        ax.set_ylabel('真实标签', fontsize=10)
        ax.set_xlabel('预测标签', fontsize=10)
    
    plt.tight_layout()
    save_path = os.path.join(cfg.data_path, "plot", f"svm_confusion_matrix_{task_name}.png")
    plt.savefig(save_path, dpi=300)
    print(f"\n混淆矩阵图已保存到: {save_path}")
    plt.close()
    
    return {
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'cm_train': cm_train,
        'cm_test': cm_test
    }


def evaluate_with_cross_validation(
    X,
    y,
    use_pca=False,
    pca_components=None,
    use_l1=False,
    C=1.0,
    n_splits=5,
    n_repeats=3,
    random_state=42
):
    """
    使用重复分层K折对Pipeline进行评估
    """
    if n_splits < 2 or n_repeats < 1:
        raise ValueError("n_splits需>=2且n_repeats需>=1")
    
    print("\n" + "=" * 80)
    print("重复分层K折交叉验证")
    print("=" * 80)
    print(f"  n_splits={n_splits}, n_repeats={n_repeats}")
    
    pipeline = build_classifier_pipeline(
        use_pca=use_pca,
        pca_components=pca_components,
        use_l1=use_l1,
        C=C,
        random_state=random_state
    )
    
    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state
    )
    
    scores = cross_val_score(pipeline, X, y, cv=rskf, n_jobs=-1)
    print(f"\n交叉验证准确率: mean={scores.mean():.4f}, std={scores.std():.4f}")
    return scores


def run_single_task(task, segments, labels, exc_indices, frame_indices):
    """
    运行单个分类任务
    """
    print("\n" + "=" * 80)
    print(f"任务: {task.upper()}")
    print("=" * 80)
    
    # 提取特征
    X, y, class_names = extract_features(
        segments, labels, exc_indices, 
        frame_indices=frame_indices, 
        task=task
    )
    
    # 根据配置确定PCA维度
    pca_components = None
    if USE_PCA:
        if PCA_COMPONENTS is None:
            pca_components = min(X.shape[1], 50)
        else:
            pca_components = min(X.shape[1], PCA_COMPONENTS)
    
    # 可选交叉验证
    cv_scores = None
    if ENABLE_CV:
        cv_scores = evaluate_with_cross_validation(
            X,
            y,
            use_pca=USE_PCA,
            pca_components=pca_components,
            use_l1=USE_L1_REGULARIZATION,
            C=SVM_C,
            n_splits=CV_SPLITS,
            n_repeats=CV_REPEATS
        )
    
    # 训练分类器
    clf, X_train, X_test, y_train, y_test = train_svm_classifier(
        X,
        y,
        use_pca=USE_PCA,
        pca_components=pca_components,
        use_l1=USE_L1_REGULARIZATION,
        C=SVM_C
    )
    
    # 评估性能
    results = evaluate_classifier(clf, X_train, X_test, y_train, y_test, class_names, task)
    
    # 显示特征重要性（SVM的权重）
    print("\n" + "=" * 80)
    print("特征重要性（SVM权重）")
    print("=" * 80)
    clf_step = clf.named_steps['clf']
    if USE_PCA:
        print("\n已启用PCA，系数对应主成分，无法直接映射至原始神经元。")
    else:
        weights = clf_step.coef_[0]
        abs_weights = np.abs(weights)
        top_k = min(10, len(weights))
        top_indices = np.argsort(abs_weights)[-top_k:][::-1]
        print(f"\n权重绝对值最大的前{top_k}个特征（兴奋性RR神经元）:")
        for i, idx in enumerate(top_indices):
            neuron_idx = exc_indices[idx]
            print(f"  {i+1}. 神经元 {neuron_idx}: 权重 = {weights[idx]:.6f}")
    
    return {
        'task': task,
        'class_names': class_names,
        'results': results,
        'cv_scores': cv_scores
    }


def main():
    """
    主函数：分别运行IC2 vs LC2和IC4 vs LC4两个任务
    """
    # 1. 加载数据
    segments, labels, exc_indices = load_rr_neurons_and_data()
    
    # 2. 提取特征的帧
    frame_indices = (13, 14)  # 刺激开始后第1、2帧
    
    # 3. 运行两个任务
    tasks = ['ic2_vs_lc2', 'ic4_vs_lc4']
    all_results = []
    
    for task in tasks:
        task_result = run_single_task(task, segments, labels, exc_indices, frame_indices)
        all_results.append(task_result)
    
    # 4. 保存汇总结果
    summary_metrics = []
    for task_result in all_results:
        task = task_result['task']
        results = task_result['results']
        cv_scores = task_result['cv_scores']
        
        summary_metrics.append({
            'task': task,
            'metric': '训练集准确率',
            'value': results['train_accuracy']
        })
        summary_metrics.append({
            'task': task,
            'metric': '测试集准确率',
            'value': results['test_accuracy']
        })
        if cv_scores is not None:
            summary_metrics.append({
                'task': task,
                'metric': '交叉验证准确率(均值)',
                'value': cv_scores.mean()
            })
            summary_metrics.append({
                'task': task,
                'metric': '交叉验证准确率(标准差)',
                'value': cv_scores.std()
            })
    
    results_df = pd.DataFrame(summary_metrics)
    results_path = os.path.join(cfg.data_path, "svm_pairwise_classification_results.csv")
    results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
    print(f"\n汇总结果已保存到: {results_path}")
    
    return all_results


if __name__ == "__main__":
    # 确保plot目录存在
    plot_dir = os.path.join(cfg.data_path, "plot")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    
    all_results = main()
    
    print("\n" + "=" * 80)
    print("所有分类任务完成！")
    print("=" * 80)


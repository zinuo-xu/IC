"""
找到每个RR细胞信号最强的时间窗
兴奋性和抑制性分别计算
"""
import numpy as np
import pandas as pd
import os
import sys

# 导入 four_class 模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from four_class import cfg, load_preprocessed_data_npz, load_data, filter_and_segment_data


def find_peak_window(neuron_traces, window_size=2, category='exc'):
    """
    找到信号最强的时间窗
    
    参数:
    neuron_traces: (n_trials, n_timepoints) 某个神经元在所有试次中的响应
    window_size: 时间窗大小（帧数）
    category: 'exc' 或 'inh'
    
    返回:
    peak_window_start: 最强信号窗口的起始帧
    peak_window_end: 最强信号窗口的结束帧
    peak_value: 最强信号值（兴奋性为正，抑制性为负）
    """
    # 计算所有试次的平均响应
    mean_trace = np.mean(neuron_traces, axis=0)  # (n_timepoints,)
    n_timepoints = len(mean_trace)
    
    if n_timepoints < window_size:
        return 0, n_timepoints - 1, np.mean(mean_trace)
    
    # 使用滑动窗口找到最强信号
    if category == 'exc':
        # 兴奋性：找正值最大的窗口
        max_window_sum = -np.inf
        peak_window_start = 0
        for start in range(n_timepoints - window_size + 1):
            window_sum = np.sum(mean_trace[start:start + window_size])
            if window_sum > max_window_sum:
                max_window_sum = window_sum
                peak_window_start = start
    else:
        # 抑制性：找负值最小（绝对值最大）的窗口
        min_window_sum = np.inf
        peak_window_start = 0
        for start in range(n_timepoints - window_size + 1):
            window_sum = np.sum(mean_trace[start:start + window_size])
            if window_sum < min_window_sum:
                min_window_sum = window_sum
                peak_window_start = start
        max_window_sum = min_window_sum  # 统一变量名
    
    peak_window_end = peak_window_start + window_size - 1
    peak_value = np.mean(mean_trace[peak_window_start:peak_window_end + 1])
    
    return peak_window_start, peak_window_end, peak_value


def find_best_category_window(segments, neuron_indices, window_size=2, category='exc'):
    """
    针对同一类别的所有RR神经元，找到共同的最显著时间窗

    参数:
    segments: (n_trials, n_neurons, n_timepoints)
    neuron_indices: 属于该类别的神经元索引列表
    window_size: 时间窗大小
    category: 'exc' 或 'inh'

    返回:
    dict 或 None
    """
    if len(neuron_indices) == 0:
        return None

    # 提取该类别所有RR神经元的响应，先对试次、后对神经元取平均
    category_traces = segments[:, neuron_indices, :]  # (n_trials, n_neurons, n_timepoints)
    mean_trace = category_traces.mean(axis=(0, 1))    # (n_timepoints,)
    n_timepoints = mean_trace.shape[0]

    if n_timepoints < window_size:
        return {
            'start': 0,
            'end': n_timepoints - 1,
            'score': float(mean_trace.mean())
        }

    if category == 'exc':
        best_score = -np.inf
        comparator = lambda x, y: x > y
    else:
        best_score = np.inf
        comparator = lambda x, y: x < y

    best_start = 0
    for start in range(n_timepoints - window_size + 1):
        window_score = mean_trace[start:start + window_size].sum()
        if comparator(window_score, best_score):
            best_score = window_score
            best_start = start

    return {
        'start': best_start,
        'end': best_start + window_size - 1,
        'score': best_score / window_size  # 平均值，便于比较
    }


def find_all_rr_peak_windows(window_size=2):
    """
    找到所有RR神经元的信号最强时间窗
    """
    print("=" * 80)
    print("开始查找RR神经元信号最强时间窗")
    print("=" * 80)
    
    # 1. 加载数据
    print("\n加载数据...")
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
    print(f"时间点数量: {segments.shape[2]}")
    
    # 2. 读取RR神经元索引
    rr_index_path = os.path.join(cfg.data_path, "rr_neuron_indices.csv")
    if not os.path.exists(rr_index_path):
        raise FileNotFoundError(f"未找到RR神经元索引文件: {rr_index_path}")
    
    rr_df = pd.read_csv(rr_index_path)
    print(f"\n读取到 {len(rr_df)} 个RR神经元索引")
    
    # 分离兴奋性和抑制性
    exc_indices = rr_df[rr_df['category'] == 'exc']['neuron_index'].values
    inh_indices = rr_df[rr_df['category'] == 'inh']['neuron_index'].values
    
    print(f"  兴奋性RR: {len(exc_indices)} 个")
    print(f"  抑制性RR: {len(inh_indices)} 个")
    
    # 3. 查找每个RR神经元的信号最强时间窗
    results = []
    
    print(f"\n使用窗口大小: {window_size} 帧")
    print("\n处理兴奋性RR神经元...")
    for idx, neuron_idx in enumerate(exc_indices):
        if (idx + 1) % 10 == 0:
            print(f"  进度: {idx + 1}/{len(exc_indices)}")
        
        # 提取该神经元在所有试次中的响应 (n_trials, n_timepoints)
        neuron_traces = segments[:, neuron_idx, :]
        
        peak_start, peak_end, peak_value = find_peak_window(
            neuron_traces, window_size=window_size, category='exc'
        )
        
        results.append({
            'neuron_index': neuron_idx,
            'category': 'exc',
            'peak_window_start': peak_start,
            'peak_window_end': peak_end,
            'peak_window_size': peak_end - peak_start + 1,
            'peak_value': peak_value,
            'peak_value_abs': abs(peak_value)
        })
    
    print("\n处理抑制性RR神经元...")
    for idx, neuron_idx in enumerate(inh_indices):
        if (idx + 1) % 10 == 0:
            print(f"  进度: {idx + 1}/{len(inh_indices)}")
        
        # 提取该神经元在所有试次中的响应
        neuron_traces = segments[:, neuron_idx, :]
        
        peak_start, peak_end, peak_value = find_peak_window(
            neuron_traces, window_size=window_size, category='inh'
        )
        
        results.append({
            'neuron_index': neuron_idx,
            'category': 'inh',
            'peak_window_start': peak_start,
            'peak_window_end': peak_end,
            'peak_window_size': peak_end - peak_start + 1,
            'peak_value': peak_value,
            'peak_value_abs': abs(peak_value)
        })
    
    # 4. 保存每个神经元的详细结果
    results_df = pd.DataFrame(results)
    output_path = os.path.join(cfg.data_path, "rr_peak_windows.csv")
    results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n每个神经元的峰值窗口已保存到: {output_path}")
    
    # 4+. 针对整个类别直接给出“最显著”的共同窗口
    print("\n" + "=" * 80)
    print("同类整体最显著的窗口（直接输出2帧）")
    print("=" * 80)
    exc_best_window = find_best_category_window(segments, exc_indices, window_size, 'exc')
    inh_best_window = find_best_category_window(segments, inh_indices, window_size, 'inh')

    if exc_best_window:
        print(f"兴奋性RR推荐窗口: [{exc_best_window['start']}, {exc_best_window['end']}] "
              f"(平均信号: {exc_best_window['score']:.4f})")
    else:
        print("兴奋性RR: 无数据")

    if inh_best_window:
        print(f"抑制性RR推荐窗口: [{inh_best_window['start']}, {inh_best_window['end']}] "
              f"(平均信号: {inh_best_window['score']:.4f})")
    else:
        print("抑制性RR: 无数据")

    # 5. 计算共同时间窗（取交集，以多胜少）
    print("\n" + "=" * 80)
    print("计算共同时间窗（取交集，以多胜少）")
    print("=" * 80)
    
    n_timepoints = segments.shape[2]
    exc_results = results_df[results_df['category'] == 'exc']
    inh_results = results_df[results_df['category'] == 'inh']
    
    # 计算兴奋性RR的共同时间窗
    if len(exc_results) > 0:
        exc_common_window = find_common_window(exc_results, n_timepoints, window_size, '兴奋性')
    else:
        exc_common_window = None
    
    # 计算抑制性RR的共同时间窗
    if len(inh_results) > 0:
        inh_common_window = find_common_window(inh_results, n_timepoints, window_size, '抑制性')
    else:
        inh_common_window = None
    
    # 6. 保存共同时间窗结果
    common_windows = []
    if exc_common_window:
        common_windows.append({
            'category': 'exc',
            'common_window_start': exc_common_window['start'],
            'common_window_end': exc_common_window['end'],
            'common_window_size': exc_common_window['end'] - exc_common_window['start'] + 1,
            'n_neurons_covered': exc_common_window['max_coverage'],
            'coverage_ratio': exc_common_window['max_coverage'] / len(exc_results),
            'all_neurons_count': len(exc_results)
        })
    
    if inh_common_window:
        common_windows.append({
            'category': 'inh',
            'common_window_start': inh_common_window['start'],
            'common_window_end': inh_common_window['end'],
            'common_window_size': inh_common_window['end'] - inh_common_window['start'] + 1,
            'n_neurons_covered': inh_common_window['max_coverage'],
            'coverage_ratio': inh_common_window['max_coverage'] / len(inh_results),
            'all_neurons_count': len(inh_results)
        })
    
    if common_windows:
        common_df = pd.DataFrame(common_windows)
        common_output_path = os.path.join(cfg.data_path, "rr_common_peak_windows.csv")
        common_df.to_csv(common_output_path, index=False, encoding='utf-8-sig')
        print(f"\n共同时间窗结果已保存到: {common_output_path}")
    
    # 7. 显示统计信息
    print("\n" + "=" * 80)
    print("统计摘要")
    print("=" * 80)
    
    if exc_common_window:
        print(f"\n兴奋性RR ({len(exc_results)} 个):")
        print(f"  共同峰值窗口: [{exc_common_window['start']}, {exc_common_window['end']}] (大小: {exc_common_window['end'] - exc_common_window['start'] + 1} 帧)")
        print(f"  覆盖神经元数: {exc_common_window['max_coverage']}/{len(exc_results)} ({exc_common_window['max_coverage']/len(exc_results)*100:.1f}%)")
        print(f"  峰值窗口起始帧范围: {exc_results['peak_window_start'].min()} - {exc_results['peak_window_start'].max()}")
        print(f"  峰值窗口结束帧范围: {exc_results['peak_window_end'].min()} - {exc_results['peak_window_end'].max()}")
        print(f"  平均峰值信号值: {exc_results['peak_value'].mean():.4f}")
    
    if inh_common_window:
        print(f"\n抑制性RR ({len(inh_results)} 个):")
        print(f"  共同峰值窗口: [{inh_common_window['start']}, {inh_common_window['end']}] (大小: {inh_common_window['end'] - inh_common_window['start'] + 1} 帧)")
        print(f"  覆盖神经元数: {inh_common_window['max_coverage']}/{len(inh_results)} ({inh_common_window['max_coverage']/len(inh_results)*100:.1f}%)")
        print(f"  峰值窗口起始帧范围: {inh_results['peak_window_start'].min()} - {inh_results['peak_window_start'].max()}")
        print(f"  峰值窗口结束帧范围: {inh_results['peak_window_end'].min()} - {inh_results['peak_window_end'].max()}")
        print(f"  平均峰值信号值: {inh_results['peak_value'].mean():.4f}")
    
    return results_df, common_windows


def find_common_window(results_df, n_timepoints, window_size, category_name):
    """
    在固定窗口大小下，找出覆盖神经元数最多的时间窗
    """
    if len(results_df) == 0:
        print(f"  {category_name}: 无神经元数据")
        return None

    if n_timepoints < window_size:
        print(f"  {category_name}: 总帧数不足以形成大小为 {window_size} 的窗口")
        return None

    max_coverage = -1
    best_start = 0
    for start in range(0, n_timepoints - window_size + 1):
        end = start + window_size - 1
        coverage = np.sum(
            (results_df['peak_window_start'] <= start) &
            (results_df['peak_window_end'] >= end)
        )
        if coverage > max_coverage:
            max_coverage = coverage
            best_start = start

    if max_coverage <= 0:
        print(f"  {category_name}: 没有窗口能覆盖任何神经元")
        return None

    best_end = best_start + window_size - 1

    print(f"\n{category_name} ({len(results_df)} 个神经元):")
    print(f"  最大覆盖数: {max_coverage}/{len(results_df)} 个神经元")
    print(f"  覆盖比例: {max_coverage / len(results_df) * 100:.1f}%")
    print(f"  共同时间窗: [{best_start}, {best_end}] (大小: {window_size} 帧)")

    return {
        'start': best_start,
        'end': best_end,
        'max_coverage': max_coverage
    }


if __name__ == "__main__":
    # 可以调整窗口大小，默认2帧
    window_size = 2
    print(f"使用窗口大小: {window_size} 帧")
    print("如需修改窗口大小，请在代码中修改 window_size 参数\n")
    
    results_df, common_windows = find_all_rr_peak_windows(window_size=window_size)


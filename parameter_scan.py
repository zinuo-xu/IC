"""
参数扫描脚本：扫描不同参数组合下的RR神经元筛选结果
"""
import numpy as np
import pandas as pd
import itertools
import time
import os
import sys

# 导入 four_class 模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from four_class import load_data, filter_and_segment_data, cfg, load_preprocessed_data_npz

# 重新定义筛选函数，返回更多信息（包括响应性神经元数量）
def _rr_selection_single_with_stats(trials, t_stimulus, l, reliability_threshold, snr_threshold, 
                                    effect_size_threshold, response_ratio_threshold, class_label="All"):
    """
    对一组试次进行RR神经元筛选，并返回统计信息
    返回: (rr_enhanced, rr_inhibitory, responsive_enhanced, responsive_inhibitory)
    """
    n_trials, n_neurons, n_timepoints = trials.shape
    
    # 定义时间窗口
    baseline_pre = np.arange(0, t_stimulus)
    baseline_post = np.arange(t_stimulus + l, n_timepoints)
    stimulus_window = np.arange(t_stimulus, t_stimulus + l)
    
    # 1. 响应性检测
    baseline_pre_mean = np.mean(trials[:, :, baseline_pre], axis=2)
    baseline_post_mean = np.mean(trials[:, :, baseline_post], axis=2)
    baseline_mean = (baseline_pre_mean + baseline_post_mean) / 2
    stimulus_mean = np.mean(trials[:, :, stimulus_window], axis=2)
    
    baseline_pre_std = np.std(trials[:, :, baseline_pre], axis=2)
    baseline_post_std = np.std(trials[:, :, baseline_post], axis=2)
    baseline_std = (baseline_pre_std + baseline_post_std) / 2
    stimulus_std = np.std(trials[:, :, stimulus_window], axis=2)
    
    # Cohen's d效应大小
    pooled_std = np.sqrt((baseline_std**2 + stimulus_std**2) / 2)
    effect_size = np.abs(stimulus_mean - baseline_mean) / (pooled_std + 1e-8)
    
    # 响应性标准
    response_ratio = np.mean(effect_size > effect_size_threshold, axis=0)
    
    # 兴奋性响应 (不加reliability)
    responsive_enhanced = np.where((response_ratio > response_ratio_threshold) & 
                                   (np.mean(stimulus_mean > baseline_mean, axis=0) > response_ratio_threshold))[0].tolist()
    
    # 抑制性响应 (不加reliability)
    responsive_inhibitory = np.where((response_ratio > response_ratio_threshold) &
                                     (np.mean(stimulus_mean < baseline_mean, axis=0) > response_ratio_threshold))[0].tolist()
    
    # 2. 可靠性检测
    signal_strength = np.abs(stimulus_mean - baseline_mean)
    noise_level = baseline_std + 1e-8
    snr = signal_strength / noise_level
    reliability_ratio = np.mean(snr > snr_threshold, axis=0)
    reliable_neurons = np.where(reliability_ratio >= reliability_threshold)[0].tolist()
    
    # 3. 最终RR神经元（加reliability）
    rr_enhanced = list(set(responsive_enhanced) & set(reliable_neurons))
    rr_inhibitory = list(set(responsive_inhibitory) & set(reliable_neurons))
    
    return set(rr_enhanced), set(rr_inhibitory), set(responsive_enhanced), set(responsive_inhibitory)


def rr_selection_by_class_with_stats(segments, labels, reliability_threshold, snr_threshold, 
                                     effect_size_threshold, response_ratio_threshold):
    """
    分类别筛选RR神经元，返回统计信息
    """
    all_class_ids = sorted(np.unique(labels))
    valid_class_ids = [cls for cls in all_class_ids if cls > 0]
    
    global_rr_enhanced_set = set()
    global_rr_inhibitory_set = set()
    global_responsive_enhanced_set = set()
    global_responsive_inhibitory_set = set()
    
    for class_id in valid_class_ids:
        class_mask = (labels == class_id)
        class_segments = segments[class_mask, :, :]
        
        if class_segments.shape[0] < 2:
            continue
            
        rr_exc, rr_inh, resp_exc, resp_inh = _rr_selection_single_with_stats(
            class_segments,
            t_stimulus=cfg.exp_info["t_stimulus"],
            l=cfg.exp_info["l_stimulus"],
            reliability_threshold=reliability_threshold,
            snr_threshold=snr_threshold,
            effect_size_threshold=effect_size_threshold,
            response_ratio_threshold=response_ratio_threshold,
            class_label=str(int(class_id))
        )
        
        global_rr_enhanced_set.update(rr_exc)
        global_rr_inhibitory_set.update(rr_inh)
        global_responsive_enhanced_set.update(resp_exc)
        global_responsive_inhibitory_set.update(resp_inh)
    
    return (sorted(list(global_rr_enhanced_set)),
            sorted(list(global_rr_inhibitory_set)),
            sorted(list(global_responsive_enhanced_set)),
            sorted(list(global_responsive_inhibitory_set)))


def parameter_scan():
    """
    执行参数扫描
    """
    print("=" * 80)
    print("开始参数扫描")
    print("=" * 80)
    
    # 加载数据
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
    
    # 定义参数范围
    reliability_thresholds = np.arange(0.55, 0.85, 0.05)  # 0.55-0.8, 步长0.05
    snr_thresholds = np.arange(0.65, 0.85, 0.05)  # 0.65-0.8, 步长0.05
    effect_size_thresholds = np.arange(0.5, 0.75, 0.05)  # 0.5-0.7, 步长0.05
    response_ratio_thresholds = np.arange(0.6, 0.85, 0.05)  # 0.6-0.8, 步长0.05
    
    # 生成所有参数组合
    param_combinations = list(itertools.product(
        reliability_thresholds,
        snr_thresholds,
        effect_size_thresholds,
        response_ratio_thresholds
    ))
    
    total_combinations = len(param_combinations)
    print(f"\n总共需要测试 {total_combinations} 组参数组合")
    
    # 存储结果
    results = []
    
    start_time = time.time()
    
    for idx, (rel_th, snr_th, eff_th, resp_th) in enumerate(param_combinations):
        if (idx + 1) % 50 == 0:
            elapsed = time.time() - start_time
            remaining = elapsed / (idx + 1) * (total_combinations - idx - 1)
            print(f"进度: {idx + 1}/{total_combinations} ({100*(idx+1)/total_combinations:.1f}%), "
                  f"已用时间: {elapsed:.1f}s, 预计剩余: {remaining:.1f}s")
        
        try:
            rr_exc, rr_inh, resp_exc, resp_inh = rr_selection_by_class_with_stats(
                segments, np.array(labels),
                reliability_threshold=rel_th,
                snr_threshold=snr_th,
                effect_size_threshold=eff_th,
                response_ratio_threshold=resp_th
            )
            
            n_rr_exc = len(rr_exc)
            n_rr_inh = len(rr_inh)
            n_resp_exc = len(resp_exc)
            n_resp_inh = len(resp_inh)
            
            # 计算reliability减少的细胞数
            reduction_exc = n_resp_exc - n_rr_exc
            reduction_inh = n_resp_inh - n_rr_inh
            reduction_total = (n_resp_exc + n_resp_inh) - (n_rr_exc + n_rr_inh)
            
            results.append({
                'reliability_threshold': rel_th,
                'snr_threshold': snr_th,
                'effect_size_threshold': eff_th,
                'response_ratio_threshold': resp_th,
                'n_responsive_exc': n_resp_exc,
                'n_responsive_inh': n_resp_inh,
                'n_responsive_total': n_resp_exc + n_resp_inh,
                'n_rr_exc': n_rr_exc,
                'n_rr_inh': n_rr_inh,
                'n_rr_total': n_rr_exc + n_rr_inh,
                'reduction_exc': reduction_exc,
                'reduction_inh': reduction_inh,
                'reduction_total': reduction_total,
                'reduction_ratio_exc': reduction_exc / n_resp_exc if n_resp_exc > 0 else 0,
                'reduction_ratio_inh': reduction_inh / n_resp_inh if n_resp_inh > 0 else 0,
                'reduction_ratio_total': reduction_total / (n_resp_exc + n_resp_inh) if (n_resp_exc + n_resp_inh) > 0 else 0,
            })
        except Exception as e:
            print(f"参数组合 ({rel_th:.2f}, {snr_th:.2f}, {eff_th:.2f}, {resp_th:.2f}) 出错: {e}")
            results.append({
                'reliability_threshold': rel_th,
                'snr_threshold': snr_th,
                'effect_size_threshold': eff_th,
                'response_ratio_threshold': resp_th,
                'n_responsive_exc': -1,
                'n_responsive_inh': -1,
                'n_responsive_total': -1,
                'n_rr_exc': -1,
                'n_rr_inh': -1,
                'n_rr_total': -1,
                'reduction_exc': -1,
                'reduction_inh': -1,
                'reduction_total': -1,
                'reduction_ratio_exc': -1,
                'reduction_ratio_inh': -1,
                'reduction_ratio_total': -1,
            })
    
    # 转换为DataFrame并保存
    df_results = pd.DataFrame(results)
    
    output_path = os.path.join(cfg.data_path, "parameter_scan_results.csv")
    df_results.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    total_time = time.time() - start_time
    print(f"\n" + "=" * 80)
    print(f"参数扫描完成！")
    print(f"总耗时: {total_time:.1f}秒 ({total_time/60:.1f}分钟)")
    print(f"结果已保存到: {output_path}")
    print(f"共 {len(results)} 组参数组合")
    print("=" * 80)
    
    # 显示一些统计信息
    print("\n统计摘要:")
    print(f"  兴奋性RR数量范围: {df_results['n_rr_exc'].min()} - {df_results['n_rr_exc'].max()}")
    print(f"  抑制性RR数量范围: {df_results['n_rr_inh'].min()} - {df_results['n_rr_inh'].max()}")
    print(f"  平均减少比例 (兴奋性): {df_results['reduction_ratio_exc'].mean():.2%}")
    print(f"  平均减少比例 (抑制性): {df_results['reduction_ratio_inh'].mean():.2%}")
    print(f"  平均减少比例 (总计): {df_results['reduction_ratio_total'].mean():.2%}")
    
    return df_results


if __name__ == "__main__":
    parameter_scan()


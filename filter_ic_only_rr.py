"""
筛选对任意一个IC条件兴奋性RR，而对2个LC都不兴奋性RR的神经元
参照 four_class.py 的预处理流程
只筛选兴奋性RR神经元，不包括抑制性
"""

import h5py
import os
import numpy as np
import scipy.io
from scipy import stats
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import ndimage
import time
import sys

# 导入 four_class.py 中的配置和函数
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from four_class import (
    ExpConfig, process_trigger, segment_neuron_data,
    filter_and_segment_data, load_data, load_preprocessed_data_npz,
    save_preprocessed_data_npz, _rr_selection_single, _rr_distribution_plot,
    reclassify
)

# 使用相同的配置
cfg = ExpConfig(r'C:\Users\xuzinuo\Desktop\79\M79.json')

def rr_selection_ic_only(segments, labels, **kwargs):
    """
    筛选对任意一个IC条件兴奋性RR，而对2个LC都不兴奋性RR的神经元
    
    参数:
    segments: (n_trials, n_neurons, n_timepoints)
    labels: (n_trials,) 包含类别标签的 NumPy 数组
            IC2->1, IC4->2, LC2->3, LC4->4
    **kwargs: 传递给 _rr_selection_single 的筛选参数
    
    返回:
    ic_only_rr_exc: 满足条件的兴奋性RR神经元索引列表
    """
    start_time = time.time()
    print("\n开始筛选：对任意一个IC条件兴奋性RR，而对2个LC都不兴奋性RR的神经元...")
    
    # IC条件：类别1 (IC2) 和类别2 (IC4)
    # LC条件：类别3 (LC2) 和类别4 (LC4)
    ic_class_ids = [1, 2]  # IC2, IC4
    lc_class_ids = [3, 4]  # LC2, LC4
    
    n_neurons = segments.shape[1]
    
    # 存储每个类别的兴奋性RR神经元
    class_rr_exc = {}  # {class_id: set of neuron indices}
    
    # 对每个类别分别进行RR筛选
    all_class_ids = sorted(np.unique(labels))
    valid_class_ids = [cls for cls in all_class_ids if cls > 0]
    
    for class_id in valid_class_ids:
        # 筛选出当前类别的试次
        class_mask = (labels == class_id)
        class_segments = segments[class_mask, :, :]
        
        # 检查试次数量
        if class_segments.shape[0] < 2:
            print(f"警告: 类别 {class_id} 试次数量不足({class_segments.shape[0]})，跳过该类别筛选。")
            class_rr_exc[class_id] = set()
            continue
        
        # 对当前类别的试次进行 RR 筛选（只关注兴奋性）
        rr_exc_indices, _ = _rr_selection_single(
            class_segments,
            class_label=str(int(class_id)),
            **kwargs
        )
        
        class_rr_exc[class_id] = rr_exc_indices
    
    # 筛选逻辑：
    # 1. 对IC条件（类别1或2）至少有一个兴奋性RR
    # 2. 对LC条件（类别3和4）都不兴奋性RR
    
    ic_only_rr_exc = set()
    
    # 遍历所有神经元
    for neuron_idx in range(n_neurons):
        # 检查IC条件：至少一个IC条件有兴奋性RR
        ic_rr_exc = any(neuron_idx in class_rr_exc.get(cls, set()) for cls in ic_class_ids)
        
        # 检查LC条件：两个LC条件都不兴奋性RR
        lc_not_rr = all(
            neuron_idx not in class_rr_exc.get(cls, set())
            for cls in lc_class_ids
        )
        
        # 如果满足条件：IC至少一个兴奋性RR，且LC都不兴奋性RR
        if ic_rr_exc and lc_not_rr:
            ic_only_rr_exc.add(neuron_idx)
    
    # 转为排序列表
    ic_only_rr_exc = sorted(list(ic_only_rr_exc))
    
    elapsed_time = time.time() - start_time
    print(f"\n筛选完成，总耗时: {elapsed_time:.2f}秒")
    print(f"筛选结果:")
    print(f"  - 兴奋性RR神经元: {len(ic_only_rr_exc)} 个")
    
    # 打印详细信息
    print(f"\n详细统计:")
    # 统计每个类别中筛选出的神经元数量
    ic2_rr = [n for n in ic_only_rr_exc if n in class_rr_exc.get(1, set())]
    ic4_rr = [n for n in ic_only_rr_exc if n in class_rr_exc.get(2, set())]
    both_ic_rr = [n for n in ic_only_rr_exc if n in class_rr_exc.get(1, set()) and n in class_rr_exc.get(2, set())]
    only_ic2_rr = [n for n in ic_only_rr_exc if n in class_rr_exc.get(1, set()) and n not in class_rr_exc.get(2, set())]
    only_ic4_rr = [n for n in ic_only_rr_exc if n not in class_rr_exc.get(1, set()) and n in class_rr_exc.get(2, set())]
    
    print(f"  IC2 (类别1) 兴奋性RR: {len(ic2_rr)} 个")
    print(f"  IC4 (类别2) 兴奋性RR: {len(ic4_rr)} 个")
    print(f"  - 仅在IC2中RR: {len(only_ic2_rr)} 个")
    print(f"  - 仅在IC4中RR: {len(only_ic4_rr)} 个")
    print(f"  - 同时在IC2和IC4中RR: {len(both_ic_rr)} 个")
    
    # 验证LC条件：应该都为0
    lc2_rr = [n for n in ic_only_rr_exc if n in class_rr_exc.get(3, set())]
    lc4_rr = [n for n in ic_only_rr_exc if n in class_rr_exc.get(4, set())]
    print(f"  LC2 (类别3) 兴奋性RR: {len(lc2_rr)} 个 (应该为0)")
    print(f"  LC4 (类别4) 兴奋性RR: {len(lc4_rr)} 个 (应该为0)")
    
    # 验证：如果LC有RR，说明筛选逻辑有问题
    if len(lc2_rr) > 0 or len(lc4_rr) > 0:
        print(f"\n⚠️  警告：筛选出的神经元中有在LC条件下是RR的，这不应该发生！")
        if len(lc2_rr) > 0:
            print(f"     LC2中RR的神经元: {lc2_rr}")
        if len(lc4_rr) > 0:
            print(f"     LC4中RR的神经元: {lc4_rr}")
    
    # 返回结果，包括分类信息
    return {
        'all': ic_only_rr_exc,
        'only_ic2': sorted(only_ic2_rr),
        'only_ic4': sorted(only_ic4_rr),
        'both_ic': sorted(both_ic_rr)
    }


if __name__ == "__main__":
    print("="*70)
    print("开始运行：筛选对任意一个IC条件RR，而对2个LC都不RR的神经元")
    print("="*70)
    
    plot_dir = os.path.join(cfg.data_path, "plot")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
        print(f"已创建图片保存目录: {plot_dir}")
    
    # 定义缓存文件路径
    cache_file = os.path.join(cfg.data_path, "preprocessed_data_cache.npz")
    print(f"预处理数据缓存文件路径: {cache_file}")
    
    # 1. 尝试加载缓存数据
    segments, labels, neuron_pos_filtered = None, None, None
    load_from_cache_successful = False
    
    if os.path.exists(cache_file):
        segments_cached, labels_cached, neuron_pos_filtered_cached = load_preprocessed_data_npz(cache_file)
        if segments_cached is not None:
            segments = segments_cached
            labels = labels_cached
            neuron_pos_filtered = neuron_pos_filtered_cached
            load_from_cache_successful = True
            print("✓ 从缓存加载预处理数据成功")
    
    # 2. 如果缓存加载失败，执行完整的加载和预处理流程
    if not load_from_cache_successful:
        print("未找到有效缓存或缓存加载失败，执行完整的加载和预处理流程...")
        
        # 2a. 加载原始数据
        neuron_data_orig, neuron_pos_orig, start_edges, stimulus_data = load_data()
        
        # 2b. 执行预处理和分割步骤
        segments, labels, neuron_pos_filtered, dff, F0_dynamic = filter_and_segment_data(
            neuron_data_orig, neuron_pos_orig, start_edges, stimulus_data, cfg
        )
        
        # 2c. 保存缓存
        save_preprocessed_data_npz(segments, labels, neuron_pos_filtered, cache_file)
    else:
        print("缓存加载成功，跳过原始数据加载和预处理步骤。")
    
    print(f"\n数据形状: segments={segments.shape}, labels={labels.shape}")
    print(f"标签分布: {dict(zip(*np.unique(labels, return_counts=True)))}")
    
    # 3. 执行筛选：对任意一个IC条件兴奋性RR，而对2个LC都不兴奋性RR
    rr_results = rr_selection_ic_only(
        segments, 
        np.array(labels)
    )
    
    # 提取分类结果
    ic_only_rr_exc = np.array(rr_results['all'], dtype=int)
    only_ic2_rr = np.array(rr_results['only_ic2'], dtype=int)
    only_ic4_rr = np.array(rr_results['only_ic4'], dtype=int)
    both_ic_rr = np.array(rr_results['both_ic'], dtype=int)
    
    # 4. 提取筛选出的神经元数据
    if ic_only_rr_exc.size > 0:
        exc_segments = segments[:, ic_only_rr_exc, :]
        exc_neuron_pos = neuron_pos_filtered[:, ic_only_rr_exc]
        
        print(f"\n提取的数据形状:")
        print(f"  - 兴奋性IC-only RR神经元位置: {exc_neuron_pos.shape}")
    else:
        print("\n警告: 未找到满足条件的神经元！")
        exc_neuron_pos = np.empty((2, 0))
    
    # 5. 可视化分布（只显示兴奋性，抑制性传入空数组）
    if ic_only_rr_exc.size > 0:
        _rr_distribution_plot(
            neuron_pos_filtered,
            exc_neuron_pos,
            np.empty((2, 0)),  # 抑制性传入空数组
            plot_dir,
            "IC_only_RR_exc"
        )
        print(f"\n✓ 已保存分布图到: {os.path.join(plot_dir, 'rr_distribution_IC_only_RR_exc.png')}")
    
    # 6. 保存结果到CSV（所有筛选出的神经元）
    output_path = os.path.join(cfg.data_path, "ic_only_rr_neuron_indices.csv")
    result_df = pd.DataFrame({
        "neuron_index": ic_only_rr_exc,
        "category": ["exc"] * len(ic_only_rr_exc)
    })
    result_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n✓ 筛选结果已保存到: {output_path}")
    
    # 7. 保存分类结果到单独的CSV文件
    if only_ic2_rr.size > 0:
        only_ic2_df = pd.DataFrame({
            "neuron_index": only_ic2_rr,
            "category": ["only_IC2"] * len(only_ic2_rr)
        })
        only_ic2_path = os.path.join(cfg.data_path, "ic_only_rr_only_IC2.csv")
        only_ic2_df.to_csv(only_ic2_path, index=False, encoding="utf-8-sig")
        print(f"✓ 仅在IC2中RR的神经元已保存到: {only_ic2_path}")
    
    if only_ic4_rr.size > 0:
        only_ic4_df = pd.DataFrame({
            "neuron_index": only_ic4_rr,
            "category": ["only_IC4"] * len(only_ic4_rr)
        })
        only_ic4_path = os.path.join(cfg.data_path, "ic_only_rr_only_IC4.csv")
        only_ic4_df.to_csv(only_ic4_path, index=False, encoding="utf-8-sig")
        print(f"✓ 仅在IC4中RR的神经元已保存到: {only_ic4_path}")
    
    if both_ic_rr.size > 0:
        both_ic_df = pd.DataFrame({
            "neuron_index": both_ic_rr,
            "category": ["both_IC"] * len(both_ic_rr)
        })
        both_ic_path = os.path.join(cfg.data_path, "ic_only_rr_both_IC.csv")
        both_ic_df.to_csv(both_ic_path, index=False, encoding="utf-8-sig")
        print(f"✓ 同时在IC2和IC4中RR的神经元已保存到: {both_ic_path}")
    
    # 8. 打印神经元索引（详细分类）
    print(f"\n" + "="*70)
    print("详细分类的神经元索引:")
    print("="*70)
    
    print(f"\n【仅在IC2中RR的神经元】({len(only_ic2_rr)}个):")
    if only_ic2_rr.size > 0:
        print(f"  {only_ic2_rr.tolist()}")
    else:
        print("  (无)")
    
    print(f"\n【仅在IC4中RR的神经元】({len(only_ic4_rr)}个):")
    if only_ic4_rr.size > 0:
        print(f"  {only_ic4_rr.tolist()}")
    else:
        print("  (无)")
    
    print(f"\n【同时在IC2和IC4中RR的神经元】({len(both_ic_rr)}个):")
    if both_ic_rr.size > 0:
        print(f"  {both_ic_rr.tolist()}")
    else:
        print("  (无)")
    
    print(f"\n【所有IC-only RR神经元（去重后）】({len(ic_only_rr_exc)}个):")
    if ic_only_rr_exc.size > 0:
        print(f"  {ic_only_rr_exc.tolist()}")
    else:
        print("  (无)")
    
    print("\n" + "="*70)
    print("程序执行完成！")
    print("="*70)


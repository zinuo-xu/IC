# IC
前沿创新课（1）
IC — Code for analysis
=====================

本仓库包含 RR 神经元筛选、功能连接分析、图论分析、SVM 分类及 IC / LC 编码相关的分析代码。
代码按“功能模块”组织，数据与中间结果不作为核心代码管理。

Structure
---------

├── .gitignore

├── README.md

├── M79.json
├── M79_trigger.txt
├── stimuli_M79.txt
│   实验参数与刺激配置文件

├── 一、数据预处理 + RR 筛选
│   ├── find_peak_window.py
│   │   寻找 RR 响应的峰值时间窗
│   ├── four_class.py
│   │   RR 神经元四分类（IC / LC × on / off）
│   ├── parameter_scan.py
│   │   RR 筛选参数扫描与稳定性测试
│   └── plot_rr_peak_frames.py
│       RR 峰值帧可视化

├── 二、RR 功能连接 + 图论分析
│   ├── analyze_corr_distance.py
│   │   功能连接与物理距离关系分析
│   ├── compute_rr_functional_connectivity.py
│   │   RR 神经元功能连接矩阵计算
│   ├── compute_rr_functional_connectivity_README.md
│   │   功能连接计算说明
│   ├── compute_rr_graph_metrics.py
│   │   图论指标计算（degree / modularity 等）
│   ├── compute_rr_physical_distance.py
│   │   RR 神经元物理距离计算
│   ├── compare_rr_modularity_on_off.py
│   │   RR on/off 模块度对比
│   └── compare_rr_modularity_on_off_thresholds.py
│       不同阈值下模块度对比分析

├── 三、SVM 分类
│   ├── analyze_rr_pca_neuron_loadings.py
│   │   RR-PCA 神经元载荷分析
│   ├── svm_classification.py
│   │   基本 SVM 分类（好像也是2分类，4分类的找不到了）
│   ├── svm_classification_pairwise.py
│   │   成对条件 SVM 分类
│   └── svm_multiclass_rr_pca.py
│       基于 RR-PCA 的多分类 SVM

├── 四、IC / LC Encoder
│   ├── filter_ic_only_rr.py
│   │   IC-only RR 神经元筛选
│   └── filter_lc_only_rr.py
│       LC-only RR 神经元筛选

├── 五、对标准地图（CCF 对齐）
│   ├── CCFalign_part1_20250211.m
│   └── CCFalign_part2_20250211.m
│      

└── （数据与中间结果）
    *.csv / *.xlsx / *.mat / *.tif
    不作为核心代码管理，默认由 .gitignore 排除

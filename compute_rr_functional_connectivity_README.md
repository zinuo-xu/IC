## compute_rr_functional_connectivity.py 使用说明

### 数据来源与预处理缓存
1. 在 `four_class.py` 中完成原始钙信号的预处理、SEGMENT 切片以及 RR 神经元筛选；生成的关键文件：
   - `preprocessed_data_cache.npz`：包含 `segments`（trial × neuron × frame）和 `labels`（trial 对应刺激类别）。
   - `rr_neuron_indices.csv`：列出所有兴奋/抑制 RR 神经元索引。脚本只会读取 `category == "exc"` 的 58 个兴奋性 RR 神经元。
2. `compute_rr_functional_connectivity.py` 会自动读取 `M79.json` 里的 `DATA_PATH` 并在该目录下寻找上述两个文件。若未找到会直接报错。

### 处理流程概览
1. **加载配置与数据**  
   - 解析 `M79.json`，获取 `DATA_PATH`、刺激窗口帧 (`t_stimulus`、`l_stimulus`) 等信息。  
   - 从缓存中拿到全部 trial 段 (`segments`) 和对应刺激标签 (`labels`)。
2. **筛选兴奋性 RR 神经元**  
   - 从 `rr_neuron_indices.csv` 读取 58 个兴奋性 RR 神经元列索引，只在这些 ROI 上计算功能连接。
3. **可选预处理**  
   - **GSR** (`--gsr`)：对每个 trial、逐帧减去所有选中神经元的平均值，去除全局成分。  
   - **Butterworth 滤波**（`--highpass/--lowpass --sampling-rate [--filter-order]`）：在时间轴上对每个 trial 做高通/低通/带通处理。若高通和低通都指定则自动成为带通。  
   - 预处理是在整段 trial 上进行，然后再截取刺激窗口，避免边缘效应。
4. **截取刺激窗口**  
   - 根据 `t_stimulus` 和 `l_stimulus` 取出每个 trial 的刺激帧（默认 12–19 帧）。
5. **按刺激条件聚合**  
   - 对四类刺激（IC2/IC4/LC2/LC4）分别取全部 trial，形状为 `(trial数, 58, 刺激帧数)`。
6. **构建功能连接矩阵**  
   - 将 trial × 时间拼接成单一长时间序列 (`flattened`)；  
   - 计算 58×58 的 Pearson 相关系数矩阵，并把对角线置 1。
7. **保存结果与可视化**  
   - 输出 `functional_connectivity_<类别>_<suffix>.csv` 和同名 `.png` 热图。  
   - `<suffix>` 由所选参数决定，例如 `raw`（无预处理）、`gsr_hp0p05_lp1p0` 等。

### 可调参数
| 参数 | 说明 | 默认 |
| --- | --- | --- |
| `--gsr` | 启用全局信号回归 | 关闭 |
| `--highpass <Hz>` | 高通截止频率；需配合 `--sampling-rate` | 无 |
| `--lowpass <Hz>` | 低通截止频率；需配合 `--sampling-rate` | 无 |
| `--sampling-rate <Hz>` | 数据采样率；使用滤波时必须指定；不滤波可不填 | 无 |
| `--filter-order <int>` | Butterworth 阶数，影响滚降陡峭程度 | 2 |

示例命令：
```bash
# 只做 GSR
python compute_rr_functional_connectivity.py --gsr

# GSR + 0.05-1 Hz 带通，采样率 4 Hz，三阶滤波
python compute_rr_functional_connectivity.py --gsr --highpass 0.05 --lowpass 1.0 --sampling-rate 4 --filter-order 3

# 仅 0.5 Hz 高通，不做 GSR
python compute_rr_functional_connectivity.py --highpass 0.5 --sampling-rate 4
```

### 输出位置
所有结果写入 `DATA_PATH/functional_connectivity/`。  
一个参数组合会生成四对文件（IC2/IC4/LC2/LC4），例如：
```
functional_connectivity_IC2_raw.csv
functional_connectivity_IC2_raw.png
...
functional_connectivity_LC4_gsr_hp0p05_lp1p0.csv
functional_connectivity_LC4_gsr_hp0p05_lp1p0.png
```

若需要进一步处理（如改变刺激窗口长度、使用不同神经元集合），可直接在 `compute_rr_functional_connectivity.py` 中修改对应逻辑或置换输入文件，然后重新运行脚本。


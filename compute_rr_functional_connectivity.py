import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal


CLASS_MAP = {
    1: "IC2",
    2: "IC4",
    3: "LC2",
    4: "LC4",
}


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_preprocessed(cache_path: Path):
    if not cache_path.exists():
        raise FileNotFoundError(
            f"未找到预处理缓存文件: {cache_path}. 请先运行 four_class.py 生成该文件。"
        )
    data = np.load(cache_path, allow_pickle=True)
    return data["segments"], data["labels"]


def load_rr_indices(index_path: Path) -> np.ndarray:
    if not index_path.exists():
        raise FileNotFoundError(
            f"未找到 RR 神经元索引文件: {index_path}. 请先完成 RR 神经元筛选流程。"
        )
    df = pd.read_csv(index_path)
    exc = df[df["category"] == "exc"]["neuron_index"].astype(int).to_numpy()
    if exc.size != 58:
        print(
            f"警告: 预期 58 个兴奋性 RR 神经元，实际找到 {exc.size} 个。"
            " 将使用当前文件中的全部兴奋性神经元。"
        )
    return exc


def compute_connectivity(class_segments: np.ndarray) -> np.ndarray:
    if class_segments.size == 0:
        raise ValueError("输入的类片段为空，无法计算功能连接矩阵。")
    flattened = class_segments.transpose(1, 0, 2).reshape(class_segments.shape[1], -1)
    corr = np.corrcoef(flattened)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def apply_gsr(trials: np.ndarray) -> np.ndarray:
    """对每个 trial 进行全局信号回归 (逐帧减去所有神经元平均)."""
    global_signal = trials.mean(axis=1, keepdims=True)
    return trials - global_signal


def apply_temporal_filter(trials: np.ndarray, cfg: Dict[str, float]) -> np.ndarray:
    """按照配置对 trial 进行 Butterworth 滤波."""
    if trials.shape[2] < 5:
        raise ValueError("时间轴长度过短，无法稳定进行滤波，请放宽窗口或跳过滤波。")

    fs = cfg["fs"]
    order = cfg["order"]
    nyquist = 0.5 * fs
    hp = cfg.get("highpass")
    lp = cfg.get("lowpass")

    if hp and lp:
        if hp >= lp:
            raise ValueError("高通截止频率必须小于低通截止频率。")
        wn = [hp / nyquist, lp / nyquist]
        btype = "bandpass"
    elif hp:
        wn = hp / nyquist
        btype = "highpass"
    elif lp:
        wn = lp / nyquist
        btype = "lowpass"
    else:
        return trials

    if np.any(np.array(wn) >= 1):
        raise ValueError("截止频率必须小于 Nyquist 频率 (fs/2)。")

    b, a = signal.butter(order, wn, btype=btype)
    padlen = min(trials.shape[2] - 1, 3 * (max(len(a), len(b)) - 1))
    filtered = signal.filtfilt(b, a, trials, axis=2, padlen=padlen)
    return filtered


def preprocess_trials(
    trials: np.ndarray,
    use_gsr: bool,
    filter_cfg: Optional[Dict[str, float]],
) -> np.ndarray:
    processed = trials.astype(float, copy=True)
    if use_gsr:
        processed = apply_gsr(processed)
    if filter_cfg is not None:
        processed = apply_temporal_filter(processed, filter_cfg)
    return processed


def build_suffix(args: argparse.Namespace) -> str:
    parts = []
    if args.gsr:
        parts.append("gsr")
    if args.highpass is not None:
        hp = str(args.highpass).replace(".", "p")
        parts.append(f"hp{hp}")
    if args.lowpass is not None:
        lp = str(args.lowpass).replace(".", "p")
        parts.append(f"lp{lp}")
    if not parts:
        return "raw"
    return "_".join(parts)


def main():
    parser = argparse.ArgumentParser(description="计算兴奋性 RR 功能连接矩阵")
    parser.add_argument("--gsr", action="store_true", help="对每个 trial 执行全局信号回归")
    parser.add_argument(
        "--highpass", type=float, default=None, help="高通截止频率 (Hz)，需同时指定采样率"
    )
    parser.add_argument(
        "--lowpass", type=float, default=None, help="低通截止频率 (Hz)，需同时指定采样率"
    )
    parser.add_argument(
        "--filter-order", type=int, default=2, help="Butterworth 滤波器阶数 (默认 2 阶)"
    )
    parser.add_argument(
        "--sampling-rate",
        type=float,
        default=None,
        help="数据采样率 (Hz)，使用滤波时必须指定",
    )
    args = parser.parse_args()

    filter_cfg = None
    if args.highpass is not None or args.lowpass is not None:
        if args.sampling_rate is None:
            raise ValueError("使用滤波时必须通过 --sampling-rate 指定采样率。")
        filter_cfg = {
            "highpass": args.highpass,
            "lowpass": args.lowpass,
            "fs": float(args.sampling_rate),
            "order": int(args.filter_order),
        }

    suffix = build_suffix(args)

    project_root = Path(__file__).resolve().parent
    config = load_config(project_root / "M79.json")
    data_path = Path(config["DATA_PATH"])
    cache_path = data_path / "preprocessed_data_cache.npz"
    rr_index_path = data_path / "rr_neuron_indices.csv"

    segments, labels = load_preprocessed(cache_path)
    labels = labels.astype(int)
    rr_exc_indices = load_rr_indices(rr_index_path)

    stim_start = int(config["EXP_INFO"]["t_stimulus"])
    stim_len = int(config["EXP_INFO"]["l_stimulus"])
    stim_slice = slice(stim_start, stim_start + stim_len)

    output_dir = data_path / "functional_connectivity"
    output_dir.mkdir(exist_ok=True)

    for class_id, class_name in CLASS_MAP.items():
        mask = labels == class_id
        class_trials = segments[mask][:, rr_exc_indices, :]
        if class_trials.size == 0:
            print(f"类别 {class_name} 没有可用试次，跳过。")
            continue
        class_trials = preprocess_trials(class_trials, args.gsr, filter_cfg)
        stim_only = class_trials[:, :, stim_slice]
        matrix = compute_connectivity(stim_only)

        row_col_labels = [str(idx) for idx in rr_exc_indices]
        df = pd.DataFrame(matrix, index=row_col_labels, columns=row_col_labels)

        csv_path = output_dir / f"functional_connectivity_{class_name}_{suffix}.csv"
        df.to_csv(csv_path, float_format="%.6f")

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            df,
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            square=True,
            cbar_kws={"label": "Pearson r"},
        )
        plt.title(f"Excitatory RR Functional Connectivity ({class_name})")
        plt.xlabel("Neuron index")
        plt.ylabel("Neuron index")
        heatmap_path = output_dir / f"functional_connectivity_{class_name}_{suffix}.png"
        plt.tight_layout()
        plt.savefig(heatmap_path, dpi=300)
        plt.close()

        print(
            f"类别 {class_name}: 使用 {mask.sum()} 个试次，输出矩阵到 {csv_path}，热图保存至 {heatmap_path}"
        )


if __name__ == "__main__":
    main()


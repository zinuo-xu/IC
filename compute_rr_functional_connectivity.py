import argparse
import json
from pathlib import Path

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
    if exc.size != 386:
        print(
            f"警告: 预期386个兴奋性 RR 神经元，实际找到 {exc.size} 个。"
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


def apply_highpass_filter(trials: np.ndarray, fs: float, cutoff: float = 0.05, order: int = 2) -> np.ndarray:
    """执行固定参数的 0.05 Hz 高通滤波。"""
    if trials.shape[2] < 5:
        raise ValueError("时间轴长度过短，无法稳定进行滤波，请放宽窗口。")
    nyquist = 0.5 * fs
    if cutoff >= nyquist:
        raise ValueError("高通截止频率必须小于 Nyquist 频率 (fs/2)。")
    wn = cutoff / nyquist
    b, a = signal.butter(order, wn, btype="highpass")
    padlen = min(trials.shape[2] - 1, 3 * (max(len(a), len(b)) - 1))
    filtered = signal.filtfilt(b, a, trials, axis=2, padlen=padlen)
    return filtered


def apply_zscore(trials: np.ndarray) -> np.ndarray:
    """按 trial×神经元对时间序列做 z-score 标准化。"""
    mean = np.mean(trials, axis=2, keepdims=True)
    std = np.std(trials, axis=2, keepdims=True)
    std = np.where(std == 0, 1.0, std)
    return (trials - mean) / std


def preprocess_trials(
    trials: np.ndarray,
    use_gsr: bool,
    sampling_rate: float,
) -> np.ndarray:
    processed = trials.astype(float, copy=True)
    if use_gsr:
        processed = apply_gsr(processed)
    processed = apply_highpass_filter(processed, fs=float(sampling_rate))
    processed = apply_zscore(processed)
    return processed


def build_suffix(args: argparse.Namespace) -> str:
    parts = ["hp0p05", "zscore"]
    if args.gsr:
        parts.insert(0, "gsr")
    return "_".join(parts)


def main():
    parser = argparse.ArgumentParser(description="计算兴奋性 RR 功能连接矩阵")
    parser.add_argument("--gsr", action="store_true", help="对每个 trial 执行全局信号回归")
    parser.add_argument(
        "--sampling-rate",
        type=float,
        default=4.0,
        help="数据采样率 (Hz)，用于 0.05 Hz 高通滤波，默认 4 Hz",
    )
    args = parser.parse_args()

    suffix = build_suffix(args)

    project_root = Path(__file__).resolve().parent
    config = load_config(project_root / "M79.json")
    data_path = Path(config["DATA_PATH"])
    cache_path = data_path / "preprocessed_data_cache.npz"
    rr_index_path = Path(r"C:\Users\xuzinuo\Desktop\new\rr_neuron_indices.csv")

    segments, labels = load_preprocessed(cache_path)
    labels = labels.astype(int)
    rr_exc_indices = load_rr_indices(rr_index_path)

    stim_start = int(config["EXP_INFO"]["t_stimulus"])
    stim_len = int(config["EXP_INFO"]["l_stimulus"])
    total_len = int(config["EXP_INFO"]["l_trials"])
    # stimulus-on: [t_stimulus, t_stimulus + l_stimulus)
    stim_on_slice = slice(stim_start, stim_start + stim_len)
    # stimulus-off: [0, 5) + [25, total_len) （假定总长为 32 帧）
    off_early_slice = slice(0, 5)
    off_late_slice = slice(25, total_len)

    output_dir = Path(r"C:\Users\xuzinuo\Desktop\new\FC")
    output_dir.mkdir(parents=True, exist_ok=True)

    for class_id, class_name in CLASS_MAP.items():
        mask = labels == class_id
        class_trials = segments[mask][:, rr_exc_indices, :]
        if class_trials.size == 0:
            print(f"类别 {class_name} 没有可用试次，跳过。")
            continue
        class_trials = preprocess_trials(class_trials, args.gsr, args.sampling_rate)

        # stimulus-on / stimulus-off 两个时间窗
        trials_on = class_trials[:, :, stim_on_slice]
        trials_off = np.concatenate(
            [
                class_trials[:, :, off_early_slice],
                class_trials[:, :, off_late_slice],
            ],
            axis=2,
        )

        matrix_on = compute_connectivity(trials_on)
        matrix_off = compute_connectivity(trials_off)

        row_col_labels = [str(idx) for idx in rr_exc_indices]

        # === 保存 stimulus-on ===
        df_on = pd.DataFrame(matrix_on, index=row_col_labels, columns=row_col_labels)
        csv_on = output_dir / f"FC_{class_name}_on.csv"
        df_on.to_csv(csv_on, float_format="%.6f")

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            df_on,
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            square=True,
            cbar_kws={"label": "Pearson r"},
        )
        plt.title(f"Excitatory RR Functional Connectivity ({class_name}, stimulus ON)")
        plt.xlabel("Neuron index")
        plt.ylabel("Neuron index")
        png_on = output_dir / f"FC_{class_name}_on.png"
        plt.tight_layout()
        plt.savefig(png_on, dpi=300)
        plt.close()

        # === 保存 stimulus-off ===
        df_off = pd.DataFrame(matrix_off, index=row_col_labels, columns=row_col_labels)
        csv_off = output_dir / f"FC_{class_name}_off.csv"
        df_off.to_csv(csv_off, float_format="%.6f")

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            df_off,
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            square=True,
            cbar_kws={"label": "Pearson r"},
        )
        plt.title(f"Excitatory RR Functional Connectivity ({class_name}, stimulus OFF)")
        plt.xlabel("Neuron index")
        plt.ylabel("Neuron index")
        png_off = output_dir / f"FC_{class_name}_off.png"
        plt.tight_layout()
        plt.savefig(png_off, dpi=300)
        plt.close()

        print(
            f"类别 {class_name}: 使用 {mask.sum()} 个试次，ON 矩阵 {csv_on}，OFF 矩阵 {csv_off}"
        )


if __name__ == "__main__":
    main()


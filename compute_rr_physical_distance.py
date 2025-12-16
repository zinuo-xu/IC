import itertools
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_rr_indices(index_path: Path) -> np.ndarray:
    df = pd.read_csv(index_path)
    exc = df[df["category"] == "exc"]["neuron_index"].astype(int).to_numpy()
    if exc.size != 58:
        print(
            f"警告: 预期 58 个兴奋性 RR 神经元，实际找到 {exc.size} 个。"
            " 将使用当前文件中的全部兴奋性神经元。"
        )
    return exc


def load_positions(mat_path: Path) -> np.ndarray:
    with h5py.File(mat_path, "r") as f:
        if "whole_center" not in f:
            raise KeyError("wholebrain_output.mat 缺少 whole_center 数据集。")
        positions = np.array(f["whole_center"])
    if positions.shape[0] < 3:
        raise ValueError(
            f"whole_center 维度为 {positions.shape}，至少需要前三个维度 (x,y,z)。"
        )
    return positions[:3, :]  # 只取 x,y,z


def main():
    project_root = Path(__file__).resolve().parent
    config = load_config(project_root / "M79.json")
    data_path = Path(config["DATA_PATH"])

    rr_indices = load_rr_indices(data_path / "rr_neuron_indices.csv")
    pos_xyz = load_positions(data_path / "wholebrain_output.mat")

    selected_pos = pos_xyz[:, rr_indices].T  # (58, 3)

    dist_matrix = np.linalg.norm(
        selected_pos[:, None, :] - selected_pos[None, :, :], axis=2
    )

    # 保存矩阵
    labels = [str(idx) for idx in rr_indices]
    dist_df = pd.DataFrame(dist_matrix, index=labels, columns=labels)
    out_dir = data_path / "functional_connectivity"
    out_dir.mkdir(exist_ok=True)
    matrix_path = out_dir / "rr_exc_physical_distance_matrix.csv"
    dist_df.to_csv(matrix_path, float_format="%.4f")

    # 保存 pair list
    records = []
    for (i_idx, i_label), (j_idx, j_label) in itertools.combinations(
        enumerate(rr_indices), 2
    ):
        records.append(
            {
                "neuron_i": int(i_label),
                "neuron_j": int(j_label),
                "distance": float(dist_matrix[i_idx, j_idx]),
            }
        )
    pair_df = pd.DataFrame(records)
    pair_df.to_csv(out_dir / "rr_exc_physical_distance_pairs.csv", index=False)

    print(f"距离矩阵保存至: {matrix_path}")
    print(
        f"pair 列表保存至: {out_dir / 'rr_exc_physical_distance_pairs.csv'} "
        f"(min={pair_df['distance'].min():.2f}, "
        f"median={pair_df['distance'].median():.2f}, "
        f"max={pair_df['distance'].max():.2f})"
    )


if __name__ == "__main__":
    main()


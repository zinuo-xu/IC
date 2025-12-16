import argparse
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

CLASS_NAMES = ["IC2", "IC4", "LC2", "LC4"]


def load_fc_matrix(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"找不到功能连接文件: {path}")
    return pd.read_csv(path, index_col=0)


def threshold_by_value(
    matrix: np.ndarray, threshold: float, use_abs: bool
) -> tuple[np.ndarray, float]:
    """
    使用固定阈值进行二值化，返回二值邻接矩阵和实际密度。

    threshold: 相关系数阈值，例如 0.2。
    """
    n = matrix.shape[0]
    if n < 2:
        return np.zeros_like(matrix, dtype=int), 0.0

    data = np.abs(matrix) if use_abs else matrix
    binary = (data >= threshold).astype(int)
    np.fill_diagonal(binary, 0)
    # 只保留对称无向图
    binary = np.triu(binary, 1)
    binary = binary + binary.T

    iu, ju = np.triu_indices(n, k=1)
    m = len(iu)  # 最大可能边数
    actual_edges = int(binary[iu, ju].sum())
    actual_density = actual_edges / m if m > 0 else 0.0
    return binary, actual_density


def build_graph(binary_adj: np.ndarray, labels: list[str]) -> nx.Graph:
    G = nx.Graph()
    for idx, label in enumerate(labels):
        G.add_node(label, index=int(label))
    rows, cols = np.where(binary_adj == 1)
    for i, j in zip(rows, cols):
        if i < j:
            G.add_edge(labels[i], labels[j])
    return G


def compute_graph_metrics(G: nx.Graph) -> dict:
    """计算模块度、全局效率、局部效率、平均聚类系数等图论指标。"""
    degrees = dict(G.degree())
    clustering = nx.clustering(G)
    avg_degree = float(np.mean(list(degrees.values()))) if degrees else 0.0
    avg_clustering = float(np.mean(list(clustering.values()))) if clustering else 0.0
    if G.number_of_nodes() > 1:
        global_eff = float(nx.global_efficiency(G))
        local_eff = float(nx.local_efficiency(G))
    else:
        global_eff = 0.0
        local_eff = 0.0

    if G.number_of_edges() == 0:
        modularity = 0.0
    else:
        import community as community_louvain  # 延迟导入

        partition = community_louvain.best_partition(G)
        modularity = float(community_louvain.modularity(partition, G))

    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "avg_degree": avg_degree,
        "avg_clustering": avg_clustering,
        "global_efficiency": global_eff,
        "local_efficiency": local_eff,
        "modularity_louvain": modularity,
    }


def main():
    parser = argparse.ArgumentParser(
        description="基于固定阈值 (0.2/0.3/0.4) 比较 stimulus ON/OFF 的 RR 图指标并估计网络密度"
    )
    parser.add_argument(
        "--fc-dir",
        default="FC",
        help="功能连接矩阵所在目录（包含 FC_IC2_on/off.csv 等）",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.2, 0.3, 0.4],
        help="阈值列表，例如: --thresholds 0.2 0.3 0.4（默认）。",
    )
    parser.add_argument(
        "--use-abs",
        action="store_true",
        help="对相关矩阵取绝对值后再阈值化。",
    )
    parser.add_argument(
        "--n-null",
        type=int,
        default=100,
        help="degree-preserving 随机重排次数 (默认 100)。",
    )
    args = parser.parse_args()

    fc_dir = Path(args.fc_dir)
    if not fc_dir.exists():
        raise FileNotFoundError(f"功能连接目录不存在: {fc_dir}")

    all_metrics: list[dict] = []
    nulltest_records: list[dict] = []

    for thr in args.thresholds:
        print(f"\n=== 处理阈值 threshold = {thr:.3f} ===")
        for class_name in CLASS_NAMES:
            phase_graphs = {}
            phase_metrics = {}
            for phase in ("off", "on"):
                fc_path = fc_dir / f"FC_{class_name}_{phase}.csv"
                df = load_fc_matrix(fc_path)

                labels = df.index.tolist()
                matrix = df.values.astype(float)
                binary, actual_density = threshold_by_value(
                    matrix, thr, args.use_abs
                )
                G = build_graph(binary, labels)

                metrics = compute_graph_metrics(G)
                metrics["class"] = class_name
                metrics["phase"] = phase
                metrics["threshold"] = thr
                metrics["actual_density"] = actual_density
                all_metrics.append(metrics)

                phase_graphs[phase] = G
                phase_metrics[phase] = metrics

                print(
                    f"{class_name} [{phase}] thr={thr:.3f}: "
                    f"density={actual_density:.3f}，节点 {metrics['nodes']}，边 {metrics['edges']}，"
                    f"avg_degree={metrics['avg_degree']:.2f}，"
                    f"avg_clustering={metrics['avg_clustering']:.2f}，"
                    f"global_eff={metrics['global_efficiency']:.3f}，"
                    f"local_eff={metrics['local_efficiency']:.3f}，"
                    f"Q={metrics['modularity_louvain']:.3f}"
                )

            # degree-preserving null: 检验 ON 的 Q 是否显著大于 OFF
            G_on = phase_graphs["on"]
            G_off = phase_graphs["off"]
            Q_on_obs = phase_metrics["on"]["modularity_louvain"]
            Q_off_obs = phase_metrics["off"]["modularity_louvain"]
            delta_obs = Q_on_obs - Q_off_obs

            n_null = int(args.n_null)
            swap_factor = 5
            null_deltas = []

            for _ in range(n_null):
                G_on_null = G_on.copy()
                G_off_null = G_off.copy()

                try:
                    nswap_on = max(1, G_on_null.number_of_edges() * swap_factor)
                    nx.double_edge_swap(
                        G_on_null, nswap=nswap_on, max_tries=nswap_on * 10
                    )
                except Exception:
                    pass

                try:
                    nswap_off = max(1, G_off_null.number_of_edges() * swap_factor)
                    nx.double_edge_swap(
                        G_off_null, nswap=nswap_off, max_tries=nswap_off * 10
                    )
                except Exception:
                    pass

                Q_on_null = compute_graph_metrics(G_on_null)["modularity_louvain"]
                Q_off_null = compute_graph_metrics(G_off_null)["modularity_louvain"]
                null_deltas.append(Q_on_null - Q_off_null)

            null_deltas = np.array(null_deltas, dtype=float)
            mean_null = float(np.mean(null_deltas))
            std_null = float(np.std(null_deltas, ddof=1)) if len(null_deltas) > 1 else 0.0
            if std_null > 0:
                z_score = float((delta_obs - mean_null) / std_null)
            else:
                z_score = float("nan")
            p_value = float(
                (1.0 + np.sum(null_deltas >= delta_obs)) / (1.0 + len(null_deltas))
            )

            nulltest_records.append(
                {
                    "class": class_name,
                    "threshold": thr,
                    "Q_on_obs": Q_on_obs,
                    "Q_off_obs": Q_off_obs,
                    "Q_delta_on_minus_off_obs": delta_obs,
                    "null_mean_delta": mean_null,
                    "null_std_delta": std_null,
                    "z_score": z_score,
                    "p_one_sided_ON>OFF": p_value,
                    "n_null": n_null,
                }
            )

    # 汇总图指标（按阈值 × 阶段）
    result_df = pd.DataFrame(all_metrics)
    summary_tables = {}
    for metric in [
        "modularity_louvain",
        "global_efficiency",
        "local_efficiency",
        "avg_clustering",
        "avg_degree",
        "edges",
        "nodes",
        "actual_density",
    ]:
        pivot = result_df.pivot_table(
            index=["class", "threshold"],
            columns="phase",
            values=metric,
        )
        pivot.columns = [f"{metric}_{c}" for c in pivot.columns]
        if metric in (
            "modularity_louvain",
            "global_efficiency",
            "local_efficiency",
            "avg_clustering",
        ):
            if f"{metric}_on" in pivot.columns and f"{metric}_off" in pivot.columns:
                pivot[f"{metric}_delta_on_minus_off"] = (
                    pivot[f"{metric}_on"] - pivot[f"{metric}_off"]
                )
        summary_tables[metric] = pivot

    from functools import reduce

    summary_df = reduce(
        lambda left, right: left.join(right, how="outer"), summary_tables.values()
    ).reset_index()

    output_dir = fc_dir / "graph_metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_metrics = output_dir / "on_off_graph_metrics_thresholds.csv"
    summary_df.to_csv(out_metrics, index=False)

    null_df = pd.DataFrame(nulltest_records)
    out_null = output_dir / "on_off_Q_nulltest_thresholds.csv"
    null_df.to_csv(out_null, index=False)

    print(f"\nON/OFF 图指标（多阈值）已写入: {out_metrics}")
    print(f"ON>OFF 模块度随机重排检验结果已写入: {out_null}")


if __name__ == "__main__":
    main()



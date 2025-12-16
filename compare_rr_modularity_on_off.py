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


def threshold_matrix(matrix: np.ndarray, threshold: float, use_abs: bool) -> np.ndarray:
    data = np.abs(matrix) if use_abs else matrix
    binary = (data >= threshold).astype(int)
    np.fill_diagonal(binary, 0)
    # 保证对称
    binary = np.triu(binary, 1)
    binary = binary + binary.T
    return binary


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
    """计算模块度、全局效率、平均聚类系数等图论指标。"""
    degrees = dict(G.degree())
    clustering = nx.clustering(G)
    avg_degree = float(np.mean(list(degrees.values()))) if degrees else 0.0
    avg_clustering = float(np.mean(list(clustering.values()))) if clustering else 0.0
    global_eff = float(nx.global_efficiency(G)) if G.number_of_nodes() > 1 else 0.0

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
        "modularity_louvain": modularity,
    }


def main():
    parser = argparse.ArgumentParser(
        description="比较 stimulus ON/OFF 阶段的 RR 功能连接模块度 (modularity_louvain)"
    )
    parser.add_argument(
        "--fc-dir",
        default="FC",
        help="功能连接矩阵所在目录（包含 FC_IC2_on/off.csv 等）",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.2,
        help="二值化阈值 (默认 0.2，与 graph_metrics 脚本保持一致)",
    )
    parser.add_argument(
        "--use-abs",
        action="store_true",
        help="对相关矩阵取绝对值后再阈值化",
    )
    args = parser.parse_args()

    fc_dir = Path(args.fc_dir)
    if not fc_dir.exists():
        raise FileNotFoundError(f"功能连接目录不存在: {fc_dir}")

    records: list[dict] = []

    for class_name in CLASS_NAMES:
        for phase in ("off", "on"):
            fc_path = fc_dir / f"FC_{class_name}_{phase}.csv"
            df = load_fc_matrix(fc_path)

            labels = df.index.tolist()
            matrix = df.values.astype(float)
            binary = threshold_matrix(matrix, args.threshold, args.use_abs)
            G = build_graph(binary, labels)

            metrics = compute_graph_metrics(G)

            records.append(
                {
                    "class": class_name,
                    "phase": phase,
                    "nodes": metrics["nodes"],
                    "edges": metrics["edges"],
                    "avg_degree": metrics["avg_degree"],
                    "avg_clustering": metrics["avg_clustering"],
                    "global_efficiency": metrics["global_efficiency"],
                    "modularity_louvain": metrics["modularity_louvain"],
                }
            )
            print(
                f"{class_name} [{phase}]: 节点 {metrics['nodes']}，边 {metrics['edges']}，"
                f"avg_degree={metrics['avg_degree']:.2f}，"
                f"avg_clustering={metrics['avg_clustering']:.2f}，"
                f"global_eff={metrics['global_efficiency']:.3f}，"
                f"modularity_louvain={metrics['modularity_louvain']:.3f}"
            )

    result_df = pd.DataFrame(records)
    # 透视成每个条件一行，列为 on/off 以及差值（对多个指标）
    summary = {}
    for metric in [
        "modularity_louvain",
        "global_efficiency",
        "avg_clustering",
        "avg_degree",
        "edges",
        "nodes",
    ]:
        pivot = result_df.pivot(index="class", columns="phase", values=metric)
        pivot.columns = [f"{metric}_{c}" for c in pivot.columns]
        if "modularity_louvain" in metric or "global_efficiency" in metric or "avg_clustering" in metric:
            # 这些指标适合看 ON-OFF 差值
            if f"{metric}_on" in pivot.columns and f"{metric}_off" in pivot.columns:
                pivot[f"{metric}_delta_on_minus_off"] = (
                    pivot[f"{metric}_on"] - pivot[f"{metric}_off"]
                )
        summary[metric] = pivot

    # 合并所有指标表
    from functools import reduce

    summary_df = reduce(
        lambda left, right: left.join(right, how="outer"), summary.values()
    ).reset_index()

    output_dir = fc_dir / "graph_metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"on_off_graph_metrics_thr{args.threshold}.csv"
    summary_df.to_csv(out_path, index=False)
    print(f"\nON/OFF 图指标对比已写入: {out_path}")


if __name__ == "__main__":
    main()



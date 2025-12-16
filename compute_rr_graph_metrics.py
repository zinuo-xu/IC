import argparse
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

try:
    import community as community_louvain
except ImportError as exc:
    raise ImportError(
        "缺少 python-louvain 包。请先运行 `pip install python-louvain`。"
    ) from exc

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


def compute_metrics(
    G: nx.Graph,
    skip_smallworld: bool = False,
    sw_nrand: int = 5,
    sw_niter: int = 50,
) -> dict:
    degrees = dict(G.degree())
    clustering = nx.clustering(G)
    avg_degree = float(np.mean(list(degrees.values()))) if degrees else 0.0
    avg_clustering = float(np.mean(list(clustering.values()))) if clustering else 0.0
    global_eff = float(nx.global_efficiency(G)) if G.number_of_nodes() > 1 else 0.0

    if G.number_of_edges() > 0:
        partition = community_louvain.best_partition(G)
        modularity = float(community_louvain.modularity(partition, G))
    else:
        partition = {node: i for i, node in enumerate(G.nodes())}
        modularity = 0.0

    if skip_smallworld:
        small_world_sigma = float("nan")
        small_world_error = ""
    else:
        try:
            small_world_sigma = float(
                nx.algorithms.smallworld.sigma(G, niter=sw_niter, nrand=sw_nrand)
            )
            small_world_error = ""
        except Exception as exc:
            small_world_sigma = float("nan")
            small_world_error = str(exc)

    node_metrics = pd.DataFrame(
        {
            "neuron_index": list(degrees.keys()),
            "degree": list(degrees.values()),
            "clustering_coeff": [clustering[n] for n in degrees.keys()],
            "community": [partition[n] for n in degrees.keys()],
        }
    )

    summary = {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "avg_degree": avg_degree,
        "avg_clustering": avg_clustering,
        "global_efficiency": global_eff,
        "modularity_louvain": modularity,
        "small_world_sigma": small_world_sigma,
        "components": nx.number_connected_components(G),
        "small_world_error": small_world_error,
    }
    return summary, node_metrics


def main():
    parser = argparse.ArgumentParser(description="RR 功能连接图论指标计算")
    parser.add_argument(
        "--fc-dir",
        default="functional_connectivity",
        help="功能连接矩阵所在目录",
    )
    parser.add_argument(
        "--suffix",
        default="gsr",
        help="文件名中的后缀，例如 raw 或 gsr_hp0p05_lp1p0",
    )
    parser.add_argument(
        "--name-template",
        default="FC_{class_name}_{suffix}.csv",
        help="功能连接文件命名模板，支持 {class_name} 与 {suffix}",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.2,
        help="二值化阈值 (默认 0.2)",
    )
    parser.add_argument(
        "--use-abs",
        action="store_true",
        help="对相关矩阵取绝对值后再阈值化",
    )
    parser.add_argument(
        "--sw-nrand",
        type=int,
        default=5,
        help="small-world sigma 计算的随机网络数量 (默认 5)",
    )
    parser.add_argument(
        "--sw-niter",
        type=int,
        default=50,
        help="small-world sigma 每个随机网络的迭代次数 (默认 50)",
    )
    parser.add_argument(
        "--skip-smallworld",
        action="store_true",
        help="跳过 small-worldness 计算以节省时间",
    )
    args = parser.parse_args()

    fc_dir = Path(args.fc_dir)
    output_dir = fc_dir / "graph_metrics"
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []

    for class_name in CLASS_NAMES:
        filename = args.name_template.format(class_name=class_name, suffix=args.suffix)
        fc_path = fc_dir / filename
        df = load_fc_matrix(fc_path)

        labels = df.index.tolist()
        matrix = df.values.astype(float)
        binary = threshold_matrix(matrix, args.threshold, args.use_abs)
        G = build_graph(binary, labels)

        summary, node_metrics = compute_metrics(
            G,
            skip_smallworld=args.skip_smallworld,
            sw_nrand=args.sw_nrand,
            sw_niter=args.sw_niter,
        )
        summary["class"] = class_name
        summaries.append(summary)

        node_metrics_path = (
            output_dir
            / f"graph_node_metrics_{class_name}_{args.suffix}_thr{args.threshold}.csv"
        )
        node_metrics.to_csv(node_metrics_path, index=False)
        print(
            f"{class_name}: 节点 {summary['nodes']}，边 {summary['edges']}，"
            f"avg_degree={summary['avg_degree']:.2f}，avg_clustering={summary['avg_clustering']:.2f}，"
            f"global_eff={summary['global_efficiency']:.3f}，modularity={summary['modularity_louvain']:.3f}，"
            f"sigma={summary['small_world_sigma']:.2f}"
        )

    summary_df = pd.DataFrame(summaries)
    summary_path = output_dir / f"graph_summary_{args.suffix}_thr{args.threshold}.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"图指标汇总已写入: {summary_path}")


if __name__ == "__main__":
    main()


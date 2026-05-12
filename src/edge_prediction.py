import os

import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse

from src.community import run_dcsbm
from src.utils import SEED, get_top_edges


def get_edge_prediction(G, net_name, results_dir, top_k=10, rwr_restart=0.15, rwr_iter=40):
    """
    Rank missing edges using local similarity, block structure and random walks.
    """
    G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    nodes = list(G.nodes())
    non_edges = list(nx.non_edges(G))
    neighbors = {node: set(G.neighbors(node)) for node in nodes}
    clustering = nx.clustering(G)
    rows = []

    # 1) Local similarity heuristics: high scores mean two proteins share similar neighborhoods.
    common_neighbors = ((u, v, len(neighbors[u] & neighbors[v])) for u, v in non_edges)
    clustering_closure = ((u, v, len(neighbors[u] & neighbors[v]) * (clustering[u] + clustering[v]) / 2) for u, v in non_edges)
    methods = {'common_neighbors': get_top_edges(common_neighbors, top_k),
               'clustering_closure': get_top_edges(clustering_closure, top_k),
               'jaccard': get_top_edges(nx.jaccard_coefficient(G, non_edges), top_k),
               'adamic_adar': get_top_edges(nx.adamic_adar_index(G, non_edges), top_k),
               'resource_allocation': get_top_edges(nx.resource_allocation_index(G, non_edges), top_k),
               'preferential_attachment': get_top_edges(nx.preferential_attachment(G, non_edges), top_k)}

    # 2) DCSBM-style score: pairs are favored if their blocks are densely connected and both nodes have high degree.
    dcsbm_labels, _ = run_dcsbm(G, seed=SEED)
    block_by_node = dict(zip(nodes, dcsbm_labels))
    degrees = dict(G.degree())
    avg_degree = np.mean(list(degrees.values()))
    block_sizes = pd.Series(dcsbm_labels).value_counts().to_dict()
    block_edges = {}

    for u, v in G.edges():
        block_1, block_2 = sorted((block_by_node[u], block_by_node[v]))
        block_edges[(block_1, block_2)] = block_edges.get((block_1, block_2), 0) + 1

    block_prob = {}
    blocks = sorted(block_sizes)
    for i, block_1 in enumerate(blocks):
        for block_2 in blocks[i:]:
            possible = block_sizes[block_1] * (block_sizes[block_1] - 1) / 2 if block_1 == block_2 else block_sizes[block_1] * block_sizes[block_2]
            block_prob[(block_1, block_2)] = (block_edges.get((block_1, block_2), 0) + 1) / (possible + 2) if possible > 0 else 0

    dcsbm_scores = []
    for u, v in non_edges:
        block_1, block_2 = sorted((block_by_node[u], block_by_node[v]))
        degree_factor = ((degrees[u] + 1) * (degrees[v] + 1)) / ((avg_degree + 1) ** 2)
        dcsbm_scores.append((u, v, block_prob.get((block_1, block_2), 0) * degree_factor))
    methods['DCSBM'] = get_top_edges(dcsbm_scores, top_k)

    # 3) Random Walk with Restart: candidates are close if a walker can reach them often without using an existing edge.
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    A = nx.to_scipy_sparse_array(G, nodelist=nodes, dtype=float, format='csr')
    row_sum = np.asarray(A.sum(axis=1)).ravel()
    P = sparse.diags(1 / np.maximum(row_sum, 1)) @ A
    I = np.eye(len(nodes))
    R = I.copy()

    for _ in range(rwr_iter):
        R = rwr_restart * I + (1 - rwr_restart) * (R @ P)

    rwr_scores = (R + R.T) / 2
    np.fill_diagonal(rwr_scores, -np.inf)
    edge_i = [node_to_idx[u] for u, v in G.edges()]
    edge_j = [node_to_idx[v] for u, v in G.edges()]
    rwr_scores[edge_i, edge_j] = -np.inf
    rwr_scores[edge_j, edge_i] = -np.inf
    upper_i, upper_j = np.triu_indices(len(nodes), 1)
    values = rwr_scores[upper_i, upper_j]
    k = min(top_k, np.isfinite(values).sum())
    if k > 0:
        top_idx = np.argpartition(-values, k - 1)[:k]
        top_idx = top_idx[np.argsort(-values[top_idx])]
        methods['RWR'] = [(nodes[upper_i[i]], nodes[upper_j[i]], values[i]) for i in top_idx]
    else:
        methods['RWR'] = []

    for method, top_edges in methods.items():
        for rank, (node_1, node_2, score) in enumerate(top_edges, start=1):
            rows.append({'method': method, 'rank': rank, 'node_1': node_1, 'node_2': node_2, 'score': score})

    predictions = pd.DataFrame(rows)

    # Consensus highlights candidate PPIs that appear in the top-k of several methods.
    if predictions.empty:
        consensus = pd.DataFrame(columns=['node_1', 'node_2', 'n_methods', 'methods', 'best_rank'])
    else:
        consensus = predictions.groupby(['node_1', 'node_2']).agg(n_methods=('method', 'nunique'), methods=('method', lambda x: ', '.join(x)), best_rank=('rank', 'min')).reset_index()
        consensus = consensus.sort_values(['n_methods', 'best_rank', 'node_1', 'node_2'], ascending=[False, True, True, True])

    predictions.to_csv(os.path.join(results_dir, f'{net_name}_edge_predictions.csv'), index=False)
    consensus.to_csv(os.path.join(results_dir, f'{net_name}_edge_prediction_consensus.csv'), index=False)
    return predictions, consensus

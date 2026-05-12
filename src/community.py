import os
import random

import graph_tool.all as gt
import igraph as ig
import leidenalg
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score

import community.community_louvain as community_louvain
from src.utils import SEED, compute_stability, labels_to_communities, plot_communities, plot_community_metrics, summarize_partition

def run_infomap(G, seed=SEED):
    nodes = list(G.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    edges = [(node_to_idx[u], node_to_idx[v]) for u, v in G.edges()]

    # igraph works with integer node ids
    G_ig = ig.Graph(n=len(nodes), edges=edges, directed=False)
    ig.set_random_number_generator(random.Random(seed))
    partition = G_ig.community_infomap(trials=10)
    return partition.membership


def run_dcsbm(G, seed=SEED):
    nodes = list(G.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    edges = [(node_to_idx[u], node_to_idx[v]) for u, v in G.edges()]

    # graph-tool also needs integer ids
    G_gt = gt.Graph(directed=False)
    G_gt.add_vertex(len(nodes))
    G_gt.add_edge_list(edges)

    gt.seed_rng(seed)
    state = gt.minimize_blockmodel_dl(G_gt, state_args={'deg_corr': True})
    blocks = state.get_blocks()
    labels = [int(blocks[v]) for v in G_gt.vertices()]
    return labels, state.entropy()


def get_community_detection(G, net_name, results_dir, n_runs=5):
    nodes = list(G.nodes())
    metrics_rows = []
    assignments = pd.DataFrame({'node': nodes})

    # 1) Louvain
    louvain_runs = []
    for i in range(n_runs):
        # Get the partition that maximizes the modularity
        partition = community_louvain.best_partition(G, random_state=SEED + i)
        louvain_runs.append([partition[node] for node in nodes])

    louvain_modularities = [nx.community.modularity(G, labels_to_communities(nodes, labels)) for labels in louvain_runs]
    louvain_labels = louvain_runs[int(np.argmax(louvain_modularities))]
    louvain_summary = summarize_partition(G, nodes, louvain_labels, 'louvain')
    louvain_summary['stability_nmi'] = compute_stability(louvain_runs)
    metrics_rows.append(louvain_summary)
    assignments['louvain'] = louvain_labels

    # 2) Greedy modularity
    greedy_communities = nx.community.greedy_modularity_communities(G)
    greedy_labels_dict = {}
    for i, community in enumerate(greedy_communities):
        for node in community:
            greedy_labels_dict[node] = i

    greedy_labels = [greedy_labels_dict[node] for node in nodes]
    greedy_summary = summarize_partition(G, nodes, greedy_labels, 'greedy')
    greedy_summary['stability_nmi'] = np.nan
    metrics_rows.append(greedy_summary)
    assignments['greedy'] = greedy_labels

    # 3) Leiden
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    edges = [(node_to_idx[u], node_to_idx[v]) for u, v in G.edges()]
    G_ig = ig.Graph(n=len(nodes), edges=edges, directed=False)

    leiden_runs = []
    for i in range(n_runs):
        partition = leidenalg.find_partition(G_ig, leidenalg.ModularityVertexPartition, seed=SEED + i)
        leiden_runs.append(partition.membership)

    leiden_modularities = [nx.community.modularity(G, labels_to_communities(nodes, labels)) for labels in leiden_runs]
    leiden_labels = leiden_runs[int(np.argmax(leiden_modularities))]
    leiden_summary = summarize_partition(G, nodes, leiden_labels, 'leiden')
    leiden_summary['stability_nmi'] = compute_stability(leiden_runs)
    metrics_rows.append(leiden_summary)
    assignments['leiden'] = leiden_labels

    # 4) Infomap
    infomap_runs = [run_infomap(G, seed=SEED + i) for i in range(n_runs)]
    infomap_modularities = [nx.community.modularity(G, labels_to_communities(nodes, labels)) for labels in infomap_runs]
    infomap_labels = infomap_runs[int(np.argmax(infomap_modularities))]
    infomap_summary = summarize_partition(G, nodes, infomap_labels, 'infomap')
    infomap_summary['stability_nmi'] = compute_stability(infomap_runs)
    metrics_rows.append(infomap_summary)
    assignments['infomap'] = infomap_labels

    # 5) Bayesian approach: degree-corrected stochastic block model
    dcsbm_labels, description_length = run_dcsbm(G, seed=SEED)
    dcsbm_summary = summarize_partition(G, nodes, dcsbm_labels, 'DCSBM', description_length) 
    dcsbm_summary['stability_nmi'] = np.nan
    metrics_rows.append(dcsbm_summary)
    assignments['DCSBM'] = dcsbm_labels 

    metrics = pd.DataFrame(metrics_rows)
    metrics = metrics[['algorithm', 'n_communities', 'modularity', 'mean_community_size', 'largest_community_size', 'stability_nmi', 'description_length']]

    # Pairwise comparison between community partitions 
    comparison_rows = []
    algorithms = ['louvain', 'greedy', 'leiden', 'infomap', 'DCSBM'] 
    for i in range(len(algorithms)):
        for j in range(i + 1, len(algorithms)):
            algo_1, algo_2 = algorithms[i], algorithms[j]
            labels_1, labels_2 = assignments[algo_1].to_numpy(), assignments[algo_2].to_numpy()
            contingency = pd.crosstab(labels_1, labels_2).to_numpy()
            row_counts, col_counts = contingency.sum(axis=1), contingency.sum(axis=0)
            same_1, same_2 = np.sum(row_counts * (row_counts - 1) / 2), np.sum(col_counts * (col_counts - 1) / 2)
            same_both = np.sum(contingency * (contingency - 1) / 2)
            union = same_1 + same_2 - same_both
            p = contingency / len(labels_1)
            row_p, col_p = p.sum(axis=1), p.sum(axis=0)
            h1 = -np.sum(row_p[row_p > 0] * np.log(row_p[row_p > 0]))
            h2 = -np.sum(col_p[col_p > 0] * np.log(col_p[col_p > 0]))
            expected = np.outer(row_p, col_p)
            mask = p > 0
            mi = np.sum(p[mask] * np.log(p[mask] / expected[mask]))
            vi = h1 + h2 - 2 * mi
            comparison_rows.append({'algorithm_1': algo_1, 'algorithm_2': algo_2, 
                                    'jaccard': same_both / union if union > 0 else 1.0, 
                                    'nmi': normalized_mutual_info_score(labels_1, labels_2), 
                                    'nvi': min(1, max(0, vi / np.log(len(labels_1)))) if len(labels_1) > 1 else 0})
    comparison = pd.DataFrame(comparison_rows)

    metrics.to_csv(os.path.join(results_dir, f'{net_name}_community_metrics.csv'), index=False)
    assignments.to_csv(os.path.join(results_dir, f'{net_name}_community_assignments.csv'), index=False)
    comparison.to_csv(os.path.join(results_dir, f'{net_name}_community_comparison.csv'), index=False) 
    plot_community_metrics(metrics, net_name, os.path.join(results_dir, f'{net_name}_community_metrics.png'))
    plot_communities(G, assignments, net_name, os.path.join(results_dir, f'{net_name}_communities.png')) 

    return metrics, assignments

import json
import os
import subprocess
import sys
import tempfile
import graph_tool.all as gt
import igraph as ig
import leidenalg
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score

import community.community_louvain as community_louvain
from src.utils import SEED


def labels_to_communities(labels, nodes):
    communities = {}
    for node, label in zip(nodes, labels):
        communities.setdefault(label, set()).add(node)
    return list(communities.values())


def get_louvain_labels(G, seed=SEED):
    partition = community_louvain.best_partition(G, random_state=seed)
    return [partition[node] for node in G.nodes()]


def get_greedy_labels(G, seed=SEED):
    communities = nx.community.greedy_modularity_communities(G)
    labels = {}
    for i, community in enumerate(communities):
        for node in community:
            labels[node] = i
    return [labels[node] for node in G.nodes()]


def get_leiden_labels(G, seed=SEED):
    nodes = list(G.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    edges = [(node_to_idx[u], node_to_idx[v]) for u, v in G.edges()]
    G_ig = ig.Graph(n=len(nodes), edges=edges, directed=False)
    partition = leidenalg.find_partition(G_ig, leidenalg.ModularityVertexPartition, seed=seed)
    return partition.membership


def get_infomap_labels(G, seed=SEED):
    nodes = list(G.nodes())
    edges = list(G.edges())
    script = """
import json
import sys
import networkx as nx
import infomap as im

input_file, output_file, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
with open(input_file, 'r') as f:
    data = json.load(f)

G = nx.Graph()
G.add_nodes_from(data['nodes'])
G.add_edges_from(data['edges'])

im_runner = im.Infomap(f'--seed {seed} --silent')
mapping = im_runner.add_networkx_graph(G)
im_runner.run()
modules = im_runner.get_modules()
if data['nodes'][0] in mapping:
    node_to_id = mapping
else:
    node_to_id = {node: node_id for node_id, node in mapping.items()}
labels = [int(modules[node_to_id[node]]) for node in data['nodes']]

with open(output_file, 'w') as f:
    json.dump(labels, f)
"""
    with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as input_file:
        json.dump({'nodes': nodes, 'edges': edges}, input_file)
        input_path = input_file.name

    with tempfile.NamedTemporaryFile('r', suffix='.json', delete=False) as output_file:
        output_path = output_file.name

    subprocess.run([sys.executable, '-c', script, input_path, output_path, str(seed)], check=True)
    with open(output_path, 'r') as f:
        labels = json.load(f)

    os.remove(input_path)
    os.remove(output_path)
    return labels


def get_dcsbm_labels(G, seed=SEED):
    nodes = list(G.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    G_gt = gt.Graph(directed=False)
    G_gt.add_vertex(len(nodes))
    G_gt.add_edge_list([(node_to_idx[u], node_to_idx[v]) for u, v in G.edges()])

    gt.seed_rng(seed)
    state = gt.minimize_blockmodel_dl(G_gt, state_args={'deg_corr': True})
    blocks = state.get_blocks()
    labels = [int(blocks[v]) for v in G_gt.vertices()]
    return labels, state.entropy()


def plot_community_metrics(metrics, net_name, plot_file):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    colors = ['lightblue', 'lightskyblue', 'cornflowerblue', 'royalblue', 'darkblue']

    axes[0].bar(metrics['algorithm'], metrics['n_communities'], color=colors[:len(metrics)])
    axes[0].set_title('NUMBER OF COMMUNITIES', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('n communities')

    axes[1].bar(metrics['algorithm'], metrics['modularity'], color=colors[:len(metrics)])
    axes[1].set_title('MODULARITY', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('modularity')

    fig.suptitle(f'COMMUNITY DETECTION: {net_name.upper()}', fontsize=16, fontweight='bold')
    fig.tight_layout()
    fig.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close(fig)


def get_community_detection(G, net_name, results_dir, n_runs=5):
    nodes = list(G.nodes())
    rows = []
    assignments = pd.DataFrame({'node': nodes})

    algorithms = {'louvain': get_louvain_labels,
                  'greedy': get_greedy_labels,
                  'leiden': get_leiden_labels,
                  'infomap': get_infomap_labels}

    for algo, function in algorithms.items():
        all_labels = [function(G, seed=SEED + i) for i in range(n_runs)]
        modularities = [nx.community.modularity(G, labels_to_communities(labels, nodes)) for labels in all_labels]
        best_idx = int(np.argmax(modularities))
        labels = all_labels[best_idx]
        communities = labels_to_communities(labels, nodes)
        sizes = [len(c) for c in communities]

        nmi_values = []
        for i in range(n_runs):
            for j in range(i + 1, n_runs):
                nmi_values.append(normalized_mutual_info_score(all_labels[i], all_labels[j]))

        rows.append({
            'algorithm': algo,
            'n_communities': len(communities),
            'modularity': modularities[best_idx],
            'mean_community_size': float(np.mean(sizes)),
            'largest_community_size': max(sizes),
            'stability_nmi': float(np.mean(nmi_values)),
            'description_length': np.nan
        })
        assignments[algo] = labels

    dcsbm_labels, description_length = get_dcsbm_labels(G, seed=SEED)
    dcsbm_communities = labels_to_communities(dcsbm_labels, nodes)
    dcsbm_sizes = [len(c) for c in dcsbm_communities]
    rows.append({
        'algorithm': 'dcsbm',
        'n_communities': len(dcsbm_communities),
        'modularity': nx.community.modularity(G, dcsbm_communities),
        'mean_community_size': float(np.mean(dcsbm_sizes)),
        'largest_community_size': max(dcsbm_sizes),
        'stability_nmi': np.nan,
        'description_length': description_length
    })
    assignments['dcsbm'] = dcsbm_labels

    metrics = pd.DataFrame(rows)
    metrics.to_csv(os.path.join(results_dir, f'{net_name}_community_metrics.csv'), index=False)
    assignments.to_csv(os.path.join(results_dir, f'{net_name}_community_assignments.csv'), index=False)
    plot_community_metrics(metrics, net_name, os.path.join(results_dir, f'{net_name}_community_metrics.png'))

    return metrics, assignments

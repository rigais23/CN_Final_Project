import heapq
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
plt.ioff()
import networkx as nx
import numpy as np
import pandas as pd
import requests
from sklearn.metrics import normalized_mutual_info_score

SEED = 1

##############################
# READ DATA AND GRAPH
##############################
def read_data(path):
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()

    single_proteins = []
    edges = []
    for line in lines:
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) == 1: # individual proteins
            single_proteins.append(parts[0])

        elif "pp" in parts: # protein-protein interaction
            i = parts.index("pp")
            p1 = parts[i - 1]
            p2 = parts[i + 1]
            edges.append((p1, p2))

    df_nodes = pd.DataFrame({"protein": single_proteins}).drop_duplicates()
    df_edges = pd.DataFrame(edges, columns=["protein1", "protein2"]).drop_duplicates()
    return df_nodes, df_edges


def load_network(path):
    _, df = read_data(path)
    G = nx.from_pandas_edgelist(df, source="protein1", target="protein2")
    return G


##############################
# MICROSCOPIC DESCRIPTORS
##############################
def get_top_k_nodes(values, k=5):
    return sorted(values.items(), key=lambda item: (-item[1], item[0]))[:k]


##############################
# FAILURES AND ATTACKS
##############################
def lcc_relative_size(G, n_original):
    if G.number_of_nodes() == 0:
        return 0
    return len(max(nx.connected_components(G), key=len)) / n_original


def plot_failure_attack(curves, thresholds, net_name, plot_file):
    net_label = net_name.capitalize()
    colors = {'random_failure': 'lightblue', 'degree_attack': 'cornflowerblue', 'betweenness_attack': 'darkblue', 'disease_genes_attack': '#0077B6'}
    labels = {'random_failure': 'Random failure', 'degree_attack': 'Degree attack', 'betweenness_attack': 'Betweenness attack', 'disease_genes_attack': 'Disease genes attack'}
    strategy_order = ['random_failure', 'degree_attack', 'betweenness_attack', 'disease_genes_attack']
    available_strategies = set(curves['strategy'])

    fig, ax = plt.subplots(figsize=(8, 6))
    for strategy in strategy_order:
        if strategy not in available_strategies:
            continue
        data = curves[curves['strategy'] == strategy].sort_values('fraction_removed')
        x = data['fraction_removed'].to_numpy()
        y = data['relative_lcc_size'].to_numpy()
        yerr = data['relative_lcc_std'].fillna(0).to_numpy()

        ax.plot(x, y, color=colors[strategy], label=labels[strategy], linewidth=2)
        if strategy == 'random_failure':
            ax.fill_between(x, y - yerr, y + yerr, color=colors[strategy], alpha=0.2)

        threshold_values = thresholds.loc[thresholds['strategy'] == strategy, 'fraction_removed_threshold']
        threshold = threshold_values.iloc[0] if not threshold_values.empty else np.nan
        if not pd.isna(threshold):
            ax.axvline(threshold, color=colors[strategy], linestyle='--', linewidth=1.5, alpha=0.8, label=f'{labels[strategy]} threshold ({threshold:.2f})')

    ax.set_title(f'FAILURES AND ATTACKS: {net_label}', fontsize=14, fontweight='bold')
    ax.set_xlabel('fraction of nodes removed', fontsize=12)
    ax.set_ylabel('relative LCC size', fontsize=12)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    plt.close('all')


##############################
# COMMUNITIES
##############################
def labels_to_communities(nodes, labels):
    communities = {}
    for node, label in zip(nodes, labels):
        communities.setdefault(label, set()).add(node)
    return list(communities.values())


def summarize_partition(G, nodes, labels, algorithm, description_length=np.nan):
    communities = labels_to_communities(nodes, labels)
    sizes = [len(c) for c in communities]

    return {'algorithm': algorithm,
            'n_communities': len(communities),
            'modularity': nx.community.modularity(G, communities),
            'mean_community_size': float(np.mean(sizes)),
            'largest_community_size': max(sizes),
            'description_length': description_length}


def compute_stability(label_runs):
    if len(label_runs) < 2:
        return np.nan

    nmi_values = []
    for i in range(len(label_runs)):
        for j in range(i + 1, len(label_runs)):
            nmi_values.append(normalized_mutual_info_score(label_runs[i], label_runs[j]))
    return float(np.mean(nmi_values))


def plot_community_metrics(metrics, net_name, plot_file):
    net_label = net_name.capitalize()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    colors = ['lightblue', 'lightskyblue', 'cornflowerblue', 'royalblue', 'darkblue']

    axes[0].bar(metrics['algorithm'], metrics['n_communities'], color=colors[:len(metrics)])
    axes[0].set_title('Number of communities', fontsize=14)
    axes[0].set_ylabel('n communities')

    axes[1].bar(metrics['algorithm'], metrics['modularity'], color=colors[:len(metrics)])
    axes[1].set_title('Modularity', fontsize=14)
    axes[1].set_ylabel('modularity')

    fig.suptitle(f'COMMUNITY DETECTION: {net_label}', fontsize=16, fontweight='bold')
    fig.tight_layout()
    fig.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    plt.close('all')


def plot_communities(G, assignments, net_name, plot_file):
    net_label = net_name.capitalize()
    algorithms = [column for column in assignments.columns if column != 'node']
    nodes = assignments['node'].tolist()
    G_plot = G.subgraph(nodes).copy()
    pos = nx.spring_layout(G_plot, seed=SEED, iterations=40)

    fig = plt.figure(figsize=(13, 8))
    grid = fig.add_gridspec(2, 6)
    axes = [fig.add_subplot(grid[0, 0:2]),
            fig.add_subplot(grid[0, 2:4]),
            fig.add_subplot(grid[0, 4:6]),
            fig.add_subplot(grid[1, 1:3]),
            fig.add_subplot(grid[1, 3:5])]
    cmap = plt.cm.Blues

    for i, algorithm in enumerate(algorithms):
        ax = axes[i]
        labels = assignments[algorithm].to_numpy()
        unique_labels = sorted(pd.unique(labels))
        label_colors = {label: cmap(0.25 + 0.7 * j / max(len(unique_labels) - 1, 1)) for j, label in enumerate(unique_labels)}
        label_by_node = dict(zip(nodes, labels))
        node_colors = [label_colors[label_by_node[node]] for node in G_plot.nodes()]

        nx.draw_networkx_edges(G_plot, pos, ax=ax, edge_color='lightsteelblue', width=0.15, alpha=0.12)
        nx.draw_networkx_nodes(G_plot, pos, ax=ax, node_color=node_colors, node_size=7, linewidths=0, alpha=0.95)
        ax.set_title('DCSBM' if algorithm == 'DCSBM' else algorithm.capitalize(), fontsize=12)
        ax.axis('off')

    fig.suptitle(f'COMMUNITIES: {net_label}', fontsize=16, fontweight='bold')
    fig.tight_layout()
    fig.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    plt.close('all')


##############################
# EPIDEMIC SPREADING
##############################
def seeds_random(G, n_seeds, rng):
    nodes = list(G.nodes())
    return list(rng.choice(nodes, size=min(n_seeds, len(nodes)), replace=False))


def seeds_top_hubs(G, n_seeds):
    ranked = sorted(G.degree(), key=lambda x: -x[1])
    return [node for node, _ in ranked[:n_seeds]]


def seeds_disease_genes(G, disease_genes, n_seeds):
    present = [g for g in disease_genes if g in G]
    seeds = sorted(present, key=lambda n: -G.degree(n))[:n_seeds]

    if len(seeds) < n_seeds:
        hubs = seeds_top_hubs(G, n_seeds * 2)
        for hub in hubs:
            if hub not in seeds:
                seeds.append(hub)
                if len(seeds) == n_seeds:
                    break

    return seeds


def load_disgenet(data_dir, gene_col='Gene'):
    files = {
        'ovarian': os.path.join(data_dir, 'disgenet_ovarian.tsv'),
        'breast':  os.path.join(data_dir, 'disgenet_breast.tsv'),
        'lung':    os.path.join(data_dir, 'disgenet_lung.tsv'),
    }

    dfs = []
    for disease, path in files.items():
        df = pd.read_csv(path, sep='\t')
        df['disease'] = disease
        dfs.append(df)

    disgenet_df = pd.concat(dfs, ignore_index=True)
    disgenet_df.to_csv(os.path.join(data_dir, 'disgenet.tsv'), sep='\t', index=False)
    return disgenet_df


def translate_genes_to_string_ids(gene_list):
    """
    Uses the STRING database API to translate Gene Symbols to STRING Ensembl IDs.
    """
    url = "https://string-db.org/api/json/get_string_ids"
    params = {"identifiers": "\r".join(gene_list), "species": 9606, "limit": 1, "echo_query": 1}
    response = requests.post(url, data=params)

    translated_ids = []
    if response.status_code == 200:
        data = response.json()
        for entry in data:
            translated_ids.append(entry["stringId"])

    return translated_ids


##############################
# EDGE PREDICTION
##############################
def get_top_edges(predictions, top_k):
    if top_k <= 0:
        return []

    # Keep only the strongest candidate missing edges.
    top_edges = heapq.nlargest(top_k, predictions, key=lambda item: item[2])
    return sorted(top_edges, key=lambda item: (-item[2], str(item[0]), str(item[1])))

import json
import os

import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from src.utils import SEED
plt.ioff()


def get_backbone_analysis(G, net_name, results_dir):
    """
    Compute the topological backbone as the maximum k-core of the LCC.
    """
    G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    original_nodes = G.number_of_nodes()
    original_edges = G.number_of_edges()
    lcc_nodes = max(nx.connected_components(G), key=len)
    G = G.subgraph(lcc_nodes).copy()

    core_numbers = nx.core_number(G)
    max_k = max(core_numbers.values())
    backbone_nodes = [node for node, k in core_numbers.items() if k == max_k]
    backbone = G.subgraph(backbone_nodes).copy()

    summary = {'network': net_name,
               'method': 'k_core',
               'max_core_k': int(max_k),
               'original_nodes': int(original_nodes),
               'original_edges': int(original_edges),
               'lcc_nodes': int(G.number_of_nodes()),
               'lcc_edges': int(G.number_of_edges()),
               'backbone_nodes': int(backbone.number_of_nodes()),
               'backbone_edges': int(backbone.number_of_edges()),
               'node_percentage': backbone.number_of_nodes() / G.number_of_nodes() * 100,
               'edge_percentage': backbone.number_of_edges() / G.number_of_edges() * 100}

    core_df = pd.DataFrame([{'node': node, 'core_number': k, 'in_backbone': k == max_k} for node, k in core_numbers.items()])
    backbone_nodes_df = core_df[core_df['in_backbone']].copy()
    backbone_edges_df = pd.DataFrame(backbone.edges(), columns=['node_1', 'node_2'])
    top_nodes_df = pd.DataFrame(sorted(backbone.degree(), key=lambda item: (-item[1], item[0]))[:20], columns=['node', 'degree_in_backbone'])

    with open(os.path.join(results_dir, f'{net_name}_backbone_summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)

    core_df.to_csv(os.path.join(results_dir, f'{net_name}_core_numbers.csv'), index=False)
    backbone_nodes_df.to_csv(os.path.join(results_dir, f'{net_name}_backbone_nodes.csv'), index=False)
    backbone_edges_df.to_csv(os.path.join(results_dir, f'{net_name}_backbone_edges.csv'), index=False)
    top_nodes_df.to_csv(os.path.join(results_dir, f'{net_name}_backbone_top_nodes.csv'), index=False)

    return summary


def get_consensus_backbone(results_dir, net_names):
    """
    Save the common nodes and edges among the three topological backbones.
    """
    node_sets = []
    edge_sets = []

    for net_name in net_names:
        nodes = set(pd.read_csv(os.path.join(results_dir, f'{net_name}_backbone_nodes.csv'))['node'])
        edges_df = pd.read_csv(os.path.join(results_dir, f'{net_name}_backbone_edges.csv'))
        edges = {tuple(sorted((row['node_1'], row['node_2']))) for _, row in edges_df.iterrows()}

        node_sets.append(nodes)
        edge_sets.append(edges)

    common_nodes = set.intersection(*node_sets)
    common_edges = set.intersection(*edge_sets)

    summary = pd.DataFrame([{'method': 'consensus_k_core', 'consensus_nodes': len(common_nodes), 'consensus_edges': len(common_edges)}])
    nodes_df = pd.DataFrame(sorted(common_nodes), columns=['node'])
    edges_df = pd.DataFrame(sorted(common_edges), columns=['node_1', 'node_2'])

    summary.to_csv(os.path.join(results_dir, 'consensus_backbone_summary.csv'), index=False)
    nodes_df.to_csv(os.path.join(results_dir, 'consensus_backbone_nodes.csv'), index=False)
    edges_df.to_csv(os.path.join(results_dir, 'consensus_backbone_edges.csv'), index=False)

    return summary, nodes_df, edges_df


def plot_backbone_networks(networks, results_dir):
    """
    Plot the backbones obtained from the LCCs with the consensus backbone highlighted.
    """
    colors = {'breast': '#9ECAE1', 'lung': '#4292C6', 'ovarian': '#08519C', 'breast_lung': '#C6DBEF', 'consensus': '#001D3D'}
    node_membership = {}
    edge_membership = {}

    for net_name in ['breast', 'lung', 'ovarian']:
        nodes = set(pd.read_csv(os.path.join(results_dir, f'{net_name}_backbone_nodes.csv'))['node'])
        edges_df = pd.read_csv(os.path.join(results_dir, f'{net_name}_backbone_edges.csv'))
        for node in nodes:
            node_membership.setdefault(node, set()).add(net_name)
        for edge in edges_df[['node_1', 'node_2']].itertuples(index=False, name=None):
            edge_membership.setdefault(tuple(sorted(edge)), set()).add(net_name)

    consensus_nodes = set(pd.read_csv(os.path.join(results_dir, 'consensus_backbone_nodes.csv'))['node'])
    consensus_edges_df = pd.read_csv(os.path.join(results_dir, 'consensus_backbone_edges.csv'))
    consensus_edges = [tuple(sorted(edge)) for edge in consensus_edges_df[['node_1', 'node_2']].itertuples(index=False, name=None)]
    consensus = nx.Graph()
    consensus.add_nodes_from(consensus_nodes)
    consensus.add_edges_from(consensus_edges)

    backbone_union = nx.Graph()
    backbone_union.add_nodes_from(node_membership.keys())
    backbone_union.add_edges_from(edge_membership.keys())
    pos = nx.spring_layout(backbone_union, seed=SEED, iterations=90, k=0.12)

    categories = {'breast_lung': [], 'breast': [], 'lung': [], 'ovarian': []}
    for node, membership in node_membership.items():
        if node in consensus_nodes:
            continue
        if membership == {'breast', 'lung'}:
            categories['breast_lung'].append(node)
        elif membership == {'breast'}:
            categories['breast'].append(node)
        elif membership == {'lung'}:
            categories['lung'].append(node)
        elif membership == {'ovarian'}:
            categories['ovarian'].append(node)

    edge_groups = {'breast_lung': [], 'breast': [], 'lung': [], 'ovarian': [], 'consensus': []}
    consensus_edges = set(consensus_edges)
    for edge, membership in edge_membership.items():
        if edge[0] not in pos or edge[1] not in pos:
            continue
        if set(edge).issubset(consensus_nodes) and edge in consensus_edges:
            edge_groups['consensus'].append(edge)
        elif membership == {'breast', 'lung'}:
            edge_groups['breast_lung'].append(edge)
        elif membership == {'breast'}:
            edge_groups['breast'].append(edge)
        elif membership == {'lung'}:
            edge_groups['lung'].append(edge)
        elif membership == {'ovarian'}:
            edge_groups['ovarian'].append(edge)

    def sample_edges(edges, max_edges):
        edges = sorted(edges)
        if len(edges) <= max_edges:
            return edges
        idx = np.linspace(0, len(edges) - 1, max_edges, dtype=int)
        return [edges[i] for i in idx]

    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    edge_limits = {'breast': 350, 'lung': 350, 'ovarian': 160, 'breast_lung': 850, 'consensus': 900}
    nx.draw_networkx_edges(nx.Graph(sample_edges(edge_groups['breast_lung'], edge_limits['breast_lung'])), pos, ax=ax, edge_color=colors['breast_lung'], width=0.20, alpha=0.18)
    for group in ['breast', 'lung', 'ovarian']:
        nx.draw_networkx_edges(nx.Graph(sample_edges(edge_groups[group], edge_limits[group])), pos, ax=ax, edge_color=colors[group], width=0.24, alpha=0.34)
    nx.draw_networkx_edges(nx.Graph(sample_edges(edge_groups['consensus'], edge_limits['consensus'])), pos, ax=ax, edge_color=colors['consensus'], width=0.20, alpha=0.22)

    nx.draw_networkx_nodes(backbone_union, pos, nodelist=categories['breast_lung'], ax=ax, node_color=colors['breast_lung'], node_size=10, linewidths=0, alpha=0.42)
    for group in ['breast', 'lung', 'ovarian']:
        nx.draw_networkx_nodes(backbone_union, pos, nodelist=categories[group], ax=ax, node_color=colors[group], node_size=13, linewidths=0, alpha=0.86)
    nx.draw_networkx_nodes(backbone_union, pos, nodelist=sorted(consensus_nodes), ax=ax, node_color=colors['consensus'], node_size=12, linewidths=0, alpha=0.95)

    legend_elements = [Line2D([0], [0], marker='o', color='none', label='Breast', markerfacecolor=colors['breast'], markersize=7),
                       Line2D([0], [0], marker='o', color='none', label='Lung', markerfacecolor=colors['lung'], markersize=7),
                       Line2D([0], [0], marker='o', color='none', label='Ovarian', markerfacecolor=colors['ovarian'], markersize=7),
                       Line2D([0], [0], marker='o', color='none', label='Consensus', markerfacecolor=colors['consensus'], markersize=7)]
    ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.03), ncol=4, frameon=False, fontsize=10, handletextpad=0.4, columnspacing=1.2, borderaxespad=0)
    ax.set_title('CONSENSUS BACKBONE', fontsize=17, fontweight='bold', pad=4)
    ax.margins(0.08)
    ax.axis('off')
    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.11)
    fig.savefig(os.path.join(results_dir, 'backbone_networks.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    plt.close('all')

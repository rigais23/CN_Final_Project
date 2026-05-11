import os
import json
import networkx as nx

from src.utils import get_top_k_nodes

def get_microscopic_descriptors(net, centrality_file, k_nodes = 5):
    # If previously computed, return the corresponding files
    if os.path.exists(centrality_file):
        with open(centrality_file, 'r') as f:
            results_value = json.load(f)
        with open(centrality_file.split('.')[0]+'_node.json', 'r') as f:
            results_node = json.load(f)
        return results_value, results_node
    
    # Get node degrees
    degree_values = dict(net.degree())
    pagerank_values = nx.pagerank(net)
    betweenness_values = nx.centrality.betweenness_centrality(net)
    closeness_values = nx.closeness_centrality(net)
    eigenvector_values = nx.eigenvector_centrality(net, max_iter=2000)
    katz_values = nx.katz_centrality_numpy(net)

    top_degree = get_top_k_nodes(degree_values, k=k_nodes)
    top_pagerank = get_top_k_nodes(pagerank_values, k=k_nodes)
    top_betweenness = get_top_k_nodes(betweenness_values, k=k_nodes)
    top_closeness = get_top_k_nodes(closeness_values, k=k_nodes)
    top_eigenvector = get_top_k_nodes(eigenvector_values, k=k_nodes)
    top_katz = get_top_k_nodes(katz_values, k=k_nodes)

    results_node = {
        'n_degree': top_degree,
        'n_pagerank': top_pagerank,
        'n_betweenness': top_betweenness,
        'n_closeness': top_closeness,
        'n_eigenvector': top_eigenvector,
        'n_katz': top_katz
    }

    results_value = {
        'degree': degree_values,
        'pagerank': pagerank_values,
        'betweenness': betweenness_values,
        'closeness': closeness_values,
        'eigenvector': eigenvector_values,
        'katz': katz_values,
    }

    with open(centrality_file, 'w') as f:
        json.dump(results_value, f, indent=4)

    with open(centrality_file.split('.')[0]+'_node.json', 'w') as f:
        json.dump(results_node, f, indent=4)

    return results_value, results_node

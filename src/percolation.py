import json
import os
import networkx as nx
import numpy as np
import pandas as pd

from src.utils import SEED, plot_failure_attack, lcc_relative_size

def removal_curve(G, removal_order, fractions):
    """
    Compute how the lcc changes with node removals
    """
    n_original = G.number_of_nodes()
    G_temp = G.copy()
    previous_k = 0
    rows = []

    # Iterate through diffrent fractions of nodes to be eliminated
    for fraction in fractions:
        k = int(round(fraction * n_original))
        G_temp.remove_nodes_from(removal_order[previous_k:k])
        previous_k = k
        rows.append({'fraction_removed': k / n_original, 'relative_lcc_size': lcc_relative_size(G_temp, n_original)})

    return pd.DataFrame(rows)


def get_failure_attack_analysis(G, net_name, results_dir, micro_file=None, n_rep=20, lcc_threshold=0.5):
    """
    Analyze the network robustness against attacks and random failures
    """
    # Retrieve the nodes belonging to the largest connected component subgraph
    lcc_nodes = max(nx.connected_components(G), key=len)
    G = G.subgraph(lcc_nodes).copy()
    nodes = list(G.nodes())
    # Generate the different fractions of nodes to be removed
    fractions = np.linspace(0, 1, 101)

    # 1) RANDOM FAILURES
    random_curves = []
    for i in range(n_rep):
        rng = np.random.default_rng(SEED + i)
        random_order = list(rng.permutation(nodes))
        random_curves.append(removal_curve(G, random_order, fractions))

    # Average and standard deviation for each curve
    random_curve = pd.concat(random_curves)
    random_curve = random_curve.groupby('fraction_removed')['relative_lcc_size'].agg(['mean', 'std']).reset_index()
    random_curve = random_curve.rename(columns={'mean': 'relative_lcc_size', 'std': 'relative_lcc_std'})
    random_curve['relative_lcc_std'] = random_curve['relative_lcc_std'].fillna(0)
    random_curve['strategy'] = 'random_failure'

    # 2) DEGREE ATTACK
    degree_order = [node for node, _ in sorted(G.degree(), key=lambda item: (-item[1], item[0]))]
    degree_curve = removal_curve(G, degree_order, fractions)
    degree_curve['relative_lcc_std'] = 0
    degree_curve['strategy'] = 'degree_attack'

    # 3) BETWEENNESS ATTACK
    if micro_file is not None and os.path.exists(micro_file):
        with open(micro_file, 'r') as f:
            betweenness = json.load(f).get('betweenness')
    else:
        betweenness = nx.betweenness_centrality(G)
    if betweenness is None:
        betweenness = nx.betweenness_centrality(G)

    betweenness = {node: value for node, value in betweenness.items() if node in G}
    betweenness_order = [node for node, _ in sorted(betweenness.items(), key=lambda item: (-item[1], item[0]))]
    betweenness_curve = removal_curve(G, betweenness_order, fractions)
    betweenness_curve['relative_lcc_std'] = 0
    betweenness_curve['strategy'] = 'betweenness_attack'

    # Concatenate results from all strategies
    curves = pd.concat([random_curve, degree_curve, betweenness_curve], ignore_index=True)

    # Compute thresholds
    thresholds = []
    for strategy in ['random_failure', 'degree_attack', 'betweenness_attack']:
        strategy_curve = curves[curves['strategy'] == strategy]
        below = strategy_curve[strategy_curve['relative_lcc_size'] <= lcc_threshold]
        threshold_fraction = np.nan if below.empty else below.iloc[0]['fraction_removed']
        thresholds.append({'strategy': strategy, 'lcc_threshold': lcc_threshold, 'fraction_removed_threshold': threshold_fraction})
    thresholds = pd.DataFrame(thresholds)

    # Save files
    curves_file = os.path.join(results_dir, f'{net_name}_failure_attack_curves.csv')
    thresholds_file = os.path.join(results_dir, f'{net_name}_failure_attack_thresholds.csv')
    plot_file = os.path.join(results_dir, f'{net_name}_failure_attack.png')
    curves.to_csv(curves_file, index=False)
    thresholds.to_csv(thresholds_file, index=False)
    plot_failure_attack(curves, net_name, plot_file)

    return curves, thresholds

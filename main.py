import os

os.environ['MPLBACKEND'] = 'Agg'

import matplotlib
matplotlib.use('Agg', force=True)
matplotlib.interactive(False)

import pandas as pd

from src.utils import load_network, translate_genes_to_string_ids
from src.macroscopic import get_macroscopic_descriptors, get_degree_distribution
from src.microscopic import get_microscopic_descriptors
from src.backbone import get_backbone_analysis, get_consensus_backbone, plot_backbone_networks
from src.percolation import get_failure_attack_analysis
from src.community import get_community_detection
from src.epidemic_spreading import get_sir_analysis, plot_cross_network_comparison
from src.edge_prediction import get_edge_prediction


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
STRUCTURAL_DIR = os.path.join(RESULTS_DIR, '1_structural_characterization')
PERCOLATION_DIR = os.path.join(RESULTS_DIR, '2_percolation')
COMMUNITY_DIR = os.path.join(RESULTS_DIR, '3_community_detection')
EPIDEMIC_DIR = os.path.join(RESULTS_DIR, '4_epidemic_spreading')
EDGE_PREDICTION_DIR = os.path.join(RESULTS_DIR, '5_edge_prediction')

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(STRUCTURAL_DIR, exist_ok=True)
os.makedirs(PERCOLATION_DIR, exist_ok=True)
os.makedirs(COMMUNITY_DIR, exist_ok=True)
os.makedirs(EPIDEMIC_DIR, exist_ok=True)
os.makedirs(EDGE_PREDICTION_DIR, exist_ok=True)


def main(net_name, disease_genes=None, mu=0.1):
    G = load_network(os.path.join(DATA_DIR, net_name))
    net = net_name.split('.')[0].split(' ')[0]  # 'ovarian', 'breast', 'lung'
    disease_genes = [] if disease_genes is None else disease_genes
    valid_disease_genes = [node for node in disease_genes if node in G.nodes]
    n_seeds_to_use = max(1, len(valid_disease_genes))
    
    # 1. STRUCTURAL CHARACTERIZATION
    out_path_macro = os.path.join(STRUCTURAL_DIR, net + '_macro.json')
    get_macroscopic_descriptors(G, out_path_macro)
    get_degree_distribution(G, net, plot_file=os.path.join(STRUCTURAL_DIR, net + '_degree_distribution.png'))
    out_path_micro = os.path.join(STRUCTURAL_DIR, net + '_micro.json')
    get_microscopic_descriptors(G, out_path_micro)
    get_backbone_analysis(G, net, STRUCTURAL_DIR)

    # 2. PERCOLATION: ATTACKS AND FAILURES
    get_failure_attack_analysis(G, net, PERCOLATION_DIR, out_path_micro, disease_genes=valid_disease_genes)

    # 3. COMMUNITY DETECTION
    get_community_detection(G, net, COMMUNITY_DIR)
    
    # 4. EPIDEMIC SPREADING: SIR DYSFUNCTION PROPAGATION
    curves, summary, sweep_df = get_sir_analysis(G, net, EPIDEMIC_DIR, disease_genes=valid_disease_genes, n_seeds=n_seeds_to_use, mu=mu)

    # 5. EDGE PREDICTION
    get_edge_prediction(G, net, EDGE_PREDICTION_DIR, top_k=10)

    return G, curves, summary, sweep_df




if __name__ == '__main__':
    disgenet_file = os.path.join(DATA_DIR, 'disgenet.tsv')
    disgenet = pd.read_csv(disgenet_file, sep='\t') if os.path.exists(disgenet_file) else None

    sweep_dfs_by_mu = {}
    networks = {}
    mu_list = [0.1, 0.3]

    for mu in mu_list:
        sweep_dfs_by_mu[mu] = {}
        for net_name in ['ovarian cancer.sif', 'breast cancer.sif', 'lung cancer.sif']:
            net = net_name.split('.')[0].split(' ')[0]
            raw_disease_genes = [] if disgenet is None else disgenet[disgenet['disease'] == net]['Gene'].tolist()
            translated_disease_genes = translate_genes_to_string_ids(raw_disease_genes) if raw_disease_genes else []
            G, curves, summary, sweep_df = main(net_name=net_name, disease_genes=translated_disease_genes, mu=mu)
            networks[net] = G
            sweep_dfs_by_mu[mu][net] = sweep_df

    get_consensus_backbone(STRUCTURAL_DIR, ['breast', 'lung', 'ovarian'])
    plot_backbone_networks(networks, STRUCTURAL_DIR)
    plot_cross_network_comparison(sweep_dfs_by_mu, results_dir=EPIDEMIC_DIR)

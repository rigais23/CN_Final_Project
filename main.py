import os
import pandas as pd

from src.utils import load_network
from src.macroscopic import get_macroscopic_descriptors, get_degree_distribution
from src.microscopic import get_microscopic_descriptors
from src.percolation import get_failure_attack_analysis
from src.community import get_community_detection
from src.dysfunction_propagation import get_sir_analysis

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

def main(net_name, disease_genes):
    G = load_network(os.path.join(DATA_DIR, net_name))
    net = net_name.split('.')[0].split(' ')[0]  # 'ovarian', 'breast', 'lung'

    # 1. STRUCTURAL CHARACTERIZATION
    out_path_macro = os.path.join(RESULTS_DIR, net + '_macro.json')
    get_macroscopic_descriptors(G, out_path_macro)
    get_degree_distribution(G, net, plot_file=os.path.join(RESULTS_DIR, net + '_degree_distribution.png'))
    out_path_micro = os.path.join(RESULTS_DIR, net + '_micro.json')
    get_microscopic_descriptors(G, out_path_micro)

    # 2. PERCOLATION: ATTACKS AND FAILURES
    get_failure_attack_analysis(G, net, RESULTS_DIR, out_path_micro)

    # 3. COMMUNITY DETECTION
    get_community_detection(G, net, RESULTS_DIR)

    # 4. SIR DYSFUNCTION PROPAGATION
    get_sir_analysis(G, net, RESULTS_DIR, disease_genes=disease_genes)


if __name__ == '__main__':
    disgenet = pd.read_csv(os.path.join(DATA_DIR, 'disgenet.tsv'), sep='\t')

    for net_name in ['ovarian cancer.sif', 'breast cancer.sif', 'lung cancer.sif']:
        net = net_name.split('.')[0].split(' ')[0]
        disease_genes = disgenet[disgenet['disease'] == net]['gene_symbol'].tolist()
        main(net_name=net_name, disease_genes=disease_genes)
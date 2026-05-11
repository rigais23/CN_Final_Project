import os

from src.utils import load_network
from src.macroscopic import get_macroscopic_descriptors, get_degree_distribution
from src.microscopic import get_microscopic_descriptors
from src.percolation import get_failure_attack_analysis
from src.community import get_community_detection

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

def main(net_name):
    G = load_network(os.path.join(DATA_DIR, net_name))
    
    # 1. STRUCTURAL CHARACTERIZATION
    out_file_macro = net_name.split('.')[0].split(' ')[0] + '_' + 'macro' + '.json'
    out_path_macro = os.path.join(RESULTS_DIR, out_file_macro)
    get_macroscopic_descriptors(G, out_path_macro)
    plot_file_macro = os.path.join(RESULTS_DIR, net_name.split('.')[0].split(' ')[0] + '_degree_distribution.png')
    get_degree_distribution(G, net_name.split('.')[0].split(' ')[0], plot_file=plot_file_macro)

    out_file_micro = net_name.split('.')[0].split(' ')[0] + '_' + 'micro' + '.json'
    out_path_micro = os.path.join(RESULTS_DIR, out_file_micro)
    get_microscopic_descriptors(G, out_path_micro)

    # 2. PERCOLATION: ATTACKS AND FAILURES
    get_failure_attack_analysis(G, net_name.split('.')[0].split(' ')[0], RESULTS_DIR, out_path_micro)

    # 3. COMMUNITY DETECTION
    get_community_detection(G, net_name.split('.')[0].split(' ')[0], RESULTS_DIR)


if __name__ == '__main__':
    net_name = 'ovarian cancer.sif'
    main(net_name=net_name)
    

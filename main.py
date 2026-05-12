import os
import pandas as pd

from src.utils import load_network, translate_genes_to_string_ids
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
    valid_disease_genes = [node for node in disease_genes if node in G.nodes]
    # Dynamically set n_seeds to the number of disease genes we found
    n_seeds_to_use = len(valid_disease_genes)

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
    # print(f" -> Running SIS with n_seeds = {n_seeds_to_use}")
    get_sir_analysis(G, net, RESULTS_DIR, disease_genes=valid_disease_genes, n_seeds=n_seeds_to_use)
    
    # CODI PER MIRAR EL OVERLAP DE DISEASE GENES AMB HUBS, I HO POSO AQUÍ PER NO FER MASSA CANVIS AL CODI, PERÒ ES POT TREURE DESPRÉS 
    '''
    degrees = dict(G.degree())
    hub_seeds = sorted(degrees, key=degrees.get, reverse=True)[:50] 

    print(f"========== DEBUGGING: {net.upper()} CANCER ==========")
    print("--- Format Check ---")
    print("Network Nodes (first 5):", list(G.nodes)[:5])
    print("Hub Seeds (first 5):", hub_seeds[:5])
    print("Disease Seeds (first 5):", disease_genes[:5])

    intersection = set(hub_seeds).intersection(set(disease_genes))
    print(f"\n--- Overlap Check ---")
    print(f"Total Hub Seeds: {len(hub_seeds)}")
    print(f"Total Disease Seeds: {len(disease_genes)}")
    print(f"Exact Overlap: {len(intersection)}")
    
    valid_disease_seeds = [node for node in disease_genes if node in G.nodes]
    print(f"Disease seeds actually found in the network: {len(valid_disease_seeds)}\n")
    '''
if __name__ == '__main__':
    '''
    disgenet = pd.read_csv(os.path.join(DATA_DIR, 'disgenet.tsv'), sep='\t')

    for net_name in ['ovarian cancer.sif', 'breast cancer.sif', 'lung cancer.sif']:
        net = net_name.split('.')[0].split(' ')[0]
        disease_genes = disgenet[disgenet['disease'] == net]['Gene'].tolist()
        main(net_name=net_name, disease_genes=disease_genes)
    '''
    # 1. Load the DisGeNET associations
    disgenet = pd.read_csv(os.path.join(DATA_DIR, 'disgenet.tsv'), sep='\t')
    
    for net_name in ['ovarian cancer.sif', 'breast cancer.sif', 'lung cancer.sif']:
        net = net_name.split('.')[0].split(' ')[0]
        # 2. Get the raw disease genes from DisGeNET
        raw_disease_genes = disgenet[disgenet['disease'] == net]['Gene'].tolist()
        # 3. Call the API to translate them dynamically
        translated_disease_genes = translate_genes_to_string_ids(raw_disease_genes)
        # 4. Run the main analysis
        main(net_name=net_name, disease_genes=translated_disease_genes)
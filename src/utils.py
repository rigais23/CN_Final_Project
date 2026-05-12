import networkx as nx
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import os
import requests

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

def plot_failure_attack(curves, net_name, plot_file):
    colors = {'random_failure': 'lightblue', 'degree_attack': 'cornflowerblue', 'betweenness_attack': 'darkblue'}
    labels = {'random_failure': 'Random failure', 'degree_attack': 'Degree attack', 'betweenness_attack': 'Betweenness attack'}

    fig, ax = plt.subplots(figsize=(8, 6))
    for strategy in ['random_failure', 'degree_attack', 'betweenness_attack']:
        data = curves[curves['strategy'] == strategy].sort_values('fraction_removed')
        x = data['fraction_removed'].to_numpy()
        y = data['relative_lcc_size'].to_numpy()
        yerr = data['relative_lcc_std'].fillna(0).to_numpy()

        ax.plot(x, y, color=colors[strategy], label=labels[strategy], linewidth=2)
        if strategy == 'random_failure':
            ax.fill_between(x, y - yerr, y + yerr, color=colors[strategy], alpha=0.2)

    ax.set_title(f'FAILURES AND ATTACKS: {net_name}', fontsize=14, fontweight='bold')
    ax.set_xlabel('fraction of nodes removed', fontsize=12)
    ax.set_ylabel('relative LCC size', fontsize=12)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close(fig)

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
    Uses the STRING database API to translate Gene Symbols (e.g., BRCA1)
    to STRING Ensembl IDs (e.g., 9606.ENSP00000357654).
    """
    url = "https://string-db.org/api/json/get_string_ids"
    
    # The API expects identifiers separated by carriage returns
    params = {
        "identifiers": "\r".join(gene_list), 
        "species": 9606, # Homo sapiens
        "limit": 1,      # Get the best match
        "echo_query": 1  # Return the original name too
    }
    
    response = requests.post(url, data=params)
    
    translated_ids = []
    if response.status_code == 200:
        data = response.json()
        for entry in data:
            translated_ids.append(entry["stringId"])
            
    return translated_ids
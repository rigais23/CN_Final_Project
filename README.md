# Comparative Topology of Multiple Onco-PPI Networks

This project analyzes protein-protein interaction (PPI) networks associated with three cancer types: breast cancer, lung cancer, and ovarian cancer. Each network is modeled as an undirected graph, where nodes represent proteins and edges represent physical or functional interactions between them.

The objective is to compare the topological organization of the three cancer-associated networks using complex network methods, with a focus on structural organization, relevant proteins, robustness, modularity, spreading dynamics, and potential missing interactions.

## Project Structure

```text
├── data/                         # Input networks and disease-gene tables
├── docs/                         # Report and project documents
├── results/                      # Generated metrics, tables and plots
├── src/                          # Analysis modules
│   ├── macroscopic.py            # Global descriptors and degree distributions
│   ├── microscopic.py            # Centrality measures
│   ├── backbone.py               # k-core and consensus backbone analysis
│   ├── percolation.py            # Random failures and targeted attacks
│   ├── community.py              # Community detection algorithms
│   ├── epidemic_spreading.py     # SIR dysfunction propagation model
│   ├── edge_prediction.py        # Candidate PPI prediction
│   └── utils.py                  # Shared helpers and plotting functions
├── community_enrichment.ipynb    # Community enrichment analysis
├── main.py                       # Main execution script
└── README.md
```

## Data

The networks were obtained from STRING/Cytoscape and exported as `.sif` files:

- `data/breast cancer.sif`
- `data/lung cancer.sif`
- `data/ovarian cancer.sif`

Disease-associated genes from DisGeNET are also included in:

- `data/disgenet_breast.tsv`
- `data/disgenet_lung.tsv`
- `data/disgenet_ovarian.tsv`
- `data/disgenet.tsv`

## Analyses

The project includes the following analyses:

1. **Structural characterization**
   - Macroscopic descriptors: number of nodes, edges, degree, clustering coefficient, assortativity, average path length and diameter.
   - Degree distribution: histogram, PDF, CCDF and logarithmic binning.
   - Microscopic descriptors: degree, PageRank, betweenness, closeness, eigenvector and Katz centrality.

2. **Backbone analysis**
   - The backbone of each network is computed using the maximum k-core of the largest connected component.
   - A consensus backbone is also computed as the set of nodes and edges shared by the three cancer-specific backbones.

3. **Percolation**
   - Random failures.
   - Degree-based attacks.
   - Betweenness-based attacks.
   - Disease-gene attacks.
   - The percolation threshold is computed as the fraction of removed nodes at which the relative size of the LCC falls below 50%.

4. **Community detection**
   - Louvain.
   - Leiden.
   - Greedy modularity.
   - Infomap.
   - Degree-Corrected Stochastic Block Model (DCSBM).

5. **Epidemic spreading**
   - SIR-based dysfunction propagation model.
   - Comparison of random, hub-based and disease-gene seed strategies.
   - Sweep over transmission rates to study propagation behavior.

6. **Edge prediction**
   - Local similarity heuristics.
   - DCSBM-based scoring.
   - Random Walk with Restart.

## Running the Project

From the repository root:

```bash
python main.py
```

The script reads the networks from `data/`, runs the analysis pipeline, and saves the generated results in `results/`.

The random seed is defined globally in `src/utils.py` as:

```python
SEED = 1
```

This is used to make stochastic analyses and layouts reproducible.

## Results

The main outputs are stored in:

```text
results/
├── 1_structural_characterization/
├── 2_percolation/
├── 3_community_detection/
├── 4_epidemic_spreading/
└── 5_edge_prediction/
```

Each folder contains the corresponding `.csv`, `.json` and `.png` files used in the final report.

Important generated outputs include:

- Macroscopic descriptors for each network.
- Degree distribution plots.
- Microscopic centrality rankings.
- Backbone and consensus backbone files.
- Percolation curves and thresholds.
- Community metrics, assignments and comparison tables.
- Epidemic spreading curves and diagrams.
- Edge prediction candidate tables.

## Notes

- Plots are saved automatically and the Matplotlib backend is set to `Agg`, so the scripts should not open interactive plot windows.
- The `.sif` networks are unweighted, so the final backbone analysis uses k-core decomposition rather than weight-based backbone extraction methods.
- Predicted edges should be interpreted as computational candidates, not experimentally validated PPIs.

## Authors

Ricard Garcia, Nuria Cardona and Laia Barcenilla.

# Comparative Topology of Multiple Onco-PPI Networks

## Phase 0 — Setup & Data Preparation

- [x] Load the three `.sif` networks into Python with NetworkX
- [ ] Clean data: remove isolates, keep only the largest connected component (LCC) for analysis (document how many nodes/edges are dropped)
- [ ] Map STRING protein IDs to gene names (use `9606.protein.info.v12.0.txt` from STRING)
- [ ] Download disease-gene metadata:
  - **DisGeNET** TSV (gene → disease associations with evidence scores)
  - **Gene Ontology** annotations for *Homo sapiens* (`goa_human.gaf` from EBI)
  - **hetionet** gene→disease edge list
- [ ] Store networks and metadata in a consistent format (e.g. node attribute tables as CSV)

**Deliverable:** Three clean NetworkX graph objects with gene-name node attributes, plus metadata tables ready for enrichment queries.

---

## Phase 1 — Structural Characterisation (+- Practical 1)

> *Research question: What are the structural properties of each cancer-specific network?*

- [x] Degree distribution — plot on log-log scale, fit power-law vs. log-normal, comment carefully (avoid claiming "scale-free" without justification; prefer "heavy-tailed")
- [ ] Strength distribution (if you add edge weights from `combined_score`)
- [x] Clustering coefficient (global and local distribution)
- [x] Average shortest path length and diameter (on LCC)
- [x] Assortativity (degree correlation)
- [ ] Identify hubs: top-k nodes by degree, betweenness centrality, and PageRank

**Deliverable:** Summary table (extend Table 1 from the proposal) + distribution plots for each network.

---

## Phase 2 — Cross-Cancer Protein Overlap

> *Research question: Is there a high overlap in the proteins involved in each type of cancer?*

- [ ] Compute pairwise Jaccard similarity of node sets (breast ∩ lung, breast ∩ ovarian, lung ∩ ovarian)
- [ ] Identify proteins shared across all three cancers. Are they hubs? What is their biological role?
- [ ] Identify cancer-specific proteins (unique to one network)
- [ ] Visualise overlaps with a Venn diagram or UpSet plot

**Deliverable:** Overlap analysis with biological interpretation of shared vs. cancer-specific proteins.

---

## Phase 3 — Community Detection & Biological Validation (+- Practical 2)

> *Research question: Which network exhibits a more defined community structure, and do communities correspond to functional/disease modules?*

### 3.1 Community Detection
- [ ] Run at least two algorithms and compare:
  - **Louvain** (fast, resolution parameter)
  - **Leiden** (more stable, recommended over Louvain)
  - **Infomap** or **SBM** (via graph-tool)
  - Buscar si hi ha alguna opció millor per PPI.
- [ ] Compute modularity Q for each method and each network
- [ ] Assess stability: run each algorithm multiple times, check NMI between runs

### 3.2 Biological Validation *(el que ha comentat el profe al mail)*
- [ ] For each detected community, run **GO enrichment** using g:Profiler (`gprofiler2` Python package):
  - Biological Process (BP)
  - Molecular Function (MF)
  - Cellular Component (CC)
- [ ] For each community, compute overlap with **DisGeNET** disease gene sets — flag communities enriched in cancer genes
- [ ] Summarise: for each cancer network, which communities are functionally coherent? Which are disease-relevant?
- [ ] Compare community structure across the three cancers: are there conserved modules? Cancer-specific ones?

**Deliverable:** Community summary table (community ID, size, top GO terms, top associated disease, modularity contribution) for each network.

---

## Phase 4 — Robustness: Random Failures vs. Targeted Attacks

> *Research question: How do the networks respond to random failures and targeted attacks?*

- [ ] Define robustness metric: relative size of the LCC after node removal
- [ ] Simulate three removal strategies:
  1. **Random failures** — remove nodes uniformly at random
  2. **Targeted attack by degree** — remove highest-degree nodes first
  3. **Targeted attack by betweenness** — remove highest-betweenness nodes first
- [ ] Plot LCC size vs. fraction of nodes removed for all three strategies, for all three cancer networks
- [ ] Identify the percolation threshold for each strategy/network
- [ ] Discuss: which cancer network is most/least robust? Are disease-associated hub proteins especially critical?

**Deliverable:** Robustness curves + percolation threshold table + biological interpretation (e.g. candidate drug targets = high-betweenness disease proteins).

---

## Phase 5 — Dysfunction Propagation (SIS Model)

> *Research question: Which proteins are most critical for the propagation of a biological alteration?*

- [ ] Implement a discrete-time **SIS model** on each network:
  - S = functional protein, I = altered/dysfunctional protein
  - Infected node spreads to each susceptible neighbour with probability λ per timestep
  - Infected node recovers with probability μ per timestep
- [ ] Explore the (λ, μ) parameter space around the epidemic threshold
- [ ] Compare propagation dynamics under three seeding strategies:
  1. Random seed proteins
  2. Seed = top hubs (high degree)
  3. Seed = disease-associated proteins (from DisGeNET)
- [ ] Metrics to track: fraction of infected nodes at steady state, time to peak, epidemic threshold
- [ ] Discuss: do disease proteins act as super-spreaders? Does the community structure slow or accelerate propagation?

**Deliverable:** SIS simulation plots (infected fraction vs. time) for each seeding strategy + comparison across networks.

---

## Phase 6 — Link Prediction *(no se si esta be)*

> *Research question: Which protein interactions are likely but undiscovered?*

- [ ] Implement and compare local similarity indices:
  - Common Neighbors
  - Adamic-Adar
  - Resource Allocation
- [ ] Evaluate with cross-validation (hide 10% of edges, predict, compute AUC)
- [ ] Discuss whether high-scoring predicted links are biologically plausible (check STRING textmining score as a proxy)

**Deliverable:** AUC comparison table for each index and each network.

---

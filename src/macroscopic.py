import os
import json
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import poisson
from collections import Counter 

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results') 

##############################
# NUMERICAL DESCRIPTORS
##############################
def get_macroscopic_descriptors(net, macroscopic_file='macroscopic.json'): 
    if os.path.exists(macroscopic_file):
        with open(macroscopic_file, 'r') as f:
            data = json.load(f)
        return data

    # 1) Nodes and edges
    n_nodes = net.number_of_nodes() # Number of nodes
    n_edges = net.number_of_edges() # Number of edges

    # 2) Degree
    degrees=dict(net.degree()) # Create a dictionary with the degree of each node
    min_degree = min(degrees.values()) # Minimum
    max_degree = max(degrees.values()) # Maximum
    avg_degree = sum(degrees.values()) / n_nodes # Average
    
    # 3) Average clustering coefficient (average of the clustering coefficient of each node)
    if net.is_multigraph():
        ccoeffs = nx.clustering(nx.Graph(net))
    else:
        ccoeffs = nx.clustering(net)
    
    min_ccoeff = min(ccoeffs.values()) # Minimum
    max_ccoeff = max(ccoeffs.values()) # Maximum
    avg_ccoeff = sum(ccoeffs.values()) / n_nodes # Average

    # 4) Assortativity
    assortativity = nx.degree_assortativity_coefficient(net)

    # 5) Average path length (average distance between all pairs of nodes)
    # Retrieve all the connected components in the network
    lcc = max(nx.connected_components(net), key=len)
    # Get the average shortest path length of the LCC
    lcc_subgraph = net.subgraph(lcc)
    avg_path_length = nx.average_shortest_path_length(lcc_subgraph)

    # 6) Diameter (maximum distance between nodes in the network)
    diameter = nx.diameter(lcc_subgraph)
    
    results = {"n_nodes": n_nodes,
               "n_edges": n_edges,
               "min_degree": min_degree,
               "max_degree": max_degree,
               "avg_degree": avg_degree,
               "min_ccoeff": min_ccoeff,
               "max_ccoeff": max_ccoeff,
               "avg_ccoeff": avg_ccoeff,
               "assortativity": assortativity,
               "lcc_avg_path_length": avg_path_length,
               "lcc_diameter": diameter}
    
    with open(macroscopic_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    return results


##############################
# VISUALIZATION
##############################
def PDF_log_binning(num_bins, min_degree, max_degree, ls_degrees):
    '''
    Computes the PDF with logaritmic binning
    '''
    ls_degrees = np.array(ls_degrees)
    ls_degrees = ls_degrees[ls_degrees > 0]
    if len(ls_degrees) == 0:
        print("Warning: No positive degrees to compute PDF")
        return np.array([]), np.nan, np.array([]), np.array([])
    min_degree = max(min_degree, 1) 
    max_degree = max(max_degree, min_degree) 
   
    # 1) Logarithmic bins: divide the entire range of degree values in bins which are equally separated in logarithmic scale
    bins = np.logspace(np.log10(min_degree), np.log10(max_degree+1),  num_bins + 1) 
    p, edges = np.histogram(ls_degrees, bins, density=True)
    # Bin centers
    bin_centers = np.sqrt(edges[:-1] * edges[1:])
    # Remove zeros
    mask = p > 0
    bin_centers = bin_centers[mask]
    p = p[mask]
    if len(bin_centers) < 2: 
        print("Warning: Not enough points to fit PDF") 
        theoretical_pdf = np.full_like(bin_centers, np.nan, dtype=float) 
        return bin_centers, np.nan, p, theoretical_pdf 

    # 3) Fit: slope m is approx -(gamma - 1) in the log-binned plot 
    m_bin, b_bin = np.polyfit(np.log10(bin_centers), np.log10(p), 1)
    theoretical_pdf = (10**b_bin) * (bin_centers**m_bin)
    
    return bin_centers, m_bin, p, theoretical_pdf


def CCDF(net, degree, degree_count):
    '''
    Compute the Complementary Cumulative Density Function
    '''
    degree = np.array(degree)
    degree_count = np.array(degree_count)
    if net.number_of_nodes() == 0: 
        print("Warning: Empty network") 
        return np.array([]), np.nan, np.array([]), np.array([]) 

    # 0) Sort by degree (to ensure cumsum works correctly)
    idx = np.argsort(degree)
    degree = degree[idx]
    degree_count = degree_count[idx]

    # 1) Compute the CCDF from the degrees of the nodes
    ccdf = np.flip(np.cumsum(np.flip(degree_count))) / net.number_of_nodes()

    # Use the mask to remove zeros (for the log)
    mask = (degree > 0) & (degree_count > 0) & (ccdf > 0) 
    degree_masked = degree[mask]
    ccdf_masked = ccdf[mask]

    # 2) Represent the CCDF in logarithmic scale
    if len(degree_masked) < 2: 
        print("Warning: Not enough points to fit CCDF") 
        theoretical = np.full_like(degree_masked, np.nan, dtype=float) 
        return ccdf, np.nan, theoretical, degree_masked 
    log_degree_fit = np.log(degree_masked)
    log_ccdf_fit = np.log(ccdf_masked)

    # 3) Fit via least-squares to a line, get the slope m.
    # gamma = -m + 1
    m,b = np.polyfit(log_degree_fit, log_ccdf_fit, 1) 
    theoretical=[np.exp(b)*k**m for k in degree_masked]
    return ccdf, m, theoretical, degree_masked


def get_degree_distribution(net, net_name='', po=False, num_bins=20, gamma_val=None, plot_file=None): 
    ls_degrees=[net.degree(node) for node in net.nodes()]
    min_degree=min(ls_degrees)
    max_degree=max(ls_degrees)
    degree = list(range(min_degree, max_degree + 1))
    degree_count = [Counter(ls_degrees).get(x, 0) for x in degree]

    # CCDF
    ccdf, m, theoretical, degree_masked = CCDF(net, degree, degree_count)
  
    # PDF with logarithmic binning
    bin_centers, m_bin, p, theoretical_log_binning = PDF_log_binning(num_bins, min_degree, max_degree, ls_degrees)

    # Sanity check: gamma comparison
    # gamma_ccdf = 1-m
    # gamma_pdf = 1-m_bin 
    # if np.isclose(gamma_ccdf, gamma_pdf, rtol=0.05, atol=0.1): 
    #     print(f"Both gammas coincide approximately: CCDF = {gamma_ccdf} = PDF = {gamma_pdf}") 
    # else:
    #     print(f"Gammas do NOT coincide: CCDF = {gamma_ccdf} | PDF = {gamma_pdf}")

    # Poisson distribution
    if po:
        # Generate poisson distribution
        n_nodes = net.number_of_nodes()
        degrees=dict(net.degree())
        avg_degree = sum(degrees.values()) / n_nodes
        poisson_yvalues = poisson.pmf(degree, avg_degree)
    
    # Theoretical Barabási-Albert line (CCDF)
    if gamma_val is not None:
        k_continuous = np.linspace(min_degree, max_degree, 100)
        # slope becomes -(gamma - 1)
        ccdf_slope = -(gamma_val - 1)        
        C_ccdf = ccdf[0] * (min_degree ** (gamma_val - 1))
        theoretical_ccdf_y = C_ccdf * (k_continuous ** ccdf_slope)

    # Plot
    fig,ax=plt.subplots(2,3,figsize=(12,8))
    if po:
        ax[0,0].hist(ls_degrees, bins=range(min_degree, max_degree + 2), color='lightblue', density=True, label='Network') 
        ax[0,0].plot(degree,poisson_yvalues, color='black', lw=2, label='Poisson')
    else:
        ax[0,0].hist(ls_degrees, color='lightblue')
    ax[0,1].hist(ls_degrees, color='lightblue')
    ax[0,2].scatter(degree, degree_count, color='lightblue')
    ax[1,0].scatter(degree, degree_count, color='lightblue')
    ax[1,1].scatter(degree, ccdf, color='lightblue')
    if gamma_val is not None:
        ax[1,1].plot(k_continuous, theoretical_ccdf_y, color='black', 
                     lw=2, label=rf'BA Theory $\gamma={gamma_val}$')             
    else:
        ax[1,1].plot(degree_masked, theoretical,color='black',label=r'$\gamma=%.2f$'%(1-m)) 
    ax[1,2].plot(bin_centers, theoretical_log_binning, color='black',label=r'$\gamma=%.2f$'%(1-m_bin)) 
    ax[1,2].scatter(bin_centers, p, color='lightblue')
    # Axis labels
    ax[0,0].set_xlabel('Degree',fontsize=15)
    ax[0,0].set_ylabel('Frequency',fontsize=15)
    ax[0,2].set_xlabel('$k$',fontsize=15)
    ax[0,2].set_ylabel('$P(k)$',fontsize=15)
    ax[1,0].set_xlabel('$k$',fontsize=15)
    ax[1,0].set_ylabel('$P(k)$',fontsize=15) 
    ax[1,1].set_xlabel('$k$',fontsize=15)
    ax[1,1].set_ylabel('$CCDF(k)$',fontsize=15)
    ax[1,2].set_xlabel('$k$',fontsize=15)
    ax[1,2].set_ylabel('$P(k)$',fontsize=15)

    # Logarithmic scale
    ax[0,1].set_yscale('log')
    ax[0,1].set_xscale('log')
    ax[1,0].set_yscale('log')
    ax[1,0].set_xscale('log')
    ax[1,1].set_yscale('log')
    ax[1,1].set_xscale('log')
    ax[1,2].set_xscale('log')
    ax[1,2].set_yscale('log')

    # Label ticks
    ax[0,2].tick_params(which='major',axis='both',labelsize=15)
    ax[1,0].tick_params(which='major',axis='both',labelsize=15)
    ax[1,1].tick_params(which='major',axis='both',labelsize=15)
    if po:
        ax[0,0].legend(frameon=False)
    ax[1,1].legend(frameon=False)
    ax[1,2].legend(frameon=False)

    # Titles
    ax[0,0].set_title('Histogram',fontsize=15)
    ax[0,1].set_title('Histogram (logarithmic scale)',fontsize=15)
    ax[0,2].set_title('PDF (linear scale)',fontsize=15)
    ax[1,0].set_title('PDF (logarithmic scale)',fontsize=15)
    ax[1,1].set_title('CCDF (logarithmic scale)',fontsize=15)
    ax[1,2].set_title('PDF with logarithmic binning', fontsize=15)

    fig.suptitle(f'DEGREE DISTRIBUTION: {net_name}', fontsize=18, fontweight='bold')
    fig.tight_layout()
    if plot_file is None: 
        plot_name = net_name.replace(' ', '_') if net_name else 'network' 
        plot_file = os.path.join(RESULTS_DIR, f'{plot_name}_degree_distribution.png') 
    fig.savefig(plot_file, dpi=300, bbox_inches='tight') 
    plt.close(fig) 
    return plot_file 

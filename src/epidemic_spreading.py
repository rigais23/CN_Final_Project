import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.utils import SEED, seeds_random, seeds_top_hubs, seeds_disease_genes


##############################
# SIR MODEL
##############################
def sir_step(G, states, lambda_, mu, rng):
    '''
    One discrete-time SIR step.
    '''
    # G --> NetworkX graph (only nodes still present matter)
    # states --> dict {node: 'S' | 'I' | 'R'}
    # lambda_ --> per-edge spreading probability (S -> I)
    # mu --> per-node degradation probability (I -> R)
    # rng --> numpy Generator

    new_states = states.copy()

    for node in G.nodes():
        if states[node] == 'S':
            # Become I if any infectious neighbour transmits
            infected_neighbours = [
                nb for nb in G.neighbors(node) if states[nb] == 'I'
            ]
            if infected_neighbours:
                # 1 - (1-lambda_)^k 
                p_infect = 1.0 - (1.0 - lambda_) ** len(infected_neighbours)
                if rng.random() < p_infect:
                    new_states[node] = 'I'

        elif states[node] == 'I':
            # Degraded by PROTEASOME
            if rng.random() < mu:
                new_states[node] = 'R'

    return new_states # Dict with updates states after one step


def run_sir(G, seeds, lambda_, mu, max_steps=500, remove_R=False, rng=None):
    '''
    Run one SIR simulation on graph G.
    '''
    # G --> NetworkX graph
    # seeds --> iterable of initial I nodes
    # lambda_ --> spreading probability per edge per step
    # mu --> degradation probability per node per step
    # max_steps --> maximum number of timesteps
    # remove_R --> if True, R nodes (and their edges) are removed from G
    # rng --> numpy Generator 

    if rng is None:
        rng = np.random.default_rng(SEED)

    G_sim = G.copy()
    n_original = G.number_of_nodes()
    states = {node: 'S' for node in G_sim.nodes()}
    for seed in seeds:
        if seed in states:
            states[seed] = 'I'

    rows = []

    for step in range(max_steps):
        counts = {'S': 0, 'I': 0, 'R': 0}
        for s in states.values():
            counts[s] += 1

        rows.append({
            'step': step,
            'S': counts['S'],
            'I': counts['I'],
            'R': counts['R'],
            'frac_S': counts['S'] / n_original,
            'frac_I': counts['I'] / n_original,
            'frac_R': counts['R'] / n_original,
        })

        # Stop early if no infectious nodes remain
        if counts['I'] == 0:
            break

        states = sir_step(G_sim, states, lambda_, mu, rng)

        if remove_R:
            r_nodes = [n for n, s in states.items() if s == 'R']
            G_sim.remove_nodes_from(r_nodes)

    return pd.DataFrame(rows) # DataFrame with columns [step, S, I, R, frac_S, frac_I, frac_R]


##############################
# PARAMETER SWEEP
##############################
def epidemic_threshold_sweep(G, lambda_values, mu=0.1, n_seeds=1, n_reps=30, max_steps=300, seed=SEED):
    """
    Sweep lambda around the epidemic threshold.
    """
    rows = []
    
    for lambda_ in lambda_values:
        final_Rs = []
        extinction_times = []
        
        for i in range(n_reps):
            rng = np.random.default_rng(seed + i)
            seeds = seeds_random(G, n_seeds, rng)
            
            df = run_sir(G, seeds, lambda_=lambda_, mu=mu,
                         max_steps=max_steps, rng=rng)
                         
            # 1. Calculate Final R
            final_Rs.append(df['frac_R'].iloc[-1])
            
            # 2. Calculate Extinction Time (Steps until I = 0)
            extinct = df[df['frac_I'] == 0]
            if not extinct.empty:
                t = int(extinct['step'].iloc[0])
            else:
                t = int(df['step'].iloc[-1]) # Default to max steps if it never hits 0
            
            extinction_times.append(t)
            
        rows.append({
            'lambda_': lambda_,
            'mean_final_R': float(np.mean(final_Rs)),
            'std_final_R': float(np.std(final_Rs)),
            'mean_extinction_time': float(np.mean(extinction_times))
        })
        
    return pd.DataFrame(rows) # DataFrame with [lambda_, mean_final_R, std_final_R, mean_extinction_time].

##############################
# FULL ANALYSIS
##############################
def get_sir_analysis(G, net_name, results_dir, lambda_=0.05, mu=0.1, n_seeds=1, n_reps=10, max_steps=500, disease_genes=None, remove_R=False):
    """
    Run SIR for all three seeding strategies and save results.
    """
    # G  --> NetworkX graph (LCC recommended)
    # net_name --> string label for filenames / plot titles
    # results_dir --> output directory
    # lambda_ --> spreading probability
    # mu --> degradation probability
    # n_seeds --> number of initially infected nodes
    # n_reps --> repetitions for random seeding (averaged)
    # max_steps --> maximum simulation timesteps
    # disease_genes -> terable of disease-associated gene names (for strategy 3)
    # remove_R --> whether to remove R nodes from the graph each step

    # Returns
        # curves --> DataFrame with [step, strategy, frac_S, frac_I, frac_R]
        # summary --> DataFrame with per-strategy summary statistics
        # sweep_df --> DataFrame from epidemic threshold sweep
    
    os.makedirs(results_dir, exist_ok=True)

    # 1. Random seeding (averaged over n_reps)
    random_runs = []
    for i in range(n_reps):
        rng = np.random.default_rng(SEED + i)
        s = seeds_random(G, n_seeds, rng)
        df = run_sir(G, s, lambda_=lambda_, mu=mu,
                     max_steps=max_steps, remove_R=remove_R, rng=rng)
        df['rep'] = i
        random_runs.append(df)

    random_all = pd.concat(random_runs)
    random_curve = (random_all
                    .groupby('step')[['frac_S', 'frac_I', 'frac_R']]
                    .mean()
                    .reset_index())
    random_curve['strategy'] = 'random'

    # 2. Hub seeding (top degree nodes)
    hub_seeds = seeds_top_hubs(G, n_seeds)
    hub_df = run_sir(G, hub_seeds, lambda_=lambda_, mu=mu,
                     max_steps=max_steps, remove_R=remove_R,
                     rng=np.random.default_rng(SEED))
    hub_df['strategy'] = 'hub'

    # 3. Disease-gene seeding 
    if disease_genes is None:
        disease_genes = []
    dis_seeds = seeds_disease_genes(G, disease_genes, n_seeds)
    dis_df = run_sir(G, dis_seeds, lambda_=lambda_, mu=mu,
                     max_steps=max_steps, remove_R=remove_R,
                     rng=np.random.default_rng(SEED))
    dis_df['strategy'] = 'disease_gene'

    # Combine Reults
    curves = pd.concat(
        [random_curve[['step', 'strategy', 'frac_S', 'frac_I', 'frac_R']],
         hub_df[['step', 'strategy', 'frac_S', 'frac_I', 'frac_R']],
         dis_df[['step', 'strategy', 'frac_S', 'frac_I', 'frac_R']]],
        ignore_index=True
    )

    # Summary statistics
    summary_rows = []
    for strat, grp in curves.groupby('strategy'):
        peak_I   = grp['frac_I'].max()
        time_peak = int(grp.loc[grp['frac_I'].idxmax(), 'step'])
        final_R  = grp['frac_R'].iloc[-1]
        summary_rows.append({
            'strategy':        strat,
            'peak_infected':   peak_I,
            'time_to_peak':    time_peak,
            'final_R_fraction': final_R,
            'lambda':          lambda_,
            'mu':              mu,
        })
    summary = pd.DataFrame(summary_rows)

    # Epidemic threshold sweep
    lambda_values =  np.logspace(-3, -0.5, 40)
    sweep_df = epidemic_threshold_sweep(G, lambda_values, mu=mu,
                                        n_seeds=n_seeds, n_reps=30,
                                        max_steps=max_steps, seed=SEED)

    # Save
    curves.to_csv(os.path.join(results_dir, f'{net_name}_sir_curves.csv'), index=False)
    summary.to_csv(os.path.join(results_dir, f'{net_name}_sir_summary.csv'), index=False)
    sweep_df.to_csv(os.path.join(results_dir, f'{net_name}_sir_sweep.csv'), index=False)

    # Plots
    plot_sir(curves, sweep_df, net_name,
             os.path.join(results_dir, f'{net_name}_sir.png'))
    
    plot_epidemic_diagram(curves, net_name,
                           os.path.join(results_dir, f'{net_name}_sir_epidemic_diagram.png'))
    
    plot_time_to_extinction(curves, net_name,
                            os.path.join(results_dir, f'{net_name}_sir_time_to_extinction.png'))

    return curves, summary, sweep_df


##############################
# PLOTTING
##############################
def plot_sir(curves, sweep_df, net_name, plot_file):
    net_label = net_name.capitalize()
    colors = {
        'random':       'lightblue',
        'hub':          'cornflowerblue',
        'disease_gene': 'darkblue',
    }
    labels = {
        'random':       'Random seed',
        'hub':          'Hub seed (top degree)',
        'disease_gene': 'Disease-gene seed',
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Left: infected fraction over time 
    ax = axes[0]
    for strategy, grp in curves.groupby('strategy'):
        grp_sorted = grp.sort_values('step')
        ax.plot(grp_sorted['step'], grp_sorted['frac_I'], color=colors[strategy], label=labels[strategy], linewidth=2)
    ax.set_title('Infected fraction over time', fontsize=13)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Fraction of nodes (I)')
    ax.legend(frameon=False, fontsize=9)

    # Centre: final R fraction over time 
    ax = axes[1]
    for strategy, grp in curves.groupby('strategy'):
        grp_sorted = grp.sort_values('step')
        ax.plot(grp_sorted['step'], grp_sorted['frac_R'], color=colors[strategy], label=labels[strategy], linewidth=2)
    ax.set_title('Cumulative dysfunction (R)', fontsize=13)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Fraction of nodes (R)')
    ax.legend(frameon=False, fontsize=9)

    # Right: threshold sweep 
    ax = axes[2]
    ax.plot(sweep_df['lambda_'], sweep_df['mean_final_R'], color='cornflowerblue', linewidth=2)
    ax.fill_between(sweep_df['lambda_'], sweep_df['mean_final_R'] - sweep_df['std_final_R'], sweep_df['mean_final_R'] + sweep_df['std_final_R'], color='lightblue', alpha=0.4)
    ax.set_title('Epidemic threshold sweep', fontsize=13)
    ax.set_xlabel('Spreading probability λ')
    ax.set_ylabel('Final R fraction')

    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle(f'SIR DYSFUNCTION PROPAGATION: {net_label}', fontsize=15, fontweight='bold')
    fig.tight_layout()
    fig.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_epidemic_diagram(curves, net_name, plot_file):
    """
    Classic SIR epidemic diagram: S, I, R all on the same axes, with one panel per seeding strategy.
    """
    net_label = net_name.capitalize()
    strategies = ['random', 'hub', 'disease_gene']
    labels_strat = {'random': 'Random seed', 'hub': 'Hub seed', 'disease_gene': 'Disease-gene seed'}
    colors = {'frac_S': 'lightblue', 'frac_I': 'cornflowerblue', 'frac_R': 'darkblue'}
    compartment_labels = {'frac_S': 'S (functional)', 'frac_I': 'I (misfolded)', 'frac_R': 'R (degraded)'}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for ax, strategy in zip(axes, strategies):
        grp = curves[curves['strategy'] == strategy].sort_values('step')
        for compartment in ['frac_S', 'frac_I', 'frac_R']:
            ax.plot(grp['step'], grp[compartment],
                    color=colors[compartment],
                    label=compartment_labels[compartment],
                    linewidth=2)
        ax.set_title(labels_strat[strategy], fontsize=12)
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Fraction of nodes')
        ax.legend(frameon=False, fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylim(-0.02, 1.02)

    fig.suptitle(f'EPIDEMIC DIAGRAM: {net_label}', fontsize=15, fontweight='bold')
    fig.tight_layout()
    fig.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_time_to_extinction(curves, net_name, plot_file):
    """
    Time to extinction: first timestep where I = 0.
    """
    net_label = net_name.capitalize()
    strategies = ['random', 'hub', 'disease_gene']
    labels_strat = {'random': 'Random seed', 'hub': 'Hub seed', 'disease_gene': 'Disease-gene seed'}
    colors = ['lightblue', 'cornflowerblue', 'darkblue']

    extinction_times = []
    for strategy in strategies:
        grp = curves[curves['strategy'] == strategy].sort_values('step').reset_index(drop=True)
        extinct = grp[grp['frac_I'] == 0]
        t_ext = int(extinct['step'].iloc[0]) if not extinct.empty else int(grp['step'].iloc[-1])
        extinction_times.append(t_ext)

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar([labels_strat[s] for s in strategies], extinction_times, color=colors)
    ax.bar_label(bars, labels=[f'{t} steps' for t in extinction_times], padding=3, fontsize=10)
    ax.set_title(f'TIME TO EXTINCTION: {net_label}', fontsize=13, fontweight='bold')
    ax.set_ylabel('Timesteps until I = 0')
    ax.set_ylim(0, max(extinction_times) * 1.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_cross_network_comparison(sweep_dfs_by_mu, results_dir):
    colors = {'breast': 'lightblue', 'lung': 'cornflowerblue', 'ovarian': 'darkblue'}
    mu_values = list(sweep_dfs_by_mu.keys())
    n_mu = len(mu_values)

    fig, axes = plt.subplots(n_mu, 3, figsize=(21, 5 * n_mu))  # 3 columns now

    if n_mu == 1:
        axes = [axes]

    for i, mu in enumerate(mu_values):
        sweep_dfs = sweep_dfs_by_mu[mu]
        ax_left, ax_right, ax_log = axes[i]  # unpack third axis

        # ── Left: final R fraction vs lambda (linear) ──
        for net_name, sweep_df in sweep_dfs.items():
            color = colors.get(net_name, '#333333')
            ax_left.plot(sweep_df['lambda_'], sweep_df['mean_final_R'],
                         color=color, label=net_name.capitalize(),
                         linewidth=2, marker='o', markersize=3)
            ax_left.fill_between(sweep_df['lambda_'],
                                 sweep_df['mean_final_R'] - sweep_df['std_final_R'],
                                 sweep_df['mean_final_R'] + sweep_df['std_final_R'],
                                 color=color, alpha=0.15)
        ax_left.set_title(f'SIR EPIDEMIC DIAGRAM (μ = {mu})', fontsize=13, fontweight='bold')
        ax_left.set_xlabel('λ')
        ax_left.set_ylabel('final R fraction (total damage)')
        ax_left.legend(frameon=False, fontsize=10)
        ax_left.spines[['top', 'right']].set_visible(False)

        # ── Centre: time to extinction vs lambda ──
        for net_name, sweep_df in sweep_dfs.items():
            color = colors.get(net_name, '#333333')
            if 'mean_extinction_time' in sweep_df.columns:
                ax_right.plot(sweep_df['lambda_'], sweep_df['mean_extinction_time'],
                              color=color, label=net_name.capitalize(),
                              linewidth=2, marker='o', markersize=3)
            else:
                print(f"Warning: 'mean_extinction_time' not found for {net_name}")
        ax_right.set_title(f'TIME TO EXTINCTION (μ = {mu})', fontsize=13, fontweight='bold')
        ax_right.set_xlabel('λ')
        ax_right.set_ylabel('t (steps to I = 0)')
        ax_right.legend(frameon=False, fontsize=10)
        ax_right.spines[['top', 'right']].set_visible(False)

        # ── Right: final R fraction vs lambda (log x-axis) ──
        for net_name, sweep_df in sweep_dfs.items():
            color = colors.get(net_name, '#333333')
            ax_log.plot(sweep_df['lambda_'], sweep_df['mean_final_R'],
                        color=color, label=net_name.capitalize(),
                        linewidth=2, marker='o', markersize=3)
            ax_log.fill_between(sweep_df['lambda_'],
                                sweep_df['mean_final_R'] - sweep_df['std_final_R'],
                                sweep_df['mean_final_R'] + sweep_df['std_final_R'],
                                color=color, alpha=0.15)
        ax_log.set_xscale('log')
        ax_log.set_title(f'EPIDEMIC THRESHOLD (μ = {mu})', fontsize=13, fontweight='bold')
        ax_log.set_xlabel('λ (log scale)')
        ax_log.set_ylabel('final R fraction')
        ax_log.legend(frameon=False, fontsize=10)
        ax_log.spines[['top', 'right']].set_visible(False)

    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, 'sir_cross_network_comparison_multi_mu.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)

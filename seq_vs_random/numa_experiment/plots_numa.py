import argparse
import pandas as pd
import matplotlib.pyplot as plt
import math

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="numa_cache_benchmark_results_std.csv")
    parser.add_argument("--metric", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    
    mean_col = f"{args.metric}_mean"
    std_col = f"{args.metric}_std"
    
    if mean_col not in df.columns or std_col not in df.columns:
        print(f"Error: Missing columns {mean_col} or {std_col} in {args.csv}")
        return

    sizes = sorted(df['size_elements'].unique())
    n_sizes = len(sizes)
    
    cols = 2
    rows = math.ceil(n_sizes / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 6 * rows), squeeze=False)
    axes = axes.flatten()

    for i, size in enumerate(sizes):
        ax = axes[i]
        df_size = df[df['size_elements'] == size]
        
        df_seq = df_size[df_size['mode'] == 'seq']
        df_rand = df_size[df_size['mode'] == 'rand']

        ax.errorbar(df_seq['numa_node'], df_seq[mean_col], yerr=df_seq[std_col], 
                    fmt='-o', label='Sequential', capsize=5, elinewidth=2, markersize=6)
        ax.errorbar(df_rand['numa_node'], df_rand[mean_col], yerr=df_rand[std_col], 
                    fmt='-s', label='Random', capsize=5, elinewidth=2, markersize=6)

        ax.set_xticks(sorted(df_size['numa_node'].unique()))
        ax.set_xlabel('Execution NUMA Node (Memory bound to Node 0)')
        ax.set_ylabel(args.metric)
        
        size_mb = df_size['size_MB'].iloc[0]
        ax.set_title(f'Array Size: {size_mb:.1f} MB')
        ax.legend()
        ax.grid(True, which="both", ls="--", alpha=0.5)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(f'{args.metric} Across NUMA Nodes', fontsize=16, y=1.02)
    plt.tight_layout()
    
    output_filename = f'plot_numa_{args.metric}_all.png'
    plt.savefig(output_filename, bbox_inches='tight')
    print(f"Saved plot to {output_filename}")

if __name__ == "__main__":
    main()

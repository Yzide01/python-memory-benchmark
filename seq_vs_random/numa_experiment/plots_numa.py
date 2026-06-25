import argparse
import pandas as pd
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="numa_cache_benchmark_results_std.csv")
    parser.add_argument("--metric", required=True)
    parser.add_argument("--size", type=int, default=100000000)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    
    mean_col = f"{args.metric}_mean"
    std_col = f"{args.metric}_std"
    
    if mean_col not in df.columns or std_col not in df.columns:
        print(f"Error: Missing columns {mean_col} or {std_col} in {args.csv}")
        return

    df_size = df[df['size_elements'] == args.size]
    
    if df_size.empty:
        print(f"Error: No data found for size {args.size}")
        return

    df_seq = df_size[df_size['mode'] == 'seq']
    df_rand = df_size[df_size['mode'] == 'rand']

    plt.figure(figsize=(10, 6))
    
    plt.errorbar(df_seq['numa_node'], df_seq[mean_col], yerr=df_seq[std_col], 
                 fmt='-o', label='Sequential', capsize=5, elinewidth=2, markersize=6)
    plt.errorbar(df_rand['numa_node'], df_rand[mean_col], yerr=df_rand[std_col], 
                 fmt='-s', label='Random', capsize=5, elinewidth=2, markersize=6)

    plt.xticks(sorted(df_size['numa_node'].unique()))
    plt.xlabel('Execution NUMA Node (Memory bound to Node 0)')
    plt.ylabel(args.metric)
    size_mb = df_size['size_MB'].iloc[0]
    plt.title(f'{args.metric} Across NUMA Nodes\n(Array Size: {size_mb:.1f} MB)')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    
    output_filename = f'plot_numa_{args.metric}_{args.size}.png'
    plt.savefig(output_filename)
    print(f"Saved plot to {output_filename}")

if __name__ == "__main__":
    main()

import argparse
import pandas as pd
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="cache_benchmark_results_std.csv")
    parser.add_argument("--metric", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    
    mean_col = f"{args.metric}_mean"
    std_col = f"{args.metric}_std"
    
    if mean_col not in df.columns or std_col not in df.columns:
        print(f"Error: Missing columns {mean_col} or {std_col} in {args.csv}")
        return

    df_seq = df[df['mode'] == 'seq']
    df_rand = df[df['mode'] == 'rand']

    plt.figure(figsize=(10, 6))
    
    plt.errorbar(df_seq['size_MB'], df_seq[mean_col], yerr=df_seq[std_col], 
                 fmt='-o', label='Sequential', capsize=5, elinewidth=2, markersize=6)
    plt.errorbar(df_rand['size_MB'], df_rand[mean_col], yerr=df_rand[std_col], 
                 fmt='-s', label='Random', capsize=5, elinewidth=2, markersize=6)

    plt.xscale('log')
    plt.xlabel('Array Size (MB)')
    plt.ylabel(args.metric)
    plt.title(f'{args.metric}: Sequential vs Random')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    
    output_filename = f'plot_{args.metric}.png'
    plt.savefig(output_filename)
    print(f"Saved plot to {output_filename}")

if __name__ == "__main__":
    main()

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import platform
import sys

def main():
    parser = argparse.ArgumentParser(description="NUMA Streaming Analytical Plotter (Clean Axes with Masking)")
    parser.add_argument("--csv", type=str, required=True, help="Path to the CSV file")
    parser.add_argument("--l3", type=float, default=32.0, help="L3 cache size (black line)")
    
    parser.add_argument("--op", type=str, default="Unknown", help="Tested operation (e.g., JOIN)")
    parser.add_argument("--cpubind", type=str, default="?", help="CPU node used")
    parser.add_argument("--membind", type=str, default="?", help="Memory node used")
    parser.add_argument("--machine", type=str, default="", help="Machine name")

    args = parser.parse_args()

    # 1. Load data
    try:
        df = pd.read_csv(args.csv)
    except FileNotFoundError:
        print(f"❌ Error: The file '{args.csv}' cannot be found.")
        sys.exit(1)

    print(f"📊 Generating clean analytical plot from {args.csv}...")

    # 2. Build Title
    hostname = args.machine if args.machine else platform.node()
    min_mo = df["Mo"].min()
    max_mo = df["Mo"].max()
    
    main_title = (
        f"NUMA STREAMING Analysis | Machine: {hostname} | Operation: {args.op.upper()}\n"
        f"Topology -> CPU-Bind: Node {args.cpubind}  |  Mem-Bind: Node {args.membind}\n"
        f"Sweep: from {min_mo} to {max_mo} MB  |  L3 Limit: {args.l3} MB"
    )

    # 3. Initialize asymmetric grid (16x11)
    fig = plt.figure(figsize=(16, 11))
    fig.suptitle(main_title, fontsize=14, fontweight='bold', color='#2c3e50', y=0.98)

    gs = gridspec.GridSpec(6, 2, figure=fig)

    ax_lat = fig.add_subplot(gs[0:3, 0])
    ax_miss = fig.add_subplot(gs[3:6, 0])
    ax_ipc = fig.add_subplot(gs[0:2, 1])
    ax_dtlb = fig.add_subplot(gs[2:4, 1])
    ax_branch = fig.add_subplot(gs[4:6, 1])

    # Configuration des graphiques avec les nouvelles colonnes (LLC_Misses et dTLB_Misses)
    plot_configs = [
        (ax_lat, "lat_ns", "Execution Latency", "Latency (ns)", "#9467bd"),
        (ax_miss, "LLC_Misses", "L3 Cache Misses (LLC Misses)", "L3 Misses", "#d62728"),
        (ax_ipc, "IPC", "Instructions Per Cycle (IPC)", "Instructions / Cycle", "#2ca02c"),
        (ax_dtlb, "dTLB_Misses", "dTLB Misses (Translation Lookaside Buffer)", "dTLB Misses", "#1f77b4"),
        (ax_branch, "Branch_Misses", "Branch Prediction Misses", "Branch Misses", "#e377c2")
    ]

    for ax, metric, title, ylabel, color in plot_configs:
        # Plot main curve
        ax.plot(df["Mo"], df[metric], marker='o', linestyle='-', linewidth=2, color=color, zorder=2)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=10, fontweight='medium')
        ax.grid(True, linestyle='--', alpha=0.5, zorder=1)
        
        # L3 Limit (in transparent black)
        ax.axvline(x=args.l3, color='black', linestyle='--', linewidth=2, label=f'L3 Limit ({args.l3} MB)', alpha=0.6, zorder=1)
        
        # Format Y-axis numbers
        if metric != "IPC":
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            format_str = "{:.2e}"
        else:
            ax.ticklabel_format(style='plain', axis='y')
            ax.legend(loc='lower right')
            format_str = "{:.2f}"

        # ==========================================
        # ANALYTICAL MODULE: PEAK DETECTION AND CLEAN MARKING (with masking)
        # ==========================================
        max_idx = df[metric].idxmax()
        x_max = df.loc[max_idx, "Mo"]
        y_max = df.loc[max_idx, metric]

        # Discreet red star
        ax.scatter(x_max, y_max, color='red', s=60, marker='*', zorder=5)

        # Crosshairs
        ymin, ymax_lim = ax.get_ylim()
        xmin, xmax_lim = ax.get_xlim()
        ax.vlines(x=x_max, ymin=ymin, ymax=y_max, color='red', linestyle=':', linewidth=1.0, alpha=0.6, zorder=3)
        ax.hlines(y=y_max, xmin=xmin, xmax=x_max, color='red', linestyle=':', linewidth=1.0, alpha=0.6, zorder=3)

        # Bounding box style (solid white background to hide ticks)
        bbox_style = dict(boxstyle="square,pad=0.2", fc="white", ec="none", alpha=1.0)

        # Coordinates on X-axis (with bbox)
        ax.annotate(f"{x_max} MB", xy=(x_max, ymin), xycoords='data',
                    xytext=(0, -6), textcoords='offset points',
                    ha='center', va='top', color='red', fontsize=9, fontweight='bold',
                    bbox=bbox_style, annotation_clip=False, zorder=10)

        # Coordinates on Y-axis (with bbox)
        ax.annotate(format_str.format(y_max), xy=(xmin, y_max), xycoords='data',
                    xytext=(-6, 0), textcoords='offset points',
                    ha='right', va='center', color='red', fontsize=9, fontweight='bold',
                    bbox=bbox_style, annotation_clip=False, zorder=10)

        # Reset limits
        ax.set_ylim(ymin, ymax_lim)
        ax.set_xlim(xmin, xmax_lim)

    # Global X-axis labels (only for bottom plots)
    ax_miss.set_xlabel("DataFrame Memory Size (MB)", fontsize=12, fontweight='bold')
    ax_branch.set_xlabel("DataFrame Memory Size (MB)", fontsize=12, fontweight='bold')

    # Adjust margins
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    
    # 4. Save output
    output_filename = f"plot_clean_STREAMING_{args.op}_c{args.cpubind}m{args.membind}.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"✅ Done! Image saved as: {output_filename}")

if __name__ == "__main__":
    main()

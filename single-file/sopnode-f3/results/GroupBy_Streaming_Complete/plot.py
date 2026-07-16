import polars as pl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# Parameters
threads = [1, 2, 3, 4]
membinds = [0, 1]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

print("📊 Generating 8 individual graphs...")
for m in membinds:
    for t in threads:
        # Make sure the script is in the same folder as the CSVs, 
        # or add the path (e.g., "GroupBy_Streaming_Complete/...")
        csv_file = f"benchmark_polars_grpby_numa_streaming_t{t}_m{m}.csv"
        
        if not os.path.exists(csv_file):
            print(f"⚠️ File not found: {csv_file}")
            continue
            
        df = pl.read_csv(csv_file)
        
        # Conversion: Memory (MB -> GB) and Latency (ns -> Seconds)
        size_gb = df["Mo"] / 1000
        lat_s = df["lat_ns"] / 1_000_000_000
        
        plt.figure(figsize=(8, 5))
        plt.plot(size_gb, lat_s, marker='o', linewidth=2, color='#1f77b4')
        
        mode_str = "LOCAL" if m == 0 else "REMOTE"
        plt.title(f"GroupBy Streaming Latency - {t} Thread(s) | {mode_str} (m{m})", fontsize=14, fontweight='bold')
        plt.xlabel("DataFrame Size (GB)", fontsize=12)
        plt.ylabel("Execution Time (Seconds)", fontsize=12)
        
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.ylim(bottom=0) # Force Y-axis to start at 0
        
        out_name = f"plot_grpby_latency_t{t}_m{m}.png"
        plt.tight_layout()
        plt.savefig(out_name, dpi=300)
        plt.close()
        print(f"  ✅ Graph generated: {out_name}")

print("\n📈 Generating 2 overlapping comparison graphs...")
for m in membinds:
    plt.figure(figsize=(10, 6))
    mode_str = "LOCAL" if m == 0 else "REMOTE"
    
    for i, t in enumerate(threads):
        csv_file = f"benchmark_polars_grpby_numa_streaming_t{t}_m{m}.csv"
        if os.path.exists(csv_file):
            df = pl.read_csv(csv_file)
            size_gb = df["Mo"] / 1000
            lat_s = df["lat_ns"] / 1_000_000_000
            
            plt.plot(size_gb, lat_s, marker='o', linewidth=2, color=colors[i], label=f"{t} Thread(s)")
            
    plt.title(f"Multi-thread Comparison GroupBy Streaming | {mode_str} (m{m})", fontsize=14, fontweight='bold')
    plt.xlabel("DataFrame Size (GB)", fontsize=12)
    plt.ylabel("Execution Time (Seconds)", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(bottom=0)
    
    out_name = f"plot_grpby_comparison_m{m}.png"
    plt.tight_layout()
    plt.savefig(out_name, dpi=300)
    plt.close()
    print(f"  ✅ Comparison graph generated: {out_name}")

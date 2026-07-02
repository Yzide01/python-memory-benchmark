import argparse
import subprocess
import sys
import os
import re
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import platform

# ==========================================
# 1. LE WORKER (Polars Streaming Engine)
# ==========================================
def worker(op, n_elements):
    # On force toujours sur 1 thread pour voir l'impact NUMA pur
    os.environ["POLARS_MAX_THREADS"] = "1"

    if op == "join":
        # ⚠️ CHANGEMENT MAJEUR : Utilisation de l'API Lazy (.lazy())
        df1 = pl.DataFrame({
            "key": np.random.randint(0, n_elements // 10, size=n_elements, dtype=np.uint32),
            "val1": np.random.randint(0, 100, size=n_elements, dtype=np.uint32)
        }).lazy() 

        df2 = pl.DataFrame({
            "key": np.random.randint(0, n_elements // 10, size=n_elements, dtype=np.uint32),
            "val2": np.random.randint(0, 100, size=n_elements, dtype=np.uint32)
        }).lazy()

        # ⚠️ Exécution avec le moteur de streaming !
        res = df1.join(df2, on="key").collect(streaming=True)

    sys.exit(0)

def get_perf_val(output, event_name):
    match = re.search(fr'([\d,]+)\s+\S*{event_name}', output, re.IGNORECASE)
    return int(match.group(1).replace(',', '')) if match else 0

# ==========================================
# 2. L'ORCHESTRATEUR
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Benchmark NUMA - Moteur Streaming Polars")
    parser.add_argument("--mode", type=str, choices=["local", "numa"], default="local")
    parser.add_argument("--cpubind", type=str, default="0")
    parser.add_argument("--membind", type=str, default="0")
    parser.add_argument("--op", type=str, default="join", help="Forcé sur join pour le streaming")
    parser.add_argument("--l3", type=float, default=32.0)
    parser.add_argument("--iter", type=int, default=3)

    parser.add_argument("--min_mo", type=float, default=8.0)
    parser.add_argument("--max_mo", type=float, default=60.0)
    parser.add_argument("--step_mo", type=float, default=2.0)

    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--elements", type=int, help=argparse.SUPPRESS)

    args = parser.parse_args()

    if args.worker:
        worker(args.op, args.elements)
        return

    print(f"\n--- Démarrage Benchmark STREAMING | Mode: {args.mode.upper()} ---")

    bytes_per_row = 16 
    sizes = []
    current_mo = args.min_mo
    while current_mo <= args.max_mo:
        sizes.append(int((current_mo * 1024 * 1024) / bytes_per_row))
        current_mo += args.step_mo

    results = {"Mo": [], "IPC": [], "Cache_Refs": [], "Cache_Misses": [], "Branch_Misses": [], "lat_ns": []}
    events = "instructions,cycles,cache-references,cache-misses,branch-misses"

    for n in sizes:
        size_mo = (n * bytes_per_row) / (1024 * 1024)
        print(f"Test Streaming avec ~{size_mo:.1f} Mo...")

        if args.mode == "local":
            cmd = f"sudo-g5k env LC_ALL=C taskset -c 0 perf stat -e {events} {sys.executable} {__file__} --worker --op {args.op} --elements {n}"
        else:
            cmd = f"sudo-g5k env LC_ALL=C numactl --cpubind={args.cpubind} --membind={args.membind} perf stat -e {events} {sys.executable} {__file__} --worker --op {args.op} --elements {n}"

        data_tmp = {"ipc": [], "refs": [], "misses": [], "b_misses": [], "lat": []}

        for _ in range(args.iter):
            proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)

            cycl = get_perf_val(proc.stderr, "cycles")
            inst = get_perf_val(proc.stderr, "instructions")
            refs = get_perf_val(proc.stderr, "cache-references")
            misses = get_perf_val(proc.stderr, "cache-misses")
            b_misses = get_perf_val(proc.stderr, "branch-misses")

            time_m = re.search(r'([\d\.]+)\s+seconds time elapsed', proc.stderr)
            time = float(time_m.group(1)) * 1_000_000_000 if time_m else 0

            data_tmp["ipc"].append(inst / cycl if cycl > 0 else 0)
            data_tmp["refs"].append(refs)
            data_tmp["misses"].append(misses)
            data_tmp["b_misses"].append(b_misses)
            data_tmp["lat"].append(time)

        results["Mo"].append(size_mo)
        results["IPC"].append(np.mean(data_tmp["ipc"]))
        results["Cache_Refs"].append(np.mean(data_tmp["refs"]))
        results["Cache_Misses"].append(np.mean(data_tmp["misses"]))
        results["Branch_Misses"].append(np.mean(data_tmp["b_misses"]))
        results["lat_ns"].append(np.mean(data_tmp["lat"]))

    # ==========================================
    # 3. TRACÉ DES GRAPHIQUES (Design Analytique Épuré)
    # ==========================================
    print("\nGénération du graphique analytique...")
    hostname = platform.node()
    
    titre_principal = (
        f"Analyse NUMA | Machine : {hostname} | Moteur : STREAMING JOIN\n"
        f"Topologie -> CPU-Bind: Node {args.cpubind}  |  Mem-Bind: Node {args.membind}\n"
        f"Balayage : de {args.min_mo} à {args.max_mo} Mo  |  Limite L3 : {args.l3} Mo"
    )

    fig = plt.figure(figsize=(16, 11))
    fig.suptitle(titre_principal, fontsize=14, fontweight='bold', color='#2c3e50', y=0.98)
    gs = gridspec.GridSpec(6, 2, figure=fig)

    ax_lat = fig.add_subplot(gs[0:3, 0])
    ax_miss = fig.add_subplot(gs[3:6, 0])
    ax_ipc = fig.add_subplot(gs[0:2, 1])
    ax_ref = fig.add_subplot(gs[2:4, 1])
    ax_branch = fig.add_subplot(gs[4:6, 1])

    plot_configs = [
        (ax_lat, "lat_ns", "Latence d'exécution", "Latence (ns)", "#9467bd"),
        (ax_miss, "Cache_Misses", "Défauts de Cache L3 (LLC Misses)", "Échecs au L3", "#d62728"),
        (ax_ipc, "IPC", "Instructions Par Cycle (IPC)", "Instructions / Cycle", "#2ca02c"),
        (ax_ref, "Cache_Refs", "Références au Cache L3 (LLC Refs)", "Accès au L3", "#1f77b4"),
        (ax_branch, "Branch_Misses", "Erreurs de Prédiction", "Erreurs de saut", "#e377c2")
    ]

    # ... (le code de tracé analytique) ...
    for ax, metric, title, ylabel, color in plot_configs:
        ax.plot(results["Mo"], results[metric], marker='o', linestyle='-', linewidth=2, color=color, zorder=2)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=10, fontweight='medium')
        ax.grid(True, linestyle='--', alpha=0.5, zorder=1)
        ax.axvline(x=args.l3, color='black', linestyle='--', linewidth=2, label=f'L3 ({args.l3} Mo)', alpha=0.6, zorder=1)
        
        if metric != "IPC":
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            format_str = "{:.2e}"
        else:
            ax.ticklabel_format(style='plain', axis='y')
            ax.legend(loc='lower right')
            format_str = "{:.2f}"

        # Détection du Pic
        max_idx = np.argmax(results[metric])
        x_max = results["Mo"][max_idx]
        y_max = results[metric][max_idx]

        ax.scatter(x_max, y_max, color='red', s=120, marker='*', zorder=5)
        ymin, ymax_lim = ax.get_ylim()
        xmin, xmax_lim = ax.get_xlim()
        ax.vlines(x=x_max, ymin=ymin, ymax=y_max, color='red', linestyle=':', linewidth=1.5, alpha=0.8, zorder=3)
        ax.hlines(y=y_max, xmin=xmin, xmax=x_max, color='red', linestyle=':', linewidth=1.5, alpha=0.8, zorder=3)

        ax.annotate(f"{x_max} Mo", xy=(x_max, ymin), xycoords='data', xytext=(0, -8), textcoords='offset points', ha='center', va='top', color='red', fontweight='bold', fontsize=10, annotation_clip=False)
        ax.annotate(format_str.format(y_max), xy=(xmin, y_max), xycoords='data', xytext=(-8, 0), textcoords='offset points', ha='right', va='center', color='red', fontweight='bold', fontsize=10, annotation_clip=False)
        
        ax.set_ylim(ymin, ymax_lim)
        ax.set_xlim(xmin, xmax_lim)

    ax_miss.set_xlabel("Taille mémoire du DataFrame (Mo)", fontsize=12, fontweight='bold')
    ax_branch.set_xlabel("Taille mémoire du DataFrame (Mo)", fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    
    # On sauvegarde le CSV en même temps au cas où !
    import pandas as pd
    df_res = pd.DataFrame(results)
    csv_filename = f"benchmark_polars_STREAMING_c{args.cpubind}m{args.membind}.csv"
    df_res.to_csv(csv_filename, index=False)
    
    output_filename = f"plot_STREAMING_c{args.cpubind}m{args.membind}.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"✅ Terminé ! CSV ({csv_filename}) et Image ({output_filename}) générés.")

if __name__ == "__main__":
    main()

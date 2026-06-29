import argparse
import subprocess
import sys
import os
import re
import polars as pl
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. LE WORKER (Polars Cache-Aware)
# ==========================================
def worker(op, n_elements):
    os.environ["POLARS_MAX_THREADS"] = "1"

    if op == "grpby":
        df = pl.DataFrame({
            "data": np.random.randint(0, 1000, size=n_elements, dtype=np.uint32),
            "groups": np.random.randint(0, 100, size=n_elements, dtype=np.uint32)
        })
        res = df.group_by("groups").agg(pl.col("data").sum())

    elif op == "join":
        df1 = pl.DataFrame({
            "key": np.random.randint(0, n_elements // 10, size=n_elements, dtype=np.uint32),
            "val1": np.random.randint(0, 100, size=n_elements, dtype=np.uint32)
        })
        df2 = pl.DataFrame({
            "key": np.random.randint(0, n_elements // 10, size=n_elements, dtype=np.uint32),
            "val2": np.random.randint(0, 100, size=n_elements, dtype=np.uint32)
        })
        res = df1.join(df2, on="key")

    sys.exit(0)

def get_perf_val(output, event_name):
    match = re.search(fr'([\d,]+)\s+\S*{event_name}', output, re.IGNORECASE)
    return int(match.group(1).replace(',', '')) if match else 0

# ==========================================
# 2. L'ORCHESTRATEUR
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Benchmark NUMA Avancé pour Polars")
    parser.add_argument("--mode", type=str, choices=["local", "numa"], default="local")
    parser.add_argument("--cpubind", type=str, default="0")
    parser.add_argument("--membind", type=str, default="0")
    parser.add_argument("--op", type=str, choices=["join", "grpby"], default="grpby")
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

    print(f"\n--- Démarrage Benchmark POLARS | Mode: {args.mode.upper()} ---")

    bytes_per_row = 8 if args.op == "grpby" else 16
    sizes = []
    current_mo = args.min_mo
    while current_mo <= args.max_mo:
        sizes.append(int((current_mo * 1024 * 1024) / bytes_per_row))
        current_mo += args.step_mo

    results = {"Mo": [], "IPC": [], "Cache_Refs": [], "Cache_Misses": [], "Branch_Misses": [], "lat_ns": []}

    events = "instructions,cycles,cache-references,cache-misses,branch-misses"

    for n in sizes:
        size_mo = (n * bytes_per_row) / (1024 * 1024)
        print(f"Test avec {n:,} éléments (~{size_mo:.1f} Mo)...")

        if args.mode == "local":
            cmd = f"sudo env LC_ALL=C taskset -c 0 perf stat -e {events} {sys.executable} {__file__} --worker --op {args.op} --elements {n}"
        else:
            cmd = f"sudo env LC_ALL=C numactl --cpubind={args.cpubind} --membind={args.membind} perf stat -e {events} {sys.executable} {__file__} --worker --op {args.op} --elements {n}"

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
    # 3. TRACÉ DES GRAPHIQUES (Écriture Scientifique + Titre Commande)
    # ==========================================
    print("\nGénération des graphiques...")
    fig, axs = plt.subplots(5, 1, figsize=(12, 23), sharex=True)
    
    # ---------------------------------------------------------
    # NOUVEAU : Récupération de la commande exacte
    # ---------------------------------------------------------
    cmd_str = "python3 " + " ".join(sys.argv)
    fig.suptitle(f"Commande d'exécution : {cmd_str}", fontsize=14, fontweight='bold', color='#333333')
    
    metrics = ["IPC", "Cache_Refs", "Cache_Misses", "Branch_Misses", "lat_ns"]
    ylabels = ["IPC (Instructions/Cycle)", "Cache References", "Cache Misses", "Branch Misses", "Latence (ns)"]
    colors = ['#2ca02c', '#1f77b4', '#d62728', '#e377c2', '#9467bd']

    for i, m in enumerate(metrics):
        axs[i].plot(results["Mo"], results[m], marker='o', linestyle='-', linewidth=2, color=colors[i])
        axs[i].set_title(f"Évolution de {m}", fontsize=12, fontweight='bold')
        axs[i].set_ylabel(ylabels[i], fontsize=10)
        axs[i].grid(True, linestyle='--', alpha=0.7)
        
        # Force l'écriture scientifique (sauf pour l'IPC qui reste de 0 à ~3)
        if m != "IPC":
            axs[i].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            
        axs[i].axvline(x=args.l3, color='red', linestyle='--', linewidth=2, label=f'Limite L3 ({args.l3} Mo)')
        if i == 0: axs[i].legend()

    axs[-1].set_xlabel("Taille mémoire du DataFrame (Mo)", fontsize=12, fontweight='bold')
    axs[-1].ticklabel_format(style='plain', axis='x')

    # On utilise rect pour laisser de la place au Super Titre (suptitle) tout en haut
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"benchmark_polars_{args.op}_{args.mode}.png", dpi=300)
    print("Terminé !")

if __name__ == "__main__":
    main()

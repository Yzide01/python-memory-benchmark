import argparse
import subprocess
import sys
import os
import re
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. LE WORKER NAÏF (Random Access)
# ==========================================
def worker(n_elements):
    data = np.ones(n_elements, dtype=np.float64)
    indices = np.random.randint(0, n_elements, size=n_elements, dtype=np.int64)
    
    for _ in range(5):
        _ = data[indices].sum()
        
    sys.exit(0)

def get_perf_val(output, event_name):
    match = re.search(fr'([\d,]+)\s+\S*{event_name}', output, re.IGNORECASE)
    return int(match.group(1).replace(',', '')) if match else 0

# ==========================================
# 2. L'ORCHESTRATEUR
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Benchmark Mémoire Naïf (NumPy)")
    parser.add_argument("--mode", type=str, choices=["local", "numa"], default="local")
    parser.add_argument("--cpubind", type=str, default="0")
    parser.add_argument("--membind", type=str, default="0")
    parser.add_argument("--l3", type=float, default=24.0)
    parser.add_argument("--iter", type=int, default=3)
    
    parser.add_argument("--min_mo", type=float, default=8.0)
    parser.add_argument("--max_mo", type=float, default=60.0)
    parser.add_argument("--step_mo", type=float, default=2.0)
    
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--elements", type=int, help=argparse.SUPPRESS)
    
    args = parser.parse_args()

    if args.worker:
        worker(args.elements)
        return

    print(f"\n--- Démarrage Benchmark NAÏF | Mode: {args.mode.upper()} ---")
    
    bytes_per_row = 16 # 8 octets data + 8 octets indices
    sizes = []
    current_mo = args.min_mo
    while current_mo <= args.max_mo:
        sizes.append(int((current_mo * 1024 * 1024) / bytes_per_row))
        current_mo += args.step_mo

    results = {"Mo": [], "IPC": [], "L1_Misses": [], "LLC_Misses": [], "dTLB_Misses": [], "Branch_Misses": [], "lat_ns": []}
    
    events = "instructions,cycles,L1-dcache-load-misses,LLC-load-misses,dTLB-load-misses,branch-misses"

    for n in sizes:
        size_mo = (n * bytes_per_row) / (1024 * 1024)
        print(f"Test avec {n:,} éléments (~{size_mo:.1f} Mo)...")
        
        if args.mode == "local":
            cmd = f"sudo env LC_ALL=C taskset -c 0 perf stat -e {events} {sys.executable} {__file__} --worker --elements {n}"
        else:
            cmd = f"sudo env LC_ALL=C numactl --cpubind={args.cpubind} --membind={args.membind} perf stat -e {events} {sys.executable} {__file__} --worker --elements {n}"
        
        data_tmp = {"ipc": [], "l1": [], "llc": [], "dtlb": [], "branch": [], "lat": []}
        
        for _ in range(args.iter):
            proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            cycl = get_perf_val(proc.stderr, "cycles")
            inst = get_perf_val(proc.stderr, "instructions")
            l1 = get_perf_val(proc.stderr, "L1-dcache-load-misses")
            llc = get_perf_val(proc.stderr, "LLC-load-misses")
            dtlb = get_perf_val(proc.stderr, "dTLB-load-misses")
            branch = get_perf_val(proc.stderr, "branch-misses")
            
            time_m = re.search(r'([\d\.]+)\s+seconds time elapsed', proc.stderr)
            time = float(time_m.group(1)) * 1_000_000_000 if time_m else 0
            
            data_tmp["ipc"].append(inst / cycl if cycl > 0 else 0)
            data_tmp["l1"].append(l1)
            data_tmp["llc"].append(llc)
            data_tmp["dtlb"].append(dtlb)
            data_tmp["branch"].append(branch)
            data_tmp["lat"].append(time)

        results["Mo"].append(size_mo)
        results["IPC"].append(np.mean(data_tmp["ipc"]))
        results["L1_Misses"].append(np.mean(data_tmp["l1"]))
        results["LLC_Misses"].append(np.mean(data_tmp["llc"]))
        results["dTLB_Misses"].append(np.mean(data_tmp["dtlb"]))
        results["Branch_Misses"].append(np.mean(data_tmp["branch"]))
        results["lat_ns"].append(np.mean(data_tmp["lat"]))

    # ==========================================
    # 3. TRACÉ DES GRAPHIQUES
    # ==========================================
    print("\nGénération des graphiques...")
    fig, axs = plt.subplots(6, 1, figsize=(12, 26), sharex=True)
    metrics = ["IPC", "L1_Misses", "LLC_Misses", "dTLB_Misses", "Branch_Misses", "lat_ns"]
    colors = ['#2ca02c', '#1f77b4', '#d62728', '#8c564b', '#e377c2', '#9467bd']

    for i, m in enumerate(metrics):
        axs[i].plot(results["Mo"], results[m], marker='o', linestyle='-', linewidth=2, color=colors[i])
        axs[i].set_title(f"Évolution de {m}", fontsize=12, fontweight='bold')
        axs[i].set_ylabel(m)
        axs[i].grid(True, linestyle='--', alpha=0.7)
        axs[i].ticklabel_format(style='plain', axis='y')
        axs[i].axvline(x=args.l3, color='red', linestyle='--', linewidth=2, label=f'Limite L3 ({args.l3} Mo)')
        if i == 0: axs[i].legend()

    axs[-1].set_xlabel("Taille mémoire de la zone de travail (Mo)", fontsize=12, fontweight='bold')
    axs[-1].ticklabel_format(style='plain', axis='x')

    plt.tight_layout()
    plt.savefig(f"benchmark_naive_{args.mode}.png", dpi=300)
    print("Terminé !")

if __name__ == "__main__":
    main()

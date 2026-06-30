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
    # 3. TRACÉ DES GRAPHIQUES (Grille Asymétrique)
    # ==========================================
    print("\nGénération des graphiques...")
    
    # 1. Récupération automatique du nom de la machine (ex: sopnode-f3)
    hostname = platform.node()
    
    # 2. Construction du nouveau Super Titre détaillé
    titre_principal = (
        f"Analyse NUMA Polars | Machine : {hostname} | Opération : {args.op.upper()}\n"
        f"Topologie -> CPU-Bind: Node {args.cpubind}  |  Mem-Bind: Node {args.membind}\n"
        f"Balayage : de {args.min_mo} à {args.max_mo} Mo (Pas: {args.step_mo} Mo)  |  Limite L3 : {args.l3} Mo"
    )

    # Création d'une figure plus large pour accueillir les deux colonnes
    fig = plt.figure(figsize=(16, 11))
    fig.suptitle(titre_principal, fontsize=14, fontweight='bold', color='#2c3e50', y=0.98)

    # 3. Définition de la grille 6x2
    # La colonne de gauche aura 2 graphes (prenant 3 cases de haut chacun)
    # La colonne de droite aura 3 graphes (prenant 2 cases de haut chacun)
    gs = gridspec.GridSpec(6, 2, figure=fig)

    # --- COLONNE GAUCHE ---
    ax_lat = fig.add_subplot(gs[0:3, 0])
    ax_miss = fig.add_subplot(gs[3:6, 0])

    # --- COLONNE DROITE ---
    ax_ipc = fig.add_subplot(gs[0:2, 1])
    ax_ref = fig.add_subplot(gs[2:4, 1])
    ax_branch = fig.add_subplot(gs[4:6, 1])

    # Dictionnaire de configuration des graphes : (Axe, Métrique, Titre, Label_Y, Couleur)
    plot_configs = [
        (ax_lat, "lat_ns", "Latence d'exécution", "Latence (nanosecondes)", "#9467bd"),
        (ax_miss, "Cache_Misses", "Défauts de Cache L3 (LLC Misses)", "Échecs au L3 (Nombre absolu)", "#d62728"),
        (ax_ipc, "IPC", "Instructions Par Cycle (IPC)", "Instructions / Cycle (Ratio)", "#2ca02c"),
        (ax_ref, "Cache_Refs", "Références au Cache L3 (LLC Refs)", "Accès au L3 (Nombre absolu)", "#1f77b4"),
        (ax_branch, "Branch_Misses", "Erreurs de Prédiction de Branchement", "Erreurs de saut (Nombre absolu)", "#e377c2")
    ]

    # Application de la configuration
    for ax, metric, title, ylabel, color in plot_configs:
        ax.plot(results["Mo"], results[metric], marker='o', linestyle='-', linewidth=2, color=color)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=10, fontweight='medium')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Ligne de démarcation L3
        ax.axvline(x=args.l3, color='red', linestyle='--', linewidth=2, label=f'Limite L3 ({args.l3} Mo)')
        
        # Écriture scientifique partout sauf pour l'IPC
        if metric != "IPC":
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        else:
            ax.ticklabel_format(style='plain', axis='y')
            # On place la légende L3 uniquement sur le premier graphique pour épurer le reste
            ax.legend(loc='lower right')

    # 4. Labels de l'axe X (uniquement sur les graphes tout en bas de chaque colonne)
    ax_miss.set_xlabel("Taille mémoire du DataFrame (Mo)", fontsize=12, fontweight='bold')
    ax_branch.set_xlabel("Taille mémoire du DataFrame (Mo)", fontsize=12, fontweight='bold')

    # Ajustement global pour éviter les chevauchements
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    
    # Sauvegarde avec un nom qui inclut la machine
    output_filename = f"benchmark_polars_{args.op}_c{args.cpubind}m{args.membind}_{hostname}.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"✅ Graphique généré : {output_filename}")

if __name__ == "__main__":
    main()

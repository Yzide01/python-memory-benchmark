import pandas as pd
import matplotlib.pyplot as plt
import argparse
import sys
import math

def main():
    parser = argparse.ArgumentParser(description="Générateur de graphiques NUMA (Grille 2x2)")
    parser.add_argument("--csv", type=str, required=True, help="Le fichier CSV généré par collect_numa.py")
    parser.add_argument("--metric", type=str, choices=["IPC", "L1_Misses", "LLC_Misses", "Lat_ns"], required=True, help="La métrique à tracer")
    
    args = parser.parse_args()

    # Charger les données
    try:
        df = pd.read_csv(args.csv)
    except FileNotFoundError:
        print(f"Erreur : Le fichier {args.csv} n'existe pas.")
        sys.exit(1)

    sizes = df['Size_MB'].unique()
    
    # Calcul de la taille de la grille (ex: 4 tailles = 2x2, 2 tailles = 1x2)
    n_plots = len(sizes)
    cols = 2
    rows = math.ceil(n_plots / cols)
    
    fig, axs = plt.subplots(rows, cols, figsize=(15, 6 * rows), squeeze=False)
    fig.suptitle(f"{args.metric} Across NUMA Nodes", fontsize=18, y=1.02)
    
    # Pour s'assurer que les axes X soient toujours des entiers
    nodes = df['CPU_Node'].unique()

    for i, size in enumerate(sizes):
        row = i // cols
        col = i % cols
        ax = axs[row, col]
        
        df_size = df[df['Size_MB'] == size]
        
        # Courbe Séquentielle
        seq_data = df_size[df_size['Pattern'] == 'Sequential']
        ax.errorbar(seq_data['CPU_Node'], seq_data[f"{args.metric}_mean"], 
                    yerr=seq_data[f"{args.metric}_std"], 
                    label="Sequential", marker='o', capsize=5, capthick=1.5, linewidth=2, color='#1f77b4')
        
        # Courbe Random
        rnd_data = df_size[df_size['Pattern'] == 'Random']
        ax.errorbar(rnd_data['CPU_Node'], rnd_data[f"{args.metric}_mean"], 
                    yerr=rnd_data[f"{args.metric}_std"], 
                    label="Random", marker='s', capsize=5, capthick=1.5, linewidth=2, color='#ff7f0e')

        ax.set_title(f"Array Size: {size} MB", fontsize=12)
        ax.set_xlabel("Execution NUMA Node (Memory bound to Node 0)")
        ax.set_ylabel(args.metric)
        ax.set_xticks(nodes)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc='lower right' if args.metric == 'IPC' else 'center right')
        
        # Force le format plain (pas de 1e7) pour plus de lisibilité
        ax.ticklabel_format(style='plain', axis='y')

    # Si le nombre de tailles est impair, on cache la dernière case vide
    for j in range(i + 1, rows * cols):
        fig.delaxes(axs[j // cols, j % cols])

    plt.tight_layout()
    output_filename = f"plot_numa_{args.metric}_all.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"✅ Graphique généré avec succès : {output_filename}")

if __name__ == "__main__":
    main()

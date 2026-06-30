import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import platform
import sys

def main():
    parser = argparse.ArgumentParser(description="Traceur NUMA Asymétrique depuis un CSV")
    parser.add_argument("--csv", type=str, required=True, help="Chemin vers le fichier CSV")
    parser.add_argument("--l3", type=float, default=32.0, help="Taille du cache L3 (ligne rouge)")
    
    # Métadonnées pour reconstruire le titre (car le CSV ne les contient pas)
    parser.add_argument("--op", type=str, default="Inconnue", help="L'opération testée (ex: JOIN, GRPBY)")
    parser.add_argument("--cpubind", type=str, default="?", help="Le nœud CPU utilisé")
    parser.add_argument("--membind", type=str, default="?", help="Le nœud Mémoire utilisé")
    parser.add_argument("--machine", type=str, default="", help="Nom de la machine (auto-détecté si vide)")

    args = parser.parse_args()

    # 1. Chargement des données
    try:
        df = pd.read_csv(args.csv)
    except FileNotFoundError:
        print(f"❌ Erreur : Le fichier '{args.csv}' est introuvable.")
        sys.exit(1)

    # Vérification rapide des colonnes (pour s'assurer qu'elles correspondent)
    expected_cols = ["Mo", "IPC", "Cache_Refs", "Cache_Misses", "Branch_Misses", "lat_ns"]
    for col in expected_cols:
        if col not in df.columns:
            print(f"⚠️ Attention: La colonne '{col}' est absente du CSV. Le script risque de planter.")
            print(f"Colonnes trouvées: {list(df.columns)}")

    print(f"📊 Génération du graphique Asymétrique depuis {args.csv}...")

    # 2. Construction du Titre
    hostname = args.machine if args.machine else platform.node()
    min_mo = df["Mo"].min()
    max_mo = df["Mo"].max()
    
    titre_principal = (
        f"Analyse NUMA Polars | Machine : {hostname} | Opération : {args.op.upper()}\n"
        f"Topologie -> CPU-Bind: Node {args.cpubind}  |  Mem-Bind: Node {args.membind}\n"
        f"Balayage : de {min_mo} à {max_mo} Mo  |  Limite L3 : {args.l3} Mo"
    )

    # 3. Initialisation de la grille asymétrique (16x11)
    fig = plt.figure(figsize=(16, 11))
    fig.suptitle(titre_principal, fontsize=14, fontweight='bold', color='#2c3e50', y=0.98)

    gs = gridspec.GridSpec(6, 2, figure=fig)

    # --- COLONNE GAUCHE (Grands graphes) ---
    ax_lat = fig.add_subplot(gs[0:3, 0])
    ax_miss = fig.add_subplot(gs[3:6, 0])

    # --- COLONNE DROITE (Petits graphes) ---
    ax_ipc = fig.add_subplot(gs[0:2, 1])
    ax_ref = fig.add_subplot(gs[2:4, 1])
    ax_branch = fig.add_subplot(gs[4:6, 1])

    # Configuration des tracés
    plot_configs = [
        (ax_lat, "lat_ns", "Latence d'exécution", "Latence (nanosecondes)", "#9467bd"),
        (ax_miss, "Cache_Misses", "Défauts de Cache L3 (LLC Misses)", "Échecs au L3 (Nombre absolu)", "#d62728"),
        (ax_ipc, "IPC", "Instructions Par Cycle (IPC)", "Instructions / Cycle (Ratio)", "#2ca02c"),
        (ax_ref, "Cache_Refs", "Références au Cache L3 (LLC Refs)", "Accès au L3 (Nombre absolu)", "#1f77b4"),
        (ax_branch, "Branch_Misses", "Erreurs de Prédiction de Branchement", "Erreurs de saut (Nombre absolu)", "#e377c2")
    ]

    for ax, metric, title, ylabel, color in plot_configs:
        ax.plot(df["Mo"], df[metric], marker='o', linestyle='-', linewidth=2, color=color)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=10, fontweight='medium')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        ax.axvline(x=args.l3, color='red', linestyle='--', linewidth=2, label=f'Limite L3 ({args.l3} Mo)')
        
        # Format scientifique sauf pour l'IPC
        if metric != "IPC":
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        else:
            ax.ticklabel_format(style='plain', axis='y')
            ax.legend(loc='lower right')

    ax_miss.set_xlabel("Taille mémoire du DataFrame (Mo)", fontsize=12, fontweight='bold')
    ax_branch.set_xlabel("Taille mémoire du DataFrame (Mo)", fontsize=12, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    
    # 4. Sauvegarde
    output_filename = f"plot_from_csv_{args.op}_c{args.cpubind}m{args.membind}.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"✅ Terminé ! Image sauvegardée sous : {output_filename}")

if __name__ == "__main__":
    main()

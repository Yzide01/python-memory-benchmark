import polars as pl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# Paramètres
threads = [1, 2, 3, 4]
membinds = [0, 1]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

print("📊 Génération des graphiques individuels (Moyennes)...")
for m in membinds:
    for t in threads:
        csv_file = f"benchmark_polars_join_numa_streaming_t{t}_m{m}.csv"
        
        if not os.path.exists(csv_file):
            print(f"  ⚠️ Fichier introuvable : {csv_file}")
            continue
            
        df = pl.read_csv(csv_file)
        
        # AGGRÉGATION : Puisqu'on a plusieurs itérations par taille, on calcule la moyenne
        df_mean = df.group_by("Size_Mo").mean().sort("Size_Mo")
        
        # Conversion : Mémoire (Mo -> Go) et Latence (ns -> Secondes)
        size_gb = df_mean["Size_Mo"] / 1000
        lat_s = df_mean["lat_ns"] / 1_000_000_000
        
        plt.figure(figsize=(8, 5))
        plt.plot(size_gb, lat_s, marker='o', linewidth=2, color='#1f77b4')
        
        mode_str = "LOCAL" if m == 0 else "DISTANT"
        plt.title(f"Latence JOIN Streaming - {t} Thread(s) | {mode_str} (m{m})", fontsize=14, fontweight='bold')
        plt.xlabel("Taille du DataFrame (Go)", fontsize=12)
        plt.ylabel("Temps d'exécution Moyen (Secondes)", fontsize=12)
        
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.ylim(bottom=0)
        
        out_name = f"plot_latency_t{t}_m{m}.png"
        plt.tight_layout()
        plt.savefig(out_name, dpi=300)
        plt.close()
        print(f"  ✅ Graphique généré : {out_name}")

print("\n📈 Génération des graphiques comparatifs superposés...")
for m in membinds:
    plt.figure(figsize=(10, 6))
    mode_str = "LOCAL" if m == 0 else "DISTANT"
    
    for i, t in enumerate(threads):
        csv_file = f"benchmark_polars_join_numa_streaming_t{t}_m{m}.csv"
        if os.path.exists(csv_file):
            df = pl.read_csv(csv_file)
            # Aggrégation par taille
            df_mean = df.group_by("Size_Mo").mean().sort("Size_Mo")
            
            size_gb = df_mean["Size_Mo"] / 1000
            lat_s = df_mean["lat_ns"] / 1_000_000_000
            
            plt.plot(size_gb, lat_s, marker='o', linewidth=2, color=colors[i], label=f"{t} Thread(s)")
            
    plt.title(f"Comparaison Multi-thread JOIN Streaming | {mode_str} (m{m})", fontsize=14, fontweight='bold')
    plt.xlabel("Taille du DataFrame (Go)", fontsize=12)
    plt.ylabel("Temps d'exécution Moyen (Secondes)", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(bottom=0)
    
    out_name = f"plot_comparison_m{m}.png"
    plt.tight_layout()
    plt.savefig(out_name, dpi=300)
    plt.close()
    print(f"  ✅ Graphique comparatif généré : {out_name}")


print("\n📦 BONUS : Génération d'un graphe 'Boxplot' (Rectangles) pour la variance...")
# Exemple de boxplot pour observer la variance (les moustaches et quartiles) sur le thread 4 en m0
test_t = 4
test_m = 0
csv_boxplot = f"benchmark_polars_join_numa_streaming_t{test_t}_m{test_m}.csv"

if os.path.exists(csv_boxplot):
    df = pl.read_csv(csv_boxplot)
    
    # On extrait la liste des tailles uniques
    unique_sizes = df["Size_Mo"].unique().sort().to_list()
    
    # On prépare une liste de listes (chaque sous-liste contient toutes les itérations d'une taille)
    latencies_for_boxplot = []
    for size in unique_sizes:
        # On filtre par taille, on convertit la colonne latence en secondes, et on en fait une liste Python
        lats_in_seconds = (df.filter(pl.col("Size_Mo") == size)["lat_ns"] / 1_000_000_000).to_list()
        latencies_for_boxplot.append(lats_in_seconds)
    
    plt.figure(figsize=(10, 6))
    
    # Le Boxplot magique
    plt.boxplot(latencies_for_boxplot, positions=range(len(unique_sizes)), patch_artist=True, boxprops=dict(facecolor='#1f77b4', alpha=0.7))
    
    # Formatage des axes (on affiche la taille en Go en bas)
    plt.xticks(range(len(unique_sizes)), [f"{s/1000:.2f} Go" for s in unique_sizes], rotation=45)
    
    plt.title(f"Boxplot de la variance d'exécution - {test_t} Thread(s) | m{test_m}", fontsize=14, fontweight='bold')
    plt.xlabel("Taille du DataFrame", fontsize=12)
    plt.ylabel("Temps d'exécution (Secondes)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(bottom=0)
    
    out_name = f"plot_boxplot_t{test_t}_m{test_m}.png"
    plt.tight_layout()
    plt.savefig(out_name, dpi=300)
    plt.close()
    print(f"  ✅ Graphique Boxplot généré : {out_name}")

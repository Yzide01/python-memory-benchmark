import polars as pl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# Paramètres
threads = [1, 2, 3, 4]
membinds = [0, 1]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

print("📊 Génération des 8 graphiques individuels...")
for m in membinds:
    for t in threads:
        # Assure-toi que ce nom correspond exactement aux fichiers générés !
        # Si tes fichiers sont dans un sous-dossier, ajoute-le (ex: f"JOIN_STREAMING/benchmark...")
        csv_file = f"t{t}_m{m}.csv"
        
        if not os.path.exists(csv_file):
            print(f"⚠️ Fichier introuvable : {csv_file}")
            continue
            
        df = pl.read_csv(csv_file)
        
        # Conversion : Mémoire (Mo -> Go) et Latence (ns -> Secondes)
        size_gb = df["Mo"] / 1000
        lat_s = df["lat_ns"] / 1_000_000_000
        
        plt.figure(figsize=(8, 5))
        plt.plot(size_gb, lat_s, marker='o', linewidth=2, color='#1f77b4')
        
        mode_str = "LOCAL" if m == 0 else "DISTANT"
        plt.title(f"Latence JOIN Streaming - {t} Thread(s) | {mode_str} (m{m})", fontsize=14, fontweight='bold')
        plt.xlabel("Taille du DataFrame (Go)", fontsize=12)
        plt.ylabel("Temps d'exécution (Secondes)", fontsize=12)
        
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.ylim(bottom=0) # On force l'axe Y à commencer à 0 pour ne pas fausser l'échelle
        
        out_name = f"plot_latency_t{t}_m{m}.png"
        plt.tight_layout()
        plt.savefig(out_name, dpi=300)
        plt.close()
        print(f"  ✅ Graphique généré : {out_name}")

print("\n📈 Génération des 2 graphiques comparatifs superposés...")
for m in membinds:
    plt.figure(figsize=(10, 6))
    mode_str = "LOCAL" if m == 0 else "DISTANT"
    
    for i, t in enumerate(threads):
        csv_file = f"t{t}_m{m}.csv"
        if os.path.exists(csv_file):
            df = pl.read_csv(csv_file)
            size_gb = df["Mo"] / 1000
            lat_s = df["lat_ns"] / 1_000_000_000
            
            plt.plot(size_gb, lat_s, marker='o', linewidth=2, color=colors[i], label=f"{t} Thread(s)")
            
    plt.title(f"Comparaison Multi-thread JOIN Streaming | {mode_str} (m{m})", fontsize=14, fontweight='bold')
    plt.xlabel("Taille du DataFrame (Go)", fontsize=12)
    plt.ylabel("Temps d'exécution (Secondes)", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(bottom=0)
    
    out_name = f"plot_comparison_m{m}.png"
    plt.tight_layout()
    plt.savefig(out_name, dpi=300)
    plt.close()
    print(f"  ✅ Graphique comparatif généré : {out_name}")

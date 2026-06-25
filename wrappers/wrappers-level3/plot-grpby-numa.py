import subprocess, pandas as pd, matplotlib.pyplot as plt, os

WRAPPER = "wrapper-polars-groupby-numa.py"
CSV_FILE = f"summary_{WRAPPER.replace('.py', '')}.csv"

def generate_plot():
    if not os.path.exists(CSV_FILE):
        subprocess.run(["python3", "collect_results.py", WRAPPER], check=True)
    
    df = pd.read_csv(CSV_FILE)
    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    for i, m in enumerate(["IPC", "LLC_misses", "lat_ns"]):
        axs[i].plot(df["Elements"], df[m], marker='o', color='b')
        axs[i].set_title(f"Évolution de {m}"); axs[i].grid(True); axs[i].set_ylabel(m)
    plt.xlabel("Taille du tableau"); plt.tight_layout()
    plt.savefig(f"plot_{WRAPPER.replace('.py', '')}.png")
    print(f"Graphique sauvegardé : plot_{WRAPPER.replace('.py', '')}.png")

if __name__ == "__main__": generate_plot()

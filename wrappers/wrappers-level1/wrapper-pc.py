import numpy as np
import subprocess
import re
import sys

# --- La fonction de calcul isolée ---
def compute_sum(n_elements):
    # On alloue dynamiquement selon la taille demandée
    data = np.arange(n_elements, dtype=np.uint32)
    result = np.sum(data)
    sys.exit(0)

# --- Wrapper configurable ---
def run_benchmark(n_elements, iterations=10):
    # On construit la commande pour appeler le script lui-même
    # On passe la taille en argument au script
    cmd = f"perf stat -e instructions,cycles,LLC-load-misses python3 wrapper-pc.py --run-compute {n_elements}"
    
    results = []
    print(f"Lancement de {iterations} itérations pour {n_elements} éléments...")
    
    for i in range(iterations):
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        matches = re.findall(r'([\d,]+)\s+cpu_\w+/instructions/', proc.stderr)
        if matches:
            results.append(sum(int(m.replace(',', '')) for m in matches))
            
    if results:
        res = np.array(results)
        moy = np.mean(res)
        std = np.std(res)
        print(f"\n--- Résultats pour {n_elements} éléments ---")
        print(f"Moyenne (des instructions CPU) : {moy:,.0f}")
        print(f"Écart-type : {std:,.0f} ({(std/moy)*100:.2f}%)")

if __name__ == "__main__":
    # Permet de lancer le calcul ou le benchmark selon l'argument
    if len(sys.argv) > 2 and sys.argv[1] == "--run-compute":
        compute_sum(int(sys.argv[2]))
    else:
        # Ici on teste avec 16 millions d'éléments
        run_benchmark(16_000_000, 10)

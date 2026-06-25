import numpy as np
import subprocess
import re
import sys
import argparse

# --- La fonction de calcul isolée ---
def compute_sum(n_elements):
    # Allocation du tableau
    data = np.arange(n_elements, dtype=np.uint32)
    # Opération simple pour tester l'accès mémoire
    result = np.sum(data)
    sys.exit(0)

# --- Fonction de mesure ---
def run_numa_benchmark(cpu_node, mem_node, n_elements, iterations):
    # La commande construit la chaîne avec numactl + perf
    # On appelle le script lui-même avec l'argument --run-compute
    cmd = (f"numactl --cpunodebind={cpu_node} --membind={mem_node} "
           f"perf stat -e instructions,cycles,LLC-load-misses "
           f"python3 wrapper-numa.py --run-compute {n_elements}")
    
    results = []
    print(f"\n--- Benchmark: CPU Nœud {cpu_node} | RAM Nœud {mem_node} ({n_elements} élém.) ---")
    
    for i in range(iterations):
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        # On extrait le nombre total d'instructions (somme des cpu_core et cpu_atom si nécessaire)
        matches = re.findall(r'([\d,]+)\s+cpu_\w+/instructions/', proc.stderr)
        if matches:
            total = sum(int(m.replace(',', '')) for m in matches)
            results.append(total)
            
    if results:
        res = np.array(results)
        moy = np.mean(res)
        std = np.std(res)
        print(f"Moyenne (des instructions CPU) : {moy:,.0f}")
        print(f"Écart-type : {std:,.0f} ({(std/moy)*100:.2f}%)")
    else:
        print("Erreur de capture perf. Vérifie que numactl et perf sont bien installés.")

# --- Gestion des arguments ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-compute", type=int, help="Exécute le calcul interne")
    parser.add_argument("--elements", type=int, default=16_000_000, help="Nombre d'éléments")
    parser.add_argument("--iter", type=int, default=10, help="Nombre d'itérations")
    args = parser.parse_args()

    if args.run_compute:
        compute_sum(args.run_compute)
    else:
        # Ici tu peux facilement boucler sur les nœuds
        # Scénario 1: Local
        run_numa_benchmark(cpu_node=0, mem_node=0, n_elements=args.elements, iterations=args.iter)
        # Scénario 2: Distant (si ton architecture le permet)
        run_numa_benchmark(cpu_node=0, mem_node=1, n_elements=args.elements, iterations=args.iter)

import argparse
import subprocess
import sys
import os
import time
import re
import csv
import numpy as np

# ==========================================
# 1. LE WORKER (Séquentiel vs Random)
# ==========================================
def worker(n_elements, pattern):
    data = np.ones(n_elements, dtype=np.float64)
    indices = np.random.randint(0, n_elements, size=n_elements, dtype=np.int64)
    
    print(f"READY_PID:{os.getpid()}", flush=True)
    sys.stdin.readline() 
    
    t0 = time.perf_counter_ns()
    
    for _ in range(5):
        if pattern == "sequential":
            _ = data.sum() 
        else:
            _ = data[indices].sum() 
            
    t1 = time.perf_counter_ns()
    
    print(f"LATENCY_NS:{t1 - t0}", flush=True)
    sys.exit(0)

def get_perf_val(output, event_name):
    match = re.search(fr'([\d,]+)\s+{event_name}', output, re.IGNORECASE)
    return int(match.group(1).replace(',', '')) if match else 0

# ==========================================
# 2. L'ORCHESTRATEUR DE COLLECTE
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Collecteur NUMA (Séquentiel vs Random)")
    parser.add_argument("--membind", type=str, default="0", help="Nœud mémoire fixe")
    parser.add_argument("--num_nodes", type=int, default=8, help="Nombre de nœuds CPU à tester")
    parser.add_argument("--sizes", type=float, nargs='+', default=[7.6, 76.3, 381.5, 762.9], help="Tailles en Mo")
    parser.add_argument("--iter", type=int, default=3, help="Itérations pour l'écart-type")
    
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--elements", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--pattern", type=str, choices=["sequential", "random"], help=argparse.SUPPRESS)
    
    args = parser.parse_args()

    if args.worker:
        worker(args.elements, args.pattern)
        return

    print(f"\n--- DÉMARRAGE DE LA COLLECTE NUMA | Membind: {args.membind} ---")
    
    # Compteurs génériques Linux (Compatibles AMD EPYC)
    events = "instructions,cycles,cache-references,cache-misses"
    bytes_per_row = 16 

    csv_filename = f"results_numa_mem{args.membind}.csv"
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Size_MB", "CPU_Node", "Pattern", "IPC_mean", "IPC_std", 
                         "Cache_Refs_mean", "Cache_Refs_std", "Cache_Misses_mean", "Cache_Misses_std", 
                         "Lat_ns_mean", "Lat_ns_std"])

        try:
            for size_mo in args.sizes:
                n_elements = int((size_mo * 1024 * 1024) / bytes_per_row)
                print(f"\n>>> Test de la taille : {size_mo} Mo ({n_elements:,} éléments)")

                for pattern in ["Sequential", "Random"]:
                    for cpu_node in range(args.num_nodes):
                        print(f"  -> Pattern: {pattern:<10} | CPU_Node: {cpu_node} ...", end=" ", flush=True)
                        
                        data_tmp = {"ipc": [], "refs": [], "misses": [], "lat": []}
                        
                        for _ in range(args.iter):
                            cmd_worker = f"sudo env LC_ALL=C numactl --cpubind={cpu_node} --membind={args.membind} {sys.executable} {__file__} --worker --elements {n_elements} --pattern {pattern.lower()}"
                            
                            worker_proc = subprocess.Popen(cmd_worker, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                            
                            worker_pid = None
                            while True:
                                line = worker_proc.stdout.readline()
                                if not line: break
                                if line.startswith("READY_PID:"):
                                    worker_pid = int(line.strip().split(":")[1])
                                    break
                                    
                            if not worker_pid: continue

                            cmd_perf = f"sudo env LC_ALL=C perf stat -e {events} -p {worker_pid}"
                            perf_proc = subprocess.Popen(cmd_perf, shell=True, stderr=subprocess.PIPE, text=True)
                            
                            time.sleep(0.5) 
                            worker_proc.stdin.write("GO\n")
                            worker_proc.stdin.flush()
                            
                            worker_out, _ = worker_proc.communicate()
                            _, perf_err = perf_proc.communicate()
                            
                            lat_ns = 0
                            for line in worker_out.split('\n'):
                                if line.startswith("LATENCY_NS:"):
                                    lat_ns = float(line.strip().split(":")[1])

                            cycl = get_perf_val(perf_err, "cycles")

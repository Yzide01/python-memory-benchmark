import argparse
import os
import signal
import subprocess
import sys
import csv
import logging

# Configuration du logging
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
LOG.addHandler(console_handler)

perf_process = None
worker_pid = None
perf_stderr_output = ""

EVENTS_2_FIELDS = {
    "cycles": "cycles",
    "instructions": "instructions",
    "L1-dcache-load-misses": "L1_misses",
    "cache-misses": "LLC_misses"
}

def start_perf_monitoring(signum, frame):
    global perf_process, worker_pid
    if worker_pid is not None and perf_process is None:
        events = list(EVENTS_2_FIELDS.keys())
        perf_cmd = [
            "perf", "stat", 
            "-e", ",".join(events), 
            "-p", str(worker_pid), 
            "-x", "|"
        ]
        perf_process = subprocess.Popen(perf_cmd, stderr=subprocess.PIPE, text=True)

def stop_perf_monitoring(signum, frame):
    global perf_process, perf_stderr_output
    if perf_process is not None:
        perf_process.send_signal(signal.SIGINT)
        _, perf_stderr_output = perf_process.communicate()
        perf_process = None

def run_experiment():
    signal.signal(signal.SIGUSR1, start_perf_monitoring)
    signal.signal(signal.SIGUSR2, stop_perf_monitoring)

    nodes = list(range(2))  # Tester sur NUMA 0 à 7
    sizes = [1_000_000, 10_000_000, 50_000_000, 100_000_000]
    modes = ["seq", "rand"]
    benchmark_cycles = 10
    
    all_results = []
    
    for node in nodes:
        for size in sizes:
            mb_size = (size * 8) / (1024 * 1024)
            for mode in modes:
                global worker_pid, perf_stderr_output
                perf_stderr_output = ""
                
                LOG.info(f"Running Node: {node} | Size: {mb_size:.1f} MB | Mode: {mode}")

                cmd = [
                    "numactl", f"--cpunodebind={node}", "--membind=0",
                    sys.executable, "np_worker.py", 
                    "--size", str(size), 
                    "--mode", mode, 
                    "--cycles", str(benchmark_cycles), 
                    "--signals"
                ]
                
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                worker_pid = proc.pid
                
                captured_stdout = []
                for line in iter(proc.stdout.readline, ""):
                    if line.startswith("PRINT:"):
                        captured_stdout.append(line.strip())
                
                proc.wait()
                
                # Failsafe si le processus a planté
                if perf_process and perf_process.poll() is None:
                    stop_perf_monitoring(None, None)

                # --- PARSING STDOUT DU WORKER ---
                total_compute_time = 0.0
                for line in captured_stdout:
                    if "Compute_Time_s:" in line:
                        total_compute_time = float(line.split(":")[2].strip())
                
                avg_time_s = total_compute_time / benchmark_cycles if benchmark_cycles > 0 else 0
                avg_bw_mb_s = mb_size / avg_time_s if avg_time_s > 0 else 0

                fields = {
                    "numa_node": node,
                    "size_elements": size,
                    "size_MB": mb_size,
                    "mode": mode,
                    "time_s_mean": avg_time_s,
                    "bandwidth_MB_s_mean": avg_bw_mb_s,
                    "IPC_mean": 0,
                }
                
                # Initialisation des compteurs
                for f in EVENTS_2_FIELDS.values():
                    fields[f"{f}_mean"] = 0

                # --- PARSING STDERR DE PERF ---
                for line in perf_stderr_output.splitlines():
                    parts = line.strip().split("|")
                    if len(parts) < 3:
                        continue

                    try:
                        val_str = parts[0].replace(",", "").replace(" ", "")
                        if not val_str or val_str == "<not":
                            continue

                        val = float(val_str)
                        event = parts[2].lower()

                        for event_name, field_name in EVENTS_2_FIELDS.items():
                            if event_name.lower() in event:
                                # Normalisation : On divise par le nombre de cycles
                                fields[f"{field_name}_mean"] = val / benchmark_cycles
                                break
                    except ValueError:
                        continue

                # Calcul de l'IPC (Instructions Per Cycle)
                inst = fields.get("instructions_mean", 0)
                cyc = fields.get("cycles_mean", 0)
                fields["IPC_mean"] = inst / cyc if cyc > 0 else 0
                
                # Ajout de champs factices d'écart-type (std) pour rester compatible avec ton script de plot !
                fields["bandwidth_MB_s_std"] = 0
                fields["IPC_std"] = 0
                fields["L1_misses_std"] = 0
                fields["LLC_misses_std"] = 0

                all_results.append(fields)

    # --- SAUVEGARDE CSV ---
    csv_file = "numa_signals_benchmark_results.csv"
    if all_results:
        with open(csv_file, mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
        LOG.info(f"Sauvegarde terminée : {csv_file}")
    else:
        LOG.error("Aucun résultat obtenu.")

if __name__ == "__main__":
    run_experiment()

import argparse
import os
import signal
import subprocess
import sys
import csv
import logging
import numpy as np

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
LOG.addHandler(console_handler)

perf_process = None
worker_pid = None
perf_stderr_all = []
current_perf_output = ""

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
    global perf_process, current_perf_output, perf_stderr_all
    if perf_process is not None:
        perf_process.send_signal(signal.SIGINT)
        _, current_perf_output = perf_process.communicate()
        perf_stderr_all.append(current_perf_output)
        perf_process = None

def run_experiment():
    signal.signal(signal.SIGUSR1, start_perf_monitoring)
    signal.signal(signal.SIGUSR2, stop_perf_monitoring)

    nodes = list(range(8))
    sizes = [1_000_000, 10_000_000, 50_000_000, 100_000_000]
    modes = ["seq", "rand"]
    benchmark_cycles = 10
    
    all_results = []
    
    for node in nodes:
        for size in sizes:
            mb_size = (size * 8) / (1024 * 1024)
            for mode in modes:
                global worker_pid, perf_stderr_all
                perf_stderr_all = []
                
                LOG.info(f"Running Node: {node} | Size: {mb_size:.1f} MB | Mode: {mode}")

                cmd = [
                    "numactl", f"--cpunodebind={node}", "--membind=0",
                    "python3.14", "np_worker.py", 
                    "--size", str(size), 
                    "--mode", mode, 
                    "--cycles", str(benchmark_cycles)
                ]
                
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                worker_pid = proc.pid
                
                time_array = []
                for line in iter(proc.stdout.readline, ""):
                    if line.startswith("PRINT: Time_s:"):
                        time_str = line.split("Time_s:")[1].strip()
                        time_array = [float(t) for t in time_str.split(",")]
                
                proc.wait()
                
                if perf_process and perf_process.poll() is None:
                    stop_perf_monitoring(None, None)

                run_bws = []
                run_ipcs = []
                run_l1s = []
                run_llcs = []

                for i, perf_out in enumerate(perf_stderr_all):
                    t_s = time_array[i] if i < len(time_array) else 0
                    if t_s > 0:
                        run_bws.append(mb_size / t_s)

                    run_fields = {f: 0 for f in EVENTS_2_FIELDS.values()}

                    for line in perf_out.splitlines():
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
                                    run_fields[field_name] = val
                                    break
                        except ValueError:
                            continue
                            
                    inst = run_fields.get("instructions", 0)
                    cyc = run_fields.get("cycles", 0)
                    if cyc > 0:
                        run_ipcs.append(inst / cyc)
                    
                    run_l1s.append(run_fields.get("L1_misses", 0))
                    run_llcs.append(run_fields.get("LLC_misses", 0))

                fields = {
                    "numa_node": node,
                    "size_elements": size,
                    "size_MB": mb_size,
                    "mode": mode,
                    "time_s_mean": np.mean(time_array) if time_array else 0,
                    "time_s_std": np.std(time_array) if time_array else 0,
                    "bandwidth_MB_s_mean": np.mean(run_bws) if run_bws else 0,
                    "bandwidth_MB_s_std": np.std(run_bws) if run_bws else 0,
                    "IPC_mean": np.mean(run_ipcs) if run_ipcs else 0,
                    "IPC_std": np.std(run_ipcs) if run_ipcs else 0,
                    "L1_misses_mean": np.mean(run_l1s) if run_l1s else 0,
                    "L1_misses_std": np.std(run_l1s) if run_l1s else 0,
                    "LLC_misses_mean": np.mean(run_llcs) if run_llcs else 0,
                    "LLC_misses_std": np.std(run_llcs) if run_llcs else 0,
                }

                all_results.append(fields)

    csv_file = "numa_signals_benchmark_results.csv"
    if all_results:
        with open(csv_file, mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
        LOG.info(f"Done: {csv_file}")

if __name__ == "__main__":
    run_experiment()

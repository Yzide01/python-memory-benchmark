import sys
import os

temp_threads = "1"
if "-t" in sys.argv:
    temp_threads = sys.argv[sys.argv.index("-t") + 1]
elif "--threads" in sys.argv:
    temp_threads = sys.argv[sys.argv.index("--threads") + 1]

os.environ["POLARS_MAX_THREADS"] = temp_threads
os.environ["RAYON_NUM_THREADS"] = temp_threads

import argparse
import subprocess
import signal
import time
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import gc

perf_process = None
worker_pid = None
perf_stderr_output = ""

# PERF EVENTS
EVENTS = "instructions,cycles,L1-dcache-load-misses,cache-misses,dTLB-load-misses,branch-misses"

def start_perf_monitoring(signum, frame):
    global perf_process, worker_pid
    if worker_pid is not None and perf_process is None:
        perf_cmd = [
            "perf", "stat", 
            "-e", EVENTS, 
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

def pin_polars_threads(start_core: int) -> None:
    pid = os.getpid()
    task_dir = f"/proc/{pid}/task"
    current_polars_core = start_core

    try:
        tids = os.listdir(task_dir)
    except FileNotFoundError:
        print(f"  [Worker] Erreur : Dossier {task_dir} introuvable.")
        return

    for tid_str in tids:
        tid = int(tid_str)
        if tid == pid:
            thread_name = "main_python_thread"
        else:
            try:
                with open(f"{task_dir}/{tid}/comm", "r") as f:
                    thread_name = f.read().strip()
            except FileNotFoundError:
                continue

        # On cherche uniquement les sous-threads de calcul Polars
        if "polars" in thread_name and "polars-ooc-clea" not in thread_name:
            target_core = str(current_polars_core)
            print(f"  [Worker] Épinglage du thread '{thread_name}' (TID {tid}) sur le cœur {target_core}")
            current_polars_core += 1

            try:
                subprocess.run(
                    ["taskset", "-p", "-c", target_core, str(tid)],
                    check=True,
                    stdout=subprocess.DEVNULL,
                )
            except Exception as e:
                print(f"  [Worker] Erreur d'épinglage : {e}")

# Worker called by the orchestrator 
# Does the benchmark, sends signal to start perf monitoring as soon as needed
def worker(op, n_elements, streaming, start_core):
    PARENT_PID = os.getppid()
    
    # disable garbage collector before
    gc.disable()

    if start_core is not None:
        pin_polars_threads(start_core)

    # Étape 1 : Génération des données (NON mesurée par perf)


#    if op == "grpby":
#        df = pl.DataFrame({
#            "data": np.random.randint(0, 1000, size=n_elements, dtype=np.uint32),
#            "groups": np.random.randint(0, 100, size=n_elements, dtype=np.uint32)
#        }).lazy()
#        query = df.group_by("groups").agg(pl.col("data").sum())
#        
#    elif op == "join":
#        df1 = pl.DataFrame({
#            "key": np.random.randint(0, n_elements // 10, size=n_elements, dtype=np.uint32),
#            "val1": np.random.randint(0, 100, size=n_elements, dtype=np.uint32)
#        }).lazy()
#        df2 = pl.DataFrame({
#            "key": np.random.randint(0, n_elements // 10, size=n_elements, dtype=np.uint32),
#            "val2": np.random.randint(0, 100, size=n_elements, dtype=np.uint32)
#        }).lazy()
#        query = df1.join(df2, on="key")

    # Copy pasted from benchmark-v2.py
        
    elements = n_elements
    if op == "grpby":
        df = pl.DataFrame(
            {
                "a": pl.int_range(0, elements, dtype=pl.UInt32, eager=True),
                "key": pl.int_range(0, elements, dtype=pl.UInt32, eager=True) % 100,
            }
        )
        # sort dataframe by key
        df = df.sort("key")
    else:
        elements = elements // 2
        df = pl.DataFrame(
            {
                "a": pl.int_range(0, elements, dtype=pl.UInt32, eager=True),
                "key": pl.int_range(0, elements, dtype=pl.UInt32, eager=True),
            }
        )
        df2 = pl.DataFrame(
            {
                "b": pl.int_range(0, elements, dtype=pl.UInt32, eager=True),
                "key": pl.int_range(0, elements, dtype=pl.UInt32, eager=True),
            }
        )
        df = df.sort("key")
        df2 = df2.sort("key")

    if not df["key"].is_sorted():
        print("DataFrame is not sorted as expected", extra={"tag": "BENCHMARK"})
        # check where the issue is
        print("df a sorted: %s, df key sorted: %s",df["a"].is_sorted(), df["key"].is_sorted())
    #else:
        #print("data is sorted")
    # Étape 2 : Signal de départ pour perf
    os.kill(PARENT_PID, signal.SIGUSR1)
    time.sleep(0.5) # Le temps que perf s'attache
    
    # Étape 3 : Exécution du calcul pur (Eager vs Streaming)
    if streaming:
        pl.Config.set_engine_affinity("streaming")

    if op == "grpby":
        t0 = time.perf_counter()
        df.group_by("key").sum()
        #df.lazy().group_by("key").agg(pl.col("a").sum()).collect()
        t1 = time.perf_counter()
    elif op == "join":
        t0 = time.perf_counter()
        df.join(df2, on="key", how="left").sum()
        #df.lazy().join(df2.lazy(), on="key", how="left").collect()
        t1 = time.perf_counter()
    
    # Étape 4 : Signal de fin
    os.kill(PARENT_PID, signal.SIGUSR2)
    time.sleep(0.5)
    
    print(f"PRINT: Time_s: {t1-t0}", flush=True)

    gc.collect()
    gc.enable()

    sys.exit(0)

# orchestrator (calls the worker)
def main():
    parser = argparse.ArgumentParser(description="Benchmark NUMA Avancé pour Polars")
    parser.add_argument("--mode", type=str, choices=["local", "numa"], default="local")
    parser.add_argument("--cpubind", type=str, default="0")
    parser.add_argument("--membind", type=str, default="0")
    parser.add_argument("--start_core", type=int, default=None, help="Cœur de départ pour le thread pinning individuel")
    parser.add_argument("--physcpubind", type=str, default=None, help="Coeurs physiques isolés (ex: '28-31' ou '60,61,62,63')")
    parser.add_argument("--op", type=str, choices=["join", "grpby"], default="grpby")
    parser.add_argument("--l3", type=float, default=32.0) # (sopnode f1 is 32, sopnode f3 is 60)
    parser.add_argument("--iter", type=int, default=3)
    
    parser.add_argument("--min_mo", type=float, default=8.0)
    parser.add_argument("--max_mo", type=float, default=60.0)
    parser.add_argument("--step_mo", type=float, default=2.0)
    
    parser.add_argument("--streaming", action="store_true", help="Active le streaming engine de Polars")
    
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--elements", type=int, help=argparse.SUPPRESS)
    parser.add_argument("-t", "--threads", type=str, default="1", help="Nombre de threads Polars")
    
    args = parser.parse_args()

    if args.worker:
        worker(args.op, args.elements, args.streaming, args.start_core)
        return

    # Enregistrement des signaux
    signal.signal(signal.SIGUSR1, start_perf_monitoring)
    signal.signal(signal.SIGUSR2, stop_perf_monitoring)

    stream_str = "AVEC STREAMING" if args.streaming else "SANS STREAMING"
    print(f"\n--- Démarrage Benchmark POLARS | Mode: {args.mode.upper()} | {stream_str} ---")
    
    bytes_per_row = 8 if args.op == "grpby" else 16
    sizes = []
    current_mo = args.min_mo
    while current_mo <= args.max_mo:
        sizes.append(int((current_mo * 1024 * 1024) / bytes_per_row))
        current_mo += args.step_mo

    results = {"Mo": [], "IPC": [], "L1_Misses": [], "LLC_Misses": [], "dTLB_Misses": [], "Branch_Misses": [], "lat_ns": []}
    
    for n in sizes:
        size_mo = (n * bytes_per_row) / (1024 * 1024)
        print(f"Test avec {n:,} éléments (~{size_mo:.1f} Mo)...")

# Choix dynamique du paramètre CPU
        cmd_base = [sys.executable, __file__, "--worker", "--op", args.op, "--elements", str(n), "--threads", str(args.threads)]
        
        # On passe le start_core au worker s'il est fourni
        if args.start_core is not None:
            cmd_base.extend(["--start_core", str(args.start_core)])

        if args.mode == "local":
            cores = args.physcpubind if args.physcpubind else "0"
            cmd = ["taskset", "-c", cores] + cmd_base
        else:
            if args.physcpubind:
                cmd = ["taskset", "-c", args.physcpubind, "numactl", f"--membind={args.membind}"] + cmd_base
            else:
                cmd = ["numactl", f"--cpubind={args.cpubind}", f"--membind={args.membind}"] + cmd_base
          
        if args.streaming:
            cmd.append("--streaming")
            
        data_tmp = {"ipc": [], "l1": [], "llc": [], "dtlb": [], "branch": [], "lat": []}
        
        for _ in range(args.iter):
            global worker_pid, perf_stderr_output
            perf_stderr_output = ""
            
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            worker_pid = proc.pid
            
            time_s = 0.0
            for line in iter(proc.stdout.readline, ""):
                if line.startswith("PRINT: Time_s:"):
                    time_s = float(line.split(":")[2].strip())
                else:
                    print("  [Worker] " + line.strip(), flush=True)
            
            proc.wait()
            
            if perf_process and perf_process.poll() is None:
                stop_perf_monitoring(None, None)
                
            # Extraction des données Perf
            run_fields = {}
            for line in perf_stderr_output.splitlines():
                parts = line.strip().split("|")
                if len(parts) < 3: continue
                try:
                    val_str = parts[0].replace(",", "").replace(" ", "")
                    if not val_str or val_str == "<not": continue
                    val = float(val_str)
                    event = parts[2].lower()
                    if "instructions" in event: run_fields["instructions"] = val
                    elif "cycles" in event: run_fields["cycles"] = val
                    elif "l1-dcache-load-misses" in event: run_fields["l1"] = val
                    elif "cache-misses" in event: run_fields["llc"] = val
                    elif "dtlb-load-misses" in event: run_fields["dtlb"] = val
                    elif "branch-misses" in event: run_fields["branch"] = val
                except ValueError:
                    continue
            
            cycl = run_fields.get("cycles", 0)
            inst = run_fields.get("instructions", 0)
            
            data_tmp["ipc"].append(inst / cycl if cycl > 0 else 0)
            data_tmp["l1"].append(run_fields.get("l1", 0))
            data_tmp["llc"].append(run_fields.get("llc", 0))
            data_tmp["dtlb"].append(run_fields.get("dtlb", 0))
            data_tmp["branch"].append(run_fields.get("branch", 0))
            data_tmp["lat"].append(time_s * 1_000_000_000)

        results["Mo"].append(size_mo)
        results["IPC"].append(np.mean(data_tmp["ipc"]))
        results["L1_Misses"].append(np.mean(data_tmp["l1"]))
        results["LLC_Misses"].append(np.mean(data_tmp["llc"]))
        results["dTLB_Misses"].append(np.mean(data_tmp["dtlb"]))
        results["Branch_Misses"].append(np.mean(data_tmp["branch"]))
        results["lat_ns"].append(np.mean(data_tmp["lat"]))

    # CSV SAVING + GRAPHS
    stream_suffix = "streaming" if args.streaming else "eager"
    csv_filename = f"benchmark_polars_{args.op}_{args.mode}_{stream_suffix}_t{args.threads}_m{args.membind}.csv"
    pl.DataFrame(results).write_csv(csv_filename)
    print(f"\nDonnées exportées avec succès dans : {csv_filename}")

    print("Génération des graphiques...")
    fig, axs = plt.subplots(6, 1, figsize=(12, 26), sharex=True)
    metrics = ["IPC", "L1_Misses", "LLC_Misses", "dTLB_Misses", "Branch_Misses", "lat_ns"]
    colors = ['#2ca02c', '#1f77b4', '#d62728', '#8c564b', '#e377c2', '#9467bd']

    for i, m in enumerate(metrics):
        axs[i].plot(results["Mo"], results[m], marker='o', linestyle='-', linewidth=2, color=colors[i])
        axs[i].set_title(f"Évolution de {m}", fontsize=12, fontweight='bold')
        axs[i].set_ylabel(m)
        axs[i].grid(True, linestyle='--', alpha=0.7)
        axs[i].ticklabel_format(style='plain', axis='y')
        axs[i].axvline(x=args.l3, color='red', linestyle='--', linewidth=2, label=f'Limite L3 ({args.l3} Mo)')
        if i == 0: axs[i].legend()

    axs[-1].set_xlabel("Taille mémoire du DataFrame (Mo)", fontsize=12, fontweight='bold')
    axs[-1].ticklabel_format(style='plain', axis='x')

    plt.tight_layout()
    plt.savefig(f"benchmark_polars_{args.op}_{args.mode}_{stream_suffix}_m{args.membind}_c{args.cpubind}_c{args.threads}.png", dpi=300)
    print("Terminé !")

if __name__ == "__main__":
    main()

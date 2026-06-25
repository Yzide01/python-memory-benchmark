import argparse
import numpy as np
import time
import sys
import subprocess
import csv

def run_compute(size, mode):
    x = np.ones(size, dtype=np.float64)
    if mode == "rand":
        indices = np.random.permutation(size)
        t0 = time.perf_counter()
        res = np.sum(x[indices])
        t1 = time.perf_counter()
    else:
        t0 = time.perf_counter()
        res = np.sum(x)
        t1 = time.perf_counter()
    print(f"COMPUTE_TIME:{t1-t0:.6f}")
    sys.exit(0)

def run_numa_benchmark():
    nodes = list(range(8))
    sizes = [1_000_000, 10_000_000, 50_000_000, 100_000_000]
    modes = ["seq", "rand"]
    iterations = 10
    events = "instructions,cycles,L1-dcache-load-misses,LLC-load-misses"
    results = []
    
    print(f"{'Node':<4} | {'Size':<10} | {'Mode':<4} | {'Mo/s':<8} | {'± Std':<7} | {'IPC':<5} | {'± Std':<6} | {'LLC Misses':<12} | {'± Std':<10}")
    print("-" * 95)

    for node in nodes:
        for size in sizes:
            mb_size = (size * 8) / (1024 * 1024)
            for mode in modes:
                times = []
                insts = []
                cycles = []
                l1_misses = []
                llc_misses = []
                
                for _ in range(iterations):
                    cmd = [
                        "numactl", f"--cpunodebind={node}", "--membind=0",
                        "perf", "stat", "-x", ",", "-e", events, 
                        sys.executable, __file__, "--run", str(size), "--mode", mode
                    ]
                    proc = subprocess.run(cmd, capture_output=True, text=True)
                    
                    cur_inst = cur_cycles = cur_l1 = cur_llc = 0
                    
                    for line in proc.stdout.split('\n'):
                        if line.startswith("COMPUTE_TIME:"):
                            times.append(float(line.split(":")[1]))
                    
                    for line in proc.stderr.split('\n'):
                        if not line: continue
                        parts = line.split(',')
                        if len(parts) >= 3 and parts[0].isdigit():
                            val = int(parts[0])
                            event_name = parts[2]
                            if "instructions" in event_name: cur_inst = val
                            elif "cycles" in event_name: cur_cycles = val
                            elif "L1-dcache-load-misses" in event_name: cur_l1 = val
                            elif "LLC-load-misses" in event_name: cur_llc = val
                    
                    insts.append(cur_inst)
                    cycles.append(cur_cycles)
                    l1_misses.append(cur_l1)
                    llc_misses.append(cur_llc)
                
                avg_time = np.mean(times) if times else 0
                bw_arr = [mb_size / t for t in times] if times else [0]
                avg_bw = np.mean(bw_arr)
                std_bw = np.std(bw_arr)
                
                ipc_arr = [i / c if c > 0 else 0 for i, c in zip(insts, cycles)]
                avg_ipc = np.mean(ipc_arr)
                std_ipc = np.std(ipc_arr)
                
                avg_l1 = np.mean(l1_misses)
                std_l1 = np.std(l1_misses)
                avg_llc = np.mean(llc_misses)
                std_llc = np.std(llc_misses)
                
                results.append({
                    "numa_node": node,
                    "size_elements": size,
                    "size_MB": mb_size,
                    "mode": mode,
                    "time_s_mean": avg_time,
                    "bandwidth_MB_s_mean": avg_bw,
                    "bandwidth_MB_s_std": std_bw,
                    "IPC_mean": avg_ipc,
                    "IPC_std": std_ipc,
                    "L1_misses_mean": avg_l1,
                    "L1_misses_std": std_l1,
                    "LLC_misses_mean": avg_llc,
                    "LLC_misses_std": std_llc
                })
                
                print(f"{node:<4} | {size:<10} | {mode:<4} | {avg_bw:<8.1f} | {std_bw:<7.1f} | {avg_ipc:<5.2f} | {std_ipc:<6.3f} | {avg_llc:<12.0f} | {std_llc:<10.0f}")

    csv_file = "numa_cache_benchmark_results_std.csv"
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=int)
    parser.add_argument("--mode", type=str, choices=["seq", "rand"], default="seq")
    args = parser.parse_args()

    if args.run:
        run_compute(args.run, args.mode)
    else:
        run_numa_benchmark()

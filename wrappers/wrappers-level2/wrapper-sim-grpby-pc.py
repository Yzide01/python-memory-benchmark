import numpy as np
import subprocess, re, sys, argparse

def compute_groupby(n_elements):
    data = np.arange(n_elements, dtype=np.uint32)
    groups = data % 100
    idx = np.argsort(groups)
    result = np.add.reduceat(data[idx], np.unique(groups[idx], return_index=True)[1])
    sys.exit(0)

def run_benchmark(n, iters):
    cmd = f"perf stat -e instructions,cycles,LLC-load-misses python3 wrapper-sim-grpby-pc.py --run {n}"
    ipc_results = []
    print(f"Lancement Groupby PC ({n} éléments, {iters} itérations)...")
    for _ in range(iters):
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        instr = re.findall(r'([\d,]+)\s+cpu_\w+/instructions/', proc.stderr)
        cycl = re.findall(r'([\d,]+)\s+cpu_\w+/cycles/', proc.stderr)
        if instr and cycl:
            total_i = sum(int(x.replace(',', '')) for x in instr)
            total_c = sum(int(x.replace(',', '')) for x in cycl)
            if total_c > 0: ipc_results.append(total_i / total_c)
    
    res = np.array(ipc_results)
    print(f"IPC Moyen : {np.mean(res):.3f} | Std : {np.std(res):.4f} ({(np.std(res)/np.mean(res))*100:.2f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=int)
    parser.add_argument("--elements", type=int, default=16_000_000)
    parser.add_argument("--iter", type=int, default=10)
    args = parser.parse_args()
    if args.run: compute_groupby(args.run)
    else: run_benchmark(args.elements, args.iter)

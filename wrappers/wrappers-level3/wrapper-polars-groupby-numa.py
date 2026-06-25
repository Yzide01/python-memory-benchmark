import polars as pl
import numpy as np
import subprocess, re, sys, argparse, os

os.environ["POLARS_MAX_THREADS"] = "1"

def compute_groupby(n_elements):
    df = pl.DataFrame({
        "data": np.random.randint(0, 1000, size=n_elements, dtype=np.uint32),
        "groups": np.random.randint(0, 100, size=n_elements, dtype=np.uint32)
    })
    result = df.group_by("groups").agg(pl.col("data").sum())
    sys.exit(0)

def print_table(metrics, label):
    print(f"\n--- Résultats : {label} ---")
    print(f"{'Metric':<15} | {'Moyenne':<15} | {'Std':<15} | {'Var (%)':<8}")
    print("-" * 60)
    for name, res in metrics.items():
        res = np.array(res)
        moy = np.mean(res); std = np.std(res)
        perc = (std / moy) * 100 if moy != 0 else 0
        val_fmt = f"{moy:.3f}" if name == "IPC" else f"{moy:,.0f}"
        print(f"{name:<15} | {val_fmt:<15} | {std:<15.2f} | {perc:<8.2f}")

def run_numa(n, iters):
    cmd = f"numactl --cpunodebind=0 --membind=0 perf stat -e instructions,cycles,LLC-load-misses python3 wrapper-polars-groupby-numa.py --run {n}"
    data = {"Instructions": [], "Cycles": [], "IPC": [], "LLC_misses": [], "lat_ns": []}
    print(f"Lancement Polars GroupBy NUMA ({n} éléments)...")
    for _ in range(iters):
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        instr = re.findall(r'([\d,]+)\s+\S*instructions', proc.stderr)
        cycl = re.findall(r'([\d,]+)\s+\S*cycles', proc.stderr)
        llc = re.findall(r'([\d,]+)\s+\S*LLC-load-misses', proc.stderr)
        time = re.search(r'([\d\.]+)\s+seconds time elapsed', proc.stderr)
        if instr and cycl:
            val_i = sum(int(x.replace(',', '')) for x in instr)
            val_c = sum(int(x.replace(',', '')) for x in cycl)
            data["Instructions"].append(val_i); data["Cycles"].append(val_c)
            if val_c > 0: data["IPC"].append(val_i / val_c)
        if llc: data["LLC_misses"].append(sum(int(x.replace(',', '')) for x in llc))
        if time: data["lat_ns"].append(float(time.group(1)) * 1_000_000_000)
    print_table(data, "Polars GroupBy NUMA")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--run", type=int); parser.add_argument("--elements", type=int, default=16_000_000); parser.add_argument("--iter", type=int, default=5)
    args = parser.parse_args()
    if args.run: compute_groupby(args.run)
    else: run_numa(args.elements, args.iter)

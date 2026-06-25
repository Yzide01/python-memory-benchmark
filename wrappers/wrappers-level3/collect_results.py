import subprocess, csv, re, sys, os

def collect(script_name):
    sizes = list(range(5_000_000, 105_000_000, 5_000_000))
    csv_file = f"summary_{script_name.replace('.py', '')}.csv"
    
    if not os.path.exists(csv_file):
        with open(csv_file, "w", newline='') as f:
            csv.writer(f).writerow(["Elements", "Instructions", "Cycles", "IPC", "LLC_misses", "lat_ns"])

    for n in sizes:
        print(f"Collecte {script_name} pour n={n}...")
        proc = subprocess.run(f"python3 {script_name} --elements {n} --iter 5", shell=True, capture_output=True, text=True)
        
        # Regex robuste pour capturer les valeurs du tableau console
        def get_v(name):
            m = re.search(fr"{name}.*?\|\s+([\d,.]+)", proc.stdout)
            return float(m.group(1).replace(',', '')) if m else 0

        with open(csv_file, "a", newline='') as f:
            csv.writer(f).writerow([n, get_v("Instructions"), get_v("Cycles"), get_v("IPC"), get_v("LLC_misses"), get_v("lat_ns")])

if __name__ == "__main__": collect(sys.argv[1])

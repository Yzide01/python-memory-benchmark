import polars as pl
import time
import gc

def micro_bench_chunks(size_gb=2.0, n_chunks=100):
    # 4 octets pour 'a' + 4 octets pour 'key' = 8 octets par ligne
    bytes_per_row = 8 
    elements = int((size_gb * 1024 * 1024 * 1024) / bytes_per_row)
    
    print(f"\n=========================================================")
    print(f" 📊 MICRO-BENCHMARK: {size_gb} Go ({elements:,} lignes) | {n_chunks} Blocs")
    print(f"=========================================================")
    
    # Génération et tri (pour correspondre exactement à ton cas réel)
    df = pl.DataFrame({
        "a": pl.int_range(0, elements, dtype=pl.UInt32, eager=True),
        "key": pl.int_range(0, elements, dtype=pl.UInt32, eager=True) % n_chunks
    }).sort("key")
    
    # ---------------------------------------------------------
    # TEST 1 : Somme globale (Le processeur lit tout d'un coup)
    # ---------------------------------------------------------
    gc.collect()
    t0 = time.perf_counter()
    _ = df["a"].sum()
    t1 = time.perf_counter()
    print(f"1. Somme globale (1 seul bloc)        : {t1-t0:.5f} secondes")
    
    # ---------------------------------------------------------
    # TEST 2 : Somme fragmentée (n * sum(total/n))
    # ---------------------------------------------------------
    gc.collect()
    chunk_size = elements // n_chunks
    t0 = time.perf_counter()
    
    res_chunks = 0
    for i in range(n_chunks):
        # On découpe physiquement le tableau en n morceaux
        chunk = df.slice(i * chunk_size, chunk_size)
        res_chunks += chunk["a"].sum()
        
    t1 = time.perf_counter()
    print(f"2. Sommes manuelles ({n_chunks} blocs)        : {t1-t0:.5f} secondes")

    # ---------------------------------------------------------
    # TEST 3 : Somme via GroupBy (La méthode de ton orchestrateur)
    # ---------------------------------------------------------
    gc.collect()
    t0 = time.perf_counter()
    _ = df.group_by("key").sum()
    t1 = time.perf_counter()
    print(f"3. Somme GroupBy (modulo {n_chunks})          : {t1-t0:.5f} secondes")


if __name__ == "__main__":
    # Test avec la bascule critique observée (6 Go vs 8 Go)
    # Tu peux modifier n_chunks pour tester modulo 2 vs modulo 100
    micro_bench_chunks(size_gb=6.0, n_chunks=100)
    micro_bench_chunks(size_gb=8.0, n_chunks=100)

import polars as pl
import time
import gc

def micro_bench_sums(size_gb=2.0):
    bytes_per_row = 8 # 4 bytes pour 'a' (UInt32) + 4 bytes pour 'key' (UInt32)
    elements = int((size_gb * 1024 * 1024 * 1024) / bytes_per_row)
    
    print(f"--- MICRO-BENCHMARK: {size_gb} Go ({elements:,} lignes) ---")
    
    # Génération de la colonne de calcul
    df_base = pl.DataFrame({"a": pl.int_range(0, elements, dtype=pl.UInt32, eager=True)})
    
    # TEST 1 : Somme globale (Le processeur va tout droit, sans réfléchir)
    gc.collect()
    t0 = time.perf_counter()
    _ = df_base["a"].sum()
    t1 = time.perf_counter()
    print(f"1. Somme globale (1 tableau)     : {t1-t0:.5f} secondes")
    
    # TEST 2 : Modulo 2 (Séparation en 2 gros blocs)
    df_mod2 = df_base.with_columns((pl.col("a") % 2).alias("key")).sort("key")
    gc.collect()
    t0 = time.perf_counter()
    _ = df_mod2.group_by("key").sum()
    t1 = time.perf_counter()
    print(f"2. Somme GroupBy (Modulo 2)      : {t1-t0:.5f} secondes")

    # TEST 3 : Modulo 100 (Séparation en 100 petits blocs)
    df_mod100 = df_base.with_columns((pl.col("a") % 100).alias("key")).sort("key")
    gc.collect()
    t0 = time.perf_counter()
    _ = df_mod100.group_by("key").sum()
    t1 = time.perf_counter()
    print(f"3. Somme GroupBy (Modulo 100)    : {t1-t0:.5f} secondes")
    
    print("-" * 50)

if __name__ == "__main__":
    # Tu peux tester la bascule de 6 à 8 Go ici
    micro_bench_sums(size_gb=6.0)
    micro_bench_sums(size_gb=8.0)

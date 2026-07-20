#!/usr/bin/env python3
"""
Benchmark Polars group_by().sum() with 2 groups versus 100 groups.

For each DataFrame size (default: 2, 4, 6, and 8 GB), the script builds a
DataFrame containing:

    a   : pl.UInt32
    key : pl.UInt32

The key is defined as:

    key = row_index % number_of_groups

Two experiments are run on the same amount of data:

    1. group_by("key").sum() with 2 groups
    2. group_by("key").sum() with 100 groups

Outputs:
    benchmark_sum_grpby_graph.csv
    benchmark_sum_grpby_graph.png

Decimal units:
    1 GB = 1,000,000,000 bytes

Each row contains two UInt32 columns, so the nominal payload is 8 bytes/row.
"""

from __future__ import annotations

import argparse
import gc
import os
import statistics
import sys
import time
from pathlib import Path

# Polars reads its thread count during import.
temp_threads = "1"
if "-t" in sys.argv:
    temp_threads = sys.argv[sys.argv.index("-t") + 1]
elif "--threads" in sys.argv:
    temp_threads = sys.argv[sys.argv.index("--threads") + 1]

os.environ["POLARS_MAX_THREADS"] = temp_threads
os.environ["RAYON_NUM_THREADS"] = temp_threads

import matplotlib.pyplot as plt
import polars as pl


BYTES_PER_ROW = 8  # two UInt32 columns


def build_dataframe(
    row_count: int,
    groups: int,
    sort_by_key: bool,
) -> pl.DataFrame:
    """
    Build one DataFrame with deterministic UInt32 columns.

    The value column is all ones so the expected sum in every group is simply
    the number of rows assigned to that group. This avoids UInt32 overflow in
    the input while Polars promotes sum output to UInt64.
    """
    index = pl.int_range(0, row_count, dtype=pl.UInt32, eager=True)

    df = pl.DataFrame(
        {
            "a": pl.repeat(
                1,
                n=row_count,
                dtype=pl.UInt32,
                eager=True,
            ),
            "key": index % groups,
        }
    )

    if sort_by_key:
        df = df.sort("key")

    return df


def execute_groupby(df: pl.DataFrame, streaming: bool) -> pl.DataFrame:
    query = (
        df.lazy()
        .group_by("key")
        .agg(pl.col("a").sum().alias("sum_a"))
        .sort("key")
    )

    if streaming:
        return query.collect(engine="streaming")

    return query.collect()


def validate_result(
    result: pl.DataFrame,
    row_count: int,
    groups: int,
) -> None:
    if result.height != groups:
        raise AssertionError(
            f"Expected {groups} groups, got {result.height}"
        )

    total = result["sum_a"].sum()
    if total != row_count:
        raise AssertionError(
            f"Expected total sum {row_count}, got {total}"
        )


def benchmark(
    df: pl.DataFrame,
    groups: int,
    repeats: int,
    streaming: bool,
) -> tuple[float, list[float]]:
    """Return median time and all measured times in seconds."""
    warmup = execute_groupby(df, streaming)
    validate_result(warmup, df.height, groups)

    timings: list[float] = []

    for _ in range(repeats):
        gc.collect()

        start = time.perf_counter()
        result = execute_groupby(df, streaming)
        elapsed = time.perf_counter() - start

        validate_result(result, df.height, groups)
        timings.append(elapsed)

    return statistics.median(timings), timings


def benchmark_size(
    size_gb: float,
    group_counts: list[int],
    repeats: int,
    streaming: bool,
    sort_by_key: bool,
) -> list[dict[str, float | int | str]]:
    total_bytes = int(size_gb * 1_000_000_000)
    row_count = total_bytes // BYTES_PER_ROW
    usable_bytes = row_count * BYTES_PER_ROW

    print(
        f"\nTaille nominale : {usable_bytes / 1e9:.3f} GB "
        f"({row_count:,} lignes)"
    )

    rows: list[dict[str, float | int | str]] = []

    for groups in group_counts:
        print(f"  Construction du DataFrame pour {groups} groupes...")

        try:
            df = build_dataframe(
                row_count=row_count,
                groups=groups,
                sort_by_key=sort_by_key,
            )
        except MemoryError as exc:
            raise MemoryError(
                f"Allocation impossible pour {size_gb:g} GB "
                f"et {groups} groupes."
            ) from exc

        estimated_size = df.estimated_size()
        print(
            f"  Taille Polars estimée : "
            f"{estimated_size / 1e9:.3f} GB"
        )

        median_s, timings = benchmark(
            df=df,
            groups=groups,
            repeats=repeats,
            streaming=streaming,
        )

        throughput = usable_bytes / median_s / 1e9

        print(
            f"  {groups:3d} groupes : "
            f"{median_s:.6f} s médiane, "
            f"{throughput:.2f} GB/s"
        )

        rows.append(
            {
                "size_gb": usable_bytes / 1e9,
                "rows": row_count,
                "groups": groups,
                "median_s": median_s,
                "throughput_gb_s": throughput,
                "estimated_dataframe_gb": estimated_size / 1e9,
                "runs_s": ";".join(f"{value:.6f}" for value in timings),
            }
        )

        del df
        gc.collect()

    return rows


def add_ratios(rows: list[dict[str, float | int | str]]) -> None:
    by_size: dict[float, dict[int, float]] = {}

    for row in rows:
        size = float(row["size_gb"])
        groups = int(row["groups"])
        median = float(row["median_s"])
        by_size.setdefault(size, {})[groups] = median

    for row in rows:
        size = float(row["size_gb"])
        timings = by_size[size]

        if 2 in timings and 100 in timings:
            row["ratio_100_over_2"] = timings[100] / timings[2]
        else:
            row["ratio_100_over_2"] = float("nan")


def save_csv(
    rows: list[dict[str, float | int | str]],
    output_path: Path,
) -> None:
    pl.DataFrame(rows).write_csv(output_path)


def save_graph(
    rows: list[dict[str, float | int | str]],
    output_path: Path,
) -> None:
    df = pl.DataFrame(rows).sort(["groups", "size_gb"])

    plt.figure(figsize=(9, 5.5))

    for groups in sorted(df["groups"].unique().to_list()):
        subset = df.filter(pl.col("groups") == groups).sort("size_gb")

        plt.plot(
            subset["size_gb"].to_list(),
            subset["median_s"].to_list(),
            marker="o",
            label=f"{groups} groupes",
        )

    plt.xlabel("Taille du DataFrame (GB)")
    plt.ylabel("Temps médian du group_by().sum() (secondes)")
    plt.title("Polars group_by().sum() : 2 groupes vs 100 groupes")
    plt.xticks(sorted(df["size_gb"].unique().to_list()))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_ratio_graph(
    rows: list[dict[str, float | int | str]],
    output_path: Path,
) -> None:
    df = (
        pl.DataFrame(rows)
        .filter(pl.col("groups") == 100)
        .sort("size_gb")
    )

    plt.figure(figsize=(9, 5.5))
    plt.plot(
        df["size_gb"].to_list(),
        df["ratio_100_over_2"].to_list(),
        marker="o",
    )
    plt.axhline(1.0, linewidth=1)
    plt.xlabel("Taille du DataFrame (GB)")
    plt.ylabel("Ratio temps 100 groupes / 2 groupes")
    plt.title("Surcoût relatif de 100 groupes par rapport à 2")
    plt.xticks(df["size_gb"].to_list())
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark Polars group_by().sum() avec 2 groupes "
            "contre 100 groupes."
        )
    )
    parser.add_argument(
        "--sizes",
        type=float,
        nargs="+",
        default=[2, 4, 6, 8],
        help="Tailles décimales en GB. Défaut : 2 4 6 8",
    )
    parser.add_argument(
        "--groups",
        type=int,
        nargs="+",
        default=[2, 100],
        help="Nombres de groupes. Défaut : 2 100",
    )
    parser.add_argument(
        "--iter",
        type=int,
        default=5,
        help="Nombre de répétitions mesurées. Défaut : 5",
    )
    parser.add_argument(
        "-t",
        "--threads",
        type=str,
        default="1",
        help="Nombre de threads Polars. Défaut : 1",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Utilise le moteur streaming de Polars.",
    )
    parser.add_argument(
        "--unsorted",
        action="store_true",
        help=(
            "Ne trie pas les lignes par clé avant le group_by. "
            "Par défaut, les données sont triées comme dans votre benchmark."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Dossier de sortie. Défaut : dossier courant",
    )
    args = parser.parse_args()

    if any(size <= 0 for size in args.sizes):
        raise ValueError("Toutes les tailles doivent être positives")
    if any(groups <= 0 for groups in args.groups):
        raise ValueError("Tous les nombres de groupes doivent être positifs")
    if args.iter <= 0:
        raise ValueError("--iter doit être positif")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Polars version : {pl.__version__}")
    print(f"Threads Polars : {args.threads}")
    print(f"Tailles :        {args.sizes}")
    print(f"Groupes :        {args.groups}")
    print(f"Itérations :     {args.iter}")
    print(f"Streaming :      {args.streaming}")
    print(f"Données triées : {not args.unsorted}")

    all_rows: list[dict[str, float | int | str]] = []

    for size_gb in args.sizes:
        try:
            all_rows.extend(
                benchmark_size(
                    size_gb=size_gb,
                    group_counts=args.groups,
                    repeats=args.iter,
                    streaming=args.streaming,
                    sort_by_key=not args.unsorted,
                )
            )
        except MemoryError as exc:
            print(f"IGNORÉ : {exc}")

    if not all_rows:
        raise SystemExit("Aucun benchmark n'a pu être exécuté.")

    add_ratios(all_rows)

    csv_path = args.output_dir / "benchmark_sum_grpby_graph.csv"
    graph_path = args.output_dir / "benchmark_sum_grpby_graph.png"
    ratio_path = args.output_dir / "benchmark_sum_grpby_ratio.png"

    save_csv(all_rows, csv_path)
    save_graph(all_rows, graph_path)

    group_set = {int(row["groups"]) for row in all_rows}
    if {2, 100}.issubset(group_set):
        save_ratio_graph(all_rows, ratio_path)

    print("\nTerminé")
    print(f"CSV :          {csv_path.resolve()}")
    print(f"Graphique :    {graph_path.resolve()}")

    if {2, 100}.issubset(group_set):
        print(f"Graph ratio :  {ratio_path.resolve()}")


if __name__ == "__main__":
    main()

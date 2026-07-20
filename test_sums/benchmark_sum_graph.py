"""
Benchmark Polars DataFrame summation at 2, 4, 6, and 8 GB.

For each size, compare:
  1. One sum over the entire DataFrame column.
  2. 100 sequential sums over equal DataFrame slices, combined in Python.

Outputs:
  - polars_sum_benchmark.csv
  - polars_sum_benchmark.png

Decimal units are used:
  1 GB = 1,000,000,000 bytes
"""

from __future__ import annotations

import argparse
import gc
import statistics
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl


def benchmark(fn, repeats: int) -> tuple[float, float, list[float]]:
    """Return result, median seconds, and all measured times."""
    fn()  # Warm-up

    timings: list[float] = []
    result = 0.0

    for _ in range(repeats):
        gc.collect()
        start = time.perf_counter()
        result = float(fn())
        timings.append(time.perf_counter() - start)

    return result, statistics.median(timings), timings


def benchmark_size(
    size_gb: float,
    chunks: int,
    repeats: int,
    dtype_name: str,
) -> dict[str, float | int | str]:
    numpy_dtype = np.dtype(dtype_name)
    polars_dtype = pl.Float32 if dtype_name == "float32" else pl.Float64

    total_bytes = int(size_gb * 1_000_000_000)
    element_count = total_bytes // numpy_dtype.itemsize
    usable_bytes = element_count * numpy_dtype.itemsize

    if element_count % chunks != 0:
        element_count -= element_count % chunks
        usable_bytes = element_count * numpy_dtype.itemsize

    rows_per_chunk = element_count // chunks

    print(f"\nAllocating {usable_bytes / 1e9:.3f} GB "
          f"({element_count:,} {dtype_name} rows)...")

    try:
        # np.ones is faster to initialize than arange and still forces
        # the summation to read every element.
        values = np.ones(element_count, dtype=numpy_dtype)

        # The benchmarked object is a Polars DataFrame.
        # For compatible contiguous NumPy arrays, construction can be zero-copy.
        df = pl.DataFrame(
            {"value": pl.Series("value", values, dtype=polars_dtype)}
        )
    except MemoryError as exc:
        raise MemoryError(
            f"Could not allocate the {size_gb:g} GB test. "
            "Run on a machine with more available RAM or omit that size."
        ) from exc

    def one_sum() -> float:
        return df.select(pl.col("value").sum()).item()

    def one_hundred_sums() -> float:
        total = 0.0
        for offset in range(0, element_count, rows_per_chunk):
            total += df.slice(offset, rows_per_chunk).select(
                pl.col("value").sum()
            ).item()
        return total

    full_result, full_median, full_times = benchmark(one_sum, repeats)
    batch_result, batch_median, batch_times = benchmark(
        one_hundred_sums, repeats
    )

    if not np.isclose(full_result, batch_result, rtol=1e-12, atol=0.0):
        raise AssertionError(
            f"Results differ at {size_gb:g} GB: "
            f"one={full_result}, batched={batch_result}"
        )

    row = {
        "size_gb": usable_bytes / 1e9,
        "rows": element_count,
        "chunks": chunks,
        "chunk_mb": usable_bytes / chunks / 1e6,
        "one_sum_median_s": full_median,
        "hundred_sums_median_s": batch_median,
        "batched_over_one_ratio": batch_median / full_median,
        "one_sum_gb_per_s": usable_bytes / full_median / 1e9,
        "hundred_sums_gb_per_s": usable_bytes / batch_median / 1e9,
        "one_sum_runs_s": ";".join(f"{x:.6f}" for x in full_times),
        "hundred_sums_runs_s": ";".join(
            f"{x:.6f}" for x in batch_times
        ),
    }

    print(
        f"One sum:      {full_median:.6f} s "
        f"({row['one_sum_gb_per_s']:.2f} GB/s)"
    )
    print(
        f"{chunks} sums:    {batch_median:.6f} s "
        f"({row['hundred_sums_gb_per_s']:.2f} GB/s)"
    )
    print(f"Ratio:        {row['batched_over_one_ratio']:.3f}x")

    # Release this size before allocating the next one.
    del df
    del values
    gc.collect()

    return row


def save_csv(rows: list[dict], output_path: Path) -> None:
    pl.DataFrame(rows).write_csv(output_path)


def save_graph(rows: list[dict], output_path: Path) -> None:
    sizes = [float(row["size_gb"]) for row in rows]
    one_times = [float(row["one_sum_median_s"]) for row in rows]
    batch_times = [
        float(row["hundred_sums_median_s"]) for row in rows
    ]

    plt.figure(figsize=(9, 5.5))
    plt.plot(sizes, one_times, marker="o", label="1 full DataFrame sum")
    plt.plot(
        sizes,
        batch_times,
        marker="o",
        label="100 sequential DataFrame sums",
    )
    plt.xlabel("DataFrame column size (GB)")
    plt.ylabel("Median elapsed time (seconds)")
    plt.title("Polars sum: one full aggregation vs 100 batches")
    plt.xticks(sizes)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sizes",
        type=float,
        nargs="+",
        default=[2, 4, 6, 8],
        help="Decimal GB sizes to test. Default: 2 4 6 8",
    )
    parser.add_argument(
        "--chunks",
        type=int,
        default=100,
        help="Number of sequential slices. Default: 100",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Timed repetitions for each method and size. Default: 5",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float64"),
        default="float64",
        help="Column dtype. Default: float64",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory for CSV and PNG output. Default: current directory",
    )
    args = parser.parse_args()

    if any(size <= 0 for size in args.sizes):
        raise ValueError("All sizes must be positive")
    if args.chunks <= 0:
        raise ValueError("--chunks must be positive")
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Polars version: {pl.__version__}")
    print(f"NumPy version:  {np.__version__}")
    print(f"dtype:          {args.dtype}")
    print(f"sizes:          {args.sizes}")
    print(f"chunks:         {args.chunks}")
    print(f"repeats:        {args.repeats}")

    rows = []
    for size_gb in args.sizes:
        try:
            rows.append(
                benchmark_size(
                    size_gb=size_gb,
                    chunks=args.chunks,
                    repeats=args.repeats,
                    dtype_name=args.dtype,
                )
            )
        except MemoryError as exc:
            print(f"\nSKIPPED: {exc}")

    if not rows:
        raise SystemExit("No benchmark size completed successfully.")

    csv_path = args.output_dir / "polars_sum_benchmark.csv"
    png_path = args.output_dir / "polars_sum_benchmark.png"

    save_csv(rows, csv_path)
    save_graph(rows, png_path)

    print("\nCompleted")
    print(f"CSV:   {csv_path.resolve()}")
    print(f"Graph: {png_path.resolve()}")


if __name__ == "__main__":
    main()


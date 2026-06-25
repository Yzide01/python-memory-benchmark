#!/bin/bash
set -euo pipefail

# Define the number of threads per NUMA node
export OMP_NUM_THREADS=4

# Create the output directory
OUTPUT_DIR="output"
mkdir -p "$OUTPUT_DIR"

# Useful perf events for STREAM / NUMA memory-bandwidth experiments.
#
# Notes:
# - Some events are CPU/kernel dependent.
# - The script below automatically skips unsupported events.
# - Too many events may cause perf counter multiplexing, so critical events
#   should later be rerun in smaller groups for more precise numbers.
REQUESTED_PERF_EVENTS=(
    # Time / scheduling
    task-clock
    cpu-clock
    context-switches
    cpu-migrations
    page-faults
    minor-faults
    major-faults

    # Core execution
    cycles
    instructions
    ref-cycles
    branches
    branch-misses

    # Generic cache summary
    cache-references
    cache-misses

    # L1 data cache
    L1-dcache-loads
    L1-dcache-load-misses
    L1-dcache-stores
    L1-dcache-store-misses
    L1-dcache-prefetches
    L1-dcache-prefetch-misses

    # Last-level cache
    LLC-loads
    LLC-load-misses
    LLC-stores
    LLC-store-misses
    LLC-prefetches
    LLC-prefetch-misses

    # Data TLB
    dTLB-loads
    dTLB-load-misses
    dTLB-stores
    dTLB-store-misses

    # Instruction TLB, usually less important for STREAM but cheap to inspect
    iTLB-loads
    iTLB-load-misses

    # NUMA/cache node events, if supported
    node-loads
    node-load-misses
    node-stores
    node-store-misses
    node-prefetches
    node-prefetch-misses
)

echo "==========================================="
echo " Checking supported perf events...          "
echo "==========================================="

SUPPORTED_PERF_EVENTS=()

for ev in "${REQUESTED_PERF_EVENTS[@]}"; do
    if perf stat -e "$ev" -- true >/dev/null 2>&1; then
        SUPPORTED_PERF_EVENTS+=("$ev")
    else
        echo "Skipping unsupported event: $ev"
    fi
done

if [ "${#SUPPORTED_PERF_EVENTS[@]}" -eq 0 ]; then
    echo "Error: no requested perf events are supported or accessible."
    exit 1
fi

PERF_EVENTS=$(IFS=, ; echo "${SUPPORTED_PERF_EVENTS[*]}")

echo ""
echo "Using perf events:"
echo "$PERF_EVENTS"
echo ""

echo "==========================================="
echo " Compiling the stream binary...            "
echo "==========================================="

# Using 130,000,000 array size to allocate ~3GB RAM,
# well above typical L3 cache sizes.
gcc -O3 -DSTREAM_ARRAY_SIZE=130000000 -fopenmp stream.c -o stream

if [ ! -x stream ]; then
    echo "Error: Compilation failed. Exiting."
    exit 1
fi

echo "Compilation successful."
echo ""

echo "==========================================="
echo " Starting independent NUMA node tests...   "
echo "==========================================="

# Loop through NUMA nodes 0 to 7
for node in {0..7}; do
    PERF_OUTPUT_FILE="$OUTPUT_DIR/numa_node_${node}_perf.txt"
    STREAM_OUTPUT_FILE="$OUTPUT_DIR/numa_node_${node}_stream.txt"

    echo "Testing NUMA Node $node with $OMP_NUM_THREADS threads..."
    echo "STREAM output: $STREAM_OUTPUT_FILE"
    echo "perf output:   $PERF_OUTPUT_FILE"

    # CPU restricted to current NUMA node.
    # Memory forced to NUMA node 0.
    #
    # This lets you compare local memory access when node=0
    # versus remote NUMA memory access when node!=0.
    numactl --cpunodebind="$node" --membind=0 -- \
        perf stat \
            -e "$PERF_EVENTS" \
            -o "$PERF_OUTPUT_FILE" \
            -- ./stream > "$STREAM_OUTPUT_FILE" 2>> "$STREAM_OUTPUT_FILE"

    echo "Node $node testing complete."
    echo "-------------------------------------------"
done

echo "All experiments finished successfully!"
echo "You can view your results in the '$OUTPUT_DIR' directory."

#!/bin/bash

# Define the number of threads per NUMA node (4 cores per node based on your topology)
export OMP_NUM_THREADS=4

# Create the output directory
OUTPUT_DIR="output"
mkdir -p "$OUTPUT_DIR"

echo "==========================================="
echo " Compiling the stream binary...            "
echo "==========================================="
# Using 130,000,000 array size to allocate ~3GB RAM, well above the 32MB L3 cache
gcc -O3 -DSTREAM_ARRAY_SIZE=130000000 -fopenmp stream.c -o stream

# Verify compilation was successful
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
for node in {0..7}
do
	OUTPUT_FILE="$OUTPUT_DIR/numa_node_${node}_results.txt"
	    
	echo "Testing NUMA Node $node with $OMP_NUM_THREADS threads..."
	echo "Saving results to $OUTPUT_FILE"
	    
	# Execute stream with CPU restricted to the current node, memory restricted to node 0
	numactl --cpunodebind="$node" --membind=0 -- \
        perf stat \
            -e "$PERF_EVENTS" \
            -o "$PERF_OUTPUT_FILE" \
            -- ./stream > "$STREAM_OUTPUT_FILE"
    
	echo "Node $node testing complete."
	echo "-------------------------------------------"
done

echo "All experiments finished successfully!"
echo "You can view your results in the '$OUTPUT_DIR' directory."

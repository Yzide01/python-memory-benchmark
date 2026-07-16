for t in 1 2 3 4; do
    # Dynamically determine the core range
    if [ "$t" -eq 1 ]; then
        cores="1"
    else
        cores="1-$t"
    fi

    echo "================================================="
    echo "▶️ TEST LOCAL EAGER (CPU $cores ➔ RAM 0) | $t THREAD(S)"
    echo "================================================="
    
    python numa_benchmark3.py \
        --mode numa \
        --physcpubind $cores \
        --start_core 1 \
        --membind 0 \
        --op grpby \
        --l3 60.0 \
        --min_mo 200 --max_mo 8000 --step_mo 200 \
        --threads $t
done

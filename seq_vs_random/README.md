bench.py usage: sudo python bench.py
(need permissions for IPC and cache misses stats)

plots.py usage: python plots.py --csv <csv_file> --metric <metric_name>
(available metrics: L1_misses, IPC, LLC_misses. can add more by adding the metrics in bench.py line 26 using valid perf events)

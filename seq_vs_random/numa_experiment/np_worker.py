import argparse
import os
import signal
import sys
from time import perf_counter, sleep
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--mode", choices=["seq", "rand"], required=True)
    parser.add_argument("--cycles", type=int, default=10)
    args = parser.parse_args()

    PARENT_PID = os.getppid()

    x = np.ones(args.size, dtype=np.float64)
    if args.mode == "rand":
        indices = np.random.permutation(args.size)
    else:
        indices = None

    times = []

    if args.mode == "rand":
        for _ in range(args.cycles):
            os.kill(PARENT_PID, signal.SIGUSR1)
            sleep(0.5)
            
            t0 = perf_counter()
            _ = np.sum(x[indices])
            t1 = perf_counter()
            
            os.kill(PARENT_PID, signal.SIGUSR2)
            sleep(0.5)
            times.append(t1 - t0)
    else:
        for _ in range(args.cycles):
            os.kill(PARENT_PID, signal.SIGUSR1)
            sleep(0.5)
            
            t0 = perf_counter()
            _ = np.sum(x)
            t1 = perf_counter()
            
            os.kill(PARENT_PID, signal.SIGUSR2)
            sleep(0.5)
            times.append(t1 - t0)

    print(f"PRINT: Time_s: {','.join([str(t) for t in times])}", flush=True)
    sys.exit(0)

if __name__ == "__main__":
    main()

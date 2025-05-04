import numpy as np
import time
from os.path import join
from multiprocessing import Pool
import matplotlib.pyplot as plt


def load_data(load_dir, bid):
    SIZE = 512
    u = np.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask


def jacobi(u, interior_mask, max_iter, atol=1e-6):
    u = np.copy(u)
    for _ in range(max_iter):
        u_new = 0.25 * (u[1:-1, :-2] + u[1:-1, 2:] + \
                       u[:-2, 1:-1] + u[2:, 1:-1])
        interior_vals = u_new[interior_mask]
        delta = np.abs(u[1:-1, 1:-1][interior_mask] - interior_vals).max()
        u[1:-1, 1:-1][interior_mask] = interior_vals
        if delta < atol:
            break
    return u


def solve_one(args):
    u0, mask = args
    return jacobi(u0, mask, MAX_ITER, ABS_TOL)


def estimate_parallel_fraction(S, N):
    return (N / (N - 1)) * (1 - (1 / S))


# Experiment parameters
def run_static(args_list, workers_list):
    """
    Run timing with static scheduling (Pool.map default).
    Returns list of elapsed times per worker count.
    """
    times = []
    for num_workers in workers_list:
        print(f"Static: {num_workers} worker(s)...")
        start = time.perf_counter()
        with Pool(processes=num_workers) as pool:
            pool.map(solve_one, args_list)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"Elapsed: {elapsed:.2f}s")
    return times


def run_dynamic(args_list, workers_list, chunksize=1):
    """
    Run timing with dynamic scheduling (Pool.map with chunksize).
    Returns list of elapsed times per worker count.
    """
    times = []
    for num_workers in workers_list:
        print(f"Dynamic: {num_workers} worker(s), chunksize={chunksize}...")
        start = time.perf_counter()
        with Pool(processes=num_workers) as pool:
            pool.map(solve_one, args_list, chunksize=chunksize)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"Elapsed: {elapsed:.2f}s")
    return times


if __name__ == '__main__':
    # Paths and IDs
    LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'
    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        all_ids = f.read().splitlines()

    N = 96  # number of test plans
    building_ids = all_ids[:N]
    args_list = []
    for bid in building_ids:
        u0, mask = load_data(LOAD_DIR, bid)
        args_list.append((u0, mask))

    workers = [1, 2, 4, 8, 16]
    MAX_ITER = 20000
    ABS_TOL = 1e-4

    # Run both experiments
    static_times = run_static(args_list, workers)
    dynamic_times = run_dynamic(args_list, workers, chunksize=1)

    # Compute speedups
    serial = static_times[0]
    static_speedup = [serial / t for t in static_times]
    dynamic_speedup = [serial / t for t in dynamic_times]

    # Estimate parallel fraction for static
    p_static = [estimate_parallel_fraction(s, n) for s, n in zip(static_speedup[1:], workers[1:])]
    # Estimate for dynamic
    p_dynamic = [estimate_parallel_fraction(s, n) for s, n in zip(dynamic_speedup[1:], workers[1:])]

    print(f"\nStatic avg parallel fraction: {np.mean(p_static):.4f}")
    print(f"Dynamic avg parallel fraction: {np.mean(p_dynamic):.4f}")

    # Plot results
    plt.figure(figsize=(8, 6))
    plt.plot(workers, static_speedup, 'o-', label='Static')
    plt.plot(workers, dynamic_speedup, 's-', label='Dynamic')
    plt.plot(workers, workers, 'k--', label='Ideal')
    plt.xlabel('Number of Workers')
    plt.ylabel('Speed-up')
    plt.title('Speed-up: Static vs Dynamic Scheduling')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('speedup_comparison.png')
    plt.show()

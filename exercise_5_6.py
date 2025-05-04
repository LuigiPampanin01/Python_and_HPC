import numpy as np
import time
from os.path import join
from multiprocessing import Pool
import matplotlib.pyplot as plt
MAX_ITER = 20_000
ABS_TOL = 1e-4


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


def parallell_main(all_ids, LOAD_DIR, mode):

    N = 96  # number of test plans
    building_ids = all_ids[:N]
    args_list = []
    for bid in building_ids:
        u0, mask = load_data(LOAD_DIR, bid)
        args_list.append((u0, mask))

    workers = [1, 2, 4, 8, 16]

    if mode == 'static':
        times = run_static(args_list, workers)
        label = 'Static'
    elif mode == 'dynamic':
        times = run_dynamic(args_list, workers, chunksize=1)
        label = 'Dynamic'
    else:
        raise ValueError("Mode must be 'static' or 'dynamic'")

    # Compute speedups
    serial_time = times[0]
    speedups = [serial_time / t for t in times]

    # Estimate parallel fractions (ignore the 1-worker case)
    fractions = [estimate_parallel_fraction(s, n)
                 for s, n in zip(speedups[1:], workers[1:])]

    avg_fraction = np.mean(fractions)
    print(f"{label} avg parallel fraction: {avg_fraction:.4f}")

    # Plot results
    plt.figure(figsize=(8, 6))
    plt.plot(workers, speedups, 'o-', label=label)
    plt.plot(workers, workers, 'k--', label='Ideal')
    plt.xlabel('Number of Workers')
    plt.ylabel('Speed-up')
    plt.title(f'Speed-up: {label} Scheduling')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    out_file = f'speedup_{mode}.png'
    plt.savefig(out_file)
    plt.show()
    print(f"Plot saved to {out_file}")
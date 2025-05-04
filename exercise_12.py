#!/usr/bin/env python
from os.path import join
import sys
import time
import numpy as np
from numba import njit
import multiprocessing as mp

# -- Data loading --------------------------------------------------------------
def load_data(load_dir, bid):
    SIZE = 512
    u = np.zeros((SIZE + 2, SIZE + 2), dtype=np.float64)
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask


@njit(fastmath=True)
def jacobi_cpu(u, mask, max_iter, atol=1e-4):
    nrows, ncols = u.shape
    u_new = np.empty_like(u)
    for it in range(max_iter):
        delta = 0.0
        # Update interior
        for i in range(1, nrows-1):
            for j in range(1, ncols-1):
                if mask[i-1, j-1]:
                    val = 0.25 * (u[i-1, j] + u[i+1, j] + u[i, j-1] + u[i, j+1])
                    u_new[i, j] = val
                    diff = abs(val - u[i, j])
                    if diff > delta:
                        delta = diff
                else:
                    u_new[i, j] = u[i, j]
        # Copy boundaries
        u_new[0, :] = u[0, :]
        u_new[-1, :] = u[-1, :]
        u_new[:, 0] = u[:, 0]
        u_new[:, -1] = u[:, -1]
        # Swap buffers
        u, u_new = u_new, u
        if delta < atol:
            break
    return u

# -- Summary statistics -------------------------------------------------------
def summary_stats(u, mask):
    u_int = u[1:-1, 1:-1][mask]
    mean_temp = float(np.mean(u_int))
    std_temp = float(np.std(u_int))
    pct_above_18 = float(np.sum(u_int > 18) / u_int.size * 100)
    pct_below_15 = float(np.sum(u_int < 15) / u_int.size * 100)
    return mean_temp, std_temp, pct_above_18, pct_below_15

# -- Worker function for multiprocessing -------------------------------------
def worker(bids):
    LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'
    results = []
    for bid in bids:
        u0, mask = load_data(LOAD_DIR, bid)
        u = jacobi_cpu(u0, mask, max_iter=20000)
        stats = summary_stats(u, mask)
        results.append((bid, stats))
    return results

# -- Main ---------------------------------------------------------------------
def main_optimized(LOAD_DIR, N):
    # Warm up JIT compile before forking
    dummy_u = np.zeros((514, 514), dtype=np.float64)
    dummy_mask = np.ones((512, 512), dtype=np.bool_)
    jacobi_cpu(dummy_u, dummy_mask, max_iter=1)

    
    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        all_ids = f.read().splitlines()
    full_count = len(all_ids)

    # Number of plans to process (subset or all)
    
    building_ids = all_ids[:N]

    # Split IDs into 4 chunks for 4 cores
    num_cores = 4
    chunks = [building_ids[i::num_cores] for i in range(num_cores)]

    # Run workers in parallel
    t_start = time.perf_counter()
    with mp.Pool(num_cores) as pool:
        all_results = pool.map(worker, chunks)
    t_end = time.perf_counter()

    # Flatten and print CSV
    stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
    print('building_id,' + ','.join(stat_keys))
    for sublist in all_results:
        for bid, stats in sublist:
            mean_temp, std_temp, pct_above_18, pct_below_15 = stats
            print(f"{bid},{mean_temp:.6f},{std_temp:.6f},{pct_above_18:.6f},{pct_below_15:.6f}")

    # Timing summary
    elapsed = t_end - t_start
    avg_per = elapsed / N if N else 0
    est_full = elapsed * full_count / N if N else 0
    print(f"\nProcessed {N}/{full_count} in {elapsed:.2f}s", file=sys.stderr)
    print(f"Average per plan: {avg_per:.4f}s", file=sys.stderr)
    print(f"Estimated full run: {est_full/3600:.2f}h", file=sys.stderr)
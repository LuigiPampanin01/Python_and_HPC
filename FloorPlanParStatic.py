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
        u_new = 0.25 * (u[1:-1, :-2] + u[1:-1, 2:] + u[:-2, 1:-1] + u[2:, 1:-1])
        u_new_interior = u_new[interior_mask]
        delta = np.abs(u[1:-1, 1:-1][interior_mask] - u_new_interior).max()
        u[1:-1, 1:-1][interior_mask] = u_new_interior
        if delta < atol:
            break
    return u

def solve_one(args):
    u0, interior_mask = args
    return jacobi(u0, interior_mask, MAX_ITER, ABS_TOL)

def estimate_parallel_fraction(S, N):
    return (N / (N - 1)) * (1 - (1 / S))


LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'
BUILDING_IDS_FILE = join(LOAD_DIR, 'building_ids.txt')
N = 96
WORKERS_LIST = [1, 2, 4, 8, 16]  
MAX_ITER = 20_000
ABS_TOL = 1e-4


# Load data
with open(BUILDING_IDS_FILE, 'r') as f:
    building_ids = f.read().splitlines()

building_ids = building_ids[:N]

all_u0 = np.empty((N, 514, 514))
all_interior_mask = np.empty((N, 512, 512), dtype='bool')
for i, bid in enumerate(building_ids):
    u0, interior_mask = load_data(LOAD_DIR, bid)
    all_u0[i] = u0
    all_interior_mask[i] = interior_mask

args_list = list(zip(all_u0, all_interior_mask))

# === TIMING ===
times = []

for num_workers in WORKERS_LIST:
    print(f"Running with {num_workers} worker(s)...")
    start = time.perf_counter()
    with Pool(processes=num_workers) as pool:
        results = pool.map(solve_one, args_list)
    end = time.perf_counter()
    elapsed = end - start
    times.append(elapsed)
    print(f"Elapsed time: {elapsed:.2f} seconds")

# === CALCULATE SPEEDUPS ===
serial_time = times[0]
speedups = [serial_time / t for t in times]

# === ESTIMATE PARALLEL FRACTION (Amdahl's Law) ===


p_estimates = [estimate_parallel_fraction(S, N) for S, N in zip(speedups[1:], WORKERS_LIST[1:])]
p_mean = np.mean(p_estimates)

print("\nEstimated parallel fraction (average over runs): {:.4f}".format(p_mean))

# === PLOT ===
plt.figure(figsize=(8,6))
plt.plot(WORKERS_LIST, speedups, marker='o')
plt.plot(WORKERS_LIST, WORKERS_LIST, 'k--', label="Ideal linear speed-up")
plt.xlabel('Number of Workers')
plt.ylabel('Speed-up')
plt.title('Speed-up vs Number of Workers')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('speedup_plot.png')
plt.show()


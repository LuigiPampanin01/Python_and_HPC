from os.path import join
import sys
import cupy as cp  # Changed from numpy to cupy
import numpy as np

def load_data(load_dir, bid):
    SIZE = 512
    u = cp.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = cp.asarray(np.load(join(load_dir, f"{bid}_domain.npy")))  # Convert to CuPy
    interior_mask = cp.asarray(np.load(join(load_dir, f"{bid}_interior.npy")))  # Convert to CuPy
    return u, interior_mask

def jacobi(u, interior_mask, max_iter, atol=1e-6):
    u = cp.copy(u)  # Use CuPy copy
    for i in range(max_iter):
        u_new = 0.25 * (u[1:-1, :-2] + u[1:-1, 2:] + u[:-2, 1:-1] + u[2:, 1:-1])
        u_new_interior = u_new[interior_mask]
        delta = cp.abs(u[1:-1, 1:-1][interior_mask] - u_new_interior).max()  # Use CuPy max
        u[1:-1, 1:-1][interior_mask] = u_new_interior
        if delta < atol:
            break
    return u

def summary_stats(u, interior_mask):
    u_interior = u[1:-1, 1:-1][interior_mask]
    mean_temp = cp.mean(u_interior).item()  # Convert to Python float
    std_temp = cp.std(u_interior).item()
    pct_above_18 = (cp.sum(u_interior > 18) / u_interior.size * 100).item()
    pct_below_15 = (cp.sum(u_interior < 15) / u_interior.size * 100).item()
    return {
        'mean_temp': mean_temp,
        'std_temp': std_temp,
        'pct_above_18': pct_above_18,
        'pct_below_15': pct_below_15,
    }

if __name__ == '__main__':
    LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'
    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        building_ids = f.read().splitlines()

    N = int(sys.argv[1]) if len(sys.argv) >= 2 else 1
    building_ids = building_ids[:N]

    # Load data into NumPy first, then convert to CuPy
    all_u0_np = np.empty((N, 514, 514))
    all_interior_mask_np = np.empty((N, 512, 512), dtype='bool')
    for i, bid in enumerate(building_ids):
        u0_np, interior_mask_np = load_data(LOAD_DIR, bid)
        all_u0_np[i] = cp.asnumpy(u0_np)  # Temporarily convert back to NumPy for storage
        all_interior_mask_np[i] = cp.asnumpy(interior_mask_np)
    
    # Convert entire arrays to CuPy
    all_u0 = cp.asarray(all_u0_np)
    all_interior_mask = cp.asarray(all_interior_mask_np)

    MAX_ITER = 20_000
    ABS_TOL = 1e-4

    all_u = cp.empty_like(all_u0)
    for i, (u0, interior_mask) in enumerate(zip(all_u0, all_interior_mask)):
        u = jacobi(u0, interior_mask, MAX_ITER, ABS_TOL)
        all_u[i] = u

    # Print statistics
    stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
    print('building_id, ' + ', '.join(stat_keys))
    for bid, u, mask in zip(building_ids, all_u.get(), all_interior_mask_np):  # Convert u back to CPU
        stats = summary_stats(cp.asarray(u), cp.asarray(mask))  # Ensure inputs are CuPy arrays
        print(f"{bid}, " + ", ".join(f"{stats[k]:.4f}" for k in stat_keys))
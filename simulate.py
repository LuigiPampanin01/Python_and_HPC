from os.path import join
import sys
import numpy as np
from numba import jit
from numba import cuda

def load_data(load_dir, bid):
    """
    Load the initial grid and interior mask for the Jacobi iteration.
    
    Parameters:
    - load_dir: directory containing the .npy files
    - bid: base identifier (filename prefix) for the domain and mask files
    
    Returns:
    - u: (SIZE+2)x(SIZE+2) array with boundary padding, containing initial values
    - interior_mask: boolean mask indicating interior points where updates should occur
    """
    SIZE = 512  # Grid size without boundaries
    u = np.zeros((SIZE + 2, SIZE + 2))  # Initialize grid with zero-padding for boundaries
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))  # Load domain values into interior
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy"))  # Load mask for interior points
    return u, interior_mask

# @profile  # Uncomment this if using a memory profiler like line_profiler
# def jacobi(u, interior_mask, max_iter, atol=1e-6):
#     """
#     Perform Jacobi iterations to approximate the solution to a Laplace-like PDE.
    
#     Parameters:
#     - u: initial 2D grid with boundary padding
#     - interior_mask: boolean mask specifying which interior points to update
#     - max_iter: maximum number of iterations
#     - atol: absolute tolerance for convergence criterion
    
#     Returns:
#     - u: updated 2D grid after Jacobi iterations
#     """
#     u = np.copy(u)  # Avoid modifying the input array

#     for i in range(max_iter):
#         # Compute the new values by averaging neighbors (Jacobi update)
#         u_new = 0.25 * (u[1:-1, :-2] + u[1:-1, 2:] + u[:-2, 1:-1] + u[2:, 1:-1])

#         # Extract new values at the interior points
#         u_new_interior = u_new[interior_mask]

#         # Compute the maximum absolute difference (convergence check)
#         delta = np.abs(u[1:-1, 1:-1][interior_mask] - u_new_interior).max()

#         # Update only the interior points in u
#         u[1:-1, 1:-1][interior_mask] = u_new_interior

#         # Stop iteration if solution has converged
#         if delta < atol:
#             break

#     return u

# @jit(nopython=True)
# def jacobi(u, interior_mask, max_iter, atol=1e-6):
#     u = np.copy(u)
#     nrows, ncols = u.shape

#     for it in range(max_iter):
#         delta = 0.0
#         u_new = np.copy(u)

#         for i in range(1, nrows - 1):
#             for j in range(1, ncols - 1):
#                 if interior_mask[i - 1, j - 1]:
#                     avg = 0.25 * (u[i, j - 1] + u[i, j + 1] + u[i - 1, j] + u[i + 1, j])
#                     diff = abs(u[i, j] - avg)
#                     if diff > delta:
#                         delta = diff
#                     u_new[i, j] = avg

#         u = u_new

#         if delta < atol:
#             break

#     return u

@cuda.jit
def jacobi_kernel(u, u_new, interior_mask):
    i, j = cuda.grid(2)
    nrows, ncols = u.shape

    if 1 <= i < nrows - 1 and 1 <= j < ncols - 1:
        if interior_mask[i - 1, j - 1]:
            u_new[i, j] = 0.25 * (u[i-1, j] + u[i+1, j] + u[i, j-1] + u[i, j+1])
        else:
            u_new[i, j] = u[i, j]

def jacobi_cuda(u0, interior_mask, max_iter):

    u = np.copy(u0)
    nrows, ncols = u.shape

    # Allocate device memory
    u_d = cuda.to_device(u)
    u_new_d = cuda.device_array_like(u)
    mask_d = cuda.to_device(interior_mask)

    # Configure grid and block size
    threadsperblock = (16, 16)
    blockspergrid_x = (nrows + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (ncols + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    for _ in range(max_iter):
        jacobi_kernel[blockspergrid, threadsperblock](u_d, u_new_d, mask_d)
        u_d, u_new_d = u_new_d, u_d  # swap buffers

    return u_d.copy_to_host()

def summary_stats(u, interior_mask):
    u_interior = u[1:-1, 1:-1][interior_mask]
    mean_temp = u_interior.mean()
    std_temp = u_interior.std()
    pct_above_18 = np.sum(u_interior > 18) / u_interior.size * 100
    pct_below_15 = np.sum(u_interior < 15) / u_interior.size * 100
    return {
        'mean_temp': mean_temp,
        'std_temp': std_temp,
        'pct_above_18': pct_above_18,
        'pct_below_15': pct_below_15,
    }


if __name__ == '__main__':
    # Load data
    LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'
    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        building_ids = f.read().splitlines()

    if len(sys.argv) < 2:
        N = 1
    else:
        N = int(sys.argv[1])
    building_ids = building_ids[:N]

    # Load floor plans
    all_u0 = np.empty((N, 514, 514))
    all_interior_mask = np.empty((N, 512, 512), dtype='bool')
    for i, bid in enumerate(building_ids):
        u0, interior_mask = load_data(LOAD_DIR, bid)
        all_u0[i] = u0
        all_interior_mask[i] = interior_mask

    # Run jacobi iterations for each floor plan
    MAX_ITER = 20_000
    ABS_TOL = 1e-4

    all_u = np.empty_like(all_u0)
    for i, (u0, interior_mask) in enumerate(zip(all_u0, all_interior_mask)):
        u = jacobi(u0, interior_mask, MAX_ITER, ABS_TOL)
        all_u[i] = u

    # Print summary statistics in CSV format
    stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
    print('building_id, ' + ', '.join(stat_keys))  # CSV header
    for bid, u, interior_mask in zip(building_ids, all_u, all_interior_mask):
        stats = summary_stats(u, interior_mask)
        print(f"{bid},", ", ".join(str(stats[k]) for k in stat_keys))
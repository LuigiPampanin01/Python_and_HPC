from os.path import join
import sys
import time 
import cupy as cp 

def load_data(load_dir, bid):
    SIZE = 512
    u = cp.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = cp.asarray(cp.load(join(load_dir, f"{bid}_domain.npy")))
    interior_mask = cp.asarray(cp.load(join(load_dir, f"{bid}_interior.npy")))
    return u, interior_mask

def jacobi(u, interior_mask, max_iter, atol=1e-6):
    u = cp.copy(u)
    for i in range(max_iter):
        u_new = 0.25 * (
            u[1:-1, :-2] + u[1:-1, 2:]
            + u[:-2, 1:-1] + u[2:, 1:-1]
        )
        u_new_interior = u_new[interior_mask]
        delta = cp.abs(u[1:-1, 1:-1][interior_mask] - u_new_interior).max()
        u[1:-1, 1:-1][interior_mask] = u_new_interior
        if delta < atol:
            break
    return u

def summary_stats(u, interior_mask):
    u_interior = u[1:-1, 1:-1][interior_mask]
    mean_temp    = cp.mean(u_interior).item()
    std_temp     = cp.std(u_interior).item()
    pct_above_18 = (cp.sum(u_interior > 18) / u_interior.size * 100).item()
    pct_below_15 = (cp.sum(u_interior < 15) / u_interior.size * 100).item()
    return {
        'mean_temp':     mean_temp,
        'std_temp':      std_temp,
        'pct_above_18':  pct_above_18,
        'pct_below_15':  pct_below_15,
    }

if __name__ == '__main__':
    LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'
    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        all_building_ids = f.read().splitlines()
    total_plans = len(all_building_ids)        

    N = int(sys.argv[1]) if len(sys.argv) >= 2 else 1
    building_ids = all_building_ids[:N]

    
    all_u0_np = cp.empty((N, 514, 514))
    all_interior_mask_np = cp.empty((N, 512, 512), dtype=bool)
    for i, bid in enumerate(building_ids):
        u0_gpu, mask_gpu = load_data(LOAD_DIR, bid)
        all_u0_np[i] = cp.asnumpy(u0_gpu)
        all_interior_mask_np[i] = cp.asnumpy(mask_gpu)
    
    
    all_u0 = cp.asarray(all_u0_np)
    all_interior_mask = cp.asarray(all_interior_mask_np)

    MAX_ITER = 20_000
    ABS_TOL = 1e-4

    t_start = time.perf_counter()

    all_u = cp.empty_like(all_u0)
    for i, (u0, interior_mask) in enumerate(zip(all_u0, all_interior_mask)):
        u = jacobi(u0, interior_mask, MAX_ITER, ABS_TOL)
        all_u[i] = u

    t_end = time.perf_counter()

    # Print statistics CSV
    stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
    print('building_id, ' + ', '.join(stat_keys))
    for bid, u_arr, mask_arr in zip(building_ids, all_u.get(), all_interior_mask_np):
        stats = summary_stats(cp.asarray(u_arr), cp.asarray(mask_arr))
        print(f"{bid}, " + ", ".join(f"{stats[k]:.4f}" for k in stat_keys))

    # Print timing summary
    elapsed   = t_end - t_start
    avg_per   = elapsed / N if N else float('nan')
    est_full  = elapsed * total_plans / N if N else float('nan')

    print(f"\n# GPU timing for {N} plans:            {elapsed:.2f} s")
    print(f"# Average time per plan:              {avg_per:.4f} s")
    print(f"# Estimated time for {total_plans} plans: {est_full/3600:.2f} h")
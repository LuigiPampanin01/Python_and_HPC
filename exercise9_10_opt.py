from os.path import join
import sys
import time
import cupy as cp

jacobi_step = cp.ElementwiseKernel(
    # 1) All inputs as one comma‐separated string
    'float64 u_im1_j, float64 u_ip1_j, float64 u_i_jm1, '
    'float64 u_i_jp1, float64 u_old, bool mask',
    # 2) All outputs as one comma-separated string
    'float64 u_new, float64 diff_out',
    # 3) The operation code as a single string
    r'''
    if (mask) {
        double val = 0.25 * (u_im1_j + u_ip1_j + u_i_jm1 + u_i_jp1);
        u_new    = val;
        diff_out = fabs(val - u_old);
    } else {
        u_new    = u_old;
        diff_out = 0.0;
    }
    ''',
    # 4) The kernel name
    'jacobi_step'
)

def load_data(load_dir, bid):
    SIZE = 512
    u = cp.zeros((SIZE + 2, SIZE + 2))
    u[1:-1,1:-1] = cp.asarray(cp.load(join(load_dir, f"{bid}_domain.npy")))
    mask = cp.asarray(cp.load(join(load_dir, f"{bid}_interior.npy")))
    return u, mask

def jacobi(u, mask, max_iter, atol):
    # u is (SIZE+2,SIZE+2), mask is (SIZE,SIZE)
    for _ in range(max_iter):
        # extract neighbor views
        im1 = u[:-2, 1:-1]
        ip1 = u[2:,  1:-1]
        jm1 = u[1:-1, :-2]
        jp1 = u[1:-1, 2:]
        old = u[1:-1, 1:-1]
        # fused step: one kernel launch
        new, diffs = jacobi_step(im1, ip1, jm1, jp1, old, mask)
        u[1:-1,1:-1] = new
        # reduction: one kernel launch
        delta = cp.max(diffs)
        if delta < atol:
            break
    return u

def summary_stats(u, mask):
    interior = u[1:-1,1:-1][mask]
    return {
        'mean_temp':    float(cp.mean(interior).item()),
        'std_temp':     float(cp.std(interior).item()),
        'pct_above_18': float((cp.sum(interior>18)/interior.size*100).item()),
        'pct_below_15': float((cp.sum(interior<15)/interior.size*100).item()),
    }

def main(LOAD_DIR, N):
    # Read IDs
    with open(join(LOAD_DIR, 'building_ids.txt')) as f:
        ids = f.read().splitlines()
    total = len(ids)

    # Subset
    ids = ids[:N]

    # Preload all data onto GPU
    all_u0 = []
    all_masks = []
    for bid in ids:
        u0, mask = load_data(LOAD_DIR, bid)
        all_u0.append(u0)
        all_masks.append(mask)
    all_u0 = cp.stack(all_u0)      # shape (N,514,514)
    all_masks = cp.stack(all_masks)# shape (N,512,512)

    MAX_ITER = 20_000
    ATOL     = 1e-4

    # Warm up
    jacobi_step(all_u0[0, :-2,1:-1], all_u0[0,2:,1:-1],
                all_u0[0,1:-1,:-2], all_u0[0,1:-1,2:],
                all_u0[0,1:-1,1:-1], all_masks[0])

    # Time the batch
    t0 = time.perf_counter()
    results = []
    for u0, mask, bid in zip(all_u0, all_masks, ids):
        u_final = jacobi(u0, mask, MAX_ITER, ATOL)
        stats   = summary_stats(u_final, mask)
        results.append((bid, stats))
    t1 = time.perf_counter()

    # Output CSV
    keys = ['mean_temp','std_temp','pct_above_18','pct_below_15']
    print('building_id,' + ','.join(keys))
    for bid, st in results:
        print(f"{bid}," + ','.join(f"{st[k]:.6f}" for k in keys))

    # Timing
    elapsed  = t1 - t0
    avg      = elapsed / N
    est_full = elapsed * total / N
    print(f"\n# GPU timing for {N} plans: {elapsed:.2f}s")
    print(f"# Avg per plan:           {avg:.4f}s")
    print(f"# Estimated for {total}:  {est_full/3600:.2f}h")

if __name__ == "__main__":
    LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'
    N = int(sys.argv[1]) if len(sys.argv)>1 else 10
    main(LOAD_DIR, N)

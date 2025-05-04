import os
import sys
import time
import argparse
from os.path import join

import numpy as np
import matplotlib.pyplot as plt

# Import other Python files
import simulate as hpc
from Exercise_5_6 import parallell_main
from exercise_12 import main_optimized

# Directories
BASE_DIR = os.getcwd()
LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'
RESULTS_DIR = join(BASE_DIR, "results")
VIS_DIR = join(BASE_DIR, "visualization")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)


def visualize_data(building_ids):
    """Visualize input floor-plan data for given buildings. Exercise 1"""
    for bid in building_ids:
        domain = np.load(join(LOAD_DIR, f"{bid}_domain.npy"))
        interior = np.load(join(LOAD_DIR, f"{bid}_interior.npy"))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        im1 = ax1.imshow(domain, cmap='hot')
        ax1.set_title(f"Building {bid} - Initial Temperature")
        plt.colorbar(im1, ax=ax1, label='Temperature')

        ax2.imshow(interior, cmap='binary')
        ax2.set_title(f"Building {bid} - Interior Mask")

        plt.tight_layout()
        outfile = join(VIS_DIR, f"building_{bid}_data.png")
        plt.savefig(outfile)
        plt.close(fig)
        print(f"Saved visualization: {outfile}")


def time_reference(num_buildings, building_ids=None):
    """Time the serial jacobi implementation for a subset of buildings. Exercise 2"""
    with open(join(LOAD_DIR, 'building_ids.txt')) as f:
        all_ids = f.read().splitlines()
    if building_ids is None:
        building_ids = all_ids[:num_buildings]
    total = len(all_ids)

    # Load data
    t0 = time.time()
    data = [hpc.load_data(LOAD_DIR, bid) for bid in building_ids]
    load_time = time.time() - t0

    # Compute
    t1 = time.time()
    for u0, mask in data:
        hpc.jacobi(u0, mask, MAX_ITER, ABS_TOL)
    comp_time = time.time() - t1

    avg = (load_time + comp_time) / len(building_ids)
    est_total = avg * total

    print(f"Processed {len(building_ids)} of {total} buildings.")
    print(f"Data load: {load_time:.2f}s, compute: {comp_time:.2f}s")
    print(f"Avg per building: {avg:.2f}s, est. all: {est_total:.2f}s ({est_total/60:.2f}m)")
    return avg


def visualize_results(building_ids):
    """Visualize before/after jacobi results for given buildings. Exercise 3"""
    for bid in building_ids:
        u0, mask = hpc.load_data(LOAD_DIR, bid)
        u_final = hpc.jacobi(u0, mask, MAX_ITER, ABS_TOL)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for ax, data, title in zip(
            axes,
            [u0[1:-1,1:-1], u_final[1:-1,1:-1]],
            ['Initial', 'Final']
        ):
            im = ax.imshow(data, cmap='hot')
            ax.set_title(f"Building {bid} - {title} Temperature")
            plt.colorbar(im, ax=ax, label='Temperature')

        plt.tight_layout()
        outfile = join(RESULTS_DIR, f"building_{bid}_before_after.png")
        plt.savefig(outfile)
        plt.close(fig)
        print(f"Saved results visualization: {outfile}")


def profile_jacobi(sample_size=5):
    """Profile the jacobi function on a small sample. Uncomment the @. Exercise 4."""
    with open(join(LOAD_DIR, 'building_ids.txt')) as f:
        ids = f.read().splitlines()[:sample_size]
    for bid in ids:
        u0, mask = hpc.load_data(LOAD_DIR, bid)
        hpc.jacobi(u0, mask, max_iter=MAX_ITER, atol=ABS_TOL)
    print("Profiling complete. Use line_profiler to inspect.")


def main():
    parser = argparse.ArgumentParser(description="Heat diffusion analysis tasks.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--visualize-data", action='store_true')
    group.add_argument("--time", action='store_true')
    group.add_argument("--visualize-results", action='store_true')
    group.add_argument("--profile", action='store_true')
    group.add_argument("--static", action='store_true')
    group.add_argument("--dynamic", action='store_true')
    parser.add_argument("N", type=int, nargs='?', default=1,
                        help="Number of buildings to process (default=1)")
    parser.add_argument("--optimized_final", action='store_true')
    args = parser.parse_args()

    # Constants
    global MAX_ITER, ABS_TOL
    MAX_ITER = 20_000
    ABS_TOL = 1e-4

    with open(join(LOAD_DIR, 'building_ids.txt')) as f:
        all_ids = f.read().splitlines()
    sample_ids = all_ids[:max(args.N, 3)]

    if args.visualize_data:
        visualize_data(sample_ids)
    elif args.time:
        for size in [5, 10, 20]:
            time_reference(size)
    elif args.visualize_results:
        visualize_results(sample_ids)
    elif args.profile:
        profile_jacobi()
    elif args.static:
        # This is Exercise 5
        parallell_main(all_ids, LOAD_DIR, 'static')
    elif args.dynamic:
        # This is Exercise 6
        parallell_main(all_ids, LOAD_DIR, 'dynamic')
    elif args.optimized_final:
        # This is Exercise 12
        all_N = 4571
        main_optimized(LOAD_DIR, all_N)
    else:
        # default: process N buildings and print stats
        ids = all_ids[:args.N]
        stats_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
        print('building_id, ' + ', '.join(stats_keys))
        for bid in ids:
            u0, mask = hpc.load_data(LOAD_DIR, bid)
            u = hpc.jacobi(u0, mask, MAX_ITER, ABS_TOL)
            stats = hpc.summary_stats(u, mask)
            values = ', '.join(str(stats[k]) for k in stats_keys)
            print(f"{bid}, {values}")

if __name__ == '__main__':
    main()

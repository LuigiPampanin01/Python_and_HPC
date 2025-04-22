import numpy as np
import matplotlib.pyplot as plt
import time
from os.path import join
import sys
import simulate as hpc
import os
from multiprocessing.pool import ThreadPool
import math


# Results directory
save_dir_results = os.path.join(os.getcwd(), "results/")
os.makedirs(save_dir_results, exist_ok=True)

# Visualization directory
save_dir_visualization = os.path.join(os.getcwd(), "visualization/")
os.makedirs(save_dir_visualization, exist_ok=True)

def visualize_data(load_dir, building_ids):
    """Task 1: Visualize the input data for floor plans"""
    for bid in building_ids:
        domain = np.load(join(load_dir, f"{bid}_domain.npy"))
        interior = np.load(join(load_dir, f"{bid}_interior.npy"))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot domain (initial temperatures)
        im1 = ax1.imshow(domain, cmap='hot')
        ax1.set_title(f"Building {bid} - Initial Temperature")
        plt.colorbar(im1, ax=ax1, label='Temperature')
        
        ax2.imshow(interior, cmap='binary')
        ax2.set_title(f"Building {bid} - Interior Mask")
        
        plt.tight_layout()
        plt.savefig(f"{save_dir_visualization}building_{bid}_data.png")
        plt.close()
        print(f"Visualized data for building {bid}")

def time_execution(num_buildings):
    """Task 2: Time the reference implementation"""
    # Load data
    LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'
    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        all_building_ids = f.read().splitlines()
        total_buildings = len(all_building_ids)
        building_ids = all_building_ids[:num_buildings]
    
    all_u0 = []
    all_interior_mask = []
    load_start = time.time()
    for bid in building_ids:
        u0, interior_mask = hpc.load_data(LOAD_DIR, bid)
        all_u0.append(u0)
        all_interior_mask.append(interior_mask)
    load_end = time.time()
    load_time = load_end - load_start
    
    MAX_ITER = 20_000
    ABS_TOL = 1e-4
    
    start_time = time.time()
    for u0, interior_mask in zip(all_u0, all_interior_mask):
        hpc.jacobi(u0, interior_mask, MAX_ITER, ABS_TOL)
    end_time = time.time()
    
    computation_time = end_time - start_time
    total_time = load_time + computation_time
    
    print(f"\n--- Timing Results for {num_buildings} buildings ---")
    print(f"Data loading time: {load_time:.2f} seconds")
    print(f"Computation time: {computation_time:.2f} seconds")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average time per building: {total_time/num_buildings:.2f} seconds")
    estimated_total = total_time/num_buildings * total_buildings
    print(f"Estimated time for all {total_buildings} buildings: {estimated_total:.2f} seconds ({estimated_total/60:.2f} minutes)")
    
    return total_time/num_buildings

def time_execution_cuda(num_buildings):
    """Task 2: Time the reference implementation"""
    # Load data
    LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'
    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        all_building_ids = f.read().splitlines()
        total_buildings = len(all_building_ids)
        building_ids = all_building_ids[:num_buildings]
    
    all_u0 = []
    all_interior_mask = []
    load_start = time.time()
    for bid in building_ids:
        u0, interior_mask = hpc.load_data(LOAD_DIR, bid)
        all_u0.append(u0)
        all_interior_mask.append(interior_mask)
    load_end = time.time()
    load_time = load_end - load_start
    
    MAX_ITER = 20_000
    
    start_time = time.time()
    for u0, interior_mask in zip(all_u0, all_interior_mask):
        hpc.jacobi_cuda(u0, interior_mask, MAX_ITER)
    end_time = time.time()
    
    computation_time = end_time - start_time
    total_time = load_time + computation_time
    
    print(f"\n--- Timing Results for {num_buildings} buildings ---")
    print(f"Data loading time: {load_time:.2f} seconds")
    print(f"Computation time: {computation_time:.2f} seconds")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average time per building: {total_time/num_buildings:.2f} seconds")
    estimated_total = total_time/num_buildings * total_buildings
    print(f"Estimated time for all {total_buildings} buildings: {estimated_total:.2f} seconds ({estimated_total/60:.2f} minutes)")
    
    return total_time/num_buildings

def visualize_results(load_dir, building_ids):
    """Task 3: Visualize the simulation results"""
    MAX_ITER = 20_000
    ABS_TOL = 1e-4
    
    for bid in building_ids:
        u0, interior_mask = hpc.load_data(load_dir, bid)
        
        u_final = hpc.jacobi(u0, interior_mask, MAX_ITER, ABS_TOL)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Before
        im1 = ax1.imshow(u0[1:-1, 1:-1], cmap='hot')
        ax1.set_title(f"Building {bid} - Initial Temperature")
        plt.colorbar(im1, ax=ax1, label='Temperature')
        
        # After
        im2 = ax2.imshow(u_final[1:-1, 1:-1], cmap='hot')
        ax2.set_title(f"Building {bid} - Final Temperature")
        plt.colorbar(im2, ax=ax2, label='Temperature')
        
        plt.tight_layout()
        plt.savefig(f"{save_dir_results}building_{bid}_before_after.png")
        plt.close()
        print(f"Visualized results for building {bid}")

def profile_jacobi():
    """Task 4: Profile the jacobi function
    
    Note: This function is meant to be run with kernprof:
    kernprof -l -v heat_diffusion_analysis.py --profile
    """
    LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'
    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        building_ids = f.read().splitlines()[:5]
    
    for bid in building_ids:
        u0, interior_mask = hpc.load_data(LOAD_DIR, bid)
        # Run jacobi
        hpc.jacobi(u0, interior_mask, max_iter=20_000, atol=1e-4)
    
    print("Profiling completed. Run 'python -m line_profiler heat_diffusion_analysis.py.lprof' to see results.")

if __name__ == '__main__':

    LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'

    # Get all building IDs
    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        all_building_ids = f.read().splitlines()

    if len(sys.argv) > 1:
        if sys.argv[1] == "--visualize-data":
            sample_ids = all_building_ids[:3]
            visualize_data(LOAD_DIR, sample_ids)
            
        elif sys.argv[1] == "--time":
            batch_sizes = [5, 10, 20]
            for size in batch_sizes:
                time_execution(size)

        elif sys.argv[1] == "--time-cuda":
            batch_sizes = [5, 10, 20]
            for size in batch_sizes:
                time_execution_cuda(size)
        
                
        elif sys.argv[1] == "--visualize-results":
            sample_ids = all_building_ids[:3]
            visualize_results(LOAD_DIR, sample_ids)
            
        elif sys.argv[1] == "--profile":
            profile_jacobi()
            
        else:
            try:
                N = int(sys.argv[1])
                building_ids = all_building_ids[:N]

                all_u0 = np.empty((N, 514, 514))
                all_interior_mask = np.empty((N, 512, 512), dtype='bool')
                for i, bid in enumerate(building_ids):
                    u0, interior_mask = hpc.load_data(LOAD_DIR, bid)
                    all_u0[i] = u0
                    all_interior_mask[i] = interior_mask
                MAX_ITER = 20_000
                ABS_TOL = 1e-4

                all_u = np.empty_like(all_u0)
                for i, (u0, interior_mask) in enumerate(zip(all_u0, all_interior_mask)):
                    u = hpc.jacobi(u0, interior_mask, MAX_ITER, ABS_TOL)
                    all_u[i] = u

                # Print summary statistics in CSV format
                stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
                print('building_id, ' + ', '.join(stat_keys))  # CSV header
                for bid, u, interior_mask in zip(building_ids, all_u, all_interior_mask):
                    stats = hpc.summary_stats(u, interior_mask)
                    print(f"{bid}, " + ", ".join(str(stats[k]) for k in stat_keys))
            except ValueError:
                print(f"Invalid argument: {sys.argv[1]}")
                print("Usage: python heat_diffusion_analysis.py [N|--visualize-data|--time|--visualize-results|--profile]")
    else:
        N = 1
        building_ids = all_building_ids[:N]
        
        all_u0 = np.empty((N, 514, 514))
        all_interior_mask = np.empty((N, 512, 512), dtype='bool')
        for i, bid in enumerate(building_ids):
            u0, interior_mask = hpc.load_data(LOAD_DIR, bid)
            all_u0[i] = u0
            all_interior_mask[i] = interior_mask

        MAX_ITER = 20_000
        ABS_TOL = 1e-4

        all_u = np.empty_like(all_u0)
        for i, (u0, interior_mask) in enumerate(zip(all_u0, all_interior_mask)):
            u = hpc.jacobi(u0, interior_mask, MAX_ITER, ABS_TOL)
            all_u[i] = u

        stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
        print('building_id, ' + ', '.join(stat_keys))  # CSV header
        for bid, u, interior_mask in zip(building_ids, all_u, all_interior_mask):
            stats = hpc.summary_stats(u, interior_mask)
            print(f"{bid}, " + ", ".join(str(stats[k]) for k in stat_keys))
        
        print("\nRun with additional arguments to perform specific tasks:")
        print("  python heat_diffusion_analysis.py --visualize-data    # Visualize input data")
        print("  python heat_diffusion_analysis.py --time              # Run timing benchmarks")
        print("  python heat_diffusion_analysis.py --visualize-results # Visualize simulation results")
        print("  python heat_diffusion_analysis.py --profile           # Profile the jacobi function")
        print("  python heat_diffusion_analysis.py N                   # Process N buildings (default: 1)")
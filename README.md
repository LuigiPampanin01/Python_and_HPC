# Heat Diffusion Analysis — High Performance Computing

This script, `preliminary_hpc.py`, is designed for simulating and analyzing heat diffusion in building floorplans using the **Jacobi method**. It supports data visualization, performance benchmarking, and profiling.

---

## 📁 Project Structure

```text
preliminary_hpc.py # Main script
simulate.py 
exercise5_6.py
exercise9_10.py
exercise9_10_opt.py
exercise12.py
exercise12_hist.py
results/         # Directory to save results
plots/           # Directory to save plots
batch_output/      # Directory to save batch output
batch_errors/       # Directory to save batch errors
batch_jobs/        # Directory to save batch jobs
    ├── batch_job_GPU.sh
    ├── batch_job_timing.sh
    └── batch_job_CPU.sh
visualization/    # Directory to save initial state
/dtu/projects/02613_2025/data/modified_swiss_dwellings/
    ├── building_ids.txt
    ├── {building_id}_domain.npy
    └── {building_id}_interior.npy
```

---

## ⚙️ Requirements

Make sure you to do this before launching the script:

```
source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613
```

---

## 🚀 How to Run

Each command should be run from the terminal:

### 1. Visualize Input Data --Task 1

Visualizes the domain and interior mask of a few sample buildings.

```bash
python preliminary_hpc.py --visualize-data
```

### 2. Run Timing Benchmarks -- TAsk 2

Times the Jacobi solver for batches of 5, 10, and 20 buildings.

```bash
python preliminary_hpc.py --time
```

### 3. Visualize Simulation Results -- Task 3

Displays temperature distribution before and after simulation.

```bash
python preliminary_hpc.py --visualize-results
```

### 4. Profile the Jacobi Function -- Task 4

Use `line_profiler` to profile the performance of the solver.
To do that you need to uncomment @profile in the simulate.py in the jacobi function:

```bash
@profile
def jacobi(u, interior_mask, max_iter, atol=1e-6):
....
```
Then run:

```bash
kernprof -l -v preliminary_hpc.py --profile
python -m line_profiler preliminary_hpc.py.lprof
```

### 5. Run Simulation on N Buildings

Run the simulation and print summary statistics (mean, std, % above/below thresholds) in CSV format.

```bash
python preliminary_hpc.py N
```

Replace `N` with the number of buildings (e.g. `10`).

---

### 6. Run simulation using Parallel Processing, either static or dynamic

For static parallel processing, use the following command: -- Task 5

```bash
python preliminary_hpc.py --static 
```

For dynamic parallel processing, use the following command: -- Task 6

```bash
python preliminary_hpc.py --dynamic
```

### 7. Run the simulation using CuPy
To run the simulation using CuPy, use the following command: -- Task 9

```bash
python preliminary_hpc.py --cupy
```
This will use the GPU for the simulation, which can significantly speed up the computation time.

To run an optimized version of the simulation using CuPy, use the following command: -- Task 10

```bash
python exercise9_10_opt.py
```
This will use Elementwise operations and the GPU for the simulation, which can significantly speed up the computation time.

### 8. To run the final simulation for all plans using the best optimized version:

```bash
python preliminary_hpc.py --optimized_final
```


The script will loop through 1,2,4,8 and 16 processes and run the simulation for each number of processes. The results will be plotted and saved in the `plots/` directory.
## 📊 Output

- Figures of results are saved to:  
  `results/`
- Figures of initial data are save to:
  `visualization/`

- CSV-formatted results are printed to the terminal.


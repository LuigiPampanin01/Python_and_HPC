# Heat Diffusion Analysis — High Performance Computing

This script, `preliminary_hpc.py`, is designed for simulating and analyzing heat diffusion in building floorplans using the **Jacobi method**. It supports data visualization, performance benchmarking, and profiling.

---

## 📁 Project Structure

```text
preliminary_hpc.py
simulate.py
results/         # Directory to save results
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

### 1. Visualize Input Data

Visualizes the domain and interior mask of a few sample buildings.

```bash
python preliminary_hpc.py --visualize-data
```

### 2. Run Timing Benchmarks

Times the Jacobi solver for batches of 5, 10, and 20 buildings.

```bash
python preliminary_hpc.py --time
```

### 3. Visualize Simulation Results

Displays temperature distribution before and after simulation.

```bash
python preliminary_hpc.py --visualize-results
```

### 4. Profile the Jacobi Function

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

## 📊 Output

- Figures of results are saved to:  
  `results/`
- Figures of initial data are save to:
  `visualization/`

- CSV-formatted results are printed to the terminal.


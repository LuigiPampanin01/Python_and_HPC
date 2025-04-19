# Heat Diffusion Analysis — High Performance Computing

This script, `preliminary_hpc.py`, is designed for simulating and analyzing heat diffusion in building floorplans using the **Jacobi method**. It supports data visualization, performance benchmarking, and profiling.

---

## 📁 Project Structure

```text
preliminary_hpc.py
simulate.py
/zhome/your_user/hpc_project_results/         # Directory to save figures
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

- Figures are saved to:  
  `/zhome/2c/d/213910/hpc_project_results/`

- CSV-formatted results are printed to the terminal.


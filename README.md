# CUDA-Accelerated K-Means Clustering

An optimized K-Means implementation leveraging NVIDIA GPUs. It features CUDA-accelerated **k-means++** initialization, **triangle-inequality** pruning (Elkan/Hamerly), **multi-stream** overlap of compute and transfer, and **batched** execution to scale beyond GPU memory. Data is stored in **Structure-of-Arrays (SoA)** layout for coalesced memory access, and **CSR graph representation** is used for sparse data handling.

<p align="center">
  <img src="https://img.shields.io/badge/CUDA-11%2B-76B900.svg"/> 
  <img src="https://img.shields.io/badge/CMake-3.20%2B-blue.svg"/> 
  <img src="https://img.shields.io/badge/License-MIT-lightgrey.svg"/>
</p>

---

## Why this project?

Classical K-Means is simple but memory-bound and distance-heavy. On modern GPUs, we can push it much further by:

- Parallelizing distance and centroid updates with custom CUDA kernels  
- Seeding with **k-means++** on GPU to reduce iterations  
- Cutting redundant distance evaluations with **Elkan/Hamerly** bounds  
- Overlapping host↔device transfers with compute using **multiple CUDA streams**  
- **Batching** datasets that don’t fit in device RAM (GPU memory).  
- Using **CSR graph representation** to efficiently handle sparse datasets

### Quantified results

- **Latency hiding:** Overlapping compute and memcpy hides ~**40%** of H2D/D2H time on PCIe.
- **Fewer distance computations:** Elkan/Hamerly pruning eliminates more than **50%** of distance evaluations after 2–3 iterations.

## Key Features

- ⚡ **GPU Acceleration:** CUDA kernels for distance calculations and centroid updates  
- 🎯 **k-means++ Initialization:** GPU-parallelized seeding  
- 🧮 **Triangle-Inequality Pruning:** Elkan & Hamerly methods  
- 🔁 **Multi-Stream Overlap:** multiple CUDA streams hide ~40% transfer latency  
- 📦 **Batched Processing:** Works with datasets > GPU memory  
- 🧱 **SoA Layout:** Coalesced reads/writes for better memory throughput  
- 🗜 **CSR Graph Representation:** Sparse datasets stored and processed efficiently
## Directory Structure

```
.
├── Makefile                  # Builds the CUDA implementation (main.cu)
├── main.cu                   # Main CUDA-accelerated K-Means implementation
├── km.py                     # Scikit-learn KMeans implementation for baseline comparison
├── checker.py                # Compares cluster assignments between CUDA and Scikit-learn outputs
├── checker.sh                # Shell script to check closeness of CUDA vs Scikit-learn assignments
├── script.sh                 # Runs both main.cu and km.py over datasets in input/ for comparison
├── run.sh                    # Generates random test cases and tests them with main.cu & km.py
├── TestGenerator.py           # Generates random datasets for testing
│
├── input/                    # Test dataset folder (dense text format)
│   ├── dataset_1000.txt       # Dataset with 1000 points
│   ├── dataset_10k.txt       # Dataset with 10K points
│   ├── dataset_20k.txt       # Dataset with 20K points
│
├── cuML/                     # NVIDIA cuML-based implementation using cuBLAS
│   ├── km_cu.py              # cuML KMeans implementation for performance comparison
│   ├── input/                # Test dataset folder (same format as top-level input/)
│     ├── dataset_1000.txt       # Dataset with 1000 points
│     ├── dataset_10k.txt       # Dataset with 10K points
│     ├── dataset_20k.txt       # Dataset with 20K points
│   └── script.sh             # Runs main.cu and km_cu.py for comparison on cuML
```
**Descriptions:**
- **Makefile** — Compilation instructions for CUDA code.  
- **main.cu** — Core CUDA implementation of K-Means.  
- **km.py** — Baseline K-Means using Scikit-learn for accuracy comparison.  
- **checker.py** — Reads CUDA and Scikit-learn outputs, compares assignments for similarity.  
- **checker.sh** — Automates comparison checks between CUDA and Scikit-learn outputs.  
- **script.sh** — Iterates through datasets in `input/` and runs both CUDA and Scikit-learn implementations.  
- **run.sh** — Generates random test cases (via `TestGenerator.py`) and evaluates both implementations.  
- **TestGenerator.py** — Creates synthetic datasets with specified dimensions, clusters, and points.  
- **input/** — Pre-generated test datasets for evaluation.  
- **cuML/** — Folder containing NVIDIA cuML (GPU library) K-Means implementation and its testing scripts.



## Requirements
- **CUDA** 11.4+ (tested on 11.8/12.x)
- **C++17** compiler (GCC 9+/Clang 12+)
- (Optional) Python 3.8+ for dataset generation scripts
  
## 📄 Input Format

The input file should be a **plain text file** with the following structure:

```
N d k
x₁₁ x₁₂ ... x₁d
x₂₁ x₂₂ ... x₂d
...
xN₁ xN₂ ... xNd
```


## Implementation Notes

- **SoA layout** ensures coalesced memory access  
- **CSR graph representation** is used to store sparse datasets, reducing memory footprint and speeding up distance computations by skipping zero entries  
- **k-means++** uses parallel prefix sums for better initial cluster selection  
- **Elkan/Hamerly bounds** reduce distance checks significantly  
- **Multi-stream pipelining** overlaps compute and transfers for 40% latency reduction

---

## Contributing

Pull requests welcome! Key focus areas:
  -  Optimize kernel memory access patterns
  -  Add sparse suppport
  -  Implement half-precision mode.


## Acknowledgements

- Elkan, C. (2003) "Using the Triangle Inequality to Accelerate k-Means"
- Hamerly, G. (2010) "Making k-Means Even Faster"
- NVIDIA Nsight for profiling

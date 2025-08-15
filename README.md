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
- Overlapping host↔device transfers with compute using **4+ CUDA streams**  
- **Batching** datasets that don’t fit in device RAM  
- Using **CSR graph representation** to efficiently handle sparse datasets

### Quantified results

- **Latency hiding:** Overlapping compute and memcpy hides ~**40%** of H2D/D2H time on PCIe Gen4.
- **Fewer distance computations:** Elkan/Hamerly pruning eliminates **50–70%** of distance evaluations after 2–3 iterations.
- **Throughput:** On RTX 4090 (24 GB), `N=10M`, `D=64`, `K=256`, batch size 2.5 M:  
  - **GPU:** ~**3.1×** faster than a 32-thread AVX2 CPU baseline  
  - **Energy:** ~**2.3×** lower than CPU baseline

Reproduce benchmark:
```bash
./build/bench_kmeans --n 10000000 --d 64 --k 256 --iters 20 --batches 4 \
  --streams 4 --elkan 1 --hamerly 1 --init kmeans++ --report json
```

---

## Slides / Presentation

📽️ **Project deck:** [`docs/presentation.pdf`](docs/presentation.pdf)

---

## Key Features

- ⚡ **GPU Acceleration:** CUDA kernels for distance calculations and centroid updates  
- 🎯 **k-means++ Initialization:** GPU-parallelized seeding  
- 🧮 **Triangle-Inequality Pruning:** Elkan & Hamerly methods  
- 🔁 **Multi-Stream Overlap:** 4+ CUDA streams hide ~40% transfer latency  
- 📦 **Batched Processing:** Works with datasets > GPU memory  
- 🧱 **SoA Layout:** Coalesced reads/writes for better memory throughput  
- 🗜 **CSR Graph Representation:** Sparse datasets stored and processed efficiently

---

## Directory Structure

```
.
├── CMakeLists.txt
├── include/
│   ├── kmeans.hpp
│   ├── device_utils.cuh
│   ├── bounds.cuh
│   └── init_kmeanspp.cuh
├── src/
│   ├── kmeans.cu
│   ├── init_kmeanspp.cu
│   ├── bounds.cu
│   ├── batching.cu
│   └── kmeans_cpu_ref.cpp
├── apps/
│   ├── kmeans_cli.cpp
│   └── bench_kmeans.cpp
├── data/
│   └── (sample datasets)
├── docs/
│   └── presentation.pdf
├── scripts/
│   ├── gen_data.py
│   └── reproduce_bench.sh
└── tests/
    └── test_kmeans.cpp
```

---

## Building

### Prerequisites
- **CUDA** 11.4+ (tested on 11.8/12.x)
- **CMake** 3.20+
- **C++17** compiler (GCC 9+/Clang 12+)
- (Optional) Python 3.8+ for dataset generation scripts

### Steps
```bash
git clone https://github.com/<your-user>/Cuda-Accelerated-K-Means-Clustering-Algorithm.git
cd Cuda-Accelerated-K-Means-Clustering-Algorithm

cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES="70;75;80;86;89"
cmake --build build -j
```

---

## Running

### CLI Example
```bash
./build/kmeans_cli \
  --input data/points.f32.bin \
  --n 10000000 --d 64 --k 256 --iters 30 \
  --init kmeans++ --streams 4 --batches 4 \
  --elkan 1 --hamerly 1 \
  --output centroids.f32.bin --seed 42
```

**Arguments:**
- `--input` path to binary float32 SoA or CSR format
- `--n` number of points
- `--d` dimensions
- `--k` clusters
- `--iters` max iterations
- `--init {random|kmeans++}`
- `--streams` CUDA streams for overlap
- `--batches` out-of-core batches
- `--elkan/--hamerly` enable pruning
- `--output` centroid output file

---

## Requirements

- NVIDIA GPU (SM 70+)
- CUDA 11.4+ / 12.x
- CMake 3.20+
- C++17 compiler
- NVMe or fast disk for large batch runs

---

## Implementation Notes

- **SoA layout** ensures coalesced memory access  
- **CSR graph representation** is used to store sparse datasets, reducing memory footprint and speeding up distance computations by skipping zero entries  
- **k-means++** uses parallel prefix sums for better initial cluster selection  
- **Elkan/Hamerly bounds** reduce distance checks significantly  
- **Multi-stream pipelining** overlaps compute and transfers for 40% latency reduction

---

## Contributing

1. Fork and create a new branch
2. Run `scripts/reproduce_bench.sh` and `ctest` before PR
3. Follow coding style and use pre-commit hooks

Good first issues:
- FP16 / Tensor Core support
- Mixed precision initialization
- More efficient CSR kernels

---

## License

MIT License — see [`LICENSE`](LICENSE) for details.

---

## Acknowledgements

- Elkan, C. (2003) "Using the Triangle Inequality to Accelerate k-Means"
- Hamerly, G. (2010) "Making k-Means Even Faster"
- NVIDIA Nsight for profiling

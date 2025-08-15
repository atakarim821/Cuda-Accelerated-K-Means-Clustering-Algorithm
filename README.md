# Cuda-Accelerated-K-Means-Clustering-Algorithm
Optimized K-Means implementation leveraging NVIDIA GPUs. Features CUDA-accelerated k-means++ initialization, triangle inequality optimizations, and batched processing for large datasets. Multiple CUDA streams were used to overlap the computation and data transfer parts, hiding memory transfer latency by 40%.

Key Features ✨
⚡ GPU Acceleration: CUDA kernels for distance calculations and centroid updates

🎯 k-means++ Initialization: Better cluster seeding using GPU-parallelized selection

📊 Elkan/Hamerly Optimizations: 50-70% fewer distance calculations via triangle inequality

🔁 Multi-Stream Processing: 4+ CUDA streams hide 40% memory transfer latency

🧩 Batched Processing: Handles datasets larger than GPU memory (tested with 100M+ points)

💾 SoA Memory Layout: Structure-of-Arrays for coalesced memory access


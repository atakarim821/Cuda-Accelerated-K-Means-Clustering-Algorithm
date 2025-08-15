# Cuda-Accelerated-K-Means-Clustering-Algorithm
An optimized K-Means implementation leveraging NVIDIA GPUs. It features CUDA‑accelerated k‑means++ initialization, **triangle‑inequality** pruning (Elkan/Hamerly), **multi-stream** overlap of compute and transfer, and **batched** execution to scale beyond GPU memory. Data is stored in a **Structure‑of‑Arrays (SoA)** layout for coalesced memory access.

<p align="center"> <img src="https://img.shields.io/badge/CUDA-11%2B-76B900.svg"/> <img src="https://img.shields.io/badge/CMake-3.20%2B-blue.svg"/> <img src="https://img.shields.io/badge/License-MIT-lightgrey.svg"/> </p>

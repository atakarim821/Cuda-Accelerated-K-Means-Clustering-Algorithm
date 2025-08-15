import numpy as np
import time
import argparse
from cuml.cluster import KMeans as cuKMeans
import cupy as cp

def load_data(filename):
    with open(filename, "r") as f:
        lines = f.readlines()

    N = int(lines[0].strip())
    d = int(lines[1].strip())
    k = int(lines[2].strip())
    data = [list(map(int, line.strip().split())) for line in lines[3:3+N]]

    return np.array(data), N, d, k

def save_labels(labels, output_filename):
    with open(output_filename, "w") as f:
        for label in labels:
            f.write(f"{int(label)}\n")

def main():
    parser = argparse.ArgumentParser(description="Run cuML KMeans clustering on a dataset.")
    parser.add_argument("input_file", help="Path to input dataset file")
    parser.add_argument("--output_file", default="gpuOutput", help="Filename to save clustering labels")
    args = parser.parse_args()

    X_cpu, N, d, k = load_data(args.input_file)
    X_cpu = X_cpu.astype(np.float64)
    X = cp.asarray(X_cpu)  # convert to GPU array

    # Step 1: KMeans++ initialization and fitting
    start_total = time.time()

    start_kpp = time.time()
    kmeans = cuKMeans(n_clusters=k, init='k-means++', max_iter=100, output_type='cupy', random_state=0)
    kmeans.fit(X)
    end_kmeans = time.time()
    print(f"Final inertia after cuML KMeans fitting: {float(kmeans.inertia_):.6f}")
    print(f"KMeans++ initialization + fitting time (cuML does both together): {end_kmeans - start_kpp:.6f} seconds")

    labels = kmeans.labels_.get()  # move from GPU to CPU
    save_labels(labels, args.output_file)

if __name__ == "__main__":
    main()


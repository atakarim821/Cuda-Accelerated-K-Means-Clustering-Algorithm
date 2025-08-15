import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import kmeans_plusplus
from sklearn.metrics.pairwise import euclidean_distances
import time
import argparse

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
            f.write(f"{label}\n")

def main():
    parser = argparse.ArgumentParser(description="Run KMeans clustering on a dataset.")
    parser.add_argument("input_file", help="Path to input dataset file")
    parser.add_argument("--output_file", default="cpuOutput", help="Filename to save clustering labels")
    args = parser.parse_args()

    X, N, d, k = load_data(args.input_file)

    # Step 1: k-means++ initialization
    start_kpp = time.time()
    init_centers, indices = kmeans_plusplus(X, n_clusters=k, random_state=0)
    end_kpp = time.time()

    # Step 2: Compute initial inertia
    distances = euclidean_distances(X, init_centers)
    closest_dist_sq = np.min(distances**2, axis=1)
    initial_inertia = np.sum(closest_dist_sq)
    print("Initial inertia after k-means++ initialization:", initial_inertia)

    # Step 3: Proceed with KMeans
    start_kmeans = time.time()
    kmeans = KMeans(n_clusters=k, init=init_centers, n_init=1, max_iter=100, algorithm='lloyd', random_state=0)
    kmeans.fit(X)
    end_kmeans = time.time()
    
    print("Final inertia after KMeans fitting:", kmeans.inertia_)
    print(f"KMeans++ initialization time: {end_kpp - start_kpp:.6f} seconds")
    print(f"KMeans fitting time: {end_kmeans - start_kmeans:.6f} seconds")

#    labels = kmeans.labels_
#    save_labels(labels, args.output_file)

if __name__ == "__main__":
    main()


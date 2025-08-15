import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import time
import argparse

def generate_and_save_data(N=1000, d=2, k=12, filename="dataset.txt",
                           center_box=(-10000000, 10000000), cluster_std=100000000): # Use current time as random seed
    random_state = int(time.time())

    # Generate synthetic data
    X, y = make_blobs(n_samples=N, centers=k, n_features=d,
                      random_state=random_state, center_box=center_box, cluster_std=cluster_std)

    # Round and convert to integers
    X_rounded = np.round(X).astype(int)

    # Save to file
    with open(filename, "w") as f:
        f.write(f"{N}\n")
        f.write(f"{d}\n")
        f.write(f"{k}\n")
        for point in X_rounded:
            f.write(" ".join(map(str, point)) + "\n")

    # Plot if d == 2
    if d == 2:
        plt.figure(figsize=(8, 6))
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap='tab10', s=10, alpha=0.6)
        plt.title(f"Generated Data (N={N}, k={k}, d=2)")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic clustered data.")
    parser.add_argument("N", type=int, help="Number of samples to generate")
    parser.add_argument("d", type=int, help="Number of dimensions (features)")
    parser.add_argument("k", type=int, help="Number of clusters")
    args = parser.parse_args()

    generate_and_save_data(N=args.N, d=args.d, k=args.k, filename="dataset.txt")


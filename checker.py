import sys
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

def read_labels(filename):
    with open(filename, 'r') as f:
        return [int(line.strip()) for line in f]

def compare_clusters(file1, file2):
    labels1 = read_labels(file1)
    labels2 = read_labels(file2)

    if len(labels1) != len(labels2):
        raise ValueError("Files must have the same number of labels.")

    ari = adjusted_rand_score(labels1, labels2)
    nmi = normalized_mutual_info_score(labels1, labels2)

    print(f"Adjusted Rand Index (ARI): {ari:.4f}")
    print(f"Normalized Mutual Information (NMI): {nmi:.4f}")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} <file1> <file2>")
        sys.exit(1)

    file1 = sys.argv[1]
    file2 = sys.argv[2]

    compare_clusters(file1, file2)


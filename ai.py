import numpy as np
from sklearn import datasets

# Load the Iris dataset and exclude some examples for unknown samples
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Select some samples to be used as unknown samples
unknown_samples = [0, 50, 100]  # Index of samples to be excluded
X_unknown = X[unknown_samples]
y_unknown = y[unknown_samples]

# Remove the unknown samples from the dataset
X = np.delete(X, unknown_samples, axis=0)
y = np.delete(y, unknown_samples)

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def kNN(u, X, Y, k=3):
    distances = [euclidean_distance(u, x) for x in X]
    k_indices = np.argsort(distances)[:k]
    k_nearest_labels = [Y[i] for i in k_indices]

    label_count = {}
    for label in k_nearest_labels:
        if label in label_count:
            label_count[label] += 1
        else:
            label_count[label] = 1

    most_common_label = None
    most_common_count = 0

    for label, count in label_count.items():
        if count > most_common_count:
            most_common_count = count
            most_common_label = label

    return most_common_label

for u in X_unknown:
    predicted_label = kNN(u, X, y, k=3)
    print(f"Unknown sample {u} is classified as class {predicted_label}")
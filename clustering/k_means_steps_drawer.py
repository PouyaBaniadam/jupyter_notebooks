"""I had to draw some diagrams to show what I am saying,
So I got this code from dear 'Gemini' to draw some fix images;
So I could include them it in my notebook."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

X, y = make_blobs(n_samples=400, centers=3, cluster_std=0.8, random_state=101)


plt.figure(figsize=(6, 6))
plt.scatter(X[:, 0], X[:, 1], c='lightgray', s=50)

initial_centroids = np.array([[0, 8], [-10, 2], [5, -2]])
colors = ['red', 'green', 'blue']

for i in range(len(initial_centroids)):
    plt.scatter(initial_centroids[i][0], initial_centroids[i][1], marker='x', c=colors[i], s=300, linewidth=4,
                zorder=10)

plt.title("Step 1: Initial Random Centroids")
plt.xticks([])
plt.yticks([])
plt.savefig("kmeans_step1_initial.png")
plt.show()

plt.figure(figsize=(6, 6))

distances = np.sqrt(((X - initial_centroids[:, np.newaxis]) ** 2).sum(axis=2))
labels = np.argmin(distances, axis=0)
point_colors = [colors[label] for label in labels]

plt.scatter(X[:, 0], X[:, 1], c=point_colors, s=50)

for i in range(len(initial_centroids)):
    plt.scatter(initial_centroids[i][0], initial_centroids[i][1], marker='x', c=colors[i], s=300, linewidth=4,
                zorder=10)

plt.title("Step 2: First Flawed Assignment")
plt.xticks([])
plt.yticks([])
plt.savefig("kmeans_step2_assignment.png")
plt.show()

plt.figure(figsize=(6, 6))
kmeans = KMeans(n_clusters=3, random_state=101, n_init=10)
kmeans.fit(X)

final_labels = kmeans.labels_
final_centroids = kmeans.cluster_centers_
final_point_colors = [colors[label] for label in final_labels]

plt.scatter(X[:, 0], X[:, 1], c=final_point_colors, s=50)

for i in range(len(final_centroids)):
    plt.scatter(final_centroids[i][0], final_centroids[i][1], marker='x', c=colors[i], s=300, linewidth=4, zorder=10)

    cluster_points = X[final_labels == i]
    radius = np.max(np.sqrt(np.sum((cluster_points - final_centroids[i]) ** 2, axis=1))) * 1.1
    circle = plt.Circle(final_centroids[i], radius, color=colors[i], fill=False, linestyle='--', linewidth=2)
    plt.gca().add_artist(circle)

plt.title("Step 3: Final Converged Clusters")
plt.xticks([])
plt.yticks([])
plt.savefig("kmeans_step3_converged.png")
plt.show()
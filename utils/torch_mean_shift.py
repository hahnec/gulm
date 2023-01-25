#https://github.com/sotte/pytorch_tutorial/blob/master/notebooks/mean_shift_clustering.ipynb

import torch
from torch import exp


def distance_batch(a, b):
    return (((a[None,:] - b[:,None]) ** 2).sum(2))**.5

def gaussian(dist, bandwidth):
    return torch.exp(-0.5 * ((dist / bandwidth))**2) / (bandwidth * (2 * torch.pi)**.5)

def meanshift_torch2(data, batch_size=500):
    n = len(data)
    X = torch.from_numpy(np.copy(data)).cuda()
    for _ in range(1):
        for i in range(0, n, batch_size):
            s = slice(i, min(n, i + batch_size))
            weight = gaussian(distance_batch(X, X[s]), 2.5)
            num = (weight[:, :, None] * X).sum(dim=1)
            X[s] = num / weight.sum(1)[:, None]
    return X

def plot_data(centroids, data, n_samples):
    import matplotlib.pyplot as plt
    color = plt.cm.rainbow(np.linspace(0,1,len(centroids)))

    fig, ax = plt.subplots(figsize=(4, 4))
    for i, centroid in enumerate(centroids):
        samples = data[i * n_samples : (i + 1) * n_samples]
        ax.scatter(samples[:, 0], samples[:, 1], color=color[i], s=1)
        ax.plot(centroid[0], centroid[1], markersize=10, marker="x", color='k', mew=5)
        ax.plot(centroid[0], centroid[1], markersize=5, marker="x", color='m', mew=2)
    plt.axis('equal')
    plt.show()

if __name__ == '__main__':

    n_clusters = 6
    n_samples = 1000

    import numpy as np
    centroids = np.random.uniform(-35, 35, (n_clusters, 2))
    slices = [np.random.multivariate_normal(centroids[i], np.diag([5., 5.]), n_samples)
            for i in range(n_clusters)]
    data = np.concatenate(slices).astype(np.float32)

    X = meanshift_torch2(data, batch_size=1).cpu().numpy()

    plot_data(centroids, data, n_samples)
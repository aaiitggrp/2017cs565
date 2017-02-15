"""
Functions to load and visualize toy datasets.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

def noisy_moons(n_samples, noise=0.05):
    X, y = datasets.make_moons(n_samples=n_samples, noise=noise)
    return X, y

def blobs(n_samples, centers = 2, random_state=8):
    X, y = datasets.make_blobs(n_samples=n_samples, random_state=8, centers=centers)
    return X, y

def visualize_2d_dataset(inputs, labels, title):
    colors = np.array([x for x in 'bgrcmy'])
    plt.scatter(inputs[:,0], inputs[:,1], color=colors[labels.tolist()])
    plt.title(title)
    plt.show()

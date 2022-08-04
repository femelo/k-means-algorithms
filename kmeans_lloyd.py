from torch import square
from base_kmeans import BaseKMeans
import numpy as np
from copy import copy
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

class KMeansLloyd(BaseKMeans):
    def __init__(self, n_clusters, max_iter=100, tol=1e-4):
        super().__init__(n_clusters, max_iter, tol)
    
    def initialize_bounds(self, i, squared_norms):
        pass

    def outer_tests(self, x_i, i):
        return False

    def inner_tests(self, x_i, i, j, squared_lower_bound):
        return False

    def get_center_indexes(self, x_i, i):
        return list(range(self.k))

    def calc_center_shift(self, c_old, c_new):
        return np.linalg.norm(c_new - c_old)

    def update_assignment(self, i, x_i, squared_distances):
        a_prv = copy(self.a[i])
        a_new = np.argmin(squared_distances)
        assignment_changed = a_new != a_prv
        if assignment_changed:
            self.update_auxiliary_variables(i, x_i, a_prv, a_new)

if __name__ == "__main__":
    # Test
    x = 100.0 * np.random.randn(1000, 2)
    init_state = 100.0 * np.random.randn(15, 2)
    kmeans = KMeansLloyd(n_clusters=15, max_iter=1000, tol=1e-4)
    kmeans.fit(x.copy(), init_state=init_state)
    print("Inertia: {}".format(kmeans.inertia))
    # Kmeans sklearn
    kmeans_sklearn = KMeans(n_clusters=15, n_init=1, init=init_state, max_iter=1000, tol=1e-4)
    kmeans_sklearn.fit(x.copy())
    print("Inertia: {}".format(kmeans_sklearn.inertia_))

    colors = ['red', 'green', 'blue', 'magenta', 'orange']
    n_c = len(colors)
    fig, ax = plt.subplots(2, 1, figsize=(8, 8))
    for i in range(x.shape[0]):
        ax[0].scatter(x[i, 0], x[i, 1], c=colors[kmeans.a[i] % n_c], alpha=0.25, marker='x')
        ax[1].scatter(x[i, 0], x[i, 1], c=colors[kmeans_sklearn.labels_[i] % n_c], alpha=0.25, marker='x')
    for i in range(kmeans.k):
        ax[0].scatter(kmeans.c[i][0], kmeans.c[i][1], edgecolors=colors[i % n_c], facecolors='none', marker='o')
        ax[1].scatter(kmeans_sklearn.cluster_centers_[i, 0], kmeans_sklearn.cluster_centers_[i, 1], 
            edgecolors=colors[i % n_c], facecolors='none', marker='o')
    plt.draw()
    plt.show()



from base_kmeans import BaseKMeans
import numpy as np
from copy import copy
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

class KMeansExponion(BaseKMeans):
    def __init__(self, n_clusters, max_iter=100, tol=1e-4):
        super().__init__(n_clusters, max_iter, tol)
        # Exponion parameters
        self.F = np.arange(1, int(np.ceil(np.log2(self.k))) + 1)
        self.w = np.zeros((self.k, self.k), dtype=np.int32)
        self.e = np.zeros((self.k, self.k))
    
    def preprocess(self):
        c = np.vstack(self.c)
        for k in range(self.k):
            distances = np.sqrt(np.sum((c - c[k]) ** 2, axis=1))
            self.D[k] = distances.copy()
            distances[k] = np.nan
            self.s[k] = np.nanmin(distances)

        # Construct layers
        indices = 2 ** (self.F - 1)
        for k in range(self.k):
            self.w[k, :] = np.argpartition(self.D[k], indices)
            self.e[k, :] = self.D[k][self.w[k, :]]

        return

    def initialize(self, x, init_state=None, random_state=None):
        super().initialize(x, init_state, random_state)
        for i in range(self.N):
            self.tight_bounds['upper'].append(True)
            self.tight_bounds['lower'].append(True)
        return

    def outer_tests(self, x_i, i):
        # First verification
        t1 = max(self.l[i], self.s[self.a[i]] / 2)
        if self.u[i] <= t1:
            return True
        # Tighen upper bound
        if not self.tight_bounds['upper'][i]:
            self.u[i] = np.sqrt(np.sum((x_i - self.c[self.a[i]]) ** 2))
            self.tight_bounds['upper'][i] = True
            # Second verification
            if self.u[i] <= t1:
                return True
        return False

    def inner_tests(self, x_i, i, j, squared_lower_bound):
        return False

    def get_center_indexes(self, x_i, i):
        # Set search radius
        R = 2 * self.u[i] + self.s[self.a[i]]

        # Calculate indices
        indices = 2 ** (self.F - 1)
        index_min = self.k - 1
        # Find candidate set of centers within search radius
        for j in indices:
            if self.e[self.a[i], j] >= R:
                index_min = j
                break
        J = self.w[self.a[i], :index_min+1].tolist()
        return J

    def calc_center_shift(self, c_old, c_new):
        return np.linalg.norm(c_new - c_old)

    def update_assignment(self, i, x_i, squared_distances):
        a_prv = copy(self.a[i])
        a_new = np.argmin(squared_distances)
        squared_distances[a_prv] = np.inf
        b_new = np.argmin(squared_distances)
        assignment_changed = a_new != a_prv
        if assignment_changed:
            self.update_auxiliary_variables(i, x_i, a_prv, a_new)
        self.b[i] = b_new
        self.u[i] = np.sqrt(squared_distances[a_new])
        self.l[i] = np.sqrt(squared_distances[b_new])

    def initialize_bounds(self, i, squared_norms):
        self.u[i] = np.sqrt(squared_norms[self.a[i]])
        self.l[i] = np.sqrt(squared_norms[self.b[i]])

    def update_bounds(self):
        N = self.N
        p = copy(self.p)
        r1 = np.argmax(p)
        p[r1] = 0.0
        r2 = np.argmax(p)
        for i in range(N):
            self.u[i] += self.p[self.a[i]]
            self.l[i] -= self.p[r1] if r1 != self.a[i] else self.p[r2]
            self.tight_bounds['upper'][i] = False
            self.tight_bounds['lower'][i] = False

if __name__ == "__main__":
    # Test
    x = 100.0 * np.random.randn(1000, 2)
    init_state = 100.0 * np.random.randn(15, 2)
    kmeans = KMeansExponion(n_clusters=15, max_iter=1000, tol=1e-4)
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



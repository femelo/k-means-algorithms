from base_kmeans import BaseKMeans
import numpy as np
from copy import copy
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster._kmeans import _k_init
from sklearn.utils.extmath import squared_norm
from sklearn.utils.validation import check_random_state

class KMeansYinYang(BaseKMeans):
    def __init__(self, n_clusters, max_iter=100, tol=1e-4, n_groups=None):
        super().__init__(n_clusters, max_iter, tol)
        # Number of groups of clusters
        if n_groups is None:
            self.n = max(1, n_clusters // 10)
        else:
            self.n = min(n_groups, n_clusters)
        # Groups of clusters
        self.G = {}
        # Assignment of points to groups of clusters
        self.g = []
    
    def preprocess(self):
        c = np.vstack(self.c)
        for k in range(self.k):
            distances = np.sqrt(np.sum((c - c[k]) ** 2, axis=1))
            self.D[k] = distances.copy()
            distances[k] = np.nan
            self.s[k] = np.nanmin(distances)
        return

    def initialize(self, x, init_state=None, random_state=None):
        if init_state is None:
            x_squared_norms = squared_norm(x)
            random_state = check_random_state(random_state)
            centers = _k_init(x, self.k, x_squared_norms, random_state, n_local_trials=None)
        elif hasattr(init_state, '__array__'):
            # ensure that the centers have the same dtype as X
            # this is a requirement of fused types of cython
            centers = np.array(init_state, dtype=x.dtype)
        else:
            raise ValueError("The initial state must be an array or None")
        # Initialize groups of clusters
        kmeans = KMeans(n_clusters=self.n).fit(centers)
        self.G['indices'] = kmeans.labels_
        for f in range(self.n):
            self.G[f] = {
                'center': kmeans.cluster_centers_[f], 
                'indices': set(np.where(kmeans.labels_ == f)[0])
            }
        super().initialize(x, centers, random_state)
        # Initialize assignment of points to groups of clusters
        for i in range(self.N):
            self.g.append(self.G['indices'][self.a[i]])
        for i in range(self.N):
            self.tight_bounds['upper'].append(True)
            self.tight_bounds['lower'].append([True for j in range(self.n)])
        return

    def outer_tests(self, x_i, i):
        # First verification
        t1 = min(self.l[i])
        if self.u[i] <= t1:
            return True
        return False

    def inner_tests(self, x_i, i, j, squared_lower_bound):
        return False

    def get_center_indexes(self, x_i, i):
        # Based on a set of group tests generate a set with the relevant center indexes
        J = []
        for f in range(self.n):
            if self.u[i] >= self.l[i][f]:
                J += list(self.G[f]['indices'])
        # This means that all group tests failed
        if len(J) == 0:
            # Tighen upper bound
            if not self.tight_bounds['upper'][i]:
                self.u[i] = np.sqrt(np.sum((x_i - self.c[self.a[i]]) ** 2))
                self.tight_bounds['upper'][i] = True
                # Second batch of group tests
                for f in range(self.n):
                    if self.u[i] >= self.l[i][f]:
                        J += list(self.G[f]['indices'])
        return J

    def calc_center_shift(self, c_old, c_new):
        return np.linalg.norm(c_new - c_old)

    def update_assignment(self, i, x_i, squared_distances):
        a_prv = copy(self.a[i])
        # g_prv = copy(self.g[i])
        a_new = np.argmin(squared_distances)
        g_new = self.G['indices'][a_new]
        assignment_changed = a_new != a_prv
        if assignment_changed:
            self.update_auxiliary_variables(i, x_i, a_prv, a_new)
        self.g[i] = g_new
        self.u[i] = np.sqrt(squared_distances[a_new])
        set_of_indices = [list(self.G[f]['indices'] - set([a_new])) for f in range(self.n)]
        squared_norms = np.array(squared_distances)
        lower_bounds = np.array(
            [np.min(squared_norms[set_of_indices[f]]) if len(set_of_indices[f]) > 0 else np.inf for f in range(self.n)]
        )
        for f in range(self.n):
            if np.isfinite(lower_bounds[f]):
                self.l[i][f] = np.sqrt(lower_bounds[f])
        return

    def initialize_bounds(self, i, squared_norms):
        self.u[i] = np.sqrt(squared_norms[self.a[i]])
        set_of_indices = [list(self.G[f]['indices'] - set([self.a[i]])) for f in range(self.n)]
        lower_bounds = [np.min(np.array(squared_norms)[set_of_indices[f]]) if len(set_of_indices[f]) > 0 else np.inf for f in range(self.n)]
        self.l[i] = np.sqrt(lower_bounds)
        return

    def update_bounds(self):
        N = self.N
        p = np.array(self.p)
        for i in range(N):
            self.u[i] += self.p[self.a[i]]
            for f in range(self.n):
                self.l[i][f] = max(self.l[i][f] - np.max(p[list(self.G[f]['indices'])]), 0)
            self.tight_bounds['upper'][i] = False
            self.tight_bounds['lower'][i] = [False for f in range(self.n)]
        return

if __name__ == "__main__":
    # Test
    x = 100.0 * np.random.randn(1000, 2)
    init_state = 100.0 * np.random.randn(15, 2)
    kmeans = KMeansYinYang(n_clusters=15, max_iter=1000, tol=1e-4, n_groups=5)
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



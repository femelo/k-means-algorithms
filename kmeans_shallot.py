from base_kmeans import BaseKMeans
import numpy as np
from copy import copy
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

class KMeansShallot(BaseKMeans):
    def __init__(self, n_clusters, max_iter=100, tol=1e-4):
        super().__init__(n_clusters, max_iter, tol)
        # Shallot parameters
        self.w = np.zeros((self.k, self.k), dtype=np.int32)
        self.e = np.zeros((self.k, self.k))
    
    def preprocess(self):
        c = np.vstack(self.c)
        for k in range(self.k):
            distances = np.sqrt(np.sum((c - c[k]) ** 2, axis=1))
            self.D[k] = distances.copy()
            distances[k] = np.nan
            self.s[k] = np.nanmin(distances)

        # Set parameters
        for k in range(self.k):
            self.w[k, :] = np.argsort(self.D[k])
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
        dist = np.linalg.norm((x_i - self.c[self.b[i]]))
        if dist < self.u[i]:
            j1 = self.b[i]
            jc = self.a[i]
            u_i = dist
        else:
            j1 = self.a[i]
            jc = self.b[i]
            u_i = self.u[i]
        dist = np.linalg.norm((x_i - self.c[jc]))
        if u_i + dist < 2 * u_i + self.s[self.a[i]]:
            # j2 = jc
            l_i = u_i + dist
        else:
            # j2 = self.w[jc, 1]
            l_i = 2 * u_i + self.s[self.a[i]]
        
        # Set search radius
        r_i = u_i + l_i

        J = [j1]
        for k in range(1, self.k):
            if self.e[j1, k] >= r_i:
                break
            J.append(self.w[j1, k])
            if self.e[j1, k] < u_i:
                # j2 = j1
                j1 = self.w[j1, k]
                l_i = u_i
                r_i = u_i + l_i
                u_i = self.e[j1, k]
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
    kmeans = KMeansShallot(n_clusters=15, max_iter=1000, tol=1e-4)
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



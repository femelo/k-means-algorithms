from attr import has
import numpy as np
from sklearn.cluster._kmeans import _k_init, _tolerance
from sklearn.utils.extmath import row_norms, squared_norm
from sklearn.utils.validation import check_random_state
from sklearn.cluster import _k_means_fast
import itertools

class BaseKMeans(object):
    def __init__(self, n_clusters, max_iter=100, tol=1e-4):
        self.k = n_clusters
        # Cluster assignments
        self.a = []
        # Assignments to the second-closest centroid
        self.b = []
        # Cluster centers
        self.c = []
        # Norm of cluster centroid shifts (from one iteration to the next)
        self.p = []
        # Cluster centroid shifts (from one iteration to the next)
        self.delta_c = []
        # Data points upper bounds
        self.u = []
        # Data points lower bounds
        self.l = []
        # Centroid distances to the closest centroid
        self.s = []
        # Number of points assigned per cluster
        self.q = []
        # Accumulation of points assigned per cluster
        self.C = []
        # Center to center distances
        self.D = []
        self.tightened = []
        self.max_iter = max_iter
        self.tol = tol
        self.iter = 0
        self.x_cross_products = None
        self.inertia = 0.0
        self.N = 0
        self.d = 0
        self.tight_bounds = {'upper': [], 'lower': []}
        self.rate = 0.0

    def converged(self):
        if self.iter > 0:
            M = np.vstack(self.delta_c)
            norm_of_shifts = squared_norm(M)
            print("Iteration {:03d}: rate = {:06.2f}%, norm = {}".format(self.iter, 100.0 * self.rate, norm_of_shifts))
            r = norm_of_shifts <= self.tol
        else:
            r = False
        return r

    def iterate(self):
        self.iter += 1

    def initial_assignment(self, x, centers):
        squared_distances = np.sum((x.reshape(1, -1) - np.vstack(centers)) ** 2, axis=1)
        a, b = np.argsort(squared_distances)[:2]
        return a, b, squared_distances

    def initialize_bounds(self, i, squared_norms):
        pass

    def initialize(self, x, init_state=None, random_state=None):
        self.tol = _tolerance(x, self.tol)
        N, d = x.shape
        self.N = N
        self.d = d
        self.x_cross_products = np.dot(x, x.T)
        if init_state is None:
            x_squared_norms = self.x_cross_products.diagonal()
            random_state = check_random_state(random_state)
            centers = _k_init(x, self.k, x_squared_norms, random_state, n_local_trials=None)
        elif hasattr(init_state, '__array__'):
            # ensure that the centers have the same dtype as X
            # this is a requirement of fused types of cython
            centers = np.array(init_state, dtype=x.dtype)
        else:
            raise ValueError("The initial state must be an array or None")
        self.c = [c for c in centers]
        for j in range(self.k):
            self.C.append(np.zeros((d, )))
            self.q.append(0)
            self.delta_c.append(None)
            self.p.append(None)
            self.D.append(None)
            self.s.append(None)
        for i in range(N):
            a_i, b_i, squared_norms = self.initial_assignment(x[i], centers)
            self.a.append(a_i)
            self.b.append(b_i)
            self.u.append(None)
            self.l.append(None)
            self.initialize_bounds(i, squared_norms)
            self.C[a_i] += x[i]
            self.q[a_i] += 1
        return

    def preprocess(self):
        pass

    def outer_tests(self, x_i, i):
        pass

    def inner_tests(self, x_i, i, j, squared_lower_bound):
        pass

    def get_center_indexes(self, x_i, i):
        pass

    def update_assignment(self, i, x_i, squared_distances, a_new = None):
        pass

    def update_inertia(self):
        self.inertia = 0.0
        a = np.array(self.a)
        for k in range(self.k):
            if self.q[k] > 1:
                indices = set(np.where(a == k)[0])
                wcss = 0.0
                for i, j in itertools.combinations(indices, 2):
                    wcss += \
                        self.x_cross_products[i, i] + \
                        self.x_cross_products[j, j] - \
                        2.0 * self.x_cross_products[i, j]
                self.inertia += (wcss / len(indices))
        return
                    
    def calc_center_shift(self, c_old, c_new):
        pass

    def update_auxiliary_variables(self, i, x_i, a_old, a_new):
        self.a[i] = a_new
        self.q[a_old] -= 1
        self.C[a_old] -= x_i
        self.q[a_new] += 1
        self.C[a_new] += x_i

    def move_centers(self):
        d = self.d
        for j in range(self.k):
            c_prv = self.c[j].copy()
            if self.q[j] > 0:
                self.c[j] = self.C[j] / self.q[j]
                self.delta_c[j] = self.c[j] - c_prv
                self.p[j] = self.calc_center_shift(c_prv, self.c[j])
            else:
                self.c[j] = np.zeros((d, ))
                self.delta_c[j] = 2.0 * self.tol * np.random.randn(d)
                self.p[j] = 0.0
        return

    def update_bounds(self):
        pass

    def fit(self, x, init_state=None, random_state=None):
        N = x.shape[0]
        x_mean = np.mean(x, axis=0)
        x -= x_mean

        if hasattr(init_state, '__array__'):
            init_state_ = init_state - x_mean
        else:
            init_state_ = init_state
        self.initialize(x, init_state_, random_state)

        while not self.converged() and self.iter < self.max_iter:
            # Preprocessing
            self.preprocess()
            self.rate = 0.0
            # Outer loop: iterate over the data points
            for i in range(N):
                # Outer tests
                if self.outer_tests(x[i], i):
                    continue
                # Inner loop: iterate over the clusters
                squared_distances = [np.inf for j in range(self.k)]
                center_indexes = self.get_center_indexes(x[i], i)
                self.rate += 1.0 / N
                for j in center_indexes:
                    # Inner tests
                    squared_lower_bound = None
                    # Set squared_lower_bound in place
                    if self.inner_tests(x[i], i, j, squared_lower_bound):
                        continue
                    # Update distances
                    if squared_lower_bound is None:
                        squared_distances[j] = np.sum((x[i] - self.c[j]) ** 2)
                    else:
                        squared_distances[j] = squared_lower_bound
                # Update assignment
                self.update_assignment(i, x[i], squared_distances)
            self.move_centers()
            self.update_bounds()
            self.update_inertia()
            self.iterate()
        self.c = [c + x_mean for c in self.c]
from dis import dis
from base_kmeans import BaseKMeans
import numpy as np
from copy import copy
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.neighbors import BallTree
from typing import Union, List, Any
import heapq

def heappush_max(heap, item):
    heap.insert(0, item)
    heapq._siftup_max(heap, 0)

class Node(object):
    def __init__(self, 
    pivot: Union[int, None] = None, 
    parent: Union[Any, None] = None,
    children: Union[List[int], None] = None,
    l_node: Union[Any, None] = None,
    r_node: Union[Any, None] = None,
    radius: Union[float, None] = None,
    centroid: Union[np.ndarray, None] = None,
    d_minnq: Union[float, None] = None):
        self.pivot = pivot
        self.parent = parent
        self.children = children
        self.l_node = l_node
        self.r_node = r_node
        self.radius = radius
        self.centroid = centroid
        self.d_minnq = d_minnq
        self.d_pq = None

class BallTree1(object):
    def __init__(self, x, distance_matrix=None):
        ids = np.arange(x.shape[0])
        self.data = x
        self.ids = ids
        if distance_matrix is None:
            n = x.shape[0]
            distance_matrix = np.zeros((n, n))
            for i in range(n):
                distance_matrix[i, :] = np.sqrt(np.sum((x - x[i]) ** 2, axis=1))
        self.distance_matrix = distance_matrix
        self.root = self.build_tree(pivot=None, parent=None, ids=self.ids)
        self.d_sofar = None
        
    def build_tree(self, pivot, parent, ids):
        if len(ids) == 1:
            node = Node(pivot=pivot, parent=parent, radius=0.0, centroid=self.data[pivot, :])
        else:
            # Split data
            data = self.data[ids]
            # Get first pivot
            centroid = np.sum(data, axis=0) / len(data)
            l1_norms = np.sum(np.abs(data - centroid), axis=1)
            p1_idx = np.argmax(l1_norms)
            p1_id = ids[p1_idx]
            # Get second pivot
            l2_norms = self.distance_matrix[p1_id, ids]
            p2_idx = np.argmax(l2_norms)
            p2_id = ids[p2_idx]
            # Get centroid
            i1, i2 = np.unravel_index(np.argmax(self.distance_matrix[np.ix_(ids, ids)]), (len(ids), len(ids)))
            if pivot is None:
                centroid = 0.5 * (data[i1] + data[i2])
            else:
                centroid = self.data[pivot, :]
            # Get current ball radius
            radius = self.distance_matrix[np.ix_(ids, ids)][i1, i2]
            # Partition left and right nodes
            mid_distance = self.distance_matrix[p1_id, p2_id] / 2
            l_node_ids = ids[self.distance_matrix[p1_id, ids] <= mid_distance]
            r_node_ids = np.array(list(set(ids) - set(l_node_ids)))
            node = Node(pivot=pivot, parent=parent, children=ids, radius=radius, centroid=centroid)
            node.l_node = self.build_tree(pivot=p1_id, parent=node, ids=l_node_ids)
            node.r_node = self.build_tree(pivot=p2_id, parent=node, ids=r_node_ids)
        return node

    def ball_knn(self, q, k, node, nn_in, r = None):
        if r is None:
            second_condition = True
        else:
            second_condition = node.d_minnq > r
        if node.d_minnq >= self.d_sofar and second_condition:
            return nn_in

        if node.children is None:
            nn_out = copy(nn_in)
            for i in [node.pivot]:
                d = node.d_pq
                if d < self.d_sofar:
                    heappush_max(nn_out, (d, i))
                    if len(nn_out) == k + 1:
                        _ = heapq._heappop_max(nn_out)
                    if len(nn_out) == k:
                        self.d_sofar = nn_out[0][0]
        else:
            # Update node's d_minnq
            if node.parent is None:
                node.d_pq = np.linalg.norm(q - node.centroid)
                node.d_minnq = max(node.d_pq - node.radius, 0.0)
            d_r = np.linalg.norm(q - node.r_node.centroid)
            d_l = np.linalg.norm(q - node.l_node.centroid)
            node.r_node.d_pq = d_r
            node.l_node.d_pq = d_l
            node.r_node.d_minnq = max(d_r - node.radius, node.d_minnq)
            node.l_node.d_minnq = max(d_l - node.radius, node.d_minnq)
            nn_tmp = copy(nn_in)
            if r is None:
                if d_l <= d_r:
                    node1 = node.l_node
                    node2 = node.r_node
                else:
                    node1 = node.r_node
                    node2 = node.l_node
                nn_tmp = self.ball_knn(q, k, node1, nn_in, r)
                nn_out = self.ball_knn(q, k, node2, nn_tmp, r)
            else:
                if d_r <= node.r_node.radius + r:
                    nn_tmp = self.ball_knn(q, k, node.r_node, nn_in, r)
                if d_l <= node.l_node.radius + r:
                    nn_out = self.ball_knn(q, k, node.l_node, nn_tmp, r)
                else:
                    nn_out = nn_tmp
        return nn_out

    def query(self, q, k, r = None):
        # Set parameters
        self.d_sofar = np.inf
        self.root.d_minnq = 0.0
        nn_out = self.ball_knn(q, k, self.root, [], r)
        heapq.heapify(nn_out)
        distances = np.array([item[0] for item in nn_out])
        indices = np.array([item[1] for item in nn_out])
        return distances, indices

class KMeansBT(BaseKMeans):
    def __init__(self, n_clusters, max_iter=100, tol=1e-4):
        super().__init__(n_clusters, max_iter, tol)
        # Shallot parameters
        self.ball_tree = None
        self.u_vector = None
        self.l_vector = None
        self.query_results = None
    
    def preprocess(self):
        c = np.vstack(self.c)
        for k in range(self.k):
            distances = np.sqrt(np.sum((c - c[k]) ** 2, axis=1))
            self.D[k] = distances.copy()
            distances[k] = np.nan
            self.s[k] = np.nanmin(distances)
        self.ball_tree = BallTree1(c, distance_matrix=np.vstack(self.D))
        # self.ball_tree1 = BallTree(c)
        return

    def initialize(self, x, init_state=None, random_state=None):
        super().initialize(x, init_state, random_state)
        c = np.vstack(self.c)
        self.u_vector = x - c[self.a]
        self.l_vector = x - c[self.b]
        for i in range(self.N):
            self.tight_bounds['upper'].append(True)
            self.tight_bounds['lower'].append(True)
        return

    def outer_tests(self, x_i, i):
        # First verification
        # t1 = self.l[i]
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
        r_i = self.u[i] + self.s[self.a[i]]
        self.query_results = self.ball_tree.query(x_i.reshape(1,-1), k=2, r=r_i)
        # dst, idx = self.ball_tree1.query(x_i.reshape(1,-1), k=2)
        # dst = dst.ravel()
        # idx = idx.ravel()
        # self.query_results = (distances, indices)
        return []

    def calc_center_shift(self, c_old, c_new):
        return np.linalg.norm(c_new - c_old)

    def update_assignment(self, i, x_i, squared_distances):
        distances, indices = self.query_results
        a_prv = copy(self.a[i])
        a_new = indices[0]
        b_new = indices[1]
        assignment_changed = a_new != a_prv
        if assignment_changed:
            self.update_auxiliary_variables(i, x_i, a_prv, a_new)
        self.b[i] = b_new
        self.u[i] = distances[0]
        self.l[i] = distances[1]

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
    n_samples = 1000
    n_clusters = 15
    n_dim = 10
    x = 100.0 * np.random.randn(n_samples, n_dim)
    init_state = 100.0 * np.random.randn(n_clusters, n_dim)
    kmeans = KMeansBT(n_clusters=n_clusters, max_iter=1000, tol=1e-4)
    kmeans.fit(x.copy(), init_state=init_state)
    print("Inertia: {}".format(kmeans.inertia))
    # Kmeans sklearn
    kmeans_sklearn = KMeans(n_clusters=n_clusters, n_init=1, init=init_state, max_iter=1000, tol=1e-4)
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



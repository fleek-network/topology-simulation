from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from constrained_kmedoids import ConstrainedKMedoids
import numpy as np
import time

embedding = MDS(n_components=2, metric=True, dissimilarity='precomputed', normalized_stress='auto')

A = np.random.rand(16, 16)
D = np.dot(A.T, A)
start = time.time()
data_points_mds = embedding.fit_transform(D)
print(f'mds duration: {time.time() - start}')


X = np.random.rand(8, 8)
pca = PCA(n_components=2)
start = time.time()
pca.fit(X)
print(f'pca duration: {time.time() - start}')


num_nodes = 10_000
target_cluster_size = 8
num_clusters = num_nodes // target_cluster_size
nodes = []
for i in range(num_clusters):
    cluster_x = np.random.uniform(low=0, high=100)
    cluster_y = np.random.uniform(low=0, high=100)
    cluster_scale = np.random.uniform(low=1, high=4)
    node_location = np.random.multivariate_normal(mean=[cluster_x, cluster_y], cov=np.eye(2) * cluster_scale, size=target_cluster_size)
    nodes.append(node_location)
nodes = np.concatenate(nodes, axis=0)
D = np.sum((np.expand_dims(nodes, axis=1) - nodes)**2, axis=-1)

kmedoids_constr = ConstrainedKMedoids(n_clusters=10_000 // 8, method='pam', min_cluster_size=6, max_cluster_size=18, metric='precomputed', random_state=0)
start = time.time()
kmedoids_constr.fit(D)
print(f'ConstrainedKMedoids duration: {time.time() - start}')

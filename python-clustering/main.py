import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
from k_means_constrained import KMeansConstrained
from sklearn.manifold import MDS


COLORS = [(230, 25, 75),
          (60, 180, 75),
          (255, 225, 25),
          (0, 130, 200),
          (245, 130, 48),
          (145, 30, 180),
          (70, 240, 240),
          (240, 50, 230),
          (210, 245, 60),
          (250, 190, 212),
          (0, 128, 128),
          (220, 190, 255),
          (170, 110, 40),
          (255, 250, 200),
          (128, 0, 0),
          (170, 255, 195),
          (128, 128, 0),
          (255, 215, 180),
          (0, 0, 128),
          (128, 128, 128),
          (0, 0, 0)]

def scatter_plot(points, assignment, title=''):
    fig = plt.figure(figsize=(14, 10))
    plt.title(title)
    for i in range(points.shape[0]):
        cluster_idx = assignment[i]
        sns.scatterplot(x=[points[i,0]], y=[points[i,1]], color=np.array(COLORS[cluster_idx % len(COLORS)]) / 255.)
    buf = io.BytesIO()
    plt.savefig(buf, format='svg')
    return buf.getvalue().decode("utf-8")

def scatter_plot_3d(points, assignment, title=''):
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(projection='3d')
    plt.title(title)
    for i in range(points.shape[0]):
        cluster_idx = assignment[i]
        ax.scatter([points[i,0]], [points[i,1]], [points[i,2]], color=np.array(COLORS[cluster_idx % len(COLORS)]) / 255.)
    buf = io.BytesIO()
    plt.savefig(buf, format='svg')
    return buf.getvalue().decode("utf-8")

def run_toy_example():
    cluster1 = np.random.multivariate_normal(mean=[1,1], cov=np.eye(2), size=10)
    cluster2 = np.random.multivariate_normal(mean=[2,2], cov=np.eye(2), size=10)
    cluster3 = np.random.multivariate_normal(mean=[4,2], cov=np.eye(2), size=10)
    
    data = np.concatenate((cluster1, cluster2, cluster3), axis=0)
    
    num_nodes = data.shape[0]
    num_clusters = num_nodes // 8

    svg_plot_before = scatter_plot(data, np.zeros(data.shape[0], dtype='int'), title='Before Clustering')

    clf = KMeansConstrained(
        n_clusters=num_clusters,
        size_min=2,
        size_max=10,
        random_state=0
    )
    clf.fit_predict(data)
    clf.cluster_centers_
    clf.labels_
    svg_plot_kmeans = scatter_plot(data, clf.labels_, title='KMeansConstrained')


    html = f"""<!DOCTYPE html>
    <html>
    <head>
        <title>Topology Simulation Report</title>
        <style>
            @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono&display=swap');
            body {{ font-family: 'IBM Plex Mono, monospace', monospace }}
            tr:nth-child(even) {{
                background-color: rgba(150, 212, 212, 0.4);
            }}
        </style>
    </head>
    <body>
        <h1>Toy Example Report</h1>
        <p>The clustering </p>
        <hr>
        {svg_plot_before}
        <hr>
        <hr>
        {svg_plot_kmeans}
        <hr>
    <body>
    </html>"""

    with open('toy_report.html', 'w') as text_file:
        text_file.write(html)


def run():
    matrix = pd.read_csv('matrix.csv', header=None)
    metadata = pd.read_csv('metadata.csv')
    data_points = np.stack((metadata['longitude'].to_numpy(), metadata['latitude'].to_numpy()), axis=1)

    svg_plot_before = scatter_plot(data_points, np.zeros(data_points.shape[0], dtype='int'), title='Before Clustering')
    
    embedding = MDS(n_components=3, metric=True, dissimilarity='precomputed', normalized_stress='auto')
    matrix = matrix.to_numpy()

    # TODO: average upper and lower triangular matrices
    i_lower = np.tril_indices(matrix.shape[0], -1)
    matrix[i_lower] = matrix.T[i_lower]
    data_points_mds = embedding.fit_transform(matrix)

    svg_plot_mds = scatter_plot_3d(data_points_mds, np.zeros(data_points.shape[0], dtype='int'), title='Before Clustering MDS')

    num_nodes = data_points.shape[0]
    num_clusters = num_nodes // 8
    clf = KMeansConstrained(
        n_clusters=num_clusters,
        size_min=2,
        size_max=10,
        random_state=0
    )
    clf.fit_predict(data_points_mds)
    clf.cluster_centers_
    clf.labels_
    svg_plot_kmeans = scatter_plot(data_points, clf.labels_, title='KMeansConstrained')
    svg_plot_kmeans_mds = scatter_plot_3d(data_points_mds, clf.labels_, title='KMeansConstrained (MDS embeddings)')

    html = f"""<!DOCTYPE html>
    <html>
    <head>
        <title>Topology Simulation Report</title>
        <style>
            @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono&display=swap');
            body {{ font-family: 'IBM Plex Mono, monospace', monospace }}
            tr:nth-child(even) {{
                background-color: rgba(150, 212, 212, 0.4);
            }}
        </style>
    </head>
    <body>
        <h1>Topology Simulation Report</h1>
        <p>Servers</p>
        <hr>
        {svg_plot_before}
        <hr>
        {svg_plot_mds}
        <hr>
        {svg_plot_kmeans_mds}
        <hr>
        {svg_plot_kmeans}
        <hr>
    <body>
    </html>"""

    with open('report.html', 'w') as text_file:
        text_file.write(html)


if __name__ == '__main__':
    # run_toy_example()
    run()
    

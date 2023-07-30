from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from collections import Counter
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import os
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

df = pd.read_csv('/home/minkyoon/crohn/csv/clam/clustering/relapse/extracted_features.csv')
filenames = df['filename'].values
labels = df['label'].values
accession_numbers = df['accession_number'].values

# Load features
features = []
for filename in filenames:
    feature_path = os.path.join('/home/minkyoon/crohn/csv/clam/clustering/relapse/', filename + '.pt')
    feature = torch.load(feature_path)
    features.append(feature.numpy())
features = np.vstack(features)



# Dimensionality reduction using t-SNE
tsne = TSNE(n_components=2, random_state=0)
features_2d = tsne.fit_transform(features)

# Clustering with DBSCAN
dbscan = DBSCAN(eps=5, min_samples=20)  # You may need to adjust these parameters depending on your data
cluster_labels = dbscan.fit_predict(features_2d)

# Plotting
plt.figure(figsize=(6, 6))
scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=cluster_labels)
plt.legend(*scatter.legend_elements(), title="Clusters")
plt.show()




from collections import Counter
from PIL import Image
import os

# Count the cluster labels
counter = Counter(cluster_labels)

# Filter out the noise cluster (label is -1 in DBSCAN)
filtered_clusters = {label: count for label, count in counter.items() if label != -1}

# Get the smallest cluster
smallest_cluster_label = min(filtered_clusters, key=filtered_clusters.get)

# Get the filenames of the images in the smallest cluster
smallest_cluster_filenames = [filename for filename, label in zip(filenames, cluster_labels) if label == smallest_cluster_label]

# Open and display the images in the smallest cluster
for filename in smallest_cluster_filenames:
    image_path = os.path.join('/path/to/your/images', filename + '.jpg')  # Modify this as needed
    image = Image.open(image_path)
    image.show()

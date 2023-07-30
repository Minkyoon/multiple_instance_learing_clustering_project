import pandas as pd
import numpy as np
import torch
import os
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Load data
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

# Clustering
kmeans = KMeans(n_clusters=2, random_state=0).fit(features)
cluster_labels = kmeans.labels_

# Visualization
tsne = TSNE(n_components=2, random_state=0)
features_2d = tsne.fit_transform(features)

plt.figure(figsize=(6, 6))
scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=cluster_labels)
plt.legend(*scatter.legend_elements(), title="Clusters")
plt.show()





import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Apply PCA to features
pca = PCA(n_components=2)
features_2d = pca.fit_transform(features)

# Plot the 2D features
plt.scatter(features_2d[:, 0], features_2d[:, 1], c='b')
plt.title("PCA")
plt.show()



import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Apply t-SNE to features
tsne = TSNE(n_components=2, random_state=0)
features_2d = tsne.fit_transform(features)

# Plot the 2D features
plt.scatter(features_2d[:, 0], features_2d[:, 1], c='r')
plt.title("t-SNE")
plt.show()

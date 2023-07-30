import pandas as pd
import numpy as np
import torch
import os
from sklearn.cluster import DBSCAN
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
dbscan = DBSCAN(eps=3, min_samples=100).fit(features)  # Modify eps and min_samples as necessary
cluster_labels = dbscan.labels_

# Visualization
tsne = TSNE(n_components=2, random_state=0)
features_2d = tsne.fit_transform(features)

plt.figure(figsize=(6, 6))
scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=cluster_labels)
plt.legend(*scatter.legend_elements(), title="Clusters")
plt.show()




## 이미지열기

from collections import Counter
from PIL import Image
import matplotlib.pyplot as plt

# Count the number of data points in each cluster
cluster_sizes = Counter(cluster_labels)

# Find the smallest cluster
# Ignore cluster -1, which represents noise
smallest_cluster = min((c for c in cluster_sizes if c != -1), key=cluster_sizes.get)

# Get the filenames of the images in the smallest cluster
smallest_cluster_filenames = [filenames[i] for i in range(len(filenames)) if cluster_labels[i] == smallest_cluster]

# Choose a filename to view
chosen_filename = smallest_cluster_filenames[0]  # Change this to view different images

# Open the image file
# Replace this with the correct path to your images
image_path = os.path.join('/path/to/your/images', chosen_filename + '.png')  # Or whatever the file extension is
image = Image.open(image_path)

# Display the image
plt.imshow(image)
plt.show()

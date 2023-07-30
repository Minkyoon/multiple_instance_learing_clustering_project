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


# Finding optimal number of clusters using silhouette score
silhouette_scores = []
max_clusters = 10  # You can adjust this value according to your needs
for n_clusters in range(2, max_clusters + 1):
    clusterer = AgglomerativeClustering(n_clusters=n_clusters)
    preds = clusterer.fit_predict(features)
    score = silhouette_score(features, preds)
    silhouette_scores.append(score)
optimal_clusters = np.argmax(silhouette_scores) + 2  # Adding 2 because range starts from 2

# Hierarchical clustering
cluster = AgglomerativeClustering(n_clusters=optimal_clusters, affinity='euclidean', linkage='ward')
cluster_labels = cluster.fit_predict(features)

# Count the number of data points in each cluster
cluster_sizes = Counter(cluster_labels)

# Find the smallest cluster
smallest_cluster = min(cluster_sizes, key=cluster_sizes.get)

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




## 개수세기
count_5 = 0  # 
count_4 = 0 
count_3 = 0  
count_2 = 0 
count_1 = 0  # 1의 개수를 저장할 변수 초기화
count_0 = 0  # 0의 개수를 저장할 변수 초기화
count_6 = 0

for element in cluster_labels:
    if element == 1:
        count_1 += 1
    elif element == 0:
        count_0 += 1
    elif element == 2:
        count_2 += 1
    elif element == 3:
        count_3 += 1
    elif element == 4:
        count_4 += 1
    elif element == 5:
        count_5 += 1
    elif element == 6:
        count_6 += 1

print("1의 개수:", count_1)
print("0의 개수:", count_0)
print("2의 개수:", count_2)
print("3의 개수:", count_3)
print("4의 개수:", count_4)
print("5의 개수:", count_5)
print("6의 개수:", count_6)

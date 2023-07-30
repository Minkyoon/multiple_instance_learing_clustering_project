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



# Compute Sum of Squared Distances (SSD) for different values of k
ssd = []
K = range(1, 15)
for k in K:
    kmeans = KMeans(n_clusters=k)
    kmeans = kmeans.fit(features)
    ssd.append(kmeans.inertia_)

# Plot SSD for each k
plt.plot(K, ssd, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum of squared distances')
plt.title('Elbow Method For Optimal k')
plt.show()



k = 2  # replace with the number of clusters you decided
kmeans = KMeans(n_clusters=k)
kmeans = kmeans.fit(features)

# Plot the clusters
for i in range(k):
    plt.scatter(features[kmeans.labels_ == i, 0], features[kmeans.labels_ == i, 1], label='Cluster ' + str(i+1))
plt.legend()
plt.title("K-means Clustering with k = " + str(k))
plt.show()



from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(features)   # X is your data
    distortions.append(kmeanModel.inertia_)

plt.figure(figsize=(16,8))
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()

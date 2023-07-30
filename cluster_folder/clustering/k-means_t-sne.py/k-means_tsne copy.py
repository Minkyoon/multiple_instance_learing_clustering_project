import pandas as pd
import numpy as np
import torch
import os
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from collections import Counter
from PIL import Image
import torch
import random

random_seed=42
np.random.seed(42)
torch.manual_seed(42)

torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)

np.random.seed(random_seed)
random.seed(random_seed)

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
kmeans = KMeans(n_clusters=8, random_state=42).fit(features)
cluster_labels = kmeans.labels_

# Visualization
tsne = TSNE(n_components=2, random_state=42)
features_2d = tsne.fit_transform(features)

plt.figure(figsize=(6, 6))
scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=cluster_labels)
plt.legend(*scatter.legend_elements(), title="Clusters")
plt.show()

# Count the cluster labels
counter = Counter(cluster_labels)
print(counter)



# Get the smallest cluster
smallest_cluster_label = min(counter, key=counter.get)

#if you want annother label chage this
smallest_cluster_label =0
# Get the filenames of the images in the smallest cluster
smallest_cluster_filenames = [filename for filename, label in zip(filenames, cluster_labels) if label == smallest_cluster_label]


## 모든 이미지확인

# Open and display the images in the smallest cluster
import os
import matplotlib.pyplot as plt
import numpy as np
# Open and display the images in the smallest cluster
for filename in smallest_cluster_filenames:
    # Split the filename and form the path
    folder, subfolder = filename.split('b')
    image_path = os.path.join('/home/minkyoon/2023_crohn_data/processed_data4', folder, subfolder + '.npy')
    
    # Check if the file exists. If not, add "0"s in front of the subfolder until it exists or until we have added 8 "0"s.
    zeros_added = 0
    while not os.path.isfile(image_path) and zeros_added < 8:
        subfolder = '0' + subfolder
        image_path = os.path.join('/home/minkyoon/2023_crohn_data/processed_data4', folder, subfolder + '.npy')
        zeros_added += 1

    # Load the .npy file
    image = np.load(image_path)

    # Display the image
    plt.figure()
    plt.imshow(image)
    plt.title(filename)
    plt.show()








## 이미지 분석

import pandas as pd

# Load the label data
label_df = pd.read_csv('/home/minkyoon/crohn/csv/label_data/relapse/relapse_label.csv')

# Extract accession_numbers from smallest_cluster_filenames
accession_numbers = [filename.split('b')[0] for filename in smallest_cluster_filenames]
unique_accession_numbers = set(accession_numbers)

print("Number of unique accession_numbers in the smallest cluster:", len(unique_accession_numbers))

# For each unique accession_number, print the label and the number of images
for accession_number in unique_accession_numbers:
    labels = label_df[label_df['accession_number'] == int(accession_number)]['label'].values
    unique_labels = set(labels)

    print("\nAccession_number:", accession_number)
    print("Number of images:", len(labels))
    
    for label in unique_labels:
        count = list(labels).count(label)
        print("Label:", label, "Count:", count)



## accession_number 단위별 이미지 분ㅅㄱ

import pandas as pd

# Load the label data
label_df = pd.read_csv('/home/minkyoon/crohn/csv/label_data/relapse/relapse_label.csv')

# Extract accession_numbers from smallest_cluster_filenames
accession_numbers = [filename.split('b')[0] for filename in smallest_cluster_filenames]
unique_accession_numbers = set(accession_numbers)

print("Number of unique accession_numbers in the smallest cluster:", len(unique_accession_numbers))

# For each unique accession_number, print the label and the number of images
for accession_number in unique_accession_numbers:
    labels = label_df[label_df['accession_number'] == int(accession_number)]['label'].values
    unique_labels = set(labels)

    print("\nAccession_number:", accession_number)
    print("Number of images:", len(labels))
    
    for label in unique_labels:
        count = list(labels).count(label)
        print("Label:", label, "Count:", count)

# Compute the label distribution in the whole dataset
total_labels = label_df['label'].values
unique_total_labels = set(total_labels)

print("\nTotal number of images:", len(total_labels))

for label in unique_total_labels:
    count = list(total_labels).count(label)
    print("Label:", label, "Count:", count)
    
import pandas as pd

# Load the label data
label_df = pd.read_csv('/home/minkyoon/crohn/csv/label_data/relapse/relapse_label.csv')

# Extract accession numbers from smallest_cluster_filenames
accession_numbers = [int(filename.split('b')[0]) for filename in smallest_cluster_filenames]


# For each unique label, count the unique accession numbers
unique_labels = label_df['label'].unique()

# Filter label data to only include accession numbers in the smallest cluster
label_df = label_df[label_df['accession_number'].isin(accession_numbers)]

# For each unique label, count the unique accession numbers
unique_labels = label_df['label'].unique()

for label in unique_labels:
    unique_accession_numbers = label_df[label_df['label'] == label]['accession_number'].unique()
    print("Label:", label, "Number of unique accession numbers:", len(unique_accession_numbers))
    
    
    
    

import pandas as pd

# Load the label data
label_df = pd.read_csv('/home/minkyoon/crohn/csv/label_data/relapse/relapse_label.csv')

# Extract accession numbers from smallest_cluster_filenames
accession_numbers = [int(filename.split('b')[0]) for filename in smallest_cluster_filenames]
order_numbers = [int(filename.split('b')[1]) for filename in smallest_cluster_filenames]

# Filter label data to only include accession numbers in the smallest cluster
label_df = label_df[label_df['accession_number'].isin(accession_numbers)]

# For each unique label, count the unique accession numbers
unique_labels = label_df['label'].unique()
for label in unique_labels:
    unique_accession_numbers = label_df[label_df['label'] == label]['accession_number'].unique()
    total_images_with_label = label_df[label_df['label'] == label].shape[0]
    print(f"Label: {label}, Number of unique accession numbers: {len(unique_accession_numbers)}, Total images with label: {total_images_with_label}")

# For each unique accession number, print the number of images in the smallest cluster, the total number of images, and the associated label
label0=0
label1=0
unique_accession_numbers = label_df['accession_number'].unique()
for accession_number in unique_accession_numbers:
    total_images = label_df[label_df['accession_number'] == accession_number].shape[0]
    images_in_smallest_cluster = accession_numbers.count(accession_number)
    associated_label = label_df[label_df['accession_number'] == accession_number]['label'].iloc[0]
    if associated_label == 1.0:
        label1 += images_in_smallest_cluster
        print(images_in_smallest_cluster)
    elif associated_label == 0.0:
        label0 += images_in_smallest_cluster
        print(images_in_smallest_cluster)
    print(f"Accession number: {accession_number}, (current images/total images): {images_in_smallest_cluster}/{total_images}, Label: {associated_label}")


print(f'label0:{label0}')
print(f'label1:{label1}')
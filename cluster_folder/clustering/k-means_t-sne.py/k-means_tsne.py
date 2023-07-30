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
smallest_cluster_label =3
# Get the filenames of the images in the smallest cluster
smallest_cluster_filenames = [filename for filename, label in zip(filenames, cluster_labels) if label == smallest_cluster_label]


## 모든 이미지확인

# Open and display the images in the smallest cluster
import os
import matplotlib.pyplot as plt
import numpy as np
# Open and display the images in the smallest cluster
for filename in smallest_cluster_filenames[:100]:
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


## accession_number 별이미지한개만확인

import os
import matplotlib.pyplot as plt
import numpy as np

# Keep track of which accession numbers we have already seen
seen_accession_numbers = set()

# Load the label data
label_df = pd.read_csv('/home/minkyoon/crohn/csv/label_data/relapse/relapse_label.csv')

# Open and display the first image for each accession number in the smallest cluster
for filename in smallest_cluster_filenames[:100]:
    # Get the accession number from the filename
    accession_number = int(filename.split('b')[0])

    # If we have already seen this accession number, skip it
    # 이미지 전체 보고싶으면 밑에 코드 주석!!
    # if accession_number in seen_accession_numbers:
    #     continue

    # Add the accession number to our set of seen accession numbers
    seen_accession_numbers.add(accession_number)

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

    # Get the label associated with the accession number
    label = label_df[label_df['accession_number'] == accession_number]['label'].iloc[0]

    # Display the image
    plt.figure()
    plt.imshow(image)
    plt.title(f"Filename: {filename}, Label: {label}")
    plt.show()








    

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




#csvforresent
import pandas as pd
import os
#if you want annother label chage this
smallest_cluster_label =2
# Get the filenames of the images in the smallest cluster
smallest_cluster_filenames = [filename for filename, label in zip(filenames, cluster_labels) if label == smallest_cluster_label]

# Load the label data
label_df = pd.read_csv('/home/minkyoon/crohn/csv/label_data/relapse/relapse_label.csv')

# Define smallest_cluster_filenames
#smallest_cluster_filenames = ['2b33', '2b39', '2b42', '42b5', '42b10', '44b5', '44b6', '44b7', '44b16']

# Convert smallest_cluster_filenames to a set of tuples for faster searching
cluster_filename_set = set()
for filename in smallest_cluster_filenames:
    # Split the filename into folder and subfolder
    folder, subfolder = filename.split('b')
    # Add the tuple to the set
    cluster_filename_set.add((int(folder), int(subfolder)))

# Define a function to check if a row matches a filename in smallest_cluster_filenames
def matches_filename(row):
    # Get the folder and subfolder from the row's filepath
    filepath = row['filepath']
    folder = os.path.basename(os.path.dirname(filepath))
    subfolder, _ = os.path.splitext(os.path.basename(filepath))
    # Check if the tuple (folder, subfolder) is in cluster_filename_set
    return (int(folder), int(subfolder)) in cluster_filename_set

# Filter the label data to only include rows where matches_filename(row) is True
filtered_df = label_df[label_df.apply(matches_filename, axis=1)]

# Save the filtered dataframe to a new CSV file
filtered_df.to_csv('/home/minkyoon/crohn/csv/clam/clustering/relapse_resnet/class2_clustering.csv', index=False)

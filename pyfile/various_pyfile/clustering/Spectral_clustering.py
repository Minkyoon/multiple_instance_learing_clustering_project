import pandas as pd
import torch
import os
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.metrics.cluster import adjusted_rand_score


df = pd.read_csv('/home/minkyoon/first/CLAM/dataset_csv/new_output_remission.csv')


folder_path = "/home/minkyoon/crohn/csv/clam/remission/512vectorpt"
files = os.listdir(folder_path)

data = []
labels = []
true_labels = []

for _, row in df.iterrows():
    file_name = str(row['slide_id']) + '.pt'
    if file_name in files:

        vector = torch.load(os.path.join(folder_path, file_name))
        data.append(vector.numpy())
        true_labels.append(row['label'])


data = np.array(data)


spectral = SpectralClustering(n_clusters=2, assign_labels="discretize", random_state=0)


spectral.fit(data)


labels = spectral.labels_


ari_score = adjusted_rand_score(true_labels, labels)
print("ARI score: ", ari_score)

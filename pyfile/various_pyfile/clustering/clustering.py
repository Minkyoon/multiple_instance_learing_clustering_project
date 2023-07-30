import pandas as pd
import torch
import os
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


df = pd.read_csv('/home/minkyoon/first/CLAM/dataset_csv/new_output.csv')


folder_path = "/home/minkyoon/crohn/csv/clam/relapse/downsampling/clam_vecotr_512/fold4"
files = os.listdir(folder_path)

data = []
labels = []

for _, row in df.iterrows():
    file_name = str(row['slide_id']) + '.pt'
    if file_name in files:
        # 각 .pt 파일에서 텐서를 로드합니다.
        vector = torch.load(os.path.join(folder_path, file_name))
        data.append(vector.numpy())
        labels.append(row['label'])


data = np.array(data)
labels = np.array(labels)

# 차원 축소를 위한 t-SNE
tsne = TSNE(n_components=2, random_state=0)
data_2d = tsne.fit_transform(data)

# Plotting
plt.figure(figsize=(6, 5))
colors = 'r', 'b'
target_ids = range(len(np.unique(labels)))

for i, c, label in zip(target_ids, colors, np.unique(labels)):
    plt.scatter(data_2d[labels == i, 0], data_2d[labels == i, 1], c=c, label=label)
plt.legend()
plt.show()

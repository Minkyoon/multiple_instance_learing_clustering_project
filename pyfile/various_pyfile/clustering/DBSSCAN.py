import pandas as pd
import torch
import os
import numpy as np
from sklearn.cluster import DBSCAN
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
        # 각 .pt 파일에서 텐서를 로드합니다.
        vector = torch.load(os.path.join(folder_path, file_name))
        data.append(vector.numpy())
        true_labels.append(row['label'])


data = np.array(data)


dbscan = DBSCAN(eps=3, min_samples=2)


dbscan.fit(data)

labels = dbscan.labels_


ari_score = adjusted_rand_score(true_labels, labels)
print("ARI score: ", ari_score)

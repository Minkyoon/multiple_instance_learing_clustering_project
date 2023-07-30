import pandas as pd
import torch
import os
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Load the labels
df = pd.read_csv('/home/minkyoon/crohn/csv/clam/clustering/relapse/extracted_features.csv')

folder_path = "/home/minkyoon/crohn/csv/clam/clustering/relapse"
files = os.listdir(folder_path)

data = []
labels = []

for _, row in df.iterrows():
    file_name = str(row['filename']) + '.pt'
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




## 3차원


import pandas as pd
import torch
import os
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the labels
df = pd.read_csv('/home/minkyoon/crohn/csv/clam/clustering/relapse/extracted_features.csv')

folder_path = "/home/minkyoon/crohn/csv/clam/clustering/relapse"
files = os.listdir(folder_path)

data = []
labels = []

for _, row in df.iterrows():
    file_name = str(row['filename']) + '.pt'
    if file_name in files:
        # Load the tensor from each .pt file
        vector = torch.load(os.path.join(folder_path, file_name))
        data.append(vector.numpy())
        labels.append(row['label'])

data = np.array(data)
labels = np.array(labels)

# Dimensionality reduction using t-SNE
tsne = TSNE(n_components=3, random_state=0)
data_3d = tsne.fit_transform(data)

# Plotting
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
colors = 'r', 'b'
target_ids = range(len(np.unique(labels)))

for i, c, label in zip(target_ids, colors, np.unique(labels)):
    ax.scatter(data_3d[labels == i, 0], data_3d[labels == i, 1], data_3d[labels == i, 2], c=c, label=label)

ax.set_xlabel('Dimension 1')
ax.set_ylabel('Dimension 2')
ax.set_zlabel('Dimension 3')
plt.legend()
plt.show()



## pca 2차원
import pandas as pd
import torch
import os
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the labels
df = pd.read_csv('/home/minkyoon/crohn/csv/clam/clustering/relapse/extracted_features.csv')

folder_path = "/home/minkyoon/crohn/csv/clam/clustering/relapse"
files = os.listdir(folder_path)

data = []
labels = []

for _, row in df.iterrows():
    file_name = str(row['filename']) + '.pt'
    if file_name in files:
        # Load the tensor from each .pt file
        vector = torch.load(os.path.join(folder_path, file_name))
        data.append(vector.numpy())
        labels.append(row['label'])

data = np.array(data)
labels = np.array(labels)

# Dimensionality reduction using PCA
pca = PCA(n_components=2)
data_2d = pca.fit_transform(data)

# Plotting
plt.figure(figsize=(6, 5))
colors = 'r', 'b'
target_ids = range(len(np.unique(labels)))

for i, c, label in zip(target_ids, colors, np.unique(labels)):
    plt.scatter(data_2d[labels == i, 0], data_2d[labels == i, 1], c=c, label=label)
plt.legend()
plt.show()



## pca 3차원
import pandas as pd
import torch
import os
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the labels
df = pd.read_csv('/home/minkyoon/crohn/csv/clam/clustering/relapse/extracted_features.csv')

folder_path = "/home/minkyoon/crohn/csv/clam/clustering/relapse"
files = os.listdir(folder_path)

data = []
labels = []

for _, row in df.iterrows():
    file_name = str(row['filename']) + '.pt'
    if file_name in files:
        # Load the tensor from each .pt file
        vector = torch.load(os.path.join(folder_path, file_name))
        data.append(vector.numpy())
        labels.append(row['label'])

data = np.array(data)
labels = np.array(labels)

# Dimensionality reduction using PCA
pca = PCA(n_components=3)
data_3d = pca.fit_transform(data)

# Plotting
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
colors = 'r', 'b'
target_ids = range(len(np.unique(labels)))

for i, c, label in zip(target_ids, colors, np.unique(labels)):
    ax.scatter(data_3d[labels == i, 0], data_3d[labels == i, 1], data_3d[labels == i, 2], c=c, label=label)

ax.set_xlabel('Dimension 1')
ax.set_ylabel('Dimension 2')
ax.set_zlabel('Dimension 3')
plt.legend()
plt.show()

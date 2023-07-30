import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from resnet_custom import resnet50_baseline
import os

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = resnet50_baseline().to(device)
model.eval()

# Define dataset
class ImageDataset(Dataset):
    def __init__(self, csv_file):
        self.dataframe = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx, 0]
        img = np.load(img_path)
        img = torch.from_numpy(img).float().permute(2, 0, 1)  # Change (H,W,C) to (C,H,W)

        label = self.dataframe.iloc[idx, 2]
        accession_number = self.dataframe.iloc[idx, 1]

        return img, label, accession_number, img_path

# Create dataloader
dataset = ImageDataset('/home/minkyoon/crohn/csv/label_data/relapse/relapse_label.csv')
dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

# Extract features and save to CSV and .pt files
data_list = []
for images, labels, acc_numbers, img_paths in dataloader:
    images = images.to(device)
    with torch.no_grad():
        features = model(images)

    for i in range(features.shape[0]):
        # Extract image index from the file path
        img_index = os.path.splitext(os.path.basename(img_paths[i]))[0]  # Remove .npy and get filename

        # Form the filename
        filename = str(int(acc_numbers[i])) + "b" + img_index

        # Save the feature to a .pt file
        feature_filepath = '/home/minkyoon/crohn/csv/clam/clustering/relapse/' + filename + '.pt'
        torch.save(features[i].cpu(), feature_filepath)
        data_list.append([filename, int(labels[i]), int(acc_numbers[i])])

df = pd.DataFrame(data_list, columns=['filename', 'label', 'accession_number'])
df.to_csv('/home/minkyoon/crohn/csv/clam/clustering/relapse/extracted_features.csv', index=False)



# import torch

# b=torch.load('/home/minkyoon/crohn/csv/clam/clustering/relapse/2b2.pt')
# b.shape
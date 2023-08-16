import random
import os 
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torch.utils.data as data
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from PIL import Image
import copy
from time import time
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt



random_seed = 2022

torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
os.environ['PYTHONHASHSEED'] = str(random_seed)

np.random.seed(random_seed)
random.seed(random_seed)

## 모델정의



transform_valid = transforms.Compose([
        transforms.ToTensor(),      
])



# Dataset Class
class CustomImageDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, 2]
        image = np.load(img_path).astype(np.float32)  # npy 파일을 로드합니다.
        if self.transform:
            image = self.transform(image)
        return image, self.data.iloc[idx, 1]  # label 대신 accession_number를 반환






# 모델 설정 값

config = {
    # Classfier 설정
    "cls_hidden_dims" : [1024, 512, 256]
    }


class ResNet(nn.Module):
    """pretrain 된 ResNet을 이
    """
    
    def __init__(self):
        """
		Args:
			base_model : resnet18 / resnet50
			config: 모델 설정 값
		"""
        super(ResNet, self).__init__()
       
        model = models.resnet50(pretrained=True)
        num_ftrs = model.fc.in_features
        self.num_ftrs = num_ftrs
                
        for name, param in model.named_parameters():
            if 'layer2' in name:
                break
            param.requires_grad = False            
        self.features = nn.Sequential(*list(model.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        b = x.size(0)
        x = x.view(b, -1)

        return x





model_image = ResNet()
model_image

class Classifier(nn.Sequential):
    """임베딩 된 feature를 이용해 classificaion
    """
    def __init__(self, model_image, **config):
        """
        Args:
            model_image : image emedding 모델
            config: 모델 설정 값
        """
        super(Classifier, self).__init__()

        self.model_image = model_image # image 임베딩 모델

        self.input_dim = model_image.num_ftrs # image feature 사이즈
        self.dropout = nn.Dropout(0.5) # dropout 적용

        self.hidden_dims = config['cls_hidden_dims'] # classifier hidden dimensions
        layer_size = len(self.hidden_dims) + 1 # hidden layer 개수
        dims = [self.input_dim] + self.hidden_dims + [2] 

        self.predictor = nn.ModuleList([nn.Linear(dims[i], dims[i+1]) for i in range(layer_size)]) # classifer layers 

    def forward(self, v):
        # Drug/protein 임베딩
        v_i = self.model_image(v) # batch_size x hidden_dim 

        for i, l in enumerate(self.predictor):
            if i == (len(self.predictor)-1):
                # If last layer,
                v_i = l(v_i)
            else:
                # If Not last layer, dropout과 ReLU 적용
                v_i = F.relu(self.dropout(l(v_i)))

        return v_i

model = Classifier(model_image, **config)
model

# 1. 라이브러리 및 모델 로드

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = "/home/minkyoon/crohn/normal_resnet/stratified_5fold/fold0_resnet50_pi.pt"
model_image = ResNet()
model = Classifier(model_image, **config).to(device)
model = torch.load(model_path, map_location=device)
model = model.to(device)
model.eval()




# DataLoader
dataset = CustomImageDataset("/home/minkyoon/crohn/csv/normal_resnet/5fold_resnet_for_startified5fold/test_fold_0.csv", transform_valid)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=1)


# Predict and Hard voting
predictions = {}
with torch.no_grad():
    for inputs, accession_numbers in dataloader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        
        for i, acc_num in enumerate(accession_numbers):
            if acc_num.item() not in predictions:
                predictions[acc_num.item()] = []
            predictions[acc_num.item()].append(predicted[i].item())

final_predictions = {}
for key, values in predictions.items():
    unique, counts = np.unique(values, return_counts=True)
    final_predictions[key] = unique[np.argmax(counts)]

for key, value in final_predictions.items():
    print(f"Accession number {key} is predicted as label {value}")
    
    


from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

# 실제 라벨을 데이터에서 가져오기
real_labels = {}
data_csv = pd.read_csv("/home/minkyoon/crohn/csv/normal_resnet/5fold_resnet_for_startified5fold/test_fold_0.csv")
for _, row in data_csv.iterrows():
    acc_number = row['accession_number']
    label = row['label']
    real_labels[acc_number] = label

# 예측 라벨과 실제 라벨 리스트 만들기
y_true = [real_labels[key] for key in final_predictions.keys()]
y_pred = list(final_predictions.values())

# 지표 계산
accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
conf_matrix = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = conf_matrix.ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)

print(f"Accuracy: {accuracy}")
print(f"Sensitivity: {sensitivity}")
print(f"Specificity: {specificity}")
print(f"F1-score: {f1}")
print("Confusion Matrix:")
print(conf_matrix)

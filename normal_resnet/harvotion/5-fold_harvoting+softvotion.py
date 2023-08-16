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
import seaborn as sns



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



#폴드명 정의
fold_num=0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = f"/home/minkyoon/crohn/normal_resnet/stratified_5fold/fold{fold_num}_resnet50_pi.pt"
model_image = ResNet()
model = Classifier(model_image, **config).to(device)
model = torch.load(model_path, map_location=device)
model = model.to(device)
model.eval()




# DataLoader
dataset = CustomImageDataset(f"/home/minkyoon/crohn/csv/normal_resnet/5fold_resnet_for_startified5fold/test_fold_{fold_num}.csv", transform_valid)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=1)

# Predict and Hard voting

# 실제 라벨을 데이터에서 가져오기
real_labels = {}
data_csv = pd.read_csv(f"/home/minkyoon/crohn/csv/normal_resnet/5fold_resnet_for_startified5fold/test_fold_{fold_num}.csv")
for _, row in data_csv.iterrows():
    acc_number = row['accession_number']
    label = row['label']
    real_labels[acc_number] = label
    
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
vote_results = {}
for key, values in predictions.items():
    unique, counts = np.unique(values, return_counts=True)
    final_predictions[key] = unique[np.argmax(counts)]
    
    # Save the vote results
    vote_results[key] = dict(zip(unique, counts))

# Print the vote results and final prediction
for key in final_predictions.keys():
    print(f"Accession number {key}:")
    print(f"  Votes: {vote_results[key]}")
    print(f"  Predicted Label: {final_predictions[key]}")
    print(f"  Actual Label: {real_labels[key]}")
    print("-----------------------------")
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score



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

plt.figure(figsize=(8, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', annot_kws={"verticalalignment": 'center', 'size': 15})
plt.title('Confusion matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks([0.5, 1.5], labels=['Negative', 'Positive'])
plt.yticks([0.5, 1.5], labels=['Negative', 'Positive'], va='center')
plt.savefig(f'hardvoting_fold_num_{fold_num}.png')
plt.show()
plt.close()



#### soft votion
print('softvotion')

# Predict and Hard voting
from torch.nn.functional import softmax

# ... [이전 코드 부분]

# Predict and Soft voting
probability_sums = {}
with torch.no_grad():
    for inputs, accession_numbers in dataloader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        probs = softmax(outputs, dim=1)
        
        for i, acc_num in enumerate(accession_numbers):
            if acc_num.item() not in probability_sums:
                probability_sums[acc_num.item()] = np.zeros(probs.shape[1])
            probability_sums[acc_num.item()] += probs[i].cpu().numpy()

final_predictions_soft = {}
for key, summed_probs in probability_sums.items():
    final_predictions_soft[key] = np.argmax(summed_probs)
    
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

# 실제 라벨을 데이터에서 가져오기
real_labels = {}
data_csv = pd.read_csv("/home/minkyoon/crohn/csv/normal_resnet/5fold_resnet_for_startified5fold/test_fold_0.csv")
for _, row in data_csv.iterrows():
    acc_number = row['accession_number']
    label = row['label']
    real_labels[acc_number] = label

# Print the soft vote results and final prediction
for key in final_predictions_soft.keys():
    print(f"Accession number {key}:")
    print(f"  Predicted Label (Soft Voting): {final_predictions_soft[key]}")
    print(f"  Actual Label: {real_labels[key]}")
    print("-----------------------------")

# 예측 라벨과 실제 라벨 리스트 만들기 (Soft Voting)
y_true_soft = [real_labels[key] for key in final_predictions_soft.keys()]
y_pred_soft = list(final_predictions_soft.values())

# 지표 계산 (Soft Voting)
accuracy_soft = accuracy_score(y_true_soft, y_pred_soft)
f1_soft = f1_score(y_true_soft, y_pred_soft)
conf_matrix_soft = confusion_matrix(y_true_soft, y_pred_soft)
tn_soft, fp_soft, fn_soft, tp_soft = conf_matrix_soft.ravel()
sensitivity_soft = tp_soft / (tp_soft + fn_soft)
specificity_soft = tn_soft / (tn_soft + fp_soft)

print("Soft Voting Results:")
print(f"Accuracy: {accuracy_soft}")
print(f"Sensitivity: {sensitivity_soft}")
print(f"Specificity: {specificity_soft}")
print(f"F1-score: {f1_soft}")
print("Confusion Matrix (Soft Voting):")
print(conf_matrix_soft)


plt.figure(figsize=(8, 8))
sns.heatmap(conf_matrix_soft, annot=True, fmt='d', cmap='Blues', annot_kws={"verticalalignment": 'center', 'size': 15})
plt.title('Confusion matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks([0.5, 1.5], labels=['Negative', 'Positive'])
plt.yticks([0.5, 1.5], labels=['Negative', 'Positive'], va='center')
plt.savefig(f'softvoting_fold_num_{fold_num}.png')
plt.show()
plt.close()

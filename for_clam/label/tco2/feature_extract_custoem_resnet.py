import random
import os 
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch
from torchvision import datasets, transforms
import torch.utils.data as data
import pandas as pd
from torch.utils.data import Dataset

# 디바이스 설정 (GPU 사용 가능하면 GPU 사용하도록)


from torchvision import models
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# 모델 설정 값


device = torch.device("cuda:3" )

# 대화2에서 구현된 모델을 import
from resnet_custom import resnet50_baseline

# 모델 로드
model = resnet50_baseline(pretrained=True)

def extract_features(model, dataloader):
    model.eval() # Set the model to evaluation mode
    with torch.no_grad(): # Do not calculate gradients
        feature_list = []
        for img, label in dataloader:
            img = img.to(device) # Move the image tensor to GPU
            features = model(img) # Forward pass to get the features
            # features shape is [batch_size, 1024]
            feature_list.append(features)
        # Concatenate all feature tensors along the batch dimension
        features_all = torch.cat(feature_list, dim=0)
    return features_all

# You can create a DataLoader for images of a certain accession number
# Assume that 'custom_dataset' is a dataset for images of that accession number



from torch.utils.data import Dataset
import numpy as np
import pandas as pd


transform = transforms.Compose([
    transforms.ToTensor(), 
           
    
])

# 사용자 정의 Dataset 클래스


class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        # CSV 파일 로딩
        self.data_info = pd.read_csv(csv_file)
        self.transform=transform

        # 이미지와 레이블 데이터의 열 이름을 알고 있어야 합니다.
        self.image_arr = self.data_info['filepath']
        self.label_arr = self.data_info['label']
        self.data_len = len(self.data_info.index)

    def __getitem__(self, index):
        # 이미지 파일을 numpy 배열로 불러오기
        single_image_name = self.image_arr[index]
        image = np.load(single_image_name).astype(np.float32)
        # numpy 배열을 PyTorch 텐서로 변환
        if isinstance(image, tuple):
            image = image[0]
            
        
        
        
        if self.transform:
           
            
            image_as_tensor = self.transform(image)
             

        # 레이블 불러오기 (레이블도 PyTorch 텐서로 변환할 수 있습니다)
        single_image_label = self.label_arr[index]
        
        return (image_as_tensor, single_image_label)

    def __len__(self):
        return self.data_len
    
    
import os

device = torch.device("cuda:3")
model = model.to(device)

#inputs = inputs.to(device)

# CSV 파일이 있는 디렉토리 경로
csv_directory = "/home/minkyoon/crohn/csv/clam/tco2/split_csv"

import csv

# 결과를 저장할 새로운 CSV 파일 열기
with open('output.csv', 'w', newline='') as csvfile:
    fieldnames = ['pt_filepath', 'label']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader() # 헤더 쓰기

    # CSV 파일이 있는 디렉토리 경로
    

    # CSV 파일들을 읽기
    for filename in os.listdir(csv_directory):
        if filename.endswith(".csv"):
            csv_path = os.path.join(csv_directory, filename)
            
            try:
                # CustomDataset과 DataLoader를 생성
                custom_dataset = CustomDataset(csv_path, transform=transform)
                custom_dataloader = torch.utils.data.DataLoader(custom_dataset, batch_size=len(custom_dataset))

                for data in custom_dataloader:
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                
                # feature 추출
                features = extract_features(model, custom_dataloader)

                # pt 파일로 저장 (파일 이름은 기존 CSV 파일과 동일하게 하되 확장자만 변경)
                pt_filepath = csv_path.replace(".csv", ".pt")
                torch.save(features, pt_filepath)

                # 라벨 추출
                label = custom_dataset.label_arr[0]  # 라벨이 하나라면 이렇게 하십시오. 아니면 더 복잡한 로직이 필요합니다.

                # pt 파일 경로와 라벨을 새로운 CSV 파일에 쓰기
                writer.writerow({'pt_filepath': pt_filepath, 'label': str(label)})
            except Exception as e:
                print(f"Error processing file: {csv_path}")
                print(e)


#import torch
#a=torch.load('/home/minkyoon/crohn/csv/clam/relapse/2.pt')
# label 이 float이라 str 로 바꾸는 코드
import pandas as pd

# csv 파일 로드
df = pd.read_csv('output.csv')

# 'label' 열의 타입을 str로 변경
df['label'] = df['label'].apply(lambda x: int(x))
df['label'] = df['label'].apply(lambda x: str(x))

# 변경된 csv 파일 저장
df.to_csv('output.csv', index=False)
df = pd.read_csv('output.csv', dtype={'label': str})



## clam csv 변경하는 코드
import pandas as pd

# 파일 읽기
df = pd.read_csv('output.csv')

# slide_id 컬럼 추가하기
df['slide_id'] = df['pt_filepath'].apply(lambda x: x.split('/')[-1].split('.')[0])

# case_id 컬럼 추가하기
df['case_id'] = ['patient_'+str(i) for i in range(df.shape[0])]

# 새로운 순서로 컬럼 재정렬하기
df = df[['case_id', 'slide_id', 'label']]


# 새로운 파일로 저장하기
df.to_csv('new_output.csv', index=False)
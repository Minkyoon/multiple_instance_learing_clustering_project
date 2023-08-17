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

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



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

device = torch.device("cpu" )



model=torch.load('/home/minkyoon/crohn/normal_resnet/stratifie_10fold_7,21/fold0_resnet50_pi.pt')
# model = model.load_state_dict(weight)

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from PIL import Image

test_transformer = transforms.Compose([
    
    transforms.ToTensor(),
    
    
])
def save_and_show_image(covid_cam, name, save_path='/home/minkyoon/crohn/for_clam/attention/image/tn_gradcam'):
    plt.figure(figsize=(6,6)) # 여기에서 이미지 크기를 조절할 수 있습니다.
    plt.imshow(covid_cam)
    plt.axis('off') # 축과 눈금을 제거
    plt.xticks([]) # x축 눈금 제거
    plt.yticks([]) # y축 눈금 제거
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0) # 여백 제거
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(f'{save_path}/{name}.png', bbox_inches='tight', pad_inches=0)
    plt.show()


def show_gradCAM(model, arr):
    """gradCAM을 이용하여 활성화맵(activation map)을 이미지 위에 시각화하기
    args:
    model (torch.nn.module): 학습된 모델 인스턴스
    arr: 시각화 할 입력 numpy 배열
    """
    # target_layers = [model.layer4[-1]] # 출력층 이전 마지막 레이어 가져오기
    target_layers = [model.model_image.features[-2]]
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)

    # Convert numpy array to torch tensor
    inp = torch.tensor(arr.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0)
    targets = [ClassifierOutputTarget(1)] # 타겟 지정
    grayscale_cam = cam(input_tensor=inp, targets=targets)
    grayscale_cam = grayscale_cam[0, :]

    # 활성화맵을 이미지 위에 표시
    visualization = show_cam_on_image(arr, grayscale_cam, use_rgb=True)

    return visualization

serial=1314
order='32'
# Load numpy array
crohn = np.load(f'/home/minkyoon/2023_crohn_data/processed_data4/{serial}/{order}.npy')

name=f'serial: {serial} order: {order}'
model=model.to(device)
#crohn=crohn.to(device)
# Get gradCAM result
covid_cam = show_gradCAM(model, crohn)
save_and_show_image(covid_cam, name)

# """# 수고하셨습니다."""



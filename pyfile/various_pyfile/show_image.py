# Create an iterator for the DataLoader
data_iter = iter(test_dataloader)

# Skip the first six batches
for _ in range(1):
    _ = next(data_iter)

# Now get the seventh batch
images_batch, labels_batch = next(data_iter)

images_batch=images_batch[0]
first_five_images = images_batch[:34] 
  # This will give us a tensor of shape [5, 3, 225, 225]

for i in range(34):
    image = first_five_images[i]
    img = image.permute(1, 2, 0)  # Switch from [C, H, W] to [H, W, C]
    
    # Normalize for viewing, if necessary
    # img = (img - img.min()) / (img.max() - img.min())

    plt.imshow(img)
    plt.show()
    
    


## show image


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os

# Read the CSV file
data = pd.read_csv('/home/minkyoon/crohn/csv/label_data/Full_label_data.csv')

# Filter the dataframe to only include rows where the accession_number is 1
filtered_data = data[data['accession_number'] == 1042]

# Sort the filtered data by the file path (assuming the file paths are in ascending order)


# Loop over the file paths in the sorted dataframe
for filepath in filtered_data['filepath']:
    # Load the image
    print(filepath)
    image = np.load(filepath)
    
    # If the images are not in the right range or orientation, 
    
    # Display the image
    plt.imshow(image, cmap='gray')
    plt.show()



import pydicom
## dcm image
path='/home/minkyoon/2023_crohn_data/original_data/2_1543635_20200601103515_ES/1.2.840.113619.2.182.140128351753206.159298744.2718.1.1.dcm'
dcm = pydicom.dcmread(path)
img = dcm.pixel_array
plt.imshow(img)


### many dcm image

import os
import pydicom
import matplotlib.pyplot as plt

# DICOM 파일이 있는 폴더
folder_path = '/home/minkyoon/2023_crohn_data/original_data/4_1543635_20210701103513_ES'

# 폴더 내의 모든 파일을 가져옵니다.
file_names = os.listdir(folder_path)
def extract_number(file_name):
    # 파일 이름에서 .dcm을 제거하고, 마지막 부분을 숫자로 변환합니다.
    return int(file_name.replace('.dcm', '').split('.')[-1])

# 파일 이름을 숫자 부분을 기준으로 내림차순으로 정렬합니다.
file_names = sorted(file_names, key=extract_number, reverse=False)


# 각 파일에 대해
for file_name in file_names:
    # 파일이 .dcm 확장자를 가지면
    if file_name.endswith('.dcm'):
        # 파일의 전체 경로를 생성합니다.
        file_path = os.path.join(folder_path, file_name)
        
        # DICOM 파일을 읽습니다.
        dcm = pydicom.dcmread(file_path)
        
        # 이미지 데이터를 가져옵니다.
        img = dcm.pixel_array
        inorder_num = dcm.InstanceNumber
        
        print(file_path)
        print(inorder_num)
        # 이미지를 표시합니다.
        plt.imshow(img)
        plt.show()



## 1300개 이미지 한장씩 가져오기
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('/home/minkyoon/crohn/csv/label_data/Full_label_data.csv')

# Group the data by 'accession_number'
grouped = data.groupby('accession_number')


# Loop over the grouped data
for name, group in grouped:
    try:
        # Get the third file path in the group
        if name <= 1100 :
            continue
        filepath = group['filepath'].iloc[2]
        
        # Load the image
        image = np.load(filepath)
    
        # Display the image
        
        plt.imshow(image, cmap='gray')
        plt.title(filepath)
        plt.show()
        
        if name > 1200:
            break
        
        
    
    except IndexError:
        print(f"Group with accession_number {name} has less than 3 images.")
    except Exception as e:
        print(f"An error occurred with the following file path: {filepath}")
        print(f"Error: {e}")

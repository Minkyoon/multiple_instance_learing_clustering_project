import os
import pandas as pd

# 파일 경로를 지정합니다.
root_dir = "/home/minkyoon/2023_crohn_data/processed_data4"

data = []

# root_dir에서 모든 .npy 파일을 찾습니다.
for dirpath, dirnames, filenames in os.walk(root_dir):
    for filename in filenames:
        if filename.endswith('.npy'):
            # 파일의 절대 경로를 구합니다.
            filepath = os.path.join(dirpath, filename)
            
            # 'accession_number'를 추출합니다. 이는 폴더 이름에서 얻습니다.
            accession_number = os.path.basename(dirpath)
            
            # 데이터를 리스트에 추가합니다.
            data.append((filepath, accession_number))

# 데이터를 DataFrame으로 변환합니다.
df = pd.DataFrame(data, columns=["filepath", "accession_number"])

# 결과를 csv 파일로 저장합니다.
df.to_csv('filepaths_and_accession_numbers.csv', index=False)




import os

root_dir = "/home/minkyoon/2023_crohn_data/processed_data3"

# os.listdir()는 지정된 디렉토리 내에 있는 모든 파일 및 디렉토리의 이름을 반환합니다.
# 이를 사용하여 root_dir 안에 있는 각 요소가 디렉토리인지 확인하고 디렉토리의 수를 세어줍니다.
num_dirs = len([name for name in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, name))])

print("Number of directories:", num_dirs)

import os
from collections import defaultdict

# 디렉토리를 설정합니다.
directory = "/home/minkyoon/2023_crohn_data/original_data"

# 디렉토리 내의 모든 폴더들을 가져옵니다.
folders = os.listdir(directory)

# serial number를 key로 하고, 해당 serial number를 가진 폴더명들을 value로 하는 딕셔너리를 생성합니다.
serial_number_dict = defaultdict(list)

# 각 폴더에 대해
for folder in folders:
    # 폴더명에서 serial number를 추출합니다.
    serial_number = folder.split('_')[0]

    # 추출한 serial number를 key로, 폴더명을 value로 딕셔너리에 추가합니다.
    serial_number_dict[serial_number].append(folder)

# serial number가 겹치는 폴더들을 출력합니다.
for serial_number, folder_list in serial_number_dict.items():
    if len(folder_list) > 1:
        print(f"Serial number {serial_number} is found in these folders: {folder_list}")





import os
import pandas as pd
import re

# 파일 경로를 지정합니다.
root_dir = "/home/minkyoon/2023_crohn_data/processed_data4"

data = []

# root_dir에서 모든 .npy 파일을 찾습니다.
for dirpath, dirnames, filenames in os.walk(root_dir):
    for filename in filenames:
        if filename.endswith('.npy'):
            # 파일의 절대 경로를 구합니다.
            filepath = os.path.join(dirpath, filename)
            
            # 'accession_number'를 추출합니다. 이는 폴더 이름에서 얻습니다.
            accession_number = int(os.path.basename(dirpath))  # int로 변환합니다.
            
            # 파일 이름에서 숫자를 추출합니다.
            num = int(re.findall('\d+', filename)[0])
            
            # 데이터를 리스트에 추가합니다.
            data.append((filepath, accession_number, num))

# 데이터를 DataFrame으로 변환합니다.
df = pd.DataFrame(data, columns=["filepath", "accession_number", "file_number"])

# 'accession_number'와 'file_number'로 데이터프레임을 정렬합니다.
df = df.sort_values(by=['accession_number', 'file_number'])

# 'accession_number'를 다시 문자열로 변환합니다.
df['accession_number'] = df['accession_number'].astype(str)

# 'file_number' 열을 삭제합니다.
df = df.drop(columns=['file_number'])

# 결과를 csv 파일로 저장합니다.
df.to_csv('filepaths_and_accession_numbers.csv', index=False)





## 라벨 데이터 불러서 serial_number를 accession_number로 여기기
import pandas as pd

df=pd.read_csv('/home/minkyoon/crohn/csv/accesion_lab_PCDAI_serial_20230628_1.csv')
selected_columns = ['PCDAI_label', 'tCO2_label', 'Hb_label', 'CRP_label', 'Crohn_label', 'serial_number']
new_df = df[selected_columns]

new_df = new_df.rename(columns={'serial_number': 'accession_number'})

# Convert the column to integers
new_df['accession_number'] = new_df['accession_number'].astype(int)

new_df.to_csv('new_df.csv', index=False)
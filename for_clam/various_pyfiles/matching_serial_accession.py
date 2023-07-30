import pandas as pd

df=pd.read_csv('/home/minkyoon/crohn/csv/original_csv/accesion_lab_PCDAI_serial_20230628_1.csv')
serial_df=df.loc[:,['accession_number','serial_number']]

serial_df['accession_number'] = serial_df['accession_number'].dropna().astype(float).astype(int).astype(str)
serial_df['serial_number'] = serial_df['serial_number'].dropna().astype(float).astype(int).astype(str)

# NaN 값을 포함하는 행을 찾습니다.
nan_rows = serial_df[serial_df.isnull().any(axis=1)]

# NaN 값을 포함하는 행을 출력합니다.
print("NaN을 포함하는 행:")
print(nan_rows)

# NaN 값을 포함하는 행을 제거합니다.
serial_df = serial_df.dropna()

serial_df.to_csv('serial_accession_match.csv', index=False, )
#serial_df.to_csv('serial_accession_match.csv', index=False, float_format='%.0f')



import pandas as pd
import os
import shutil

# CSV 파일을 불러와서 'serial_number'와 'accession_number'를 매핑하는 dictionary를 만듭니다.
mapping_df = pd.read_csv('/home/minkyoon/crohn/csv/original_csv/serial_accession_match.csv')
serial_to_accession = mapping_df.set_index('serial_number')['accession_number'].to_dict()

# 원본 파일 경로
source_dir = '/home/minkyoon/crohn/csv/clam/remission/512vectorpt/'

# 대상 디렉토리가 존재하지 않으면 만듭니다.
target_dir = '/home/minkyoon/crohn/csv/clam/remission/512vectorpt/512vector_accession/'
os.makedirs(target_dir, exist_ok=True)

## 991 serial_number가없음 991 그래서 날아간듯
# 원본 디렉토리에서 모든 파일을 순회합니다.
for filename in os.listdir(source_dir):
    # 파일인지 확인합니다.
    if os.path.isfile(os.path.join(source_dir, filename)):
        # 파일명에서 확장자를 제외한 부분을 serial number로 사용합니다.
        serial_number = int(os.path.splitext(filename)[0])

        # 만약 serial number가 매핑 dictionary에 있으면, 새로운 파일명을 생성합니다.
        if serial_number in serial_to_accession:
            new_filename = str(int(serial_to_accession[serial_number])) + '.pt'
            
            # 파일을 새 위치에 복사합니다.
            shutil.copy(os.path.join(source_dir, filename), os.path.join(target_dir, new_filename))





import pandas as pd
import numpy as np

# CSV 파일을 불러와서 'serial_number'와 'accession_number'를 매핑하는 dictionary를 만듭니다.
mapping_df = pd.read_csv('/home/minkyoon/crohn/csv/original_csv/serial_accession_match.csv')
serial_to_accession = mapping_df.set_index('serial_number')['accession_number'].to_dict()

# 원본 파일 경로
source_file = '/home/minkyoon/first/CLAM/splits/task_1_tumor_vs_normalnew_100/splits_4.csv'

# 데이터 불러오기
data_df = pd.read_csv(source_file)

# 각 column에 대해 serial_number를 accession_number로 변경
for column in ['train', 'val', 'test']:
    data_df[column] = data_df[column].map(serial_to_accession)
    
#    data_df[column].dropna(inplace=True)
    data_df[column] = data_df[column].astype(np.int64)

# 결과를 CSV 파일로 저장
data_df.to_csv('remission_train_test_valid.csv', index=False)
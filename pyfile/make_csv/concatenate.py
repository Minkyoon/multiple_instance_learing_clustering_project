


import pandas as pd

# 데이터 파일 경로
data_file_path = '/home/minkyoon/crohn/csv/whole_data_patinet_id.csv'
patient_file_path = '/home/minkyoon/crohn/csv/patient_id_accession_number.csv'

# 데이터 파일 불러오기
data_df = pd.read_csv(data_file_path)
patient_df = pd.read_csv(patient_file_path)

# patient_id와 accession number 연결하여 새로운 열 생성
accession_dict = {}

for index, row in patient_df.iterrows():
    patient_id = row['PATIENT ID']
    accession_no = row['ACCESS NO.']
    if patient_id in accession_dict:
        accession_dict[patient_id].append(accession_no)
    else:
        accession_dict[patient_id] = [accession_no]

data_df['accession_number'] = data_df['patient_id'].map(lambda x: accession_dict.get(x, []))

# fake_accession_number 열 추가
data_df['fake_accession_number'] = data_df['filepath'].str.extract(r'\/(\w{7})b')

# 데이터 저장
data_df.to_csv('fake_whole_data_patinet_id_with_accession.csv', index=False)


import os
import glob

# 폴더 경로
folder_path = '/home/minkyoon/2023_crohn_data/original_data'

# 모든 dcm 파일 개수 세기
file_count = len(glob.glob(os.path.join(folder_path, '**/*.dcm'), recursive=True))

# 결과 출력
print(f"폴더 안에 있는 dcm 파일 개수: {file_count}개")



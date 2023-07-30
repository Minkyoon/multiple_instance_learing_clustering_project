import pandas as pd

# CSV 파일 불러오기
df = pd.read_csv('/home/minkyoon/crohn/csv/patient_id_accession_number.csv', encoding='cp949')

# 겹치는 PATIENT ID 찾기
duplicates = df['PATIENT ID'][df['PATIENT ID'].duplicated()]

# 겹치는 PATIENT ID와 갯수 출력
for patient_id in duplicates:
    count = df[df['PATIENT ID'] == patient_id].shape[0]
    print(f"{patient_id}는 {count}개")
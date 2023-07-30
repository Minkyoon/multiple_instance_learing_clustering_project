import pandas as pd

# 파일들을 불러옵니다.
df1 = pd.read_csv('/home/minkyoon/crohn/for_clam/label/remission/new_output.csv')
df2 = pd.read_csv('/home/minkyoon/crohn/csv/label_data/accesion_lab_PCDAI_serial_20230628_1 (1).csv')

# slide_id 와 serial_number를 integer 형태로 변환합니다.
df1['slide_id'] = df1['slide_id'].astype(int)
df2['serial_number'] = df2['serial_number'].astype(int)

# 두 개의 데이터프레임을 'slide_id'와 'serial_number'를 기준으로 병합합니다.
merged_df = pd.merge(df1, df2, left_on='slide_id', right_on='serial_number', how='inner')

# 'patient_id'의 고유한 값들의 수를 확인합니다.
unique_patient_id = merged_df['ID'].unique()

print(f'The number of unique patient ids: {unique_patient_id}')
unique_patient_id_df = pd.DataFrame(unique_patient_id, columns=['unique_patient_id'])

unique_patient_id_df.to_csv('remission_patinet_id.csv', index=False)
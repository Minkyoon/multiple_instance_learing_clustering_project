import pandas as pd

# 파일 불러오기
df1 = pd.read_csv('/home/minkyoon/crohn/csv/label_data/accesion_lab_PCDAI_serial_20230628_1 (1).csv')
df2 = pd.read_csv('/home/minkyoon/crohn/csv/label_data/1yr_relapse_dx_date.csv')

# ID를 기준으로 두 데이터프레임을 합치기 (inner join 사용)
merged_df = pd.merge(df1, df2, on='ID', how='inner')

print(merged_df)

merged_df = merged_df.drop(columns=['Unnamed: 0'])
merged_df.to_csv('relapse.csv', index=False)





import pandas as pd

# 파일 불러오기
df = pd.read_csv('/home/minkyoon/crohn/csv/label_data/relapse.csv')

# date 및 dx_date 열을 날짜 형식으로 변환
df['date'] = pd.to_datetime(df['date'])
df['dx_date'] = pd.to_datetime(df['dx_date'])

# date와 dx_date 간의 차이를 계산 (단위: 일)
df['diff_days'] = (df['date'] - df['dx_date']).dt.days

# diff_days가 30일 이하인 행만 남기기
df_30 = df[df['diff_days'] <= 30]
print(df_30)

# diff_days가 60일 이하인 행의 수 확인
count_60 = df[df['diff_days'] <= 60].shape[0]
print(f"Number of rows where difference is less than or equal to 60 days: {count_60}")
df_30 = df_30[['serial_number', 'relapse']]
df_30.to_csv('real_relapse.csv', index=False)






import pandas as pd

# 파일 불러오기
df1 = pd.read_csv('/home/minkyoon/crohn/csv/label_data/Full_label_data.csv')
df2 = pd.read_csv('/home/minkyoon/crohn/csv/label_data/relapse/real_relapse.csv')

# serial_number를 기준으로 두 데이터프레임을 합치기 (inner join 사용)
merged_df = pd.merge(df1, df2, left_on='accession_number', right_on='serial_number', how='inner')

# label이라는 새 열을 만들고 relapse 값을 할당
merged_df['label'] = merged_df['relapse']

# 필요한 열만 남기기
merged_df = merged_df[['filepath', 'accession_number', 'label']]

merged_df.to_csv('relapse_label.csv', index=False)

print(merged_df)

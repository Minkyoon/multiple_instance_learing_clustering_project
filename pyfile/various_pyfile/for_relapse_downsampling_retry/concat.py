import pandas as pd

# CSV 파일들을 pandas DataFrame으로 읽습니다.
df1 = pd.read_csv('/home/minkyoon/crohn/csv/original_csv/serial_accession_match.csv')
df2 = pd.read_csv('/home/minkyoon/crohn/csv/label_data/relapse/real_relapse.csv')

# 두 DataFrame을 'serial_number' 열을 기준으로 병합합니다.
merged_df = pd.merge(df1, df2, how='inner', on='serial_number')

# 병합된 데이터프레임을 확인합니다.
print(merged_df)
merged_df.to_csv('serial_accession_relapse_merged.csv', index=False)


import pandas as pd

# CSV 파일들을 pandas DataFrame으로 읽습니다.
df1 = pd.read_csv('/home/minkyoon/crohn/csv/label_data/accesion_lab_PCDAI_20230714 (2).csv')
df2 = pd.read_csv('/home/minkyoon/crohn/csv/label_data/relapse/serial_accession_relapse_merged.csv')

# 필요한 열만 선택합니다.
df1 = df1[['accession_number', 'birth', 'gender', 'date']]

# 'birth' 열의 데이터 타입을 datetime으로 변환합니다.
df1['birth'] = pd.to_datetime(df1['birth'])
df1['date'] = pd.to_datetime(df1['date'])

# 연도 부분만 추출하여 'birth' 열을 업데이트합니다.
df1['birth'] = df1['birth'].dt.year
df1['date'] = df1['date'].dt.year
df1['age']=df1['date']-df1['birth']
# 중복된 행을 제거합니다.
df1 = df1.drop_duplicates(subset=['accession_number', 'age', 'gender'])
df1 = df1[['accession_number', 'age', 'gender']]

# 두 DataFrame을 'accession_number' 열을 기준으로 병합합니다.
merged_df = pd.merge(df2, df1, how='left', on='accession_number')

# 병합된 데이터프레임을 확인합니다.
print(merged_df)



merged_df.to_csv('serial_accession_relapse_merged_with_age_sex.csv', index=False)

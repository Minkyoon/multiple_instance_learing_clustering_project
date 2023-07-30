import pandas as pd

# 두 csv 파일을 읽어옵니다.
df1 = pd.read_csv('/home/minkyoon/crohn/csv/for_model_csv/model_base.csv')
df2 = pd.read_csv('/home/minkyoon/crohn/csv/label_data/label.csv')

# 'accession_number'를 기준으로 두 DataFrame을 합칩니다.
merged_df = pd.merge(df1, df2, on='accession_number' ,)

# 결과를 출력합니다.
merged_df
merged_df.to_csv('Full_label_data.csv', index=False)



## pcdai label

import pandas as pd

# csv 파일을 읽어옵니다.
df = pd.read_csv('/home/minkyoon/crohn/csv/label_data/Full_label_data.csv')

# 필요한 열만 선택합니다.
df = df[['filepath', 'accession_number', 'PCDAI_label']]

# 'PCDAI_label' 열에서 NaN 값을 가진 행을 제거합니다.
df = df.dropna(subset=['PCDAI_label'])

# 'PCDAI_label' 열의 이름을 'label'로 바꿉니다.
df = df.rename(columns={'PCDAI_label': 'label'})

# 결과를 csv 파일로 저장합니다.
df.to_csv('/home/minkyoon/crohn/csv/label_data/Cleaned_label_data.csv', index=False)



### 라벨 개수세기

label_counts = df['label'].value_counts()
print(label_counts)


## crohn

df = pd.read_csv('/home/minkyoon/crohn/csv/label_data/Full_label_data.csv')

# 필요한 열만 선택합니다.
df = df[['filepath',  'CRP_label','accession_number']]

# 'PCDAI_label' 열에서 NaN 값을 가진 행을 제거합니다.
df = df.dropna(subset=['CRP_label'])

# 'PCDAI_label' 열의 이름을 'label'로 바꿉니다.
df = df.rename(columns={'CRP_label': 'label'})

# 결과를 csv 파일로 저장합니다.
df.to_csv('/home/minkyoon/crohn/csv/label_data/crp_label_data.csv', index=False)



### 라벨 개수세기

label_counts = df['label'].value_counts()
print(label_counts)



# hb

## crohn

df = pd.read_csv('/home/minkyoon/crohn/csv/label_data/Full_label_data.csv')

# 필요한 열만 선택합니다.
df = df[['filepath', 'accession_number', 'Hb_label']]

# 'PCDAI_label' 열에서 NaN 값을 가진 행을 제거합니다.
df = df.dropna(subset=['Hb_label'])

# 'PCDAI_label' 열의 이름을 'label'로 바꿉니다.
df = df.rename(columns={'Hb_label': 'label'})

# 결과를 csv 파일로 저장합니다.
df.to_csv('hb_label_data.csv', index=False)



### 라벨 개수세기

label_counts = df['label'].value_counts()
print(label_counts)



## tco


## crohn

df = pd.read_csv('/home/minkyoon/crohn/csv/label_data/Full_label_data.csv')

# 필요한 열만 선택합니다.
df = df[['filepath',  'tCO2_label','accession_number']]

# 'PCDAI_label' 열에서 NaN 값을 가진 행을 제거합니다.
df = df.dropna(subset=['tCO2_label'])

# 'PCDAI_label' 열의 이름을 'label'로 바꿉니다.
df = df.rename(columns={'tCO2_label': 'label'})

# 결과를 csv 파일로 저장합니다.
df.to_csv('/home/minkyoon/crohn/csv/label_data/tCO2_label_data.csv', index=False)
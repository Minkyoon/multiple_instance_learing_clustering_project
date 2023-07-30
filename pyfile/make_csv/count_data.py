import pandas as pd

# csv 파일 로드
df = pd.read_csv('/home/minkyoon/crohn/csv/label_data/crohn/crohn_label_data.csv')

# 'accession_number'로 그룹화하고, 각 그룹의 크기를 셉니다.
grouped = df.groupby('accession_number').size()

# 각 라벨에 대한 데이터 개수를 셉니다.
label_counts = df['label'].value_counts()

print(label_counts)


# 'accession_number'와 'label'로 그룹화하고, 각 그룹의 크기를 셉니다.
grouped = df.groupby(['accession_number', 'label']).size()

print(grouped)



import pandas as pd

# csv 파일 로드
df = pd.read_csv('/home/minkyoon/crohn/csv/label_data/relapse/relapse_label.csv')

# 'accession_number'를 기준으로 중복 제거 (처음 나타나는 항목을 유지)
df = df.drop_duplicates(subset='accession_number')

# 각 라벨에 대한 데이터 개수를 셉니다.
label_counts = df['label'].value_counts()

print(label_counts)

import pandas as pd

# csv 파일 로드
df = pd.read_csv('/home/minkyoon/crohn/for_clam/label/relapse/new_output.csv')

# 'accession_number'를 기준으로 중복 제거 (처음 나타나는 항목을 유지)
#df = df.drop_duplicates(subset='accession_number')

# 각 라벨에 대한 데이터 개수를 셉니다.
label_counts = df['label'].value_counts()

print(label_counts)



## 중복되는 csv 확인
import pandas as pd

# csv 파일 로드
df1 = pd.read_csv('/home/minkyoon/crohn/csv/label_data/relapse/relapse_label.csv')
df2 = pd.read_csv('/home/minkyoon/crohn/csv/label_data/relapse/relapse_label_filtered.csv')

# 라벨 1.0만 필터링
df1 = df1[df1['label'] == 1.0]
df2 = df2[df2['label'] == 1.0]

# 첫 번째 파일의 'accession_number'가 두 번째 파일에 없는 경우 찾기
diff_accession_number = df1[~df1['accession_number'].isin(df2['accession_number'])]['accession_number'].unique()

# 결과 출력
print(diff_accession_number)

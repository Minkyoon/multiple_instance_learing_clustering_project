import pandas as pd

# CSV 파일 읽기
df = pd.read_csv('/home/minkyoon/crohn/csv/label_data/relapse/relapse_label_filtered.csv')

# 'accession_number'에 따라 데이터 분할
grouped = df.groupby('accession_number')

# 각 그룹을 별도의 CSV 파일로 저장
for name, group in grouped:
    group.to_csv(f'/home/minkyoon/crohn/csv/clam/relapse/downsampling/first/{name}.csv', index=False)


## to make real clam csv


import pandas as pd
from sklearn.model_selection import train_test_split

# 파일 읽기
df = pd.read_csv('/home/minkyoon/crohn/for_clam/new_output.csv')

# 필요한 열만 남기기
df = df[['slide_id', 'label']]

# 데이터를 train, valid, test로 분할
train, temp = train_test_split(df, test_size=0.2, random_state=42)
valid, test = train_test_split(temp, test_size=0.5, random_state=42)

# 각각의 데이터를 csv 파일로 저장
train.to_csv('/home/minkyoon/crohn/csv/label_data/clam/for_clam_relapse/train_valid_test_split/train.csv', index=False)
valid.to_csv('/home/minkyoon/crohn/csv/label_data/clam/for_clam_relapse/train_valid_test_split/valid.csv', index=False)
test.to_csv('/home/minkyoon/crohn/csv/label_data/clam/for_clam_relapse/train_valid_test_split/test.csv', index=False)


## train 에 있는 accession number만사용
import pandas as pd

# 파일 읽기
relapse_df = pd.read_csv('/home/minkyoon/crohn/csv/label_data/clam/for_clam_relapse/relapse_csv/relapse_label.csv')
train_df = pd.read_csv('/home/minkyoon/crohn/csv/label_data/clam/for_clam_relapse/train_valid_test_split/train.csv')

# 'relapse_label.csv' 파일에서 'train.csv' 파일의 'slide_id'와 겹치는 'accession_number'를 가진 행만 남기기
filtered_relapse_df = relapse_df[relapse_df['accession_number'].isin(train_df['slide_id'])]

# 제외된 'accession_number' 출력
excluded_accession_numbers = relapse_df[~relapse_df['accession_number'].isin(train_df['slide_id'])]['accession_number'].unique()
print("제외된 accession_number: ", excluded_accession_numbers)

# 필터링된 데이터를 csv 파일로 저장
filtered_relapse_df.to_csv('filtered_relapse_label.csv', index=False)

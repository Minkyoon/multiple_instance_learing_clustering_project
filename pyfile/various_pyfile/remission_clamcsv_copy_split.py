import pandas as pd

# split 데이터 읽기
split_df = pd.read_csv('/home/minkyoon/first/CLAM/splits/task_1_tumor_vs_normal_remission_for_multimodaal_100/splits_0.csv')

# 학습, 검증, 테스트에 사용할 accession_number 추출
train_acc_nums = split_df['train'].values
valid_acc_nums = split_df['val'].values
test_acc_nums = split_df['test'].values

# 원본 데이터 불러오기
data_df = pd.read_csv('/home/minkyoon/crohn/csv/end_to_end/remission/remission_label.csv')

# 각각의 accession_number에 대한 데이터만 필터링
train_data = data_df[data_df['accession_number'].isin(train_acc_nums)]
valid_data = data_df[data_df['accession_number'].isin(valid_acc_nums)]
test_data = data_df[data_df['accession_number'].isin(test_acc_nums)]

# csv 파일로 저장
train_data.to_csv('/home/minkyoon/crohn/csv/end_to_end/remission/train_data.csv', index=False)
valid_data.to_csv('/home/minkyoon/crohn/csv/end_to_end/remission/valid_data.csv', index=False)
test_data.to_csv('/home/minkyoon/crohn/csv/end_to_end/remission/test_data.csv', index=False)

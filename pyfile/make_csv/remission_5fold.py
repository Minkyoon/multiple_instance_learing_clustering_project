import pandas as pd
from sklearn.model_selection import GroupKFold, train_test_split

# Load data
df = pd.read_csv('/home/minkyoon/crohn/csv/end_to_end/remission/remission_label.csv')

# Add a new column with only the filename part of the filepath
df['filename'] = df['filepath'].apply(lambda x: int(x.split('/')[-1].split('.')[0]))

# Sort by 'accession_number' first and then by 'filename'
df = df.sort_values(['accession_number', 'filename'])

# Remove the 'filename' column as it's not needed anymore
df = df.drop('filename', axis=1)

# Implement 5-fold cross-validation on the whole dataset
gkf = GroupKFold(n_splits=5)

for i, (train_valid_index, test_index) in enumerate(gkf.split(df, groups=df['accession_number'])):
    # Get train_valid and test subsets for the current fold
    train_valid_subset = df.iloc[train_valid_index]
    test_subset = df.iloc[test_index]
    
    # Split train_valid subset into train and valid
    train_subset, valid_subset = train_test_split(train_valid_subset, test_size=1/8, stratify=train_valid_subset['accession_number'], random_state=42)
    
    # Save each fold as separate CSV files
    train_subset.to_csv(f'/home/minkyoon/crohn/csv/end_to_end/remission/5FOLD/train_fold{i}.csv', index=False)
    valid_subset.to_csv(f'/home/minkyoon/crohn/csv/end_to_end/remission/5FOLD/valid_fold{i}.csv', index=False)
    test_subset.to_csv(f'/home/minkyoon/crohn/csv/end_to_end/remission/5FOLD/test_fold{i}.csv', index=False)




import pandas as pd
from sklearn.model_selection import train_test_split, KFold

# 데이터 불러오기
df = pd.read_csv('/home/minkyoon/crohn/csv/end_to_end/remission/remission_label.csv')

# accession_number를 기준으로 데이터를 분할
unique_acc_nums = df['accession_number'].unique()

# K-Fold 객체 생성
kf = KFold(n_splits=5, random_state=42, shuffle=True)

for fold, (train_val_index, test_index) in enumerate(kf.split(unique_acc_nums)):
    # 테스트셋 지정
    test_acc_nums = unique_acc_nums[test_index]
    test_data = df[df['accession_number'].isin(test_acc_nums)]

    # 나머지 데이터에서 훈련 및 검증셋 생성
    train_val_acc_nums = unique_acc_nums[train_val_index]
    train_acc_nums, valid_acc_nums = train_test_split(train_val_acc_nums, test_size=0.125, random_state=42, shuffle=False)

    train_data = df[df['accession_number'].isin(train_acc_nums)]
    valid_data = df[df['accession_number'].isin(valid_acc_nums)]

    # Fold 별 데이터를 csv로 저장
    train_data.to_csv(f'/home/minkyoon/crohn/csv/end_to_end/remission/5FOLD/fold_{fold}_train_data.csv', index=False)
    valid_data.to_csv(f'/home/minkyoon/crohn/csv/end_to_end/remission/5FOLD/fold_{fold}_valid_data.csv', index=False)
    test_data.to_csv(f'/home/minkyoon/crohn/csv/end_to_end/remission/5FOLD/fold_{fold}_test_data.csv', index=False)

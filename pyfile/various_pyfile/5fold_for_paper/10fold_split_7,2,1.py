import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

# csv 파일 로드
df = pd.read_csv('/home/minkyoon/first/CLAM/dataset_csv/new_output.csv')

# case_id 컬럼 삭제
df = df.drop(columns=['case_id'])

# label에 대한 StratifiedKFold
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)  # test를 안 겹치게 10-fold

# train, valid, test를 저장할 dataframe
idx = 0

# test set 분할
for train_val_index, test_index in skf.split(df['slide_id'], df['label']):
    test = df.loc[test_index, 'slide_id'].values.tolist()

    # train과 valid set 분할
    train_val = df.loc[train_val_index]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.222, random_state=42)  # 7:2의 비율에서 validation이 차지하는 비율은 0.222 (2/9)
    train_index, val_index = next(sss.split(train_val['slide_id'], train_val['label']))

    train = train_val.iloc[train_index]['slide_id'].values.tolist()
    val = train_val.iloc[val_index]['slide_id'].values.tolist()

    # 분할된 train, valid, test를 저장
    split_df = pd.DataFrame()
    split_df['train'] = pd.Series(train)
    split_df['val'] = pd.Series(val)
    split_df['test'] = pd.Series(test)

    # Float을 int로 변환 (".0" 제거)
    split_df = split_df.astype('Int64')

    # 각 fold를 별도의 csv 파일로 저장
    split_df.to_csv(f'/home/minkyoon/first/CLAM/splits/remission_stratified_7,2,1/splits_{idx}.csv', index=False)  # 10-fold로 변경
    idx += 1

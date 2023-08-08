import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


random_seed=42
# 원본 데이터셋을 읽어옵니다.
total_dataset = pd.read_csv('/data/gongmo/team1/gongmo_2023/pyfile/normal_syntheic.csv')

# StratifiedKFold 인스턴스를 생성합니다.
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)

# StratifiedKFold를 사용하여 데이터를 5-fold로 나눕니다.
for fold, (train_index, test_index) in enumerate(skf.split(total_dataset, total_dataset['label'])):
    # 학습 데이터와 테스트 데이터를 분리합니다.
    train_data = total_dataset.iloc[train_index]
    test_data = total_dataset.iloc[test_index]
    
    # 학습 데이터에서 validation 데이터를 분리합니다.
    train_data, valid_data = train_test_split(train_data, test_size=0.2, stratify=train_data['label'], random_state=random_seed)

    # 각 데이터셋을 CSV 파일로 저장합니다.
    # train_data.to_csv(f'/data/gongmo/team1/gongmo_2023/csv/covidcollection_5fold/train_fold{fold}.csv', index=False)
    # valid_data.to_csv(f'/data/gongmo/team1/gongmo_2023/csv/covidcollection_5fold/valid_fold{fold}.csv', index=False)
    # test_data.to_csv(f'/data/gongmo/team1/gongmo_2023/csv/covidcollection_5fold/test_fold{fold}.csv', index=False)
    train_data.to_csv(f'/data/gongmo/team1/gongmo_2023/csv/dcgan_normal_vssyn/train_fold{fold}.csv', index=False)
    valid_data.to_csv(f'/data/gongmo/team1/gongmo_2023/csv/dcgan_normal_vssyn/valid_fold{fold}.csv', index=False)
    test_data.to_csv(f'/data/gongmo/team1/gongmo_2023/csv/dcgan_normal_vssyn/test_fold{fold}.csv', index=False)

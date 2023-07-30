import pandas as pd

# 각 CSV 파일에서 'slide_id' 열을 읽습니다.
train = pd.read_csv('/home/minkyoon/crohn/csv/label_data/clam/for_clam_relapse/train_valid_test_split/train.csv')['slide_id']
test = pd.read_csv('/home/minkyoon/crohn/csv/label_data/clam/for_clam_relapse/train_valid_test_split/test.csv')['slide_id']
valid = pd.read_csv('/home/minkyoon/crohn/csv/label_data/clam/for_clam_relapse/train_valid_test_split/valid.csv')['slide_id']

# 새로운 데이터프레임을 생성합니다.
df = pd.DataFrame()

# 'slide_id'를 순서대로 새로운 데이터프레임에 추가합니다.
df['train'] = train
df['valid'] = valid
df['test'] = test




# 결과를 CSV 파일로 저장합니다.
df.to_csv('splits_0.csv', index_label='Unnamed: 0')

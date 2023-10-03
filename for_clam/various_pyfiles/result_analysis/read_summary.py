import pandas as pd

# CSV 파일을 읽어서 데이터 프레임으로 변환
file_path = '/home/minkyoon/CLAM2/results/remission_multimodal_stratified_721_s1/summary.csv'
data = pd.read_csv(file_path)

# 'test_auc'와 'test_acc' 열의 평균 계산
test_auc_mean = data['test_auc'].mean()
test_acc_mean = data['test_acc'].mean()

# 데이터 프레임 출력
data

# 평균 값 출력
2

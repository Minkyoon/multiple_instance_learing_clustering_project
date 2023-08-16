import pandas as pd

# CSV 파일을 읽어서 데이터 프레임으로 변환
file_path = '/home/minkyoon/first/CLAM/results/remission_stratified_for_sysam_recommend_setting_7,2,1/task_1_tumor_vs_normal_CLAM_50_s1/summary.csv'
data = pd.read_csv(file_path)

# 'test_auc'와 'test_acc' 열의 평균 계산
test_auc_mean = data['test_auc'].mean()
test_acc_mean = data['test_acc'].mean()

# 데이터 프레임 출력
print(data)

# 평균 값 출력
print("Average test_auc:", test_auc_mean)
print("Average test_acc:", test_acc_mean)

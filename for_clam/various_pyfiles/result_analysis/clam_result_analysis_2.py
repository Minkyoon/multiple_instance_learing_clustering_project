import pandas as pd

df = pd.read_csv('/home/minkyoon/first/CLAM/results/remission_stratified_for_sysam_recommend_setting/task_1_tumor_vs_normal_CLAM_50_s1/summary.csv')

print('auc: ', df['test_auc'].mean())
print('acc: ', df['test_acc'].mean())
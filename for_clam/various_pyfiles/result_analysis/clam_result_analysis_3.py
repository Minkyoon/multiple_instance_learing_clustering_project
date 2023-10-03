import pickle
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, f1_score
import matplotlib.pyplot as plt
import seaborn as sns



# 결과 파일들의 경로
# result_files = [
#     '/home/minkyoon/first/CLAM/results/remission_stratified_for_sysam_recommend_setting_7,2,1/task_1_tumor_vs_normal_CLAM_50_s1/split_0_results.pkl'
#     '/home/minkyoon/first/CLAM/results/remission_stratified_for_sysam_recommend_setting_7,2,1/task_1_tumor_vs_normal_CLAM_50_s1/split_0_results.pkl'
#     '/home/minkyoon/first/CLAM/results/remission_stratified_for_sysam_recommend_setting_7,2,1/task_1_tumor_vs_normal_CLAM_50_s1/split_0_results.pkl'
#     '/home/minkyoon/first/CLAM/results/remission_stratified_for_sysam_recommend_setting_7,2,1/task_1_tumor_vs_normal_CLAM_50_s1/split_0_results.pkl'
#     '/home/minkyoon/first/CLAM/results/remission_stratified_for_sysam_recommend_setting_7,2,1/task_1_tumor_vs_normal_CLAM_50_s1/split_0_results.pkl'
#     '/home/minkyoon/first/CLAM/results/remission_stratified_for_sysam_recommend_setting_7,2,1/task_1_tumor_vs_normal_CLAM_50_s1/split_0_results.pkl'
#     '/home/minkyoon/first/CLAM/results/remission_stratified_for_sysam_recommend_setting_7,2,1/task_1_tumor_vs_normal_CLAM_50_s1/split_0_results.pkl'
#     '/home/minkyoon/first/CLAM/results/remission_stratified_for_sysam_recommend_setting_7,2,1/task_1_tumor_vs_normal_CLAM_50_s1/split_0_results.pkl'
#     '/home/minkyoon/first/CLAM/results/remission_stratified_for_sysam_recommend_setting_7,2,1/task_1_tumor_vs_normal_CLAM_50_s1/split_0_results.pkl'
# ]

result_files=[]

for i in range(10):
    name=f'/home/minkyoon/CLAM2/results2/update_multihead1_add_911_230915_forpaper_save_multihead/remission_multimodal_stratified_721_s1/split_{i}_results.pkl'
    result_files.append(name)

# 모든 결과를 저장할 리스트
all_true_labels = []
all_predicted_probs = []

# 각 결과 파일에서 데이터를 로드
for file_path in result_files:
    with open(file_path, 'rb') as file:
        results = pickle.load(file)
        all_true_labels.extend([v['label'] for v in results.values()])
        all_predicted_probs.extend([v['prob'][0, 1] for v in results.values()])

# numpy 배열로 변환
all_true_labels = np.array(all_true_labels)
all_predicted_probs = np.array(all_predicted_probs)

threshold = 0.5
all_predicted_labels = (all_predicted_probs >= threshold).astype(int)

# confusion matrix 계산
cm = confusion_matrix(all_true_labels, all_predicted_labels)

# AUROC 계산
fpr, tpr, _ = roc_curve(all_true_labels, all_predicted_probs)
roc_auc = auc(fpr, tpr)

# Sensitivity, Specificity, F1-Score 계산
tn, fp, fn, tp = cm.ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
f1 = f1_score(all_true_labels, all_predicted_labels)
precision = tp / (tp + fp)
recall = sensitivity
accuracy = (tp + tn) / (tp + tn + fp + fn)

# Confusion matrix 그리기
plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', annot_kws={"verticalalignment": 'center', 'size': 15})
plt.title('Confusion matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks([0.5, 1.5], labels=['Negative', 'Positive'])
plt.yticks([0.5, 1.5], labels=['Negative', 'Positive'], va='center')
plt.show()

# ROC curve 그리기
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# 지표 출력
print(f'Sensitivity: {sensitivity:.3f}')
print(f'Specificity: {specificity:.3f}')
print(f'F1-Score: {f1:.3f}')
print(f'Recall: {recall:.3f}')
print(f'Precision: {precision:.3f}')
print(f'accuracy:{accuracy:.3f}')




confusion_matrix = np.array([[90, 9], [16, 11]])

# 혼동 행렬의 각 요소 추출
TN = confusion_matrix[0, 0]
FP = confusion_matrix[0, 1]
FN = confusion_matrix[1, 0]
TP = confusion_matrix[1, 1]

# Sensitivity(민감도) 또는 Recall(재현율) 계산
sensitivity = TP / (TP + FN)
print(f"Sensitivity: {sensitivity:.2f}")

# Specificity(특이도) 계산
specificity = TN / (TN + FP)
print(f"Specificity: {specificity:.2f}")

# Precision(정밀도) 계산
precision = TP / (TP + FP)
print(f"Precision: {precision:.2f}")

# F1 Score 계산
f1_score = 2 * (precision * sensitivity) / (precision + sensitivity)
print(f"F1 Score: {f1_score:.2f}")

# Accuracy(정확도) 계산
accuracy = (TP + TN) / (TP + TN + FP + FN)
print(f"Accuracy: {accuracy:.2f}")





# Initialize variables to store Youden's index and corresponding threshold
best_youden = 0
best_threshold = 0

# Calculate Youden's index for each threshold
for thr in np.linspace(0, 1, 100):
    predicted_at_thr = (all_predicted_probs >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(all_true_labels, predicted_at_thr).ravel()
    sensitivity_at_thr = tp / (tp + fn)
    specificity_at_thr = tn / (tn + fp)
    youden_at_thr = sensitivity_at_thr + specificity_at_thr - 1
    
    if youden_at_thr > best_youden:
        best_youden = youden_at_thr
        best_threshold = thr

# Calculate metrics at best threshold
all_predicted_labels = (all_predicted_probs >= best_threshold).astype(int)
cm = confusion_matrix(all_true_labels, all_predicted_labels)
tn, fp, fn, tp = cm.ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
f1 = f1_score(all_true_labels, all_predicted_labels)
precision = tp / (tp + fp)
recall = sensitivity
accuracy = (tp + tn) / (tp + tn + fp + fn)

print(f"Best Youden's index: {best_youden:.3f} at threshold: {best_threshold:.3f}")
print(f'Sensitivity: {sensitivity:.3f}')
print(f'Specificity: {specificity:.3f}')
print(f'F1-Score: {f1:.3f}')
print(f'Recall: {recall:.3f}')
print(f'Precision: {precision:.3f}')
print(f'Accuracy: {accuracy:.3f}')

# %%

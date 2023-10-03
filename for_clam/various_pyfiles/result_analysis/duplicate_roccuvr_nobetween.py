

### fold 마다 roc curve 그리고 결과보기


import pickle
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
def find_optimal_point(mean_fpr, mean_tpr):
    youden_index = mean_tpr - mean_fpr
    optimal_idx = np.argmax(youden_index)
    optimal_fpr = mean_fpr[optimal_idx]
    optimal_tpr = mean_tpr[optimal_idx]
    return optimal_fpr, optimal_tpr


def plot_roc_curve(result_files, color, label):
    tprs = []
    mean_fpr = np.linspace(0, 1, 100)
    aurocs = []

    for file_path in result_files:
        with open(file_path, 'rb') as file:
            results = pickle.load(file)
            true_labels = [v['label'] for v in results.values()]
            predicted_probs = [v['prob'][0, 1] for v in results.values()]

            fpr, tpr, _ = roc_curve(true_labels, predicted_probs)
            tprs.append(np.interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    
    optimal_fpr, optimal_tpr = find_optimal_point(mean_fpr, mean_tpr)
    plt.scatter(optimal_fpr, optimal_tpr, marker='o', color='black', s=15)

    plt.plot(mean_fpr, mean_tpr, color=color, label=f'{label} (AUC = {round(mean_auc, 3)})')
    #plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color=color, alpha=.2)
    
    
def plot_roc_curve2(result_files, color, label):
    tprs = []
    mean_fpr = np.linspace(0, 1, 100)
    aurocs = []

    for file_path in result_files:
        with open(file_path, 'rb') as file:
            results = pickle.load(file)
            true_labels = [v['label'] for v in results.values()]
            predicted_probs = [v['prob'][1] for v in results.values()]

            fpr, tpr, _ = roc_curve(true_labels, predicted_probs)
            tprs.append(np.interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    
    optimal_fpr, optimal_tpr = find_optimal_point(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color=color, label=f'{label} (AUC = {round(mean_auc, 3)})')
    plt.scatter(optimal_fpr, optimal_tpr, marker='o', color='black', label='Optimal Point' , s=15)

    
    #plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color=color, alpha=.2)

plt.figure(figsize=(6, 6))

result_files_1 = []
result_files_2 = []
result_files_3=[]

for i in range(10):
    name1=f'/home/minkyoon/CLAM2/results2/update_multihead1_add_911_230915_forpaper_save_multihead/remission_multimodal_stratified_721_s1/split_{i}_results.pkl'
    result_files_1.append(name1)
    name2=f'/home/minkyoon/first/CLAM/results/remission_stratified_for_recommend_setting_7,2,1_learningragechange/task_1_tumor_vs_normal_CLAM_50_s1/split_{i}_results.pkl'
    result_files_2.append(name2)
    name3=f'/home/minkyoon/xgboost/pkl/splits_{i}_results.pkl'
    result_files_3.append(name3)

plot_roc_curve(result_files_1, 'b', 'multi_modal')
plot_roc_curve(result_files_2, 'r', 'clam')
plot_roc_curve2(result_files_3, 'g', 'xgboost')

plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


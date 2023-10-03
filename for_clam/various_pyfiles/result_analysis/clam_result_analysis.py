import pickle
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, f1_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns

# Load the results
fold=0
with open(f'/home/minkyoon/CLAM2/results/update_multihead7_add_911_230823_forpaper/remission_multimodal_stratified_721_s1/split_1_results.pkl', 'rb') as file:
    results = pickle.load(file)

# Extract true labels and predicted probabilities
true_labels = np.array([v['label'] for v in results.values()])
predicted_probs = np.array([v['prob'][0, 1] for v in results.values()])  # Probability of positive class


threshold = 0.5  
predicted_labels = (predicted_probs >= threshold).astype(int)


#  confusion matrix
cm = confusion_matrix(true_labels, predicted_labels)



# Compute AUROC
fpr, tpr, _ = roc_curve(true_labels, predicted_probs)
roc_auc = auc(fpr, tpr)

# Compute Sensitivity, Specificity, F1-Score, Recall
tn, fp, fn, tp = cm.ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
f1 = f1_score(true_labels, predicted_labels)
recall = sensitivity  # in binary classification, recall is the same as sensitivity

# Plotting the confusion matrix
plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', annot_kws={"verticalalignment": 'center', 'size': 15})
plt.title('Confusion matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks([0.5, 1.5], labels=['Negative', 'Positive'])
plt.yticks([0.5, 1.5], labels=['Negative', 'Positive'], va='center')
plt.show()


# ROC curve
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

# metrics
print(f'Sensitivity: {sensitivity:.2f}')
print(f'Specificity: {specificity:.2f}')
print(f'F1-Score: {f1:.2f}')
print(f'Recall: {recall:.2f}')




import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Sample function to compute bootstrap for ROC
def bootstrap_roc(true_labels, predicted_probs, n_bootstraps=1000, alpha=.05):
    n_samples = len(true_labels)
    bootstrapped_scores = []

    for i in range(n_bootstraps):
        # Bootstrap by sampling with replacement on the prediction indices
        indices = np.random.randint(0, len(predicted_probs), len(predicted_probs))
        if len(np.unique(true_labels[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue

        fpr, tpr, _ = roc_curve(true_labels[indices], predicted_probs[indices])
        bootstrapped_scores.append(auc(fpr, tpr))
        
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()

    # Computing the lower and upper bound of the 95% confidence interval
    #alpha =.5
    confidence_lower = sorted_scores[int((alpha / 2) * n_bootstraps)]
    confidence_upper = sorted_scores[int((1 - alpha / 2) * n_bootstraps)]

    return confidence_lower, confidence_upper

# Actual ROC curve
fpr, tpr, _ = roc_curve(true_labels, predicted_probs)
roc_auc = auc(fpr, tpr)

confidence_lower, confidence_upper = bootstrap_roc(true_labels, predicted_probs)

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color='darkorange', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (roc_auc, (confidence_upper-confidence_lower)))
plt.fill_between(fpr, tpr - (confidence_upper-confidence_lower),
                 tpr + (confidence_upper-confidence_lower), color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()



#
cm=np.array([[28 ,17],
            [29 ,16]])
cm=np.array([[95,5],
             [14,13]])

# find tp tn fp fn

# Initialize counters for tp, tn, fp, fn
tp = []
tn = []
fp = []
fn = []

# Iterate over the dictionary items
for slide_id, info in results.items():
    # Determine the predicted label (class with highest probability)
    pred_label = np.argmax(info['prob'])
    
    # Compare the predicted and true labels
    if pred_label == info['label']:
        # If they're the same, it's either a true positive or true negative
        if pred_label == 1:
            tp.append(int(slide_id))  # True positive
        else:
            tn.append(int(slide_id))  # True negative
    else:
        # If they're different, it's either a false positive or false negative
        if pred_label == 1:
            fp.append(int(slide_id))  # False positive
        else:
            fn.append(int(slide_id))  # False negative

print("True Positives:", tp)
print("True Negatives:", tn)
print("False Positives:", fp)
print("False Negatives:", fn)



import numpy as np

# 주어진 혼동 행렬을 numpy 배열로 생성
confusion_matrix = np.array([[84,14], [15, 2]])

# 혼동 행렬의 각 요소 추출
TN = confusion_matrix[0, 0]
FP = confusion_matrix[0, 1]
FN = confusion_matrix[1, 0]
TP = confusion_matrix[1, 1]

# Sensitivity(민감도) 또는 Recall(재현율) 계산
sensitivity = TP / (TP + FN)
print(f"Sensitivity: {sensitivity:.3f}")

# Specificity(특이도) 계산
specificity = TN / (TN + FP)
print(f"Specificity: {specificity:.3f}")

# Precision(정밀도) 계산
precision = TP / (TP + FP)
print(f"Precision: {precision:.3f}")

# F1 Score 계산
f1_score = 2 * (precision * sensitivity) / (precision + sensitivity)
print(f"F1 Score: {f1_score:.3f}")

# Accuracy(정확도) 계산
accuracy = (TP + TN) / (TP + TN + FP + FN)
print(f"Accuracy: {accuracy:.3f}")


import torch

# 주어진 텐서 생성
tensor = torch.tensor([
    [0.0141, 0.0354, 0.0245, 0.0136, 0.0177, 0.0247, 0.0140, 0.0312, 0.0188,
    0.0206, 0.0154, 0.0126, 0.0285, 0.0231, 0.0307, 0.0377, 0.0305, 0.0166,
    0.0211, 0.0167, 0.0173, 0.0147, 0.0121, 0.0217, 0.0199, 0.0265, 0.0149,
    0.0293, 0.0190, 0.0231, 0.0267, 0.0100, 0.0195, 0.0222, 0.0169, 0.0206,
    0.0106, 0.0158, 0.0218, 0.0165, 0.0189, 0.0142, 0.0252, 0.0195, 0.0121,
    0.0228, 0.0174, 0.0127, 0.0306]
], device='cuda:0')

# attention score가 높은 순서대로 인덱스를 얻음
sorted_indices = torch.argsort(tensor, descending=True)
top_indices = sorted_indices[0, :4]

print("Top 4 indices:", top_indices)


import pandas as pd

df=pd.read_csv('/home/minkyoon/CLAM2/data/processed/remission_under_10.csv')


df=df.iloc[:, 1:50]







import pandas as pd
import torch



# 주어진 텐서 생성
tensor = torch.tensor([
    [0.0141, 0.0354, 0.0245, 0.0136, 0.0177, 0.0247, 0.0140, 0.0312, 0.0188,
    0.0206, 0.0154, 0.0126, 0.0285, 0.0231, 0.0307, 0.0377, 0.0305, 0.0166,
    0.0211, 0.0167, 0.0173, 0.0147, 0.0121, 0.0217, 0.0199, 0.0265, 0.0149,
    0.0293, 0.0190, 0.0231, 0.0267, 0.0100, 0.0195, 0.0222, 0.0169, 0.0206,
    0.0106, 0.0158, 0.0218, 0.0165, 0.0189, 0.0142, 0.0252, 0.0195, 0.0121,
    0.0228, 0.0174, 0.0127, 0.0306]
], device='cuda:0')
tensor = torch.tensor([
[ 0.0017,0.0016, 0.0013, 0.0016, 0.0022, 0.0022, 0.0017, 0.0020, 0.0015, 0.0021,
         0.0022, 0.0020, 0.0021, 0.0014, 0.0016, 0.0019, 0.0011, 0.0012, 0.0013,
         0.0015, 0.0014, 0.0019, 0.0012, 0.0024, 0.0023, 0.0016, 0.0016, 0.0018,
         0.0014, 0.0013, 0.0014, 0.0012, 0.0013, 0.0014, 0.0020, 0.0020, 0.0014,
         0.0015, 0.0022, 0.0016, 0.0017, 0.0011, 0.0017, 0.0017, 0.0015, 0.0014,
          0.0023, 0.0020, 0.0016]
])

# attention score가 높은 순서대로 인덱스를 얻음
sorted_indices = torch.argsort(tensor, descending=True)
top_indices = sorted_indices[0, :10].cpu().numpy()

# 가장 높은 attention score와 해당 인덱스를 출력
print("Top 10 features with their attention scores:")
for i in top_indices:
    feature_name = df.columns[i]
    feature_values = df[feature_name]
    print(f"Feature: {feature_name}, Attention Score: {tensor[0, i]:.4f}, ")

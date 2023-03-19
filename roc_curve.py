import pickle
from sklearn.metrics import accuracy_score, roc_curve, auc, classification_report, det_curve
import matplotlib.pyplot as plt
import numpy as np

with open('true_label.pkl', 'rb') as f:
    true_label = pickle.load(f)

with open('predict_label.pkl', 'rb') as f:
    predict_label = pickle.load(f)

# Áp đặt ngưỡng phân loại 0.6
y_pred_threshold = []
for pred in predict_label:
    if pred >= 0.2:
        y_pred_threshold.append(1)
    else:
        y_pred_threshold.append(0)


# Calculate the accuracy and AUC score
accuracy = accuracy_score(true_label, y_pred_threshold)
fpr, tpr, thresholds = roc_curve(true_label, predict_label)
auc_score = auc(fpr, tpr)

print("accuracy: ", accuracy)
# Vẽ ROC curve
plt.plot(fpr, tpr, label='ROC curve (area = %0.4f)' % auc_score)
plt.plot([0, 1], [0, 1], 'k--', label='Random guessing')
plt.xlim([-0.02, 1.0])
plt.ylim([-0.02, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


threshold_idx = np.argmin(np.abs(thresholds - 0.2))
threshold_fpr = fpr[threshold_idx]
threshold_tpr = tpr[threshold_idx]

print("False Positive Rate at threshold 0.6:", threshold_fpr)
print("True Positive Rate at threshold 0.6:", threshold_tpr)


# Tìm threshold tương ứng với FPR = 0.2
fpr_idx = np.argmin(np.abs(fpr - 0.2))
threshold = thresholds[fpr_idx]

print("Threshold for FPR = 0.2:", threshold)

# Tính toán DET curves
fpr, fnr, thresholds = det_curve(true_label, predict_label)

# Vẽ DET curves
plt.figure()
plt.plot(fpr, fnr)
plt.xlabel('False Positive Rate')
plt.ylabel('False Negative Rate')
plt.title('DET curves')
plt.show()
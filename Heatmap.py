import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# สร้าง Confusion Matrix
confusion_matrix = np.array([[6, 2],  # TP, FP
                             [2, 40]])  # FN, TN

# สร้าง Heatmap พร้อม Label ใหม่
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Users to Be Recommended', 'Users Not to Be Recommended'],
            yticklabels=['Recommended Users', 'Not Recommended Users'])

plt.title('Confusion Matrix for Recommendation System')
plt.xlabel('Ground Truth (Actual Class)')
plt.ylabel('System Prediction (Predicted Class)')
plt.show()

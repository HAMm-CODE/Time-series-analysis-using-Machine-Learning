# -*- coding: utf-8 -*-
"""
Improved SVM_example.py – higher recognition accuracy
"""

from sklearn import datasets, svm, metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

# Load full Iris dataset (use all 4 features)
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Feature scaling for better SVM performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

# Choose one improved model (RBF kernel)
clf = svm.SVC(kernel="rbf", gamma=0.5, C=10)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Compute and display recognition accuracy
acc = metrics.accuracy_score(y_test, y_pred)
print(f"Improved Recognition Accuracy (RBF kernel): {acc:.2f}")

# Plot 3x3 confusion matrix
cm = metrics.confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.title("Confusion Matrix – Improved RBF SVM")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

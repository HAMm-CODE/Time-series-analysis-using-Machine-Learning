# -*- coding: utf-8 -*-
"""
Modified SVM_example.py to include 3x3 confusion matrix
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

# Load Iris dataset
iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train SVM (RBF kernel example)
clf = svm.SVC(kernel="rbf", gamma=0.7, C=1.0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Create and plot 3x3 confusion matrix
cm = metrics.confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.title("3x3 Confusion Matrix for RBF SVM")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

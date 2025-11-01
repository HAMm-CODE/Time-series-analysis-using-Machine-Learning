
from aeon.datasets import load_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# Load GunPoint dataset
X, y = load_classification("GunPoint")

# Flatten time series so RandomForest can handle it (2D input)
n_samples, n_channels, n_timepoints = X.shape
X_reshaped = X.reshape(n_samples, n_channels * n_timepoints)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_reshaped, y, test_size=0.3, random_state=42, stratify=y
)

# Train Random Forest
clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc * 100:.2f}%")

# Confusion matrix figure
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix – GunPoint Random Forest")
plt.show()

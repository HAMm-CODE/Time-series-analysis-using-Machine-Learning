
import matplotlib.pyplot as plt
from matplotlib import colormaps
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.inspection import DecisionBoundaryDisplay

# Load Iris dataset
iris = datasets.load_iris()
X = iris['data'][:, :2]   # Use only first two features
y = iris['target']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

C = 1.0  # SVM regularization parameter
models = [
    ("SVC with linear kernel", svm.SVC(kernel="linear", C=C)),
    ("LinearSVC (linear kernel)", svm.LinearSVC(C=C, max_iter=10000)),
    ("SVC with RBF kernel", svm.SVC(kernel="rbf", gamma=0.7, C=C)),
    ("SVC with polynomial (degree 3) kernel", svm.SVC(kernel="poly", degree=3, gamma="auto", C=C)),
]

# Train, test, and display accuracy for each model
print("Recognition Accuracy Results:")
for name, clf in models:
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    print(f"{name}: {acc:.2f}")

# Plot decision boundaries
fig, sub = plt.subplots(2, 2, figsize=(8, 8))
plt.subplots_adjust(wspace=0.4, hspace=0.4)
X0, X1 = X[:, 0], X[:, 1]

for (name, clf), ax in zip(models, sub.flatten()):
    disp = DecisionBoundaryDisplay.from_estimator(
        clf,
        X,
        response_method="predict",
        cmap=colormaps['coolwarm'],
        alpha=0.8,
        ax=ax,
        xlabel=iris.feature_names[0],
        ylabel=iris.feature_names[1],
    )
    ax.scatter(X0, X1, c=y, cmap=colormaps['coolwarm'], s=20, edgecolors="k")
    ax.set_title(name)

plt.show()

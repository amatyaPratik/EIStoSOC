import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, matthews_corrcoef

# Load PCA-transformed dataset
# Replace with your actual filename
pca_data = pd.read_csv("./Output/pca_features.csv")
X = pca_data[["PC1", "PC2", "battery", "cycle"]]  # Features
y = pca_data["soc"]  # Target

# Preprocess categorical features (if any, like "battery")
X = pd.get_dummies(X, columns=["battery"], drop_first=True)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=17)

# Define models and hyperparameter grids
svc = SVC()
knn = KNeighborsClassifier()
rf = RandomForestClassifier()

param_grid_svm = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto'],
    'kernel': ['linear', 'rbf']
}

param_grid_knn = {
    'n_neighbors': [3, 5, 7],
    'weights': ['uniform', 'distance']
}

param_grid_rf = {
    'n_estimators': [100, 200],
    'criterion': ["gini", "entropy"]
}

models = [svc, knn, rf]
params = [param_grid_svm, param_grid_knn, param_grid_rf]

# Perform Grid Search for each model
for i, model in enumerate(models):
    grid_search = GridSearchCV(
        estimator=model, param_grid=params[i], scoring='accuracy', cv=5, verbose=1)
    grid_search.fit(X_train, y_train)

    print(f"Model: {model.__class__.__name__}")
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Training Accuracy: {grid_search.best_score_:.2f}")

    y_pred = grid_search.predict(X_test)
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(
        f"Matthews Correlation Coefficient: {matthews_corrcoef(y_test, y_pred):.2f}")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

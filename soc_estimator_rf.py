from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# Load the saved PCA CSV file
pca_df = pd.read_csv("./Output/pca_features.csv")

# Split data
X = pca_df[["PC1", "PC2"]]   # Features
y = pca_df["soc"]  # Target variable (categorical soc)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Define the model and parameter grid
rf_classifier = RandomForestClassifier(random_state=42)
param_grid_rf = {
    'n_estimators': [50, 100, 200],        # Number of trees in the forest
    'max_depth': [None, 10, 20],           # Depth of the tree
    # Min samples to split an internal node
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],         # Min samples at a leaf node
}

# Grid Search with Cross-Validation
grid_search = GridSearchCV(
    estimator=rf_classifier, param_grid=param_grid_rf, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model and evaluation
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)
print("Best Parameters:", grid_search.best_params_)
print("Accuracy on Test Set:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# This is a sample Python script.
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, matthews_corrcoef
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import Util.util as ut

DATASET_PATH = "./Data/Data_IFR14500_Batt_1-11_cycle_1_2.csv"
FREQ_PATH = "./Data/Frequencies.csv"

if __name__ == '__main__':

    # Loading dataset
    DATASET, FREQUENCIES, PERIODS = ut.load_data(
        data_path=DATASET_PATH, data_freq_path=FREQ_PATH, sep=",")  # Default paths

    # Data display
    ut.display_with_plotly(DATASET, FREQUENCIES, 1)

    # Organize data for Machine Learning models
    DATASET = DATASET.drop(columns=["cycle"])
    y = DATASET["SoC"]  # Label array
    X = DATASET.drop(columns=['Battery', 'SoC'])  # Feature matrix

    # Create SoC step as labels -> 10 class problem
    y_1 = [(s + 5 if s % 2 != 0 else s)
           for s in y]  # Simplified label creation
    y = pd.Series(y_1)

    # Convert to numpy arrays with contiguous memory layout
    X = np.ascontiguousarray(X.values)
    y = np.ascontiguousarray(y.values)

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.20, random_state=17
    )

    # Parameter definition for each model type with updated range
    param_grid_svm = {
        'C': [0.1, 1, 10, 100, 1000],
        'gamma': [1, 0.1, 0.01, 0.001, 'scale', 'auto'],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'decision_function_shape': ['ovo', 'ovr']
    }

    param_grid_knn = {
        'n_neighbors': [1, 2, 3, 4, 5, 6, 7],
        'weights': ['uniform', 'distance'],
        'p': [1, 2]
    }

    param_grid_rf = {
        'criterion': ["gini", "entropy"],
        'max_features': ['sqrt', 'log2']
    }

    # Model definitions
    svc = SVC()
    knn = KNeighborsClassifier()
    rf = RandomForestClassifier()

    classifiers = [svc, knn, rf]
    parameters = [param_grid_svm, param_grid_knn, param_grid_rf]

    # Grid search and evaluation for each model
    for i, clf in enumerate(classifiers):
        grid_search = GridSearchCV(
            estimator=clf,
            param_grid=parameters[i],
            scoring='accuracy',
            cv=5,
            verbose=1
        )

        grid_search.fit(X_train, y_train)

        print(f"\nModel: {clf.__class__.__name__}")
        print(f"Best parameters: {grid_search.best_params_}")
        best_score = grid_search.best_score_ * 100
        print(f"Training accuracy with tuning: {best_score:.2f}%")

        # Predict and evaluate on test set
        predicted = grid_search.predict(X_test)
        test_accuracy = accuracy_score(y_test, predicted) * 100
        print(f"Test accuracy with tuning: {test_accuracy:.2f}%")
        print(
            f"MCC for test dataset with tuning: {matthews_corrcoef(y_test, predicted):.2f}")
        print("-- TEST CLASSIFICATION REPORT --")
        print(classification_report(y_test, predicted))
        print("-- CONFUSION MATRIX --")
        print(confusion_matrix(y_test, predicted))

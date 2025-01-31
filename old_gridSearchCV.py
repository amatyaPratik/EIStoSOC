# This is a sample Python script.
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, matthews_corrcoef
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

import Util.util as ut

DATASET_PATH = "./Data/Data_IFR14500_Batt_1-11_cycle_1_2.csv"
FREQ_PATH = "Data/Frequencies.csv"

if __name__ == '__main__':

    # Loading dataset
    DATASET, FREQUENCIES, PERIODS = \
        ut.load_data(data_path=DATASET_PATH,
                     data_freq_path=FREQ_PATH, sep=",")  # Default paths

    # Data display
    ut.display_with_plotly(DATASET, FREQUENCIES, 1)

    # Organize data for Machine Learning models
    DATASET = DATASET.drop(columns=["cycle"])
    y = DATASET["SoC"]  # label array
    X = DATASET.drop(columns=['Battery', 'SoC'])  # feature matrix

    # Create SoC step as labels -> 10 class problem
    y_1 = []
    for s in y:
        if s % 2 != 0:
            s = s + 5
        y_1.append(s)
    y = pd.Series(y_1)

    # Split data in train and test
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        stratify=y,
                                                        test_size=0.20,
                                                        random_state=17)

    # Parameter definition for each model type
    param_grid_svm = {
        'C': [0.1, 1, 10, 100, 1000, 10000, 100000, 100000000],
        'gamma': [1, 0.1, 0.01, 0.001, 0.0001, 'scale', 'auto'],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'decision_function_shape': ['ovo', 'ovr']
    }

    param_grid_knn = {
        'n_neighbors': [1, 2, 3, 4, 5, 6, 7],
        'weights': ['uniform', 'distance'],
        'p': [1, 2]
    }

    param_grid_rf = {
        'criterion': ["gini", "entropy", "log_loss"],
        'max_features': ['sqrt', 'log2', None]
    }

    svc = SVC()
    knn = KNeighborsClassifier()
    rf = RandomForestClassifier()

    classifiers = [svc, knn, rf]
    parameters = [param_grid_svm, param_grid_knn, param_grid_rf]

    # For each of the models, a GridSearch is performed to find the best hyperparameters and the results are displayed.
    for i in range(0, len(classifiers)):
        grid_search = GridSearchCV(estimator=classifiers[i],
                                   param_grid=parameters[i],
                                   scoring='accuracy',
                                   cv=5,
                                   verbose=1)

        grid_search.fit(X_train, y_train)

        print(classifiers[i].__class__)
        print(f"\tBest param: {grid_search.best_params_}")
        accuracy = grid_search.best_score_ * 100
        print("\tAccuracy for our Training dataset with tuning is : {:.2f}%.".format(
            accuracy))
        predicted = grid_search.predict(X_test)
        accuracy_test = accuracy_score(y_test, predicted) * 100
        print("\tAccuracy for our Test dataset with tuning is : {:.2f}%.".format(
            accuracy_test))
        print("\tMCC for our Test dataset with tuning is : {:.2f}%.".format(
            matthews_corrcoef(y_test, predicted)))
        print("\t-- TEST CLASSIFICATION REPORT --")
        print(classification_report(y_test, predicted))
        print("\t-- CONFUSION MATRIX --")
        print(confusion_matrix(y_test, predicted))

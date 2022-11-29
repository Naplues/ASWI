# -*-coding:utf-8 -*-
from statistics import mean

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

from helper import *
import warnings
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.metrics import *

__author__ = 'Naples'

warnings.filterwarnings("ignore", category=DeprecationWarning)
np.random.seed(0)


# ############################### Test data set split ########################################
def allocate_data_set():
    for project in PROJECT:
        for x in range(1, 5):  # Iterator 1, 2, 3, 4
            # Warning data in previous and next release, respectively.
            w_set_1, w_list_1 = get_warning_set(f'{data_path}/{project}/warnings/warningInfo{x}.csv',
                                                f'{data_path}/{project}/golden/goldenFeatures{x}.csv')

            w_set_2, w_list_2 = get_warning_set(f'{data_path}/{project}/warnings/warningInfo{x + 1}.csv',
                                                f'{data_path}/{project}/golden/goldenFeatures{x + 1}.csv')

            # Overlapping warnings
            intersection = list(w_set_1.intersection(w_set_2))
            # Linking the indices of same warnings in two consecutive releases.
            index_map = list()
            for instance in intersection:
                cur_index = [w_list_1.index(instance), w_list_2.index(instance)]
                index_map.append(cur_index)

            # Data set split
            # test B
            data_2 = read_data_from_file(f'{data_path}/{project}/golden/goldenFeatures{x + 1}.csv')
            text_B = data_2[0]
            for index in range(len(data_2[1:])):
                warning = w_list_2[index]
                if warning in intersection:
                    text_B += data_2[index + 1]
            save_csv_result(f'{data_path}/{project}/golden_B/', f'goldenFeatures{x + 1}.csv', text_B)

            # test C
            data_2 = read_data_from_file(f'{data_path}/{project}/golden/goldenFeatures{x + 1}.csv')
            text_C = data_2[0]
            for index in range(len(data_2[1:])):
                warning = w_list_2[index]
                if warning not in intersection:
                    text_C += data_2[index + 1]
            save_csv_result(f'{data_path}/{project}/golden_C/', f'goldenFeatures{x + 1}.csv', text_C)


# ############################### Test data set split ########################################

CLASSIFIERS = [
    LogisticRegression(max_iter=3000, random_state=0),
    RandomForestClassifier(random_state=0),
    DecisionTreeClassifier(random_state=0),
    AdaBoostClassifier(random_state=0),
    svm.SVC(kernel="linear", random_state=0)]


def open_file(train_project_path, test_project_path, release=1):
    train_file = rf"{train_project_path}/goldenFeatures{release}.csv"
    test_file = rf"{test_project_path}/goldenFeatures{release + 1}.csv"

    train = pd.read_csv(train_file, index_col=None, skiprows=0, low_memory=False)
    test = pd.read_csv(test_file, index_col=None, skiprows=0, low_memory=False)

    header_list = [train.columns.tolist(), test.columns.tolist()]  # list of header for training set

    return train, test, header_list


def data_preparing(train_project, test_project, release):
    train_data, test_data, feature_list = open_file(train_project, test_project, release)

    # Extract training and test data set
    x_train_numerical = train_data.iloc[:, :-6]  # pandas.core.frame.DataFrame
    x_train_nominal = train_data.iloc[:, -6:-1]  # pandas.core.frame.DataFrame
    y_train = train_data.iloc[:, -1]  # pandas.core.series.Series

    x_test_numerical = test_data.iloc[:, :-6]
    x_test_nominal = test_data.iloc[:, -6:-1]
    y_test = test_data.iloc[:, -1]

    x_train_numerical = np.asanyarray(x_train_numerical)
    x_test_numerical = np.asanyarray(x_test_numerical)

    # OneHotEncoder
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    x_train_nominal = encoder.fit_transform(x_train_nominal)
    x_test_nominal = encoder.transform(x_test_nominal)

    # x_train, x_test
    x_train = np.c_[x_train_numerical, x_train_nominal]
    x_test = np.c_[x_test_numerical, x_test_nominal]

    # Normalize the x for training and test data sets
    # Linear svm needs normalization to reduce the running cost on server
    min_max_scalar = preprocessing.MinMaxScaler()

    # Convert dataframe into numpy.array to normalize

    x_train = min_max_scalar.fit_transform(np.asarray(x_train))
    x_test = min_max_scalar.transform(np.asarray(x_test))

    return x_train, y_train, x_test, y_test


# ######################################  Cross-release experimental scenario #####################################
def build_model(train_file, test_file, clf, release):
    x_train, y_train, x_test, y_test = data_preparing(train_file, test_file, release)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    # Evaluation
    precision = precision_score(y_test, y_pred, pos_label='close')
    recall = recall_score(y_test, y_pred, pos_label='close')
    f1 = f1_score(y_test, y_pred, pos_label='close')
    return round(precision, 3), round(recall, 3), round(f1, 3)


strategies = [["golden", "golden_B"], ["golden", "golden_C"], ["golden", "golden"], ]
test_data_set = ['B', 'C', 'B+C']


def run_prediction(to_file=False):
    performance_dict, performance_avg_dict = dict(), dict()
    for clf in CLASSIFIERS:
        print(f'========== {clf}')
        p_list, r_list, f_list = [], [], []
        p_avg_list, r_avg_list, f_avg_list = [], [], []
        for project in PROJECT:
            print(f'{project}')
            p_avg, r_avg, f_avg = [], [], []
            for release in [1, 2, 3, 4]:
                train_file = rf"{data_path}/{project}/golden/"
                test_file = rf"{data_path}/{project}/golden/"
                p, r, f = build_model(train_file, test_file, clf, release)
                p_list.append(p)
                r_list.append(r)
                f_list.append(f)

                p_avg.append(p)
                r_avg.append(r)
                f_avg.append(f)
            p_avg_list.append(mean(p_avg))
            r_avg_list.append(mean(r_avg))
            f_avg_list.append(mean(f_avg))

        performance_dict[str(clf) + '-p'] = p_list
        performance_dict[str(clf) + '-r'] = r_list
        performance_dict[str(clf) + '-f'] = f_list

        performance_avg_dict[str(clf) + 'P'] = p_avg_list
        performance_avg_dict[str(clf) + 'R'] = r_avg_list
        performance_avg_dict[str(clf) + 'F1'] = f_avg_list

    # Output result in test B and C, respectively.
    df = pd.DataFrame(performance_dict)
    df.to_csv(f'{root_path}/analysis/Supervised.csv') if to_file else None
    df = pd.DataFrame(performance_avg_dict)
    df.to_csv(f'{root_path}/analysis/Supervised-avg.csv') if to_file else None
    pass


def run_prediction_2():
    p_dict, r_dict, f_dict = dict(), dict(), dict()
    for index in range(len(test_data_set)):
        for clf in CLASSIFIERS:
            print(f'========== {clf}')
            p_list, r_list, f_list = [], [], []
            for project in PROJECT:
                print(f'{project}')
                for release in [1, 2, 3, 4]:
                    train_file = rf"{data_path}/{project}/{strategies[index][0]}/"
                    test_file = rf"{data_path}/{project}/{strategies[index][1]}/"
                    p, r, f = build_model(train_file, test_file, clf, release)
                    p_list.append(p)
                    r_list.append(r)
                    f_list.append(f)

            p_dict[str(clf)] = p_list
            r_dict[str(clf)] = r_list
            f_dict[str(clf)] = f_list

        # Output result in test B and C, respectively.
        df = pd.DataFrame(p_dict)
        df.to_csv(f'{root_path}/analysis/Supervised-{test_data_set[index]}_p.csv')
        df = pd.DataFrame(r_dict)
        df.to_csv(f'{root_path}/analysis/Supervised-{test_data_set[index]}_r.csv')
        df = pd.DataFrame(f_dict)
        df.to_csv(f'{root_path}/analysis/Supervised-{test_data_set[index]}_f.csv')


if __name__ == '__main__':
    run_prediction(to_file=True)
    # run_prediction_2()
    # allocate_data_set()
    pass

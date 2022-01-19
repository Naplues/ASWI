# -*-coding:utf-8 -*-

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


################################ Dataset allocation ########################################
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
                if len(cur_index) != 2:
                    print('Would never execute')
                    continue
                index_map.append(cur_index)

            # Data set allocation
            # train A
            data_1 = read_data_from_file(f'{data_path}/{project}/golden/goldenFeatures{x}.csv')
            text_A = data_1[0]
            for index in range(len(data_1[1:])):
                warning = w_list_1[index]
                if warning in intersection:
                    continue
                text_A += data_1[index + 1]
            save_csv_result(f'{data_path}/{project}/golden_A/', f'goldenFeatures{x}.csv', text_A)

            # test C
            data_2 = read_data_from_file(f'{data_path}/{project}/golden/goldenFeatures{x + 1}.csv')
            text_C = data_2[0]
            for index in range(len(data_2[1:])):
                warning = w_list_2[index]
                if warning in intersection:
                    continue
                text_C += data_2[index + 1]
            save_csv_result(f'{data_path}/{project}/golden_C/', f'goldenFeatures{x + 1}.csv', text_C)


################################ Dataset allocation ########################################

CLASSIFIERS = [
    LogisticRegression(max_iter=3000),
    RandomForestClassifier(),
    DecisionTreeClassifier(),
    AdaBoostClassifier(),
    svm.SVC(kernel="linear", probability=True)]


def open_file(train_project, test_project, release=1):
    trainFile = rf"{train_project}/goldenFeatures{release}.csv"
    testFile = rf"{test_project}/goldenFeatures{release + 1}.csv"

    train = pd.read_csv(trainFile, index_col=None, skiprows=0, low_memory=False)
    test = pd.read_csv(testFile, index_col=None, skiprows=0, low_memory=False)

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


#######################################  Cross-release experimental scenario #####################################
def build_WP_model(train_file, test_file, clf, release):
    x_train, y_train, x_test, y_test = data_preparing(train_file, test_file, release)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    y_score = clf.predict_proba(x_test)
    # Evaluation
    precision = precision_score(y_test, y_pred, pos_label='close')
    recall = recall_score(y_test, y_pred, pos_label='close')
    f1 = f1_score(y_test, y_pred, pos_label='close')
    auc = roc_auc_score(y_test, y_score[:, 1])
    # print(f'{round(precision, 3)},{round(recall, 3)},{round(f1, 3)},{round(auc, 3)}', end='\t')
    return round(precision, 3), round(recall, 3), round(f1, 3), round(auc, 3)


strategies = [["golden", "golden"], ["golden_A", "golden"], ["golden", "golden_C"], ]


def runWP():
    result_p_dict = dict()
    result_r_dict = dict()
    result_f_dict = dict()
    result_a_dict = dict()
    for clf in CLASSIFIERS:
        print(f"===== Classifier {clf} =====")
        for s in strategies:
            print(f'{s}')
            result_p_list = []
            result_r_list = []
            result_f_list = []
            result_a_list = []
            for project in PROJECT:
                for release in [1, 2, 3, 4]:
                    train_file = rf"{data_path}/{project}/{s[0]}/"
                    test_file = rf"{data_path}/{project}/{s[1]}/"
                    p, r, f, a = build_WP_model(train_file, test_file, clf, release)
                    result_p_list.append(p)
                    result_r_list.append(r)
                    result_f_list.append(f)
                    result_a_list.append(a)
                    print(f'{project}-{release} => {release + 1}')
            result_p_dict[str(clf) + str(s)] = result_p_list
            result_r_dict[str(clf) + str(s)] = result_r_list
            result_f_dict[str(clf) + str(s)] = result_f_list
            result_a_dict[str(clf) + str(s)] = result_a_list

    df = pd.DataFrame(result_p_dict)
    df.to_csv(f'{root_path}/analysis/RQ2-supervised_p.csv')
    df = pd.DataFrame(result_r_dict)
    df.to_csv(f'{root_path}/analysis/RQ2-supervised_r.csv')
    df = pd.DataFrame(result_f_dict)
    df.to_csv(f'{root_path}/analysis/RQ2-supervised_f.csv')
    df = pd.DataFrame(result_a_dict)
    df.to_csv(f'{root_path}/analysis/RQ2-supervised_a.csv')


if __name__ == '__main__':
    runWP()
    # allocate_data_set()
    pass

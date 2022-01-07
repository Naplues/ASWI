# -*-coding:utf-8 -*-

import numpy as np
from helper import *
import warnings
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.metrics import *

__author__ = 'Naples'

warnings.filterwarnings("ignore", category=DeprecationWarning)
np.random.seed(0)

CLASSIFIERS = [RandomForestClassifier(),
               DecisionTreeClassifier(),
               svm.SVC(kernel="linear", probability=True)]

# F71, F3, F15 多值类特征
# F20, F21 名词类特征
golden_features = ["F116", "F115", "F117", "F120", "F123", "F110", "F105", "F68", "F101", "F104", "F65", "F22",
                   "F94", "F71-", "F72", "F25", "F3-", "F15-", "F126", "F41", "F77", 'category']


def open_file(train_project, test_project):
    trainFile = rf"{data_path}/{train_project}/features/totalFeatures4.csv"
    testFile = rf"{data_path}/{test_project}/features/totalFeatures5.csv"

    train = pd.read_csv(trainFile, index_col=None, skiprows=0, low_memory=False)
    test = pd.read_csv(testFile, index_col=None, skiprows=0, low_memory=False)

    header_list = [train.columns.tolist(), test.columns.tolist()]  # list of header for training set

    return test, train, header_list


def get_common_feature_names(header_list):
    train_golden = set([feature for feature in header_list[0] if feature.startswith(tuple(golden_features))])
    test_golden = set([feature for feature in header_list[1] if feature.startswith(tuple(golden_features))])

    common_features = train_golden.intersection(test_golden)
    return common_features


def trim(df, common_features):
    # Filter uncommon features from dataset
    return df.drop(set(df.columns.tolist()) - common_features, axis=1)


def data_preparing_CP(train_project, test_project):
    test, train, feature_list = open_file(train_project, test_project)
    common_features = get_common_feature_names(feature_list)

    # training and set
    train_data = trim(train, common_features)
    test_data = trim(test, common_features)

    # Extract training and test data set
    x_train = train_data.iloc[:, :-1]  # pandas.core.frame.DataFrame
    y_train = train_data.iloc[:, -1]  # pandas.core.series.Series

    x_test = test_data.iloc[:, :-1]
    y_test = test_data.iloc[:, -1]

    # Normalize the x for training and test data sets
    # Linear svm needs normalization to reduce the running cost on server
    min_max_scalar = preprocessing.MinMaxScaler()

    # Convert dataframe into numpy.array to normalize
    x_train = min_max_scalar.fit_transform(np.asarray(x_train))
    x_test = min_max_scalar.transform(np.asarray(x_test))

    return x_train, y_train, x_test, y_test


def data_preparing_WP(project):
    return data_preparing_CP(project, project)


#######################################  Within Project #################################################
def build_WP_model(project, clf):
    x_train, y_train, x_test, y_test = data_preparing_WP(project)
    clf.fit(x_train, y_train)
    y_label = clf.predict(x_test)
    y_score = clf.predict_proba(x_test)
    # Evaluation
    recall = recall_score(y_test, y_label, pos_label='close')
    auc = roc_auc_score(y_test, y_score[:, 1])
    print(round(recall, 3), round(auc, 3), end="\t")
    print(f"- {project}")


def runWP():
    for clf in CLASSIFIERS:
        print(f"-- {clf}")
        for project in PROJECT:
            build_WP_model(project, clf)


#######################################  Cross Project #################################################
def build_CP_model(train_project, test_project, clf):
    x_train, y_train, x_test, y_test = data_preparing_CP(train_project, test_project)
    clf.fit(x_train, y_train)
    y_label = clf.predict(x_test)
    y_score = clf.predict_proba(x_test)
    # Evaluation
    precision = precision_score(y_test, y_label, pos_label='close')
    recall = recall_score(y_test, y_label, pos_label='close')
    f1 = f1_score(y_test, y_label, pos_label='close')
    auc = roc_auc_score(y_test, y_score[:, 1])
    print(round(precision, 3), round(recall, 3), round(f1, 3), end="\t")
    print(f"- {train_project} ==> {test_project}")


def runCP():
    for clf in CLASSIFIERS:
        print(f"-- {clf}")
        for train_project in PROJECT:
            for test_project in PROJECT:
                if train_project == test_project:
                    continue
                build_CP_model(train_project, test_project, clf)
            # break
        break

    pass


if __name__ == '__main__':
    runWP()
    # runCP()

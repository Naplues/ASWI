# -*-coding:utf-8 -*-

import numpy as np
import glob
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


def open_file(path):
    trainingFiles = glob.glob(path + "/training_set/*.csv")
    testFile = rf"{path}/test_set/totalFeatures5.csv"

    training_list = []  # training set list
    header_list = []  # get the list of headers of dfs for training & testing

    for file in trainingFiles:
        df = pd.read_csv(file, index_col=None, skiprows=0, low_memory=False)
        training_list.append(df)  # list of training set
        header_list.append(df.columns.tolist())  # list of header for training set

    test = pd.read_csv(testFile, index_col=None, skiprows=0, low_memory=False)
    header_list.append(test.columns.tolist())

    return test, training_list, header_list


def get_common_feature_names(header_list):
    # "F116", "F115",
    golden_features = ["F117", "F120", "F123", "F110", "F105", "F68", "F101", "F104", "F65", "F22",
                       " F94", "F72", "F25", "F126", "F41", "F77", "category"]
    # golden_features = ["F116", "F115", "F117", "F120", "F123", "F110", "F105", "F68", "F101", "F104", "F65", "F22",
    #                    " F94", "F71", "F72", "F25", "F3-", "F15", "F126", "F41", "F77", "category"]
    # golden_fea = ["F116", "F115", "F117", "F120", "F123", "F110", "F105", "F68", "F101", "F104", "F65", "F22", "F20",
    #  "F21", "F94", "F71", "F72", "F25", "F3-", "F15", "F126", "F41", "F77"]
    golden_list = []
    for header in header_list:
        golden = []
        for feature in header:
            if feature.startswith(tuple(golden_features)):
                golden.append(feature)
        golden_list.append(golden)

    common_features = set(golden_list[0])
    for s in golden_list[1:]:
        common_features.intersection_update(s)

    return common_features


def trim(df, common_features):
    # Filter uncommon features from dataset
    return df.drop(set(df.columns.tolist()) - common_features, axis=1)


def data_preparing(path):
    test, training_list, feature_list = open_file(path)
    common_features = get_common_feature_names(feature_list)

    # training and set # ONLY USE THE VERSION 4 FOR TRAINING
    training_data = trim(training_list[-1], common_features)
    test_data = trim(test, common_features)

    # Extract training and test data set
    x_train = training_data.iloc[:, :-1]  # pandas.core.frame.DataFrame
    y_train = training_data.iloc[:, -1]  # pandas.core.series.Series

    x_test = test_data.iloc[:, :-1]
    y_test = test_data.iloc[:, -1]

    # Normalize the x for training and test data sets
    # Linear svm needs normalization to reduce the running cost on server
    min_max_scalar = preprocessing.MinMaxScaler()

    # Convert dataframe into numpy.array to normalize
    x_train = min_max_scalar.fit_transform(np.asarray(x_train))
    x_test = min_max_scalar.transform(np.asarray(x_test))

    return x_train, y_train, x_test, y_test


#######################################  Within Project #################################################
def build_WP_model(project, clf):
    x_train, y_train, x_test, y_test = data_preparing(rf'data/{project}')
    clf.fit(x_train, y_train)
    y_label = clf.predict(x_test)
    y_score = clf.predict_proba(x_test)
    # Evaluation
    recall = recall_score(y_test, y_label, pos_label='close')
    auc = roc_auc_score(y_test, y_score[:, 1])
    print(round(recall, 3), round(auc, 3), end="\t")
    print(f"- {project}")


def runWP():
    projects = ['derby', 'mvn', 'lucence', 'phoenix', 'cass', 'jmeter', 'tomcat', 'ant', 'commons']
    classifiers = [RandomForestClassifier(),
                   DecisionTreeClassifier(),
                   svm.SVC(kernel="linear", probability=True)]
    for clf in classifiers:
        print(f"-- {clf}")
        for project in projects:
            build_WP_model(project, clf)


#######################################  Cross Project #################################################
def build_CP_model(train_project, test_project, clf):
    _, _, x_train, y_train = data_preparing(rf'data/{train_project}')
    _, _, x_test, y_test = data_preparing(rf'data/{test_project}')
    clf.fit(x_train, y_train)
    y_label = clf.predict(x_test)
    y_score = clf.predict_proba(x_test)
    # Evaluation
    precision = precision_score(y_test, y_label, pos_label='close')
    recall = recall_score(y_test, y_label, pos_label='close')
    f1 = f1_score(y_test, y_label, pos_label='close')
    auc = roc_auc_score(y_test, y_score[:, 1])
    print(round(precision, 3), round(recall, 3), round(f1, 3), round(auc, 3), end="\t")
    print(f"- {train_project} ==> {test_project}")


def runCP():
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    projects = ['derby', 'mvn', 'lucence', 'phoenix', 'cass', 'jmeter', 'tomcat', 'ant', 'commons']
    classifiers = [RandomForestClassifier(),
                   DecisionTreeClassifier(),
                   svm.SVC(kernel="linear", probability=True)]
    for clf in classifiers:
        print(f"-- {clf}")
        for train_project in projects:
            for test_project in projects:
                if train_project == test_project:
                    continue
                build_CP_model(train_project, test_project, clf)
            # break
        break

    pass


if __name__ == '__main__':
    # runWP()
    runCP()

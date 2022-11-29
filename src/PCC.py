# -*- coding: utf-8 -*-
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.metrics import *
from scipy import stats

from src.RQ2 import open_file
from src.helper import *

np.seterr(divide='ignore', invalid='ignore')

CATEGORY = 'category'


################################## feature selection ##########################################
def binary_to_value(binary):
    label = []
    for x in binary:
        if x == 'close':
            label.append(1)
        else:
            label.append(0)
    return pd.Series(label)


def numerical_feature_correlation(project, feature_name, to_file=False):
    """
    RQ3
    Analyzing the correlation between numerical features and warning category
    Pearson correlation coefficient (PCC)
    """
    total_path = f'{data_path}/{project}/golden/goldenFeatures1.csv'
    df = pd.read_csv(total_path)
    # Remove category in list of feature name
    f_name = feature_name.copy()
    f_name.remove(CATEGORY)

    feature = df[f_name]

    # Transform warning category into numerical values and add it to dataframe
    feature.insert(len(feature.columns), 'category', binary_to_value(df[CATEGORY]))
    # Transposed dataframe
    feature = pd.DataFrame(feature.values.T, index=feature.columns, columns=feature.index)
    # Calculate PCC values of all features (including category)
    pcc_value = np.corrcoef(feature)
    result = pd.DataFrame(pcc_value, index=feature.index, columns=feature.index)
    result.to_csv(f'{root_path}/analysis/RQ3-pcc-{project}.csv') if to_file else None

    # Obtain most relative features
    pcc_category = result[CATEGORY]
    result_pcc = pcc_category

    pcc_category = pcc_category.sort_values(ascending=False)
    top_features = [index for index in pcc_category.index if pcc_category[index] > 0.5]

    return result_pcc, top_features


def run_numerical():
    # Create dataframe using dict
    pcc_dict = dict()
    for project in PROJECT:
        pcc_list, top_feature = numerical_feature_correlation(project, numerical_feature_names)
        pcc_dict[project] = pcc_list

    df = pd.DataFrame(pcc_dict)
    df = pd.DataFrame(df.values.T, index=df.columns, columns=list(numerical_feature_names))
    df.to_csv(f'{root_path}/analysis/RQ3-numerical-pcc.csv')
    pass


def nominal_feature_correlation(project, feature_name):
    """
    RQ3
    Pearson Chi-Squared Test
    Analyzing the correlation between nominal features and warning category
    """
    total_path = f'{data_path}/{project}/golden/goldenFeatures1.csv'
    df = pd.read_csv(total_path)
    # Remove category in list of feature name
    f_names = feature_name.copy()
    f_names.remove(CATEGORY)
    print(project, 'Nominal features: ', f_names)
    chi2_list, p_list, d_list = [], [], []
    for f_name in f_names:
        # Generate cross table
        observed_table = pd.crosstab(index=df[f_name], columns=df[CATEGORY])
        chi2_statistic, p_value, d, _ = stats.chi2_contingency(observed=observed_table)
        chi2_list.append(chi2_statistic)
        p_list.append(p_value)
        d_list.append(d)

        observed_table['ratio'] = observed_table['close'] / observed_table['open']
        # print(observed_table, '\n\n')

    return chi2_list, p_list, d_list


def run_nominal():
    chi2_dict, p_dict, d_dict = dict(), dict(), dict()
    for project in PROJECT:
        chi2_list, p_list, d_list = nominal_feature_correlation(project, nominal_feature_names)
        chi2_dict[project] = chi2_list
        p_dict[project] = p_list
        d_dict[project] = d_list

    feature_name = nominal_feature_names.copy()
    feature_name.remove(CATEGORY)

    df = pd.DataFrame(chi2_dict)
    df = pd.DataFrame(df.values.T, index=df.columns, columns=feature_name)
    df.to_csv(f'{root_path}/analysis/RQ3-nominal_chi2.csv')

    df = pd.DataFrame(p_dict)
    df = pd.DataFrame(df.values.T, index=df.columns, columns=feature_name)
    df.to_csv(f'{root_path}/analysis/RQ3-nominal_p.csv')

    df = pd.DataFrame(d_dict)
    df = pd.DataFrame(df.values.T, index=df.columns, columns=feature_name)
    df.to_csv(f'{root_path}/analysis/RQ3-nominal_df.csv')


################################## Model test ##########################################
CLASSIFIERS = [
    LogisticRegression(max_iter=3000),
    RandomForestClassifier(),
    DecisionTreeClassifier(),
    AdaBoostClassifier(),
    svm.SVC(kernel="linear", probability=True)]


def data_preparing(train_project, test_project, release):
    train_data, test_data, feature_list = open_file(train_project, test_project, release)

    # Extract training and test data set
    x_train_numerical = train_data.iloc[:, :-6]  # pandas.core.frame.DataFrame
    x_train_nominal = train_data.iloc[:, -6:-1]  # pandas.core.frame.DataFrame
    y_train = train_data.iloc[:, -1]  # pandas.core.series.Series

    x_test_numerical = test_data.iloc[:, :-6]
    x_test_nominal = test_data.iloc[:, -6:-1]
    y_test = test_data.iloc[:, -1]

    # select informative features
    # -6:DWT, -5:DM, -4:DF, -3:DLP
    x_train_numerical = np.asanyarray(x_train_numerical)[:, -6:-2]
    x_test_numerical = np.asanyarray(x_test_numerical)[:, -6:-2]
    # print(x_train_numerical)
    # 0:SD, 3:WP, 4:WT
    x_train_nominal = x_train_nominal.iloc[:, [0, 3, 4]]
    x_test_nominal = x_test_nominal.iloc[:, [0, 3, 4]]

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
    return round(precision, 3), round(recall, 3), round(f1, 3), round(auc, 3)


strategies = [["golden_A", "golden"], ["golden", "golden_C"], ]


def runWP(to_file=False):
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
    df.to_csv(f'{root_path}/analysis/RQ3-inf_p.csv') if to_file else None
    df = pd.DataFrame(result_r_dict)
    df.to_csv(f'{root_path}/analysis/RQ3-inf_r.csv') if to_file else None
    df = pd.DataFrame(result_f_dict)
    df.to_csv(f'{root_path}/analysis/RQ3-inf_f.csv') if to_file else None
    df = pd.DataFrame(result_a_dict)
    df.to_csv(f'{root_path}/analysis/RQ3-inf_a.csv') if to_file else None


def main():
    # run_numerical()
    # run_nominal()
    # runWP(to_file=True)
    pass


if __name__ == '__main__':
    main()
    pass

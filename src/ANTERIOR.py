# -*- coding: utf-8 -*-
from statistics import mean

import numpy as np

from helper import *
import pandas as pd
from sklearn.metrics import *


# Do we really need a training model
def get_overlapping_dict(project, release):
    # Warning data in previous and next release, respectively.
    w_set_1, w_list_1 = get_warning_set(f'{data_path}/{project}/warnings/warningInfo{release - 1}.csv',
                                        f'{data_path}/{project}/golden/goldenFeatures{release - 1}.csv')

    w_set_2, w_list_2 = get_warning_set(f'{data_path}/{project}/warnings/warningInfo{release}.csv',
                                        f'{data_path}/{project}/golden/goldenFeatures{release}.csv')

    # Overlapping warnings
    intersection = list(w_set_1.intersection(w_set_2))
    # Linking the indices of same warnings in two consecutive releases.
    index_map = list()
    for instance in intersection:
        # [index in previous release, index in next release]
        cur_index = [w_list_1.index(instance), w_list_2.index(instance)]
        index_map.append(cur_index)

    # 两版本的警报特征
    data_1 = pd.read_csv(f'{data_path}/{project}/golden/goldenFeatures{release - 1}.csv')

    overlapping_dict = {}
    for index in index_map:
        previous_release_index, next_release_index = index[0], index[1]
        overlapping_dict[next_release_index] = data_1[CATEGORY][previous_release_index]
    return overlapping_dict


def first_stage(project, release):
    """
    Building simple two-stage approach.
    """
    df = pd.read_csv(f'{data_path}/{project}/golden/goldenFeatures{release}.csv')
    # Ground truth
    y_test_all = df[CATEGORY]
    # Default label for each warning is 'open'
    y_test, y_pred = [], []

    # ========== First stage: Overlapping warnings
    overlap_dict = get_overlapping_dict(project, release)
    overlapping_indices = overlap_dict.keys()

    print(f'{project}, {len(overlapping_indices)}')
    for index in overlapping_indices:
        y_test.append(y_test_all[index])
        y_pred.append(overlap_dict[index])

    # Performance indicators
    precision = precision_score(y_test, y_pred, pos_label='close')
    recall = recall_score(y_test, y_pred, pos_label='close')
    f1 = f1_score(y_test, y_pred, pos_label='close')
    print(f'{round(precision, 3)},{round(recall, 3)},{round(f1, 3)},{project}')
    return precision, recall, f1


def second_stage(project, release, feature_name):
    """
    Building simple two-stage approach.
    """
    df = pd.read_csv(f'{data_path}/{project}/golden_C/goldenFeatures{release}.csv')
    # Ground truth
    y_test = df[CATEGORY]
    # Default label for each warning is 'open'
    y_pred = ['open'] * len(df)

    # ========== Second stage: Calculate the classification score
    feature = np.array([0] * len(df))
    for name in feature_name:
        feature = feature + np.array(df[name])

    for index in range(len(y_pred)):
        if feature[index] > 0:
            y_pred[index] = 'close'

    # Performance indicators
    precision = precision_score(y_test, y_pred, pos_label='close')
    recall = recall_score(y_test, y_pred, pos_label='close')
    f1 = f1_score(y_test, y_pred, pos_label='close')
    print(f'{round(precision, 3)},{round(recall, 3)},{round(f1, 3)},{project},{feature_name}')
    return precision, recall, f1


def iota(project, release, feature_name):
    """
    Building simple two-stage approach.
    """
    df = pd.read_csv(f'{data_path}/{project}/golden/goldenFeatures{release}.csv')
    # Ground truth
    y_test = df[CATEGORY]
    # Default label for each warning is 'open'
    y_pred = ['open'] * len(df)

    # ========== First stage: Overlapping warnings
    overlap_dict = get_overlapping_dict(project, release)

    # ========== Second stage: Calculate the classification score
    feature = np.array([0] * len(df))
    for name in feature_name:
        feature = feature + np.array(df[name])

    for index in range(len(y_pred)):
        if index in overlap_dict:
            y_pred[index] = overlap_dict[index]
        elif feature[index] > 0:
            y_pred[index] = 'close'

    # Performance indicators
    precision = precision_score(y_test, y_pred, pos_label='close')
    recall = recall_score(y_test, y_pred, pos_label='close')
    f1 = f1_score(y_test, y_pred, pos_label='close')
    print(f'{round(precision, 3)},{round(recall, 3)},{round(f1, 3)},{project},{feature_name}')
    return precision, recall, f1


def run_first_stage(to_file=False):
    test_releases = [2, 3, 4, 5]
    # 要注意overlapping檢測程序
    performance_dict = dict()
    p_list, r_list, f1_list = [], [], []
    for project in PROJECT:
        for release in test_releases:
            p, r, f1 = first_stage(project=project, release=release)
            # p, r, f1 = second_stage(project=project, release=release, feature_name=feature)
            p_list.append(p)
            r_list.append(r)
            f1_list.append(f1)
    performance_dict['P'] = p_list
    performance_dict['R'] = r_list
    performance_dict['F1'] = f1_list

    df = pd.DataFrame(performance_dict)
    df.to_csv(f'{root_path}/analysis/IOTA-First.csv') if to_file else None
    pass


def run_second_stage(to_file=False):
    test_releases = [2, 3, 4, 5]
    features = [['DM'], ['DF'], ['DM', 'DF', ]]
    p_dict = dict()
    for feature in features:
        print(feature)
        p_list, r_list, f1_list = [], [], []
        for project in PROJECT:
            for release in test_releases:
                p, r, f1 = second_stage(project=project, release=release, feature_name=feature)
                p_list.append(p)
                r_list.append(r)
                f1_list.append(f1)
        p_dict["+".join(feature) + '-p'] = p_list
        p_dict["+".join(feature) + '-r'] = r_list
        p_dict["+".join(feature) + '-f'] = f1_list
    df = pd.DataFrame(p_dict)
    df.to_csv(f'{root_path}/analysis/IOTA-Second.csv') if to_file else None
    pass


def run_prediction(to_file=False):
    feature = ['DM', 'DF', ]
    test_releases = [2, 3, 4, 5]
    performance_dict, performance_avg_dict = dict(), dict()
    p_list, r_list, f_list = [], [], []
    p_avg_list, r_avg_list, f_avg_list = [], [], []
    for project in PROJECT:
        p_avg, r_avg, f_avg = [], [], []
        for release in test_releases:
            p, r, f1 = iota(project=project, release=release, feature_name=feature)
            p_list.append(p)
            r_list.append(r)
            f_list.append(f1)
            # average
            p_avg.append(p)
            r_avg.append(r)
            f_avg.append(f1)
        p_avg_list.append(mean(p_avg))
        r_avg_list.append(mean(r_avg))
        f_avg_list.append(mean(f_avg))
    performance_dict['P'] = p_list
    performance_dict['R'] = r_list
    performance_dict['F1'] = f_list

    performance_avg_dict['P'] = p_avg_list
    performance_avg_dict['R'] = r_avg_list
    performance_avg_dict['F1'] = f_avg_list

    df = pd.DataFrame(performance_dict)
    df.to_csv(f'{root_path}/analysis/IOTA.csv') if to_file else None
    df = pd.DataFrame(performance_avg_dict)
    df.to_csv(f'{root_path}/analysis/IOTA-avg.csv') if to_file else None
    pass


def main():
    # run_first_stage(to_file=True)
    # run_second_stage(to_file=True)
    # run_prediction(to_file=True)
    pass


if __name__ == '__main__':
    main()
    pass

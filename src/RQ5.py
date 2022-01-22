# -*- coding: utf-8 -*-

import numpy as np

from helper import *
import pandas as pd
from sklearn.metrics import *


# Do we really need a training model
def get_steal_dict(project, release):
    # Warning data in previous and next release, respectively.
    w_set_1, w_list_1 = get_warning_set(f'{data_path}/{project}/warnings/warningInfo{release}.csv',
                                        f'{data_path}/{project}/golden/goldenFeatures{release}.csv')

    w_set_2, w_list_2 = get_warning_set(f'{data_path}/{project}/warnings/warningInfo{release + 1}.csv',
                                        f'{data_path}/{project}/golden/goldenFeatures{release + 1}.csv')

    # Overlapping warnings
    intersection = list(w_set_1.intersection(w_set_2))
    # Linking the indices of same warnings in two consecutive releases.
    index_map = list()
    for instance in intersection:
        cur_index = [w_list_1.index(instance), w_list_2.index(instance)]
        index_map.append(cur_index)

    # 两版本的警报特征
    data_1 = pd.read_csv(f'{data_path}/{project}/golden/goldenFeatures{release}.csv')

    steal_dict = {}
    for index in index_map:
        steal_dict[index[1]] = data_1[CATEGORY][index[0]]
    return steal_dict


def unsupervised(project, feature_name, release=4, steal=None):
    """
    Building simple unsupervised models.
    :return:
    """
    if steal is None:
        steal = {}
    total_path = f'{data_path}/{project}/golden/goldenFeatures{release}.csv'
    df = pd.read_csv(total_path)
    # Calculate the ranking score
    feature = [0] * len(df)
    for name in feature_name:
        feature = feature + np.array(df[name])

    # Sort all instances in descending order
    sorted_index = np.argsort(feature)[::-1]

    # 记录预测值
    y_test = df[CATEGORY]
    y_score = feature[sorted_index]
    y_pred = ['open'] * len(feature)
    for index in range(len(y_pred)):
        if index in steal:
            y_pred[index] = steal[index]
        elif feature[index] > 0:
            y_pred[index] = 'close'

    # 计算分类指标
    precision = precision_score(y_test, y_pred, pos_label='close')
    recall = recall_score(y_test, y_pred, pos_label='close')
    f1 = f1_score(y_test, y_pred, pos_label='close')
    auc = roc_auc_score(y_test, y_score)
    print(f'{round(precision, 3)},{round(recall, 3)},{round(f1, 3)},{round(auc, 3)},{project},{feature_name}')
    return recall


def run_unsupervised(to_file=False):
    releases = [2, 3, 4, 5]
    features = [['DM'], ['DF'], ['DM', 'DF', ]]
    f1_dict = dict()
    for feature in features:
        print(feature)
        f1_list = []
        for project in PROJECT:
            for release in releases:
                f1 = unsupervised(project, feature, release, steal=get_steal_dict(project, release - 1))
                f1_list.append(f1)
            f1_dict["-".join(feature)] = f1_list
    df = pd.DataFrame(f1_dict)
    filepath = f'{root_path}/analysis/RQ5-unsupervised-II-f.csv'
    df.to_csv(filepath) if to_file else None
    pass


def main():
    run_unsupervised(to_file=True)
    pass


if __name__ == '__main__':
    main()
    pass

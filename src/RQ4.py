# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.metrics import *
from src.helper import *


def unsupervised_explore_threshold(project, feature_name, release=4, threshold=0.5):
    """
    Explore the proper classification threshold in simple unsupervised models.
    """
    total_path = f'{data_path}/{project}/golden/goldenFeatures{release}.csv'
    df = pd.read_csv(total_path)
    # Calculate the ranking score
    num_instance = len(df)
    feature = [0] * num_instance
    for name in feature_name:
        feature = feature + np.array(df[name])

    # Sort all instances in descending order
    sorted_index = np.argsort(feature)[::-1]
    selected_index = sorted_index[:int(threshold * num_instance)]
    # 记录预测值
    y_test = df[CATEGORY]
    y_score = feature[sorted_index]
    y_pred = ['open'] * len(feature)
    for index in selected_index:
        y_pred[index] = 'close'

    # 计算分类指标
    precision = precision_score(y_test, y_pred, pos_label='close')
    recall = recall_score(y_test, y_pred, pos_label='close')
    f1 = f1_score(y_test, y_pred, pos_label='close')
    auc = roc_auc_score(y_test, y_score)
    print(f'{round(precision, 3)},{round(recall, 3)},{round(f1, 3)},{round(auc, 3)},{project},{feature_name}')
    return round(f1, 3)


def unsupervised(project, feature_name, release=4, strategy='II'):
    """
    Building simple unsupervised models.
    :return:
    """
    total_path = f'{data_path}/{project}/golden/goldenFeatures{release}.csv'
    if strategy == 'III':
        total_path = f'{data_path}/{project}/golden_C/goldenFeatures{release}.csv'
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
    for index in sorted_index:
        if feature[index] > 0:
            y_pred[index] = 'close'

    # 计算分类指标
    precision = precision_score(y_test, y_pred, pos_label='close')
    recall = recall_score(y_test, y_pred, pos_label='close')
    f1 = f1_score(y_test, y_pred, pos_label='close')
    auc = roc_auc_score(y_test, y_score)
    print(f'{round(precision, 3)},{round(recall, 3)},{round(f1, 3)},{round(auc, 3)},{project},{feature_name}')
    return recall


def threshold_exploration(to_file=False):
    feature = [['DLP'], ['DM'], ['DF'], ['DWT'], ['DLP', 'DM', 'DF', 'DWT']]
    threshold = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, ]

    for fea in feature:
        print(fea)
        f1_dict = dict()
        for th in threshold:
            print(f'threshold: {th}')
            f1_list = []
            for project in PROJECT:
                for release in [2, 3]:
                    f1 = unsupervised_explore_threshold(project, fea, release, threshold=th)
                    f1_list.append(f1)
            f1_dict[str(th)] = f1_list
        df = pd.DataFrame(f1_dict)
        df.to_csv(f'{root_path}/analysis/RQ4-th-{"-".join(fea)}.csv') if to_file else None
    pass


def run_unsupervised(strategy='II', to_file=False):
    releases = [2, 3, 4, 5]
    features = [['DM'], ['DF'], ['DM', 'DF', ]]
    f1_dict = dict()
    for feature in features:
        print(feature)
        f1_list = []
        for project in PROJECT:
            for release in releases:
                f1 = unsupervised(project, feature, release, strategy=strategy)
                f1_list.append(f1)
            f1_dict["-".join(feature)] = f1_list
    df = pd.DataFrame(f1_dict)
    filepath = f'{root_path}/analysis/RQ4-unsupervised-II-f.csv'
    if strategy == 'III':
        filepath = f'{root_path}/analysis/RQ4-unsupervised-III-f.csv'
    df.to_csv(filepath) if to_file else None
    pass


def main():
    # threshold_exploration(to_file=True)
    # run_unsupervised(strategy='II', to_file=True)
    # run_unsupervised(strategy='III', to_file=True)
    pass


if __name__ == '__main__':
    main()
    pass

# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.metrics import *
from src.helper import *


def binary_to_value(binary):
    label = []
    for x in binary:
        if x == 'close':
            label.append(1)
        else:
            label.append(0)
    return pd.Series(label)


def unsupervised(project, feature_name, release=4):
    """
    构建简单的无监督模型进行识别
    :return:
    """
    total_path = f'{data_path}/{project}/golden/goldenFeatures{release}.csv'
    df = pd.read_csv(total_path)
    # 按照某组特征排序
    feature = [0] * len(df)
    for name in feature_name:
        feature = feature + np.array(df[name])

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
    print(f'{round(precision, 3)},{round(recall, 3)},{round(f1, 3)},{round(auc, 3)},{project}')


def run():
    releases = [2, 3, 4, 5]
    feature = ['DLP', 'DM', 'DF', 'DWT']
    print(feature)
    for release in releases:
        for project in PROJECT:
            unsupervised(project, feature, release)
            # break
        print()
    pass


def run2():
    releases = [2, 3, 4, 5]
    features = [['DLP'], ['DM'], ['DF'], ['DWT']]
    for feature in features:
        print(feature)
        for release in releases:
            for project in PROJECT:
                unsupervised(project, feature, release)
                # break
            print()
    pass


if __name__ == '__main__':
    # run()
    run2()
    pass

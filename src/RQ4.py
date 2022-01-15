# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.metrics import *
from src.helper import *

# F15, F71, F3 多变量型特征
# F20, F21 字符型特征, 需要转换为数值进行分析
CATEGORY = 'category'
golden_feature_names = ['F25', 'F72',  # 'F71',
                        'F104', 'F105', 'F101', 'F65', 'F68',
                        'F126', 'F41',
                        # 'F3', 'F15',
                        'F22', 'F94',  # 'F20', 'F21',
                        'F77',
                        'F110', 'F116',
                        'F115', 'F117', 'F120', 'F123']

golden_feature_with_label_names = ['F25', 'F72', 'F71',
                                   'F104', 'F105', 'F101', 'F65', 'F68',
                                   'F126', 'F41',
                                   'F3', 'F15',
                                   'F20', 'F21', 'F22', 'F94',
                                   'F77',
                                   'F110', 'F116',
                                   'F115', 'F117', 'F120', 'F123', 'category']


def binary_to_value(binary):
    label = []
    for x in binary:
        if x == 'close':
            label.append(1)
        else:
            label.append(0)
    return pd.Series(label)


def unsupervised(project, feature_name):
    """
    构建简单的无监督模型进行识别
    Pearson correlation coefficient 皮尔逊相关系数
    :param project:
    :param feature_name:
    :return:
    """
    total_path = f'{data_path}/{project}/features/totalFeatures4.csv'
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
    for project in PROJECT:
        # ['F115', 'F116', 'F117', 'F110']
        unsupervised(project, ['F116', 'F115', 'F117', 'F110'])
        # break
    pass


if __name__ == '__main__':
    run()
    pass

# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

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

#
all_feature_names = ['F23', 'F25', 'F34', 'F35', 'F36', 'F37', 'F38', 'F39', 'F40', 'F41', 'F42', 'F43', 'F44', 'F45',
                     'F46', 'F61', 'F62', 'F64', 'F65', 'F66', 'F67', 'F68', 'F69', 'F70', 'F72', 'F73', 'F74', 'F77',
                     'F79', 'F83', 'F84', 'F88', 'F94', 'F95', 'F101', 'F102', 'F103', 'F104', 'F105', 'F107', 'F108',
                     'F109', 'F110', 'F111', 'F112', 'F113', 'F114', 'F115', 'F116', ]

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


def feature_correlation(project, feature_name, scale='golden'):
    """
    分析指定特征与标签的关系 F116
    Pearson correlation coefficient 皮尔逊相关系数
    :param project:
    :param feature_name:
    :param scale:
    :return:
    """
    total_path = f'{data_path}/{project}/features/totalFeatures4.csv'
    df = pd.read_csv(total_path)
    feature = df[feature_name]
    # 将标签表示为数字并且加入成为新的特征列
    feature.insert(len(feature.columns), 'category', binary_to_value(df[CATEGORY]))
    # 转置后的矩阵
    feature = pd.DataFrame(feature.values.T, index=feature.columns, columns=feature.index)
    # 计算所有特征的相关系数
    pcc = np.corrcoef(feature)
    result = pd.DataFrame(pcc, index=feature.index, columns=feature.index)
    result.to_csv(f'{root_path}/analysis/pcc-{project}-{scale}.csv')

    # 取出最相关的特征
    pcc_category = result[CATEGORY]
    pcc_category = pcc_category.sort_values(ascending=False)
    top_features = [index for index in pcc_category.index if pcc_category[index] > 0.5]

    print(top_features)


def run():
    for project in PROJECT:
        feature_correlation(project, golden_feature_names)
        # feature_correlation(project, all_feature_names, scale='all')
        # break
    pass


if __name__ == '__main__':
    run()
    pass

# 构建简单的无监督模型进行识别

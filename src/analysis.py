# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

from src.helper import *

CATEGORY = 'category'
golden_feature_names = ['F25', 'F72',  # 'F71',
                        'F104', 'F105', 'F101', 'F65', 'F68',
                        'F126', 'F41',
                        # 'F3', 'F15',
                        'F22', 'F94',  # 'F20', 'F21',
                        'F77',
                        'F110', 'F116',
                        'F115', 'F117', 'F120', 'F123']


def binary_to_value(binary):
    label = []
    for x in binary:
        if x == 'close':
            label.append(1)
        else:
            label.append(0)
    return pd.Series(label)


def feature_correlation(project, feature_name):
    """
    RQ3
    分析指定特征与标签的关系 F116
    Pearson correlation coefficient 皮尔逊相关系数
    :param project:
    :param feature_name:
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
    pcc_value = np.corrcoef(feature)
    result = pd.DataFrame(pcc_value, index=feature.index, columns=feature.index)
    # result.to_csv(f'{root_path}/analysis/pcc-{project}-{scale}.csv')

    # 取出最相关的特征
    pcc_category = result[CATEGORY]
    result_pcc = pcc_category

    pcc_category = pcc_category.sort_values(ascending=False)
    top_features = [index for index in pcc_category.index if pcc_category[index] > 0.5]

    return result_pcc, top_features


def run():
    # 使用字典创建,字典的健是列名
    pcc_dict = dict()
    for project in PROJECT:
        pcc_list, top_feature = feature_correlation(project, golden_feature_names)
        pcc_dict[project] = pcc_list

    df = pd.DataFrame(pcc_dict)
    # 转置后的数据帧
    df = pd.DataFrame(df.values.T, index=df.columns, columns=list(feature_map.values()))
    df.to_csv(f'{root_path}/analysis/pcc-all.csv')
    pass


if __name__ == '__main__':
    run()
    print(list(feature_map.values()))
    pass

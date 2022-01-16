# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from scipy import stats

from src.helper import *

np.seterr(divide='ignore', invalid='ignore')

CATEGORY = 'category'


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
    :return:
    """
    total_path = f'{data_path}/{project}/features/totalFeatures1.csv'
    df = pd.read_csv(total_path)
    # Remove category in list of feature name
    f_name = feature_name.copy()
    f_name.remove(CATEGORY)

    feature = df[f_name]

    # 将标签表示为数字并且加入成为新的特征列
    feature.insert(len(feature.columns), 'category', binary_to_value(df[CATEGORY]))
    # 转置后的矩阵
    feature = pd.DataFrame(feature.values.T, index=feature.columns, columns=feature.index)
    # 计算所有特征的相关系数
    pcc_value = np.corrcoef(feature)
    result = pd.DataFrame(pcc_value, index=feature.index, columns=feature.index)
    result.to_csv(f'{root_path}/analysis/pcc-{project}.csv') if to_file else None

    # 取出最相关的特征
    pcc_category = result[CATEGORY]
    result_pcc = pcc_category

    pcc_category = pcc_category.sort_values(ascending=False)
    top_features = [index for index in pcc_category.index if pcc_category[index] > 0.5]

    return result_pcc, top_features


def nominal_feature_correlation(project, feature_name, to_file=False):
    """
    RQ3
    Pearson correlation coefficient 卡方检验
    Analyzing the correlation between numerical features and warning category
    Chi-Squared Test
    :return:
    """
    total_path = f'{data_path}/{project}/features/totalFeatures1.csv'
    df = pd.read_csv(total_path)
    # Remove category in list of feature name
    f_names = feature_name.copy()
    f_names.remove(CATEGORY)
    print(f_names)
    for f_name in f_names:
        # Generate cross table
        observed_table = pd.crosstab(index=df[f_name], columns=df[CATEGORY])
        chi2_statistic, p_value, d, _ = stats.chi2_contingency(observed=observed_table)
        print(nominal_feature_map[f_name], chi2_statistic, p_value, d)


def run():
    # 使用字典创建,字典的健是列名
    pcc_dict = dict()
    for project in PROJECT:
        pcc_list, top_feature = numerical_feature_correlation(project, golden_feature_names)
        pcc_dict[project] = pcc_list

    df = pd.DataFrame(pcc_dict)
    df = pd.DataFrame(df.values.T, index=df.columns, columns=list(feature_map.values()))
    df.to_csv(f'{root_path}/analysis/pcc-all.csv')
    pass


if __name__ == '__main__':
    # run()
    for project in PROJECT:
        nominal_feature_correlation(project, golden_nominal_feature_names)
        break
    pass

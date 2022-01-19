# -*- coding: utf-8 -*-
import os
import pandas as pd

root_path = rf'C:/Users/gzq-712/Desktop/Git/SAWI'
data_path = f'{root_path}/data'

# Total 9 projects
PROJECT = ['ant', 'cass', 'commons', 'derby', 'jmeter', 'lucence', 'mvn', 'phoenix', 'tomcat']

CATEGORY = 'category'
# ============================= 数值型特征
# 原始名称
golden_numerical_feature_names = ['F25', 'F72', 'F104', 'F105', 'F101', 'F65', 'F68', 'F126', 'F41', 'F22', 'F94',
                                  'F77', 'F110', 'F116', 'F115', 'F117', 'F120', 'F123', 'category']
# 修改后名称
numerical_feature_names = ['ND', 'FCR', 'RCC', 'DWM', 'DWF', 'NM', 'NC', 'LAR', 'LAM', 'WR', 'NWP', 'NR', 'DWT', 'DM',
                           'DF', 'DLP', 'DDL', 'ALT', 'category']
# 特征映射
numerical_feature_map = {'F25': 'ND', 'F72': 'FCR', 'F104': 'RCC', 'F105': 'DWM', 'F101': 'DWF', 'F65': 'NM',
                         'F68': 'NC', 'F126': 'LAR', 'F41': 'LAM', 'F22': 'WR', 'F94': 'NWP', 'F77': 'NR',
                         'F110': 'DWT', 'F116': 'DM', 'F115': 'DF', 'F117': 'DLP', 'F120': 'DDL', 'F123': 'ALT',
                         'category': 'category'}

# ============================= 标称型特征 F71, F3, F15, F20, F21
# 原始名称
golden_nominal_feature_names = ['F71', 'F3', 'F15', 'F20', 'F21', 'category']
# 修改后名称
nominal_feature_names = ['SD', 'PS', 'MV', 'WP', 'WT', 'category']
# 特征映射
nominal_feature_map = {'F71': 'SD', 'F3': 'PS', 'F15': 'MV', 'F20': 'WP', 'F21': 'WT', 'category': 'category'}

# new name for each feature
all_feature_names = ['ND', 'FCR', 'RCC', 'DWM', 'DWF', 'NM', 'NC', 'LAR', 'LAM', 'WR', 'NWP', 'NR', 'DWT', 'DM', 'DF',
                     'DLP',
                     'DDL', 'ALT', 'SD', 'PS', 'MV', 'WP', 'WT', 'category']


def make_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def read_data_from_file(path):
    with open(path, 'r', encoding='utf-8', errors="ignore") as fr:
        lines = fr.readlines()
    return lines


def save_csv_result(file_path, file_name, data):
    make_path(file_path)
    with open(f'{file_path}{file_name}', 'w', encoding='utf-8') as file:
        file.write(data)
    print(f'Result has been saved to {file_path}{file_name} successfully!')


def get_warning_set(warning_path, label_path):
    """
    Get warning set of specific release.
    """
    warning_set = set()
    warning_list = list()
    labels = list(pd.read_csv(label_path)[CATEGORY])
    data = read_data_from_file(warning_path)
    for index in range(len(data)):
        line = data[index]
        # Category, Pattern, File, Method, Lines, Code
        ss = line.split(',', maxsplit=5)
        # Remove Lines from warning information.
        identifier = line.replace(ss[4], '') + labels[index]
        warning_set.add(identifier)
        warning_list.append(identifier)
    return warning_set, warning_list

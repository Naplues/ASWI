# -*- coding: utf-8 -*-
import os

root_path = rf'C:/Users/gzq-712/Desktop/Git/SAWI'
data_path = f'{root_path}/data'

# Total 9 projects
PROJECT = ['ant', 'cass', 'commons', 'derby', 'jmeter', 'lucence', 'mvn', 'phoenix', 'tomcat']

# F71, F3, F15, F20, F21
golden_feature_names = ['F25', 'F72', 'F104', 'F105', 'F101', 'F65', 'F68', 'F126', 'F41', 'F22', 'F94', 'F77', 'F110',
                        'F116', 'F115', 'F117', 'F120', 'F123', 'category']

feature_map = {'F25': 'ND', 'F72': 'FCR', 'F104': 'RCC', 'F105': 'DWM', 'F101': 'DWF', 'F65': 'NM', 'F68': 'NC',
               'F126': 'LAR', 'F41': 'LAM', 'F22': 'WR', 'F94': 'NWP', 'F77': 'NR', 'F110': 'DWT', 'F116': 'DM',
               'F115': 'DF', 'F117': 'DLP', 'F120': 'DDL', 'F123': 'ALT', 'category': 'category'}

golden_nominal_feature_names = ['F20', 'F21', 'category']
nominal_feature_map = {'F71': 'SD', 'F3': 'PS', 'F15': 'MV', 'F20': 'WP', 'F21': 'WT', 'category': 'category'}


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

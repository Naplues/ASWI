# -*- coding: utf-8 -*-
from helper import *
import pandas as pd


def get_warning_set(path):
    """
    Get warning set of specific release.
    """
    warning_set = set()
    warning_list = list()
    data = read_data_from_file(path)
    for line in data:
        # Category, Pattern, File, Method, Lines, Code
        ss = line.split(',', maxsplit=5)
        # Remove Lines from warning information.
        warning_set.add(line.replace(ss[4], ''))
        warning_list.append(line.replace(ss[4], ''))
    return warning_set, warning_list


def measure_consecutive_data():
    print(f'Project, v1 & v2 => O12, Ratio, v2 & v3 => O23, Ratio, v3 & v4 => O34, Ratio, v4 & v5 => O45, Ratio')
    for project in PROJECT:
        print(f'{project}', end=', ')
        for x in range(1, 5):  # Iterator 1, 2, 3, 4
            w_set_1, _ = get_warning_set(f'{data_path}/{project}/warnings/warningInfo{x}.csv')
            w_set_2, _ = get_warning_set(f'{data_path}/{project}/warnings/warningInfo{x + 1}.csv')
            # Obtain the intersection of previous and next release.
            intersection = w_set_1.intersection(w_set_2)
            ratio = round(len(intersection) / len(w_set_2), 4)
            print(f'{len(w_set_1)} & {len(w_set_2)} => {len(intersection)}, {ratio}', end=', ')
        print()
    pass


def measure_consecutive_feature():
    print(f'Project, v1 & v2 => O12, Ratio, v2 & v3 => O23, Ratio, v3 & v4 => O34, Ratio, v4 & v5 => O45, Ratio')
    for feature in golden_numerical_feature_names:
        print(f'========== {numerical_feature_map[feature]} ==========')
        for project in PROJECT:
            print(f'{project}', end='\t')
            for x in range(1, 5):
                # 两版本的警报数据
                w_set_1, w_list_1 = get_warning_set(f'{data_path}/{project}/warnings/warningInfo{x}.csv')
                w_set_2, w_list_2 = get_warning_set(f'{data_path}/{project}/warnings/warningInfo{x + 1}.csv')

                # 两版本重合的警报数据
                intersection = list(w_set_1.intersection(w_set_2))
                # 关联两个版本相同的警报索引
                index_map = list()
                for instance in intersection:
                    cur_index = [w_list_1.index(instance), w_list_2.index(instance)]
                    if len(cur_index) != 2:
                        continue
                    index_map.append(cur_index)

                # 两版本的警报特征
                data_1 = pd.read_csv(f'{data_path}/{project}/features/totalFeatures{x}.csv')
                data_2 = pd.read_csv(f'{data_path}/{project}/features/totalFeatures{x + 1}.csv')
                # print(len(w_set_1), len(w_list_1), len(data_1), len(w_set_2), len(w_list_2), len(data_2))
                overlap = 0
                for index in index_map:
                    if index[0] >= len(data_1) or index[1] >= len(data_2):
                        continue
                    f1, f2 = data_1[feature][index[0]], data_2[feature][index[1]]
                    if f1 == f2:
                        overlap += 1
                ratio = round(overlap / len(intersection), 4)
                print(f'{overlap} / {len(intersection)} =>, {round(ratio * 100, 2)}%', end=', ')
            print()
            # break
    pass


def main():
    # measure_consecutive_data()
    # measure_consecutive_feature()
    pass


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
from helper import *
import pandas as pd


def measure_overlapping_in_consecutive_releases():
    """
    Measure the overlapping instances between consecutive releases.
    """
    text = f'Project, R1 & R2 => O12, Ratio, R2 & R3 => O23, Ratio, R3 & R4 => O34, Ratio, R4 & R5 => O45, Ratio\n'
    for project in PROJECT:
        text += f'{project},'
        for x in range(1, 5):  # Iterator 1, 2, 3, 4
            # Warning data in previous and next release, respectively.
            w_set_1, _ = get_warning_set(f'{data_path}/{project}/warnings/warningInfo{x}.csv',
                                         f'{data_path}/{project}/golden/goldenFeatures{x}.csv')

            w_set_2, _ = get_warning_set(f'{data_path}/{project}/warnings/warningInfo{x + 1}.csv',
                                         f'{data_path}/{project}/golden/goldenFeatures{x + 1}.csv')
            # Obtain the intersection of previous and next release.
            intersection = w_set_1.intersection(w_set_2)
            ratio = round(len(intersection) / len(w_set_2) * 100, 2)
            text += f'{len(w_set_1)} & {len(w_set_2)} => {len(intersection)},{ratio}%,'
        text += '\n'
    print(text)
    save_csv_result(f'{root_path}/analysis/', 'RQ1-overlapping-warnings.csv', text)
    pass


def measure_overlapping_features_in_consecutive_releases():
    text = f'Project, R1 & R2 => O12, Ratio, R2 & R3 => O23, Ratio, R3 & R4 => O34, Ratio, R4 & R5 => O45, Ratio\n'
    for feature in all_feature_names:
        text += f'========== {feature} ==========\n'
        for project in PROJECT:
            text += f'{project}\t'
            for x in range(1, 5):  # Iterator 1, 2, 3, 4
                # Warning data in previous and next release, respectively.
                w_set_1, w_list_1 = get_warning_set(f'{data_path}/{project}/warnings/warningInfo{x}.csv',
                                                    f'{data_path}/{project}/golden/goldenFeatures{x}.csv')

                w_set_2, w_list_2 = get_warning_set(f'{data_path}/{project}/warnings/warningInfo{x + 1}.csv',
                                                    f'{data_path}/{project}/golden/goldenFeatures{x + 1}.csv')

                # Overlapping warnings
                intersection = list(w_set_1.intersection(w_set_2))
                # Linking the indices of same warnings in two consecutive releases.
                index_map = list()
                for instance in intersection:
                    cur_index = [w_list_1.index(instance), w_list_2.index(instance)]
                    if len(cur_index) != 2:
                        print('Would never execute')
                        continue
                    index_map.append(cur_index)

                # 两版本的警报特征
                data_1 = pd.read_csv(f'{data_path}/{project}/golden/goldenFeatures{x}.csv')
                data_2 = pd.read_csv(f'{data_path}/{project}/golden/goldenFeatures{x + 1}.csv')
                # print(len(w_set_1), len(w_list_1), len(data_1), len(w_set_2), len(w_list_2), len(data_2))
                overlap = 0
                for index in index_map:
                    if index[0] >= len(data_1) or index[1] >= len(data_2):
                        print('Would never execute')
                        continue
                    f1, f2 = data_1[feature][index[0]], data_2[feature][index[1]]
                    if f1 == f2:
                        overlap += 1
                ratio = round(overlap / len(intersection), 4)
                text += f'{overlap} / {len(intersection)} =>, {round(ratio * 100, 2)}%,'
            text += '\n'

    print(text)
    save_csv_result(f'{root_path}/analysis/', 'RQ1-overlapping-features.csv', text)
    pass


def main():
    # measure_overlapping_in_consecutive_releases()
    # measure_overlapping_features_in_consecutive_releases()
    pass


if __name__ == '__main__':
    main()

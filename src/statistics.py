# -*- coding: utf-8 -*-
from helper import *
import pandas as pd

"""
(3) File history
F25  -> 15 : ND  : Int // file age, number of days the file has existed
F72  -> 16 : FCR : Int // file creation number of revision since file creation
F71  -> 18 : SD  : Set // XXX developers, set of developers who have made changes to the file

(5) Code Characteristic
F104 -> 23 : RCC : Float // comment-code ratio, ratio of comment length and code length in file
F105 -> 24 : DWM : Float // method depth, depth of warned line in method
F101 -> 25 : DWF : Float // file depth, depth of warned line in file
F65  -> 28 : NM  : Int   // methods in file, number of methods in file
F68  -> 31 : NC  : Int   // class in package, number of (inner) class in package

(2) Code history
F126 -> 40 : LAR : Int // Lines of code added in file during the past 25 revisions
F41  -> 46 : LAM : Int // Lines of code added in package during the past 3 months

(2) Code analysis
F3   -> 72 :  PS : Nominal // XXX parameter signature
F15  -> 84 :  MV : Nominal // XXX method visibility

(4) Warning characteristic
F20  -> 89 : WP  : Nominal // XXX warning pattern
F21  -> 90 : WT  : Nominal // XXX warning type
F22  -> 91 : WR  : Int     // warning priority
F94  -> 96 : NWP : Int     // warnings in package, number of warnings in package

(1) Warning history
F77  -> 99 : NR  : Int // warning lifetime by revision, number of revisions between current revision and open revision

(6) Warning combination
F110 -> 106 : DWT : Float // warning context for warning type, difference of actionable and unactionable warnings for a warning type normalized by S
F116 -> 107 : DM  : Float // warning context in method, difference of actionable and unactionable warnings for the method normalized by S
F115 -> 108 : DF  : Float // warning context in file, difference of actionable and unactionable warnings for the file normalized by S
F117 -> 112 : DLP : Float // defect likelihood for warning pattern
F120 -> 115 : DDL : Float  //  discretization of defect likelihood
F123 -> 116 : ALT : Float // average lifetime for warning type, average value for feature F100 for a warning type
    
"""


def data_summary():
    """
    数据集统计信息
    :return:
    """
    for project in PROJECT:
        stat_total, stat_close, stat_ratio = [], [], []
        for x in range(1, 6):
            path = f'{data_path}/{project}/features/totalFeatures{x}.csv'
            data = pd.read_csv(path)
            category = data['category']
            num_close = len([i for i in category if i == 'close'])
            num_open = len([i for i in category if i == 'open'])
            num_total = len(category)
            ratio_close = round(num_close / num_total, 3)
            stat_total.append(num_total)
            stat_close.append(num_close)
            stat_ratio.append(ratio_close)
            # print(project, num_total, num_close, num_open, ratio_close)
        print(f'{project}, {min(stat_total)}-{max(stat_total)},'
              f'{min(stat_close)}-{max(stat_close)},'
              f'{round(min(stat_ratio) * 100, 1)}%-{round(max(stat_ratio) * 100, 1)}%')
    pass


def export_golden_dataset(to_file=False):
    for project in PROJECT:
        make_path(f'{data_path}/{project}/golden/')
        for x in range(1, 6):
            total_path = f'{data_path}/{project}/features/totalFeatures{x}.csv'
            golden_path = f'{data_path}/{project}/golden/goldenFeatures{x}.csv'
            df = pd.read_csv(total_path)

            golden_features = []
            for feature in df.columns:
                if feature in golden_feature_names:
                    golden_features.append(feature)
                elif '-' in feature and feature.split('-')[0] in golden_feature_names:
                    golden_features.append(feature)

            df = df[golden_features]

            df.to_csv(golden_path, index=False) if to_file else None
            print(f'{project}-{x}: {len(df.columns)}')


# 相邻版本的数据是十分相似的
# 如果数据本身相似，是否有必要进行预测
def get_warning_set(path):
    warning_set = set()
    warning_list = list()
    data = read_data_from_file(path)
    for line in data:
        ss = line.split(',', maxsplit=5)
        warning_set.add(line.replace(ss[4], ''))
        warning_list.append(line.replace(ss[4], ''))
    return warning_set, warning_list


def measure_consecutive_data():
    print(f'Project, v1 & v2 => O12, Ratio, v2 & v3 => O23, Ratio, v3 & v4 => O34, Ratio, v4 & v5 => O45, Ratio')
    for project in PROJECT:
        print(f'{project}', end=', ')
        for x in range(1, 5):
            w_set_1, _ = get_warning_set(f'{data_path}/{project}/warnings/warningInfo{x}.csv')
            w_set_2, _ = get_warning_set(f'{data_path}/{project}/warnings/warningInfo{x + 1}.csv')
            intersection = w_set_1.intersection(w_set_2)
            ratio = round(len(intersection) / len(w_set_2), 4)
            print(f'{len(w_set_1)} & {len(w_set_2)} => {len(intersection)}, {ratio}', end=', ')
        print()
    pass


def measure_consecutive_feature():
    print(f'Project, v1 & v2 => O12, Ratio, v2 & v3 => O23, Ratio, v3 & v4 => O34, Ratio, v4 & v5 => O45, Ratio')
    for feature in golden_feature_names:
        print(f'========== {feature_map[feature]} ==========')
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
    # data_summary()

    # 导出Golden Feature Set 数据集
    export_golden_dataset(to_file=True)
    # measure_consecutive_data()
    # measure_consecutive_feature()
    pass


if __name__ == '__main__':
    main()

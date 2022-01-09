# -*- coding: utf-8 -*-
from helper import *
import pandas as pd


def data_summary():
    """
    数据集统计信息
    :return:
    """
    for project in PROJECT:
        path = f'{data_path}/{project}/features/totalFeatures4.csv'
        data = pd.read_csv(path)
        category = data['category']
        num_close = len([i for i in category if i == 'close'])
        num_open = len([i for i in category if i == 'open'])
        num_total = len(category)
        ratio_close = round(num_close / num_total, 3)
        print(project, num_close, num_open, num_total, ratio_close)
    pass


def export_golden_dataset():
    """
    Category 1 -- File history \n
    F25  -> 15 : file age, number of days the file has existed \n
    F72  -> 16 : file creation \n
    F71  -> 18 : developers, set of developers who have made changes to the file \n

    Category 2 -- Code Characteristic \n
    F104 -> 23 : comment-code ratio, ratio of comment length and code length in file \n
    F105 -> 24 : method depth, depth of warned line in method \n
    F101 -> 25 : file depth, depth of warned line in file \n
    F65  -> 28 : methods in file, number of methods in file \n
    F68  -> 31 : class in package, number of (inner) class in package \n

    Category 3 -- Code history \n
    F126 -> 40 : added percentage of lines of code in file during the past 25 revisions \n
    F41  -> 46 : added percentage of lines of code in package during the past 3 months \n

    Category 4 -- Code analysis \n
    F3   -> 72 : parameter signature, \n
    F15  -> 84 : method visibility \n

    Category 5 -- Warning characteristic \n
    F20  -> 89 : warning pattern xxx \n
    F21  -> 90 : warning type xxx \n
    F22  -> 91 : warning priority \n
    F94  -> 96 : warnings in package, number of warnings in package \n

    Category 6 -- Warning history \n
    F77  -> 99 : warning lifetime by revision, number of revisions between current revision and open revision \n

    Category 7 -- Warning combination \n
    F110 -> 106 : warning context for warning type, difference of actionable and unactionable warnings for a warning type normalized by S \n
    F116 -> 107 : warning context in method, difference of actionable and unactionable warnings for the method normalized by S \n
    F115 -> 108 : warning context in file, difference of actionable and unactionable warnings for the file normalized by S \n
    F117 -> 112 : defect likelihood for warning pattern \n
    F120 -> 115 : discretization of defect likelihood \n
    F123 -> 116 : average lifetime for warning type, average value for feature F100 for a warning type \n
    :return:
    """
    # F71, F3, F15
    golden_feature_names = ['F25', 'F72', 'F71',
                            'F104', 'F105', 'F101', 'F65', 'F68',
                            'F126', 'F41',
                            'F3', 'F15',
                            'F20', 'F21', 'F22', 'F94',
                            'F77',
                            'F110', 'F116',
                            'F115', 'F117', 'F120', 'F123', 'category']

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

            df.to_csv(golden_path, index=False)
            print(f'{project}-{x}: {len(df.columns)}')


# 相邻版本的数据是十分相似的
# 如果数据本身相似，是否有必要进行预测
def get_warning_set(path):
    warning_set = set()
    data = read_data_from_file(path)
    for line in data:
        ss = line.split(',', maxsplit=5)
        warning_set.add(line.replace(ss[4], ''))
    return warning_set


def measure_consecutive_data():
    print(f'Project, v1 & v2 => O12, Ratio, v2 & v3 => O23, Ratio, v3 & v4 => O34, Ratio, v4 & v5 => O45, Ratio')
    for project in PROJECT:
        print(f'{project}', end=', ')
        for x in range(1, 5):
            w_set_1 = get_warning_set(f'{data_path}/{project}/warnings/warningInfo{x}.csv')
            w_set_2 = get_warning_set(f'{data_path}/{project}/warnings/warningInfo{x + 1}.csv')
            intersection = w_set_1.intersection(w_set_2)
            ratio = round(len(intersection) / len(w_set_2), 4)
            print(f'{len(w_set_1)} & {len(w_set_2)} => {len(intersection)}, {ratio}', end=', ')
        print()
    pass


def main():
    # data_summary()
    # export_golden_dataset()
    measure_consecutive_data()
    pass


if __name__ == '__main__':
    main()

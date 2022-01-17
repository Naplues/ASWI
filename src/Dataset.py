# -*- coding: utf-8 -*-
from helper import *
import pandas as pd

__author__ = 'Naples'
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
    Summary of dataset.
    :return:
    """
    for project in PROJECT:
        stat_total, stat_close, stat_ratio = [], [], []
        for x in range(1, 6):
            path = f'{data_path}/{project}/features/totalFeatures{x}.csv'
            data = pd.read_csv(path)
            category = data['category']
            num_close = len([i for i in category if i == 'close'])
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
    """
    1. Export 23 golden feature set from total 116 features.
    """
    for project in PROJECT:
        make_path(f'{data_path}/{project}/golden/')
        for x in range(1, 6):
            total_path = f'{data_path}/{project}/features/totalFeatures{x}.csv'
            golden_path = f'{data_path}/{project}/golden/goldenFeatures{x}.csv'
            df = pd.read_csv(total_path)

            # Add Numerical features
            new_df = df[golden_numerical_feature_names]

            # Add Nominal features - 'F71', 'F3', 'F15', 'F20', 'F21'
            nominal_feature_dict = {'F71': set(), 'F3': set(), 'F15': set()}
            for feature in df.columns:
                if '-' in feature:
                    ss = feature.split('-')
                    nominal_feature_dict[ss[0]].add(feature) if ss[0] in nominal_feature_dict else None

            for nominal in nominal_feature_dict.keys():
                # Process each feature
                nominal_values = [''] * len(df)
                for feature_value in nominal_feature_dict[nominal]:
                    # Process each feature value
                    exists = list(df[feature_value])
                    indices = [index for index in range(len(exists)) if exists[index] == 1]
                    for i in indices:
                        nominal_values[i] = feature_value
                # Add each feature into dataframe
                new_df.insert(len(new_df.columns) - 1, nominal, nominal_values)

            # 'F20', 'F21'
            new_df.insert(len(new_df.columns) - 1, 'F20', df['F20'])
            new_df.insert(len(new_df.columns) - 1, 'F21', df['F21'])

            # Rename feature names
            print(new_df)
            new_df.columns = new_name
            print(new_df)
            # Output to file
            new_df.to_csv(golden_path, index=False) if to_file else None
            print(f'{project}-{x}: {len(new_df.columns)}')


def check_consistency():
    """
    2. Check the consistency between the number of warnings and feature instances.
    """
    for project in PROJECT:
        for x in range(1, 6):  # Iterator 1, 2, 3, 4, 6
            list_warnings = read_data_from_file(f'{data_path}/{project}/warnings/warningInfo{x}.csv')
            list_instances = read_data_from_file(f'{data_path}/{project}/golden/goldenFeatures{x}.csv')
            len_warnings, len_instances = len(list_warnings), len(list_instances) - 1
            # ######### Detect consistency
            w_index, i_index = 0, 1  # warning(instance) index start from 0(1)
            redundant_warning_index = []  # need to be record and removed from warning_d
            for i in range(len_warnings):
                cur_warning = list_warnings[w_index].split(',')[0] + list_warnings[w_index].split(',')[1]
                cur_instance = list_instances[i_index].split(',')[-2] + list_instances[i_index].split(',')[-3]
                if cur_warning == cur_instance:
                    i_index += 1
                else:
                    redundant_warning_index.append(w_index)
                w_index += 1

            print(f'{project}-{x} warning: {len_warnings}, feature: {len_instances} '
                  f'Redundant: {len_warnings - len_instances} == {len(redundant_warning_index)}')

            # ######### Correct inconsistency
            warning_text = ''
            for index in range(len_warnings):
                if index in redundant_warning_index:
                    continue
                warning_text += list_warnings[index]
            save_csv_result(f'{data_path}/{project}/warnings/', f'warningInfo{x}.csv', warning_text)
    pass


def main():
    # data_summary()
    # First, export golden dataset. Second, check consistency
    # export_golden_dataset(to_file=True)
    # check_consistency()
    pass


if __name__ == '__main__':
    main()
    pass

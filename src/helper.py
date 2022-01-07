# -*- coding: utf-8 -*-
import os


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

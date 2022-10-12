import pandas as pd
import numpy as np

COLUMNS_RENAMES = {
    'Untitled': 'FREQ',
    'Untitled 1': 'CV_4A_READ',
    'Untitled 2': 'PT_1A',
    'Untitled 3': 'PT_1B',
    'Untitled 4': 'PT_1C',
    'Untitled 5': 'PT_1D',
    'Untitled 6': 'PT_2A',
    'Untitled 7': 'PT_2B',
    'Untitled 8': 'PT_2C',
    'Untitled 9': 'PT_3A',
    'Untitled 10': 'PT_3B',
    'Untitled 11': 'PT_4A',
    'Untitled 12': 'PT_5A',
    'Untitled 13': 'PT_6A',
    'Untitled 14': 'FT_1A',
    'Untitled 15': 'FT_2A',
    'Untitled 16': 'FT_2B',
    'Untitled 17': 'FT_3A',
    'Untitled 18': 'FT_4A',
    'Untitled 19': 'FT_4B',
    'Untitled 20': 'FT_5A',
    'Untitled 21': 'FT_6A'
}


def read_excel(path, add_angle):
    df = pd.read_excel(path)
    df = df.rename(columns=COLUMNS_RENAMES)
    columns = ['FREQ', 'FT_1A', 'FT_3A']
    if add_angle:
        columns.append('CV_4A_READ')
    df = df[columns]
    df = df[df['FT_1A'] > 0]
    df = df[df['FT_3A'] > 0]
    return df
 

def read_training_data(name, add_angle=False):
    path = f"./training_data/{name}.xlsx"
    df = read_excel(path, add_angle)
    return df


def read_test_data(name, add_angle=False):
    path = f"./test_data/{name}.xlsx"
    df = read_excel(path, add_angle)
    return df


def concat_delayed_flows(data):
    delayed_data = data[['FT_1A', 'FT_3A']].shift(1)
    delay_renames = {'FT_1A': 'FT_1A(K-1)', 'FT_3A': 'FT_3A(K-1)'}
    delayed_data = delayed_data.rename(columns=delay_renames)
    data = pd.concat([delayed_data, data], axis=1).dropna(axis=0)
    return data


def pd2dataarray(data, with_angle=False):
    data_array = np.array(data)
    index = 4 if with_angle else 3
    x = data_array[:, 0:index]
    y = data_array[:, index:index+2]
    return (x, y)

import pandas as pd
import numpy as np


def read_training_data(name):
    df = pd.read_excel(f"./training_data/{name}.xlsx")
    complete_renames = {
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
    df = df.rename(columns=complete_renames)
    df = df[['FREQ', 'FT_1A', 'FT_3A']]
    return df


def read_test_data(name):
    df = pd.read_excel(f"./test_data/{name}.xlsx")
    complete_renames = {
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
    df = df.rename(columns=complete_renames)
    df = df[['FREQ', 'FT_1A', 'FT_3A']]
    return df

def concat_delayed_flows(data):
    delayed_data = data[['FT_1A', 'FT_3A']].shift(1)
    delay_renames = {'FT_1A': 'FT_1A(K-1)', 'FT_3A': 'FT_3A(K-1)'}
    delayed_data = delayed_data.rename(columns=delay_renames)
    data = pd.concat([delayed_data, data], axis=1).dropna(axis=0)
    return data


def pd2dataarray(data):
    data_array = np.array(data)
    x = data_array[:, 0:3]
    y = data_array[:, 3:5]
    return (x, y)

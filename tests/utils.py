import numpy as np


def convert_r_data_to_py_array(data: str):
    data = data.replace('\n', ' ')
    data = [float(d) for d in data.split(' ') if d != '']
    return data

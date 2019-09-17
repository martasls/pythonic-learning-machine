from os import pardir, makedirs, listdir
from os.path import join, dirname, exists
from pickle import dump, load
from pmlb import fetch_data

from pandas import read_csv, read_pickle

import utils
from utils.environment_constants import SAMPLE_LABELS, SAMPLE_LABELS_NO_VAL


def _get_path_to_data_dir():
    """Returns path to data_sets directory from src/data_sets."""
    return join(dirname(__file__), pardir, pardir, "data_sets")


def read_raw_data_set(data_set_name):
    """Reads 01_raw data_sets set from data_sets/01_raw/'data_set_name'."""
    file_path = join(_get_path_to_data_dir(), "01_raw", data_set_name)
    file_name = 'data_set.csv'
    return read_csv(join(file_path, file_name), header=None)


def read_cleaned_data_set(data_set_name):
    """Reads 02_cleaned data_sets set from data_sets/02_cleaned/'data_set_name'."""
    file_path = join(_get_path_to_data_dir(), "02_cleaned")
    file_name = data_set_name + '.csv'
    return read_csv(join(file_path, file_name), header=None)


def read_standardized_data_set(data_set_name):
    file_path = join(_get_path_to_data_dir(), '03_standardized')
    file_name = data_set_name + '.pkl'
    return read_pickle(join(file_path, file_name))


def list_files(dir):
    """Lists files in dir."""
    file_path = join(_get_path_to_data_dir(), dir)
    return listdir(file_path)


def remove_extension(file):
    return file.split('.')[0]


def benchmark_to_pickle(benchmark, file_path_ext=None):
    # Appends path to data folder to file path.
    if file_path_ext == None:
        file_path_ext = join(_get_path_to_data_dir(), '05_benchmark', benchmark.data_set_name)
    else:
        file_path_ext = join(file_path_ext, benchmark.data_set_name)

    # If 'file_path_ext' does not exist, create 'file_path_ext'.
    if not exists(file_path_ext):
        makedirs(file_path_ext)

    file_name_ext = benchmark.file_name + '.pkl'

    with open(join(file_path_ext, file_name_ext), 'wb') as f:
        dump(benchmark, f)


def benchmark_from_pickle(data_set_name, file_name):
    file_path_ext = join(_get_path_to_data_dir(), '05_benchmark', data_set_name, file_name)

    if exists(file_path_ext):
        with open(file_path_ext, 'rb') as f:
            return load(f)


def data_set_to_pickle(data_set, file_path, file_name):
    """Serializes 'data_set' as 'file_name'.pkl file in 'file_path'."""

    # Appends path to data folder to file path.
    file_path_ext = join(_get_path_to_data_dir(), file_path)

    # If 'file_path_ext' does not exist, create 'file_path_ext'.
    if not exists(file_path_ext):
        makedirs(file_path_ext)

    # Add extension .pkl to file_name.
    file_name_ext = file_name + ".pkl"

    data_set.to_pickle(path=join(file_path_ext, file_name_ext))


def read_pickle(file_path):
    
    with open(file_path, 'rb') as p:
        return load(p)

def data_set_from_pickle(file_path, file_name):
    """"""

    file_path_ext = join(_get_path_to_data_dir(), file_path)

    file_name_ext = file_name + ".pkl"

    return read_pickle(join(file_path_ext, file_name_ext))


def load_samples(data_set_name, index):

    file_path = join('04_resampled', data_set_name)
    file_names = [sample_label + "_" + str(index) for sample_label in SAMPLE_LABELS]

    return [data_set_from_pickle(file_path, file_name) for file_name in file_names]


def load_samples_no_val(data_set_name, index):
    file_path = join('04_resampled', data_set_name)
    file_names = [sample_label + "_" + str(index) for sample_label in SAMPLE_LABELS_NO_VAL]

    return [data_set_from_pickle(file_path, file_name) for file_name in file_names]


def load_standardized_samples(data_set_name, file_path=None):
    if file_path == None:
        file_path = '03_standardized'
    
    return data_set_from_pickle(file_path, data_set_name)

def load_pmlb_samples(data_set_name):
    if data_set_name == 'clean1' or data_set_name == 'clean2':
        df = fetch_data(data_set_name)
        df = df.drop(['molecule_name', 'conformation_name'], axis=1)
    elif data_set_name == 'breast-cancer-wisconsin':
        df = load_standardized_samples('c_cancer')
    elif data_set_name == 'diabetes':
        df = load_standardized_samples('c_diabetes')
    else:
        df = fetch_data(data_set_name)
    return df


def get_standardized_folder(): 
    return join(_get_path_to_data_dir(), '03_standardized')


def get_resampled_folder(): 
    return join(_get_path_to_data_dir(), '04_resampled')


def get_benchmark_folder():
    return join(_get_path_to_data_dir(), '05_benchmark')

    
def get_formatted_folder(): 
    return join(_get_path_to_data_dir(), '06_formatted')


def get_results_folder(): 
    return join(_get_path_to_data_dir(), '07_results')


def read_csv_(path):
    data_frame = read_csv(path)
    data_frame = data_frame.drop(data_frame.columns[0], axis=1) 
    return data_frame

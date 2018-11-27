import numpy as np
import pandas as pd
from random import seed, random, randrange

def get_input_variables(data_set):
    """Returns independent variables."""
    target = data_set.columns[len(data_set.columns) - 1]
    return data_set.drop(target, axis=1)

def get_target_variable(data_set):
    """Returns target variable."""
    return data_set[data_set.columns[len(data_set.columns) - 1]]

def is_classification(data_set):
    target = data_set[data_set.columns[len(data_set.columns) - 1]]
    return all(map(lambda x: x in (0, 1), target))

def generate_sub_training_set(training_set_values, ratio=1.0):
    """Reorganizes a given training set. It doesn't change its size (because ratio=1), but reorganizes it with replacement
    (ie the same instance can show up more than once)"""
    training_set_df = pd.DataFrame(training_set_values)
    sub_training_set = pd.DataFrame()
    n_sample = round(len(training_set_df) * ratio)
    while len(sub_training_set) < n_sample:
        index = randrange(len(training_set_df))
        sub_training_set = sub_training_set.append(training_set_df.iloc[index])
    sub_training_set_values = sub_training_set.values
    return sub_training_set_values

def generate_training_target(training_set_values, target_vector, ratio=1.0):
    """returns a subset (if ratio != 1) of the training set and its target values in separate ndarrays"""
    training_set_df = pd.DataFrame(training_set_values)
    target_vector_df = pd.DataFrame(target_vector)
    sub_training_set = pd.DataFrame()
    sub_target_vector = pd.DataFrame()
    n_sample = round(len(training_set_df) * ratio)
    while len(sub_training_set) < n_sample:
        index = randrange(len(training_set_df))
        sub_training_set = sub_training_set.append(training_set_df.iloc[index])
        sub_target_vector = sub_target_vector.append(target_vector_df.iloc[index])
    sub_training_set_values = sub_training_set.values 
    sub_target_vector_values = sub_target_vector.values
    sub_target_vector_values = sub_target_vector_values.ravel()
    return sub_training_set_values, sub_target_vector_values

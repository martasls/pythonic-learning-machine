import random
import numpy as np
from numpy import sqrt, mean, square 
from numba import jit 

@jit(nopython=True)
def generate_random_weight_vector(length, max_range): 
    random_weights = [random.uniform(0, max_range) for i in range(length)]
    return np.array(random_weights)

def generate_weight_vector(length):
    return np.array([(1/length) for i in range(length)])


from numpy import sqrt, mean, square, array, count_nonzero
from numba import jit

class Metric():

    greater_is_better = None

    @staticmethod
    def evaluate(prediction, target):
        pass

class RootMeanSquaredError(Metric):

    greater_is_better = False

    
    @staticmethod 
    @jit(nopython=True, nogil=True)
    def evaluate(prediction, target):
        return sqrt(mean(square(prediction - target)))

class WeightedRootMeanSquaredError(Metric):

    greater_is_better = False 
    
    def __init__(self, weight_vector):
        self.weight_vector = weight_vector

    def evaluate(self, prediction, target):
        """Calculates RMSE taking into account a weight vector"""
        return sqrt(mean(square((prediction - target) * self.weight_vector)))
    
class Accuracy(Metric):

    greater_is_better = True

    
    @staticmethod
    def evaluate(prediction, target):
        prediction_label = prediction > 0.5
        return count_nonzero(prediction_label == target) / prediction.shape[0]

def is_better(value_1, value_2, metric):
    if metric.greater_is_better:
        return value_1 > value_2
    else:
        return value_1 < value_2

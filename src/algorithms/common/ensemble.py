import numpy as np 
from numpy import mean, median
import random
from sklearn.utils.extmath import stable_cumsum
from copy import deepcopy
from data.extract import generate_sub_training_set
from utils.useful_methods import generate_random_weight_vector, generate_weight_vector
from algorithms.common.metric import WeightedRootMeanSquaredError, RootMeanSquaredError
import time
from threading import Thread 
from multiprocessing import Process

class Ensemble(object):
    """
    Class represents ensemble learning technique. In short, ensemble techniques predict output over a meta learner
    that it self is supplied with output of a number of base learners.

    Attributes:
        base_learner: Base learner algorithms that supplies meta learner.
        number_learners: Number of base learners.
        meta_learner: Meta learner that predicts output, based on base learner predictions.
        learners: List, containing the trained base learners.

    Notes:
        base_learner needs to support fit() and predict() function.
        meta_learner function needs to support numpy ndarray as input.
    """

    def __init__(self, base_learner, number_learners, meta_learner=mean):
        self.base_learner = base_learner
        self.number_learners = number_learners
        self.meta_learner = meta_learner
        self.learners = list()
    
    def _fit_learner(self, i, input_matrix, target_vector, metric, verbose):
        if verbose: print(i)
        # Creates deepcopy of base learner.
        learner = deepcopy(self.base_learner)
        # Trains base learner.
        learner.fit(input_matrix, target_vector, metric) 
        # Adds base learner to list.
        self.learners.append(learner)
        print()
    
    def fit(self, input_matrix, target_vector, metric, verbose=False):
        """Trains learner to approach target vector, given an input matrix, based on a defined metric."""
        # threads = [] 
        for i in range(self.number_learners):
        #     t = Process(target=self._fit_learner, args=(i, input_matrix, target_vector, metric, verbose))
        #     t.daemon = True
        #     threads.append(t) 

        # for t in threads: 
        #     t.start() 

        # for t in threads: 
        #     t.join()
            if verbose: print(i)
            # Creates deepcopy of base learner.
            learner = deepcopy(self.base_learner)
            # Trains base learner.
            learner.fit(input_matrix, target_vector, metric, verbose) 
            # Adds base learner to list.
            self.learners.append(learner)
            


    def predict(self, input_matrix):
        """Predicts target vector, given input_matrix, based on trained ensemble."""

        # Creates prediction matrix.
        predictions = np.zeros([input_matrix.shape[0], self.number_learners])
        # Supplies prediction matrix with predictions of base learners.
        for learner, i in zip(self.learners, range(len(self.learners))):
            predictions[:, i] = learner.predict(input_matrix)
        # Applies meta learner to prediction matrix.
        return self.meta_learner(predictions, axis=1)

class EnsembleBagging(Ensemble): 

    def __init__(self, base_learner, number_learners, meta_learner=mean):
        Ensemble.__init__(self, base_learner, number_learners, meta_learner)

    def fit(self, input_matrix, target_vector, metric, verbose=False):

        original_input_matrix = input_matrix
        original_target_vector = target_vector
        size = input_matrix.shape[0]

        for i in range(self.number_learners):
            if verbose: print(i)
            # Creates deepcopy of base learner.
            learner = deepcopy(self.base_learner)
            ## Reorganizes the input matrix 
            idx = np.random.choice(np.arange(size), size, replace=True)
            input_matrix = original_input_matrix[idx]
            target_vector = original_target_vector[idx]
            # Trains base learner.
            learner.fit(input_matrix, target_vector, metric, verbose) 
            # Adds base learner to list.
            self.learners.append(learner)


class EnsembleRandomIndependentWeighting(Ensemble): 

    def __init__(self, base_learner, number_learners, meta_learner=mean, weight_range=1):
        Ensemble.__init__(self, base_learner, number_learners, meta_learner)
        self.weight_range = weight_range
    
    def fit(self, input_matrix, target_vector, metric, verbose=False):
        
        for i in range(self.number_learners):
            if verbose: print(i)
            # Creates deepcopy of base learner.
            learner = deepcopy(self.base_learner)
            # Generates a random weight vector
            weight_vector = generate_random_weight_vector(input_matrix.shape[0], self.weight_range)
            # Instatiates the WeightedRootMeanSquaredError object with the weight vector
            metric = WeightedRootMeanSquaredError(weight_vector)
            # Trains base learner #
            learner.fit(input_matrix, target_vector, metric)
            # Adds base learner to list.
            self.learners.append(learner)

class EnsembleBoosting(Ensemble):

    def __init__(self, base_learner, number_learners, meta_learner=mean, learning_rate=1):
        Ensemble.__init__(self, base_learner, number_learners, meta_learner)
        self.learning_rate = learning_rate
        self.estimator_weights = np.zeros(self.number_learners, dtype=np.float64)

    def _get_learning_rate(self, learning_rate): 
        if(self.learning_rate == 'random'):
            #return random generated learning rate between 0 and 1 
            return random.uniform(0, 1)
        else: 
            return 1
      
    def fit(self, input_matrix, target_vector, metric, verbose=False):
        """Trains learner to approach target vector, given an input matrix, based on a defined metric."""
        # Initialize the weights with 1/n where n is the size of the input matrix
        weight_vector = generate_weight_vector(input_matrix.shape[0])
        size = input_matrix.shape[0]
        original_input_matrix = input_matrix
        original_target_vector = target_vector
        for i in range(self.number_learners):
            if verbose: print(i)
            # Creates deepcopy of base learner.
            learner = deepcopy(self.base_learner)
            # select the training instances
            idx = np.random.choice(np.arange(size), size, p=weight_vector)
            input_matrix = original_input_matrix[idx]
            target_vector = original_target_vector[idx]
            # Trains base learner.
            learner.fit(input_matrix, target_vector, metric)
            # calculate the output (semantics) of the model for every instance even the ones not used for the training 
            # learner.predict(self, input_matrix)
            y_predict = learner.predict(original_input_matrix)
            # calculate the absolute error vector: Ei = |yi - ti| 
            error_vector = np.abs(target_vector - y_predict)
            # calculate the maximum absolute error 
            max_abs_error = error_vector.max() 
            # calculate the normalized error vector (with values between 0 and 1): ENi = Ei / max absolute error 
            error_vector = error_vector / max_abs_error  
            # take into account the loss function - square in this case
            error_vector **= 2 
            # calculate the weighted error of this element: EEk = sum(wi*Ei)
            learner_error = (weight_vector * error_vector).sum()
            # calculate the beta used in the weight update: beta = EEk/(1-EEk)
            beta = learner_error / (1. - learner_error)
            # calculate the weight that the elem will have on the final ensemble
            # self.learning_rate*np.log(1/beta)
            learning_rate = self._get_learning_rate(self.learning_rate)
            estimator_weight = learning_rate * np.log(1. / beta)
            self.estimator_weights[i] = estimator_weight
            # update the weights for the next iteration 
            # sample_weight *= np.power(beta, (1-error_vect) *self.learning_rate)
            weight_vector *= np.power(beta, (1. - error_vector) * learning_rate)
            # doing this, because otherwise will get an error that probabilities do not sum to 1
            weight_vector /= weight_vector.sum() 
            # Adds base learner to list.
            self.learners.append(learner)
            
    def _get_median_predict(self, input_matrix, limit):
        # Evaluate predictions of all estimators
        predictions = np.array([
            learner.predict(input_matrix) for learner in self.learners[:limit]]).T
        # Sort the predictions
        sorted_idx = np.argsort(predictions, axis=1)
        # Find index of median prediction for each sample
        weight_cdf = stable_cumsum(self.estimator_weights[sorted_idx], axis=1)
        median_or_above = weight_cdf >= 0.5 * weight_cdf[:, -1][:, np.newaxis]
        median_idx = median_or_above.argmax(axis=1)

        median_estimators = sorted_idx[np.arange(input_matrix.shape[0]), median_idx]

        # Return median predictions
        return predictions[np.arange(input_matrix.shape[0]), median_estimators]
    
    def _get_mean_predict(self, input_matrix):
        # Creates prediction matrix.
        predictions = np.zeros([input_matrix.shape[0], self.number_learners])
        # Supplies prediction matrix with predictions of base learners.
        for learner, i in zip(self.learners, range(len(self.learners))):
            predictions[:, i] = learner.predict(input_matrix)
        # Applies meta learner to prediction matrix.
        return self.meta_learner(predictions, axis=1)


    def predict(self, input_matrix):
        # verify what is the meta learner, if it's median then return get_median_predict else return the mean predictions  
        if(self.meta_learner == median):
            return self._get_median_predict(input_matrix, self.number_learners)
        else:
            return self._get_mean_predict(input_matrix)

    

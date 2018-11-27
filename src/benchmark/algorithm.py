import numpy as np 
from algorithms.semantic_learning_machine.algorithm import SemanticLearningMachine
from algorithms.neat_python.algorithm import Neat
from algorithms.simple_genetic_algorithm.algorithm import SimpleGeneticAlgorithm
from data.extract import generate_sub_training_set, generate_training_target
from timeit import default_timer
from utils.useful_methods import generate_random_weight_vector
from algorithms.common.metric import RootMeanSquaredError, WeightedRootMeanSquaredError

_time_seconds = lambda: default_timer()

def _get_parent(algorithm):
    return algorithm.__class__.__bases__[0]

def _benchmark_fit(algorithm, input_matrix, target_vector, metric, verbose):
    parent = _get_parent(algorithm) 
    parent.fit(algorithm, input_matrix, target_vector, metric, verbose)
    return algorithm.log

def _benchmark_run(algorithm, verbose=False):
    parent = _get_parent(algorithm)
    time_log = list()
    solution_log = list()
    stopping_criterion = False
    original_input_matrix = algorithm.input_matrix
    original_target_vector = algorithm.target_vector
    original_metric = algorithm.metric 
    while (not stopping_criterion):
        start_time = _time_seconds()
        if(algorithm.random_sampling_technique): # do i have to check if the algorithm is SLM ? i think so 
            """if subset_training_data is set to true then the training instances change at each iteration"""
            # algorithm.input_matrix, algorithm.target_vector = generate_training_target(original_input_matrix, original_target_vector, ratio=0.4)
            size = int(original_input_matrix.shape[0] * algorithm.subset_ratio) # THIS IS THE RATIO WHICH SHOULD BE A PARAMETER 
            idx = np.random.choice(np.arange(size), size, replace=False)
            algorithm.input_matrix = original_input_matrix[idx]
            algorithm.target_vector = original_target_vector[idx]
        if(algorithm.random_weighting_technique): 
            """if random_weighting_technique is set to true then the weights change at each iteration """ 
            weight_vector = generate_random_weight_vector(original_input_matrix.shape[0], algorithm.weight_range)
            metric = WeightedRootMeanSquaredError(weight_vector)
            algorithm.metric = metric
        stopping_criterion = parent._epoch(algorithm)
        end_time = _time_seconds()
        time_log.append(end_time - start_time)
        solution_log.append(algorithm.champion)
        if verbose:
            parent._print_generation(algorithm)
        algorithm.current_generation += 1
        algorithm.log = {
        'time_log': time_log,
        'solution_log': solution_log}
    if(algorithm.random_sampling_technique or algorithm.random_weighting_technique):
        algorithm.metric = RootMeanSquaredError
        algorithm.champion.predictions = algorithm.champion.neural_network.predict(original_input_matrix)
        algorithm.champion.value = algorithm.metric.evaluate(algorithm.champion.predictions, original_target_vector)


class BenchmarkSLM(SemanticLearningMachine):

    fit = _benchmark_fit
    _run = _benchmark_run

class BenchmarkNEAT(Neat):

    fit = _benchmark_fit
    _run = _benchmark_run

class BenchmarkSGA(SimpleGeneticAlgorithm):

    fit = _benchmark_fit
    _run = _benchmark_run

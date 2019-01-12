import argparse
import os
from threading import Thread
from sys import argv
from time import sleep
from benchmark.benchmarker import Benchmarker, continue_benchmark, SLM_MODELS, MLP_MODELS, ENSEMBLES
from data.io_plm import get_benchmark_folder, read_pickle, get_resampled_folder, get_formatted_folder, read_csv_, \
                        get_standardized_folder, remove_extension
from data.extract import is_classification
from benchmark.formatter import format_benchmark
from benchmark.results_extractor import extract_results 


def start_b(data_set_name, file_name=None, models_to_run=SLM_MODELS, models_to_run_2=MLP_MODELS): #change to None to choose from console
    """ starts benchmark """
    if models_to_run == 'SLM_MODELS':
        print(1)
        models_to_run = SLM_MODELS
    if models_to_run == 'MLP_MODELS':
        print(2)
        models_to_run = MLP_MODELS
    if models_to_run_2 == 'SLM_MODELS':
        print(3)
        models_to_run_2 = SLM_MODELS
    if models_to_run_2 == 'MLP_MODELS':
        print(4)
        models_to_run_2 = MLP_MODELS
    # SLM MODELS 
    if models_to_run is not None:
        benchmarker = Benchmarker(data_set_name, models=models_to_run, ensembles=ENSEMBLES)
        # benchmarker.run()
        benchmarker.run_nested_cv()
    # MLP MODELS 
    if models_to_run_2 is not None: 
        benchmarker = Benchmarker(data_set_name, models=models_to_run_2, ensembles=ENSEMBLES)
        benchmarker.run_nested_cv()

def continue_b(data_set_name, file_name):
    """ continues benchmark """
    continue_benchmark(data_set_name, file_name)

if __name__ == '__main__':

    # for data_set in os.listdir(get_standardized_folder()):
    #     start_b(remove_extension(data_set))


    # start_b("c_diabetes")

    """ this block of code formats the benchmark files into csv files """
    benchmark_paths = []
    for folder in os.listdir(get_benchmark_folder()):
        path = os.path.join(get_benchmark_folder(), folder)
        for file in os.listdir(path):
            benchmark_paths.append(os.path.join(get_benchmark_folder(), folder, file))

    for benchmark_path in benchmark_paths:
        benchmark = read_pickle(benchmark_path)
        benchmark_formatted = format_benchmark(benchmark)

    """ this block of code generates the results automatically""" 
    # for folder in os.listdir(get_formatted_folder()):
    #     path = os.path.join(get_formatted_folder(), folder)
    #     extract_results(path)

    # parser = argparse.ArgumentParser(description='Runs benchmark for data set.')
    # parser.add_argument('-d', metavar='data_set_name', type=str, dest='data_set_name',
    #                     help='a name of a data set')
    # parser.add_argument('-f', metavar='file_name', type=str, dest='file_name',
    #                     help='a file name of an existing benchmark')
    # parser.add_argument('-m1', metavar='models_to_run', dest='models_to_run',
    #                     help='MLP_MODELS or SLM_MODELS')
    # parser.add_argument('-m2', metavar='models_to_run_2', dest='models_to_run_2',
    #                     help='MLP_MODELS or SLM_MODELS but different than the previous one')
    # args = parser.parse_args()
    
    # if args.file_name:
    #     thread = Thread(target=continue_b, kwargs=vars(args))
    # else:
    #     thread = Thread(target=start_b, kwargs=vars(args))
    
    # try:
    #     thread.daemon = True
    #     thread.start()
    #     while True: sleep(100)
    # except (KeyboardInterrupt, SystemExit):
    #     print('\n! Received keyboard interrupt, quitting threads.\n')
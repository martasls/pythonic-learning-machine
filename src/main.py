import argparse
import os
from threading import Thread
from sys import argv
from time import sleep
from benchmark.benchmarker import Benchmarker, continue_benchmark
from data.io_plm import get_benchmark_folder, read_pickle, get_resampled_folder, get_formatted_folder, read_csv_
from data.extract import is_classification
from benchmark.formatter import format_benchmark
from benchmark.extracter import extract_results 


def start_b(data_set_name, file_name=None):
    """ starts benchmark """
    benchmarker = Benchmarker(data_set_name)
    # benchmarker.run()
    benchmarker.run_nested_cv()



def continue_b(data_set_name, file_name):
    """ continues benchmark """
    continue_benchmark(data_set_name, file_name)


if __name__ == '__main__':

    # for folder in os.listdir(get_resampled_folder()):
    #     start_b(folder)


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

    # """ this block of code is supposed to generate the results automatically""" 
    # results_paths = [] 
    # for folder in os.listdir(get_formatted_folder()):
    #     path = os.path.join(get_formatted_folder(), folder)
    #     for file in os.listdir(path):
    #         results_paths.append(os.path.join(get_formatted_folder(), folder, file))

    # for results_path in results_paths: 
    #     results = read_csv_(results_path)
    #     results_extracted = extract_results(results)

    # parser = argparse.ArgumentParser(description='Runs benchmark for data set.')
    # parser.add_argument('-d', metavar='data_set_name', type=str, dest='data_set_name',
    #                     help='a name of a data set')
    # parser.add_argument('-f', metavar='file_name', type=str, dest='file_name',
    #                     help='a file name of an existing benchmark')
    #
    # args = parser.parse_args()
    #
    # if args.file_name:
    #     thread = Thread(target=continue_b, kwargs=vars(args))
    # else:
    #     thread = Thread(target=start_b, kwargs=vars(args))
    #
    # try:
    #     thread.daemon = True
    #     thread.start()
    #     while True: sleep(100)
    # except (KeyboardInterrupt, SystemExit):
    #     print('\n! Received keyboard interrupt, quitting threads.\n')

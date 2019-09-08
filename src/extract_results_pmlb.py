import os

from data.io_plm import get_benchmark_folder, get_formatted_folder, read_pickle
from benchmark.formatter import format_benchmark, merge_best_results

from data.io_plm import get_benchmark_folder, read_pickle
from extract_to_csv import process_selected_benchmarks_csv as process_to_csv
from extract_to_generalization_boxplots import process_all as process_to_boxplot
from extract_to_latex_tables import process_all as process_to_latex

if __name__ == '__main__':


    
    # benchmark_paths = []

    # for folder in os.listdir(get_benchmark_folder()):
    #     path = os.path.join(get_benchmark_folder(), folder)
    #     for file in os.listdir(path):
    #         benchmark_paths.append(os.path.join(get_benchmark_folder(), folder, file))

    # for benchmark_path in benchmark_paths:
    #     benchmark = read_pickle(benchmark_path)
    #     benchmark_formatted = format_benchmark(benchmark)

    for folder in os.listdir(get_formatted_folder()):
        path = os.path.join(get_formatted_folder(), folder)
        merge_best_results(path)
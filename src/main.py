import argparse
import os
import numpy as np
from threading import Thread
from sys import argv
from time import sleep
#from benchmark.benchmarker import Benchmarker, continue_benchmark, pickup_benchmark, SLM_MODELS, MLP_MODELS, ENSEMBLES
from data.io_plm import get_benchmark_folder, read_pickle, get_resampled_folder, get_formatted_folder, read_csv_, \
                        get_standardized_folder, remove_extension
from data.extract import is_classification
from benchmark.formatter import format_benchmark, merge_best_results
from benchmark.results_extractor import extract_results
from data.extract import is_classification, get_input_variables, get_target_variable 
from data.io_plm import load_samples, load_samples_no_val, benchmark_to_pickle, benchmark_from_pickle, load_standardized_samples
from algorithms.common.metric import RootMeanSquaredError, is_better, Accuracy

# def start_b(data_set_name, file_name=None, models_to_run=None, models_to_run_2=MLP_MODELS, ensembles=ENSEMBLES):  # change to None to choose from console
#     """ starts benchmark """
#     if models_to_run == 'SLM_MODELS':
#         models_to_run = SLM_MODELS
#     if models_to_run == 'MLP_MODELS':
#         models_to_run = MLP_MODELS
#     if models_to_run_2 == 'SLM_MODELS':
#         models_to_run_2 = SLM_MODELS
#     if models_to_run_2 == 'MLP_MODELS':
#         models_to_run_2 = MLP_MODELS
#     # SLM MODELS 
#     if models_to_run is not None:
#         benchmarker = Benchmarker(data_set_name, models=models_to_run, ensembles=ensembles, benchmark_id='slm')
#         # benchmarker.run()
#         benchmarker.run_nested_cv()
#     # MLP MODELS 
#     if models_to_run_2 is not None: 
#         benchmarker = Benchmarker(data_set_name, models=models_to_run_2, ensembles=ensembles, benchmark_id='mlp')
#         benchmarker.run_nested_cv()


# def continue_b(data_set_name, file_name):
#     """ continues benchmark """
#     continue_benchmark(data_set_name, file_name)


# def pickup_b(data_set_name, file_name):
#     """continues benchmark after parameter tuning""" 
#     pickup_benchmark(data_set_name, file_name)

if __name__ == '__main__':
 
    # from algorithms.xcs.xcs.algorithms.xcs import XCSAlgorithm
    # from algorithms.xcs.xcs.scenarios import MUXProblem, ScenarioObserver, PreClassifiedData, UnclassifiedData

    # samples = load_standardized_samples('c_cancer')
    # X = get_input_variables(samples).values
    # y = get_target_variable(samples).values
    # assert len(X) == len(y)
    # scenario = PreClassifiedData(X, y)


    # algorithm = XCSAlgorithm() 
    # algorithm.exploration_probability = .1
    # algorithm.discount_factor = 0
    # algorithm.do_ga_subsumption = True
    # algorithm.do_action_set_subsumption = True

    # model = algorithm.new_model(scenario) #classifier set
    # model = algorithm.run(scenario)
    

    # scenario = UnclassifiedData(X)
    # scenario.possible_actions = model.possible_actions
    # model = algorithm.run(scenario)
    # predictions = np.array(scenario.get_classifications())

    # print("Accuracy " + str(Accuracy.evaluate(predictions, y)))

    """ this block of code formats the benchmark files into csv files """
    benchmark_paths = []
    for folder in os.listdir(get_benchmark_folder()):
        path = os.path.join(get_benchmark_folder(), folder)
        for file in os.listdir(path):
            benchmark_paths.append(os.path.join(get_benchmark_folder(), folder, file))

    for benchmark_path in benchmark_paths:
        benchmark = read_pickle(benchmark_path)
        benchmark_formatted = format_benchmark(benchmark)

    """ merge best results """
    # for folder in os.listdir(get_formatted_folder()):
    #     path = os.path.join(get_formatted_folder(), folder)
    #     merge_best_results(path)

    """ this block of code generates the results automatically""" 
    # for folder in os.listdir(get_formatted_folder()):
    #     path = os.path.join(get_formatted_folder(), folder)
    #     extract_results(path)
    """
    Para o r_bio, as runs com problemas são: 2, 16, 24 e 27. 

    Para o r_ppb, as runs com problemas são: 16.

    Para o r_student, as runs com problemas são: 0 e 16 (para o Boosting Mean + FLR, pelo menos) e 8 e 24.
    
    #####
    
    marta_c_credit_slm__2019_01_16__20_11_08
    marta_c_credit_mlp__2019_01_16__20_11_08
    
    marta_c_diabetes_slm__2019_01_15__22_30_24
    marta_c_diabetes_mlp__2019_01_15__22_30_24
    
    marta_r_bio_slm__2019_01_15__22_33_01
    marta_r_bio_mlp__2019_01_15__22_33_01
    
    marta_r_ppb_slm__2019_01_16__20_11_04
    marta_r_ppb_mlp__2019_01_16__20_11_04
    
    marta_r_student_slm__2019_01_15__22_30_14
    marta_r_student_mlp__2019_01_15__22_30_14
    
    """
    # pickup_b("r_bio", "marta_r_bio_slm__2019_01_15__22_33_01.pkl")
    
    # pickup_b("r_bio", "marta_r_bio_mlp__2019_01_15__22_33_01.pkl")
    # pickup_b("r_ppb", "marta_r_ppb_mlp__2019_01_16__20_11_04.pkl")
    # pickup_b("r_student", "marta_r_student_mlp__2019_01_15__22_30_14.pkl")
    
    #start_b("r_student", models_to_run='SLM_MODELS', models_to_run_2=None, ensembles=None)
    #start_b("r_ppb", models_to_run='SLM_MODELS', models_to_run_2=None, ensembles=None)
    #start_b("r_bio", models_to_run='SLM_MODELS', models_to_run_2=None, ensembles=None)
    #start_b("c_diabetes", models_to_run='SLM_MODELS', models_to_run_2=None, ensembles=None)
    #start_b("c_credit", models_to_run='SLM_MODELS', models_to_run_2=None, ensembles=None)
    
    #start_b("r_student", models_to_run_2='MLP_MODELS', models_to_run=None)
    #start_b("r_ppb", models_to_run_2='MLP_MODELS', models_to_run=None)
    #start_b("r_bio", models_to_run_2='MLP_MODELS', models_to_run=None)
    #start_b("c_diabetes", models_to_run_2='MLP_MODELS', models_to_run=None)
    #start_b("c_credit", models_to_run_2='MLP_MODELS', models_to_run=None)
    
    # start_b("r_bio", models_to_run_2='MLP_MODELS', models_to_run=None)
    
    #pickup_b("r_bio", "marta_r_bio_slm__2019_01_15__22_33_01.pkl")
    #pickup_b("r_ppb", "marta_r_ppb_slm__2019_01_16__20_11_04.pkl")
    #pickup_b("r_student", "marta_r_student_slm__2019_01_15__22_30_14.pkl")
    #pickup_b("c_credit", "marta_c_credit_slm__2019_01_16__20_11_08.pkl")
    #pickup_b("c_diabetes", "marta_c_diabetes_slm__2019_01_15__22_30_24.pkl")
    
    #pickup_b("r_bio", "marta_r_bio_mlp__2019_01_15__22_33_01.pkl")
    #pickup_b("r_ppb", "marta_r_ppb_mlp__2019_01_16__20_11_04.pkl")
    #pickup_b("r_student", "marta_r_student_mlp__2019_01_15__22_30_14.pkl")
    #pickup_b("c_credit", "marta_c_credit_mlp__2019_01_16__20_11_08.pkl")
    #pickup_b("c_diabetes", "marta_c_diabetes_mlp__2019_01_15__22_30_24.pkl")
    
    """
    data_set_name = argv[1]
    if len(argv) > 2:
        models_to_run = argv[2]
    else:
        models_to_run='SLM_MODELS'
    
    print('Dataset:', data_set_name, ', models:', models_to_run)
    start_b(data_set_name, models_to_run=models_to_run, models_to_run_2=None)
    """

    # for data_set in os.listdir(get_standardized_folder()):
    #     start_b(remove_extension(data_set))
    
    # start_b("c_credit", models_to_run='SLM_MODELS', models_to_run_2=None)
    # start_b("c_diabetes", models_to_run='SLM_MODELS', models_to_run_2=None)
    # start_b("r_bio", models_to_run='SLM_MODELS', models_to_run_2=None)
    #start_b("r_ppb", models_to_run='SLM_MODELS', models_to_run_2=None)
    # start_b("r_student", models_to_run='SLM_MODELS', models_to_run_2=None)

    # start_b("r_ld50")

    # -IG- probably the following code can be removed <start>
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
    # -IG- probably the following code can be removed <end>

from benchmark.pmlb_benchmarker import Benchmarker, continue_benchmark, pickup_benchmark, SLM_MODELS, MLP_MODELS_SGD_ADAM, ENSEMBLES
from pmlb import classification_dataset_names
from benchmark.training_benchmarker import TrainingBenchmarker
from utils.environment_constants import SELECTED_DATASETS
from data.extract import get_target_variable
from data.io_plm import load_pmlb_samples

MODELS_TO_RUN = {'slm': SLM_MODELS, 'mlp': MLP_MODELS_SGD_ADAM}
test = {'mlp' : MLP_MODELS_SGD_ADAM}

def continue_b(data_set_name, file_name):
    """ continues benchmark """
    continue_benchmark(data_set_name, file_name)

def pickup_b(data_set_name, file_name):
    """continues benchmark after parameter tuning""" 
    pickup_benchmark(data_set_name, file_name)

def start_b(data_set_name, models=None, ensembles=ENSEMBLES, benchmark_id=None):
    benchmarker = Benchmarker(data_set_name, models=models, ensembles=ensembles, benchmark_id=benchmark_id)
    benchmarker.run_nested_cv()

if __name__ == '__main__':
    for classification_dataset in SELECTED_DATASETS:
        # samples = load_pmlb_samples(classification_dataset)
        # target_var = get_target_variable(samples).values
        # print("\n"+classification_dataset)
        # print(target_var)
        # print("\n")
        for model_to_run in test: 
            start_b(classification_dataset, models=MODELS_TO_RUN[model_to_run], ensembles=None, benchmark_id=model_to_run)
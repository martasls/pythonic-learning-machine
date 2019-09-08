
from benchmark.pmlb_benchmarker import Benchmarker, continue_benchmark, pickup_benchmark, SLM_MODELS, MLP_MODELS_SGD_ADAM, ENSEMBLES
from pmlb import classification_dataset_names
from benchmark.training_benchmarker import TrainingBenchmarker

models_to_run = {'slm': SLM_MODELS,
            'mlp': MLP_MODELS_SGD_ADAM}

selected_data_sets = []

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
    
    for classification_dataset in classification_dataset_names: 
        for model_to_run in models_to_run: 
            start_b(classification_dataset, models=models_to_run[model_to_run], ensembles=None, benchmark_id=model_to_run)

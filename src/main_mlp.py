from benchmark.benchmarker import Benchmarker, continue_benchmark, pickup_benchmark, MLP_MODELS, ENSEMBLES, MLP_MODELS_SGD_ADAM

def start_mlp(data_set_name, ensembles=ENSEMBLES, benchmark_id='mlp'):
    # benchmarker = Benchmarker(data_set_name, models=MLP_MODELS, ensembles=ensembles, benchmark_id=benchmark_id)
    benchmarker = Benchmarker(data_set_name, models=MLP_MODELS_SGD_ADAM, ensembles=ensembles, benchmark_id=benchmark_id)
    benchmarker.run_nested_cv()

def continue_b(data_set_name, file_name):
    """ continues benchmark """
    continue_benchmark(data_set_name, file_name)


def pickup_b(data_set_name, file_name):
    """continues benchmark after parameter tuning""" 
    pickup_benchmark(data_set_name, file_name)

if __name__ == '__main__':
    
    #start_mlp("r_student", ensembles=None, benchmark_id='mlp-sgd-adam')
    #start_mlp("r_ppb", ensembles=None, benchmark_id='mlp-sgd-adam')
    #start_mlp("r_bio", ensembles=None, benchmark_id='mlp-sgd-adam')
    
    #start_mlp("r_music", ensembles=None, benchmark_id='mlp-sgd-adam')
    #start_mlp("r_parkinsons", ensembles=None, benchmark_id='mlp-sgd-adam')
    
    #start_mlp("r_concrete", ensembles=None, benchmark_id='mlp-sgd-adam')
    
    #start_mlp("c_diabetes", ensembles=None, benchmark_id='mlp-sgd-adam')
    start_mlp("c_credit", ensembles=None, benchmark_id='mlp-sgd-adam')
    
    #start_mlp("c_sonar", ensembles=None, benchmark_id='mlp-sgd-adam')
    #start_mlp("c_cancer", ensembles=None, benchmark_id='mlp-sgd-adam')
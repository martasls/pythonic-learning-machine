from benchmark.benchmarker import Benchmarker, continue_benchmark, pickup_benchmark, SLM_MODELS, MLP_MODELS, ENSEMBLES, \
    MLP_MODELS_SGD_ADAM


def start_b(data_set_name, file_name=None, models_to_run=None, models_to_run_2=MLP_MODELS, ensembles=ENSEMBLES):
    """ starts benchmark """
    if models_to_run == 'SLM_MODELS':
        models_to_run = SLM_MODELS
    if models_to_run == 'MLP_MODELS':
        models_to_run = MLP_MODELS
    if models_to_run_2 == 'SLM_MODELS':
        models_to_run_2 = SLM_MODELS
    if models_to_run_2 == 'MLP_MODELS':
        models_to_run_2 = MLP_MODELS
    
    # SLM MODELS 
    if models_to_run is not None:
        benchmarker = Benchmarker(data_set_name, models=models_to_run, ensembles=ensembles, benchmark_id='slm')
        benchmarker.run_nested_cv()
    
    # MLP MODELS 
    if models_to_run_2 is not None: 
        benchmarker = Benchmarker(data_set_name, models=models_to_run_2, ensembles=ensembles, benchmark_id='mlp')
        benchmarker.run_nested_cv()


def continue_b(data_set_name, file_name):
    """ continues benchmark """
    continue_benchmark(data_set_name, file_name)


def pickup_b(data_set_name, file_name):
    """continues benchmark after parameter tuning""" 
    pickup_benchmark(data_set_name, file_name)


def start_mlp(data_set_name, ensembles=ENSEMBLES, benchmark_id='mlp'):
    # benchmarker = Benchmarker(data_set_name, models=MLP_MODELS, ensembles=ensembles, benchmark_id=benchmark_id)
    benchmarker = Benchmarker(data_set_name, models=MLP_MODELS_SGD_ADAM, ensembles=ensembles, benchmark_id=benchmark_id)
    benchmarker.run_nested_cv()


if __name__ == '__main__':
    
    #start_mlp("r_student", ensembles=None, benchmark_id='mlp-sgd-adam')
    #start_mlp("r_ppb", ensembles=None, benchmark_id='mlp-sgd-adam')
    #start_mlp("r_bio", ensembles=None, benchmark_id='mlp-sgd-adam')
    
    #start_mlp("r_music", ensembles=None, benchmark_id='mlp-sgd-adam')
    #start_mlp("r_parkinsons", ensembles=None, benchmark_id='mlp-sgd-adam')
    
    #start_mlp("r_concrete", ensembles=None, benchmark_id='mlp-sgd-adam')
    
    #start_mlp("c_diabetes", ensembles=None, benchmark_id='mlp-sgd-adam')
    #start_mlp("c_credit", ensembles=None, benchmark_id='mlp-sgd-adam')
    
    #start_mlp("c_sonar", ensembles=None, benchmark_id='mlp-sgd-adam')
    #start_mlp("c_cancer", ensembles=None, benchmark_id='mlp-sgd-adam')
    
    """
    # start_b("r_student", models_to_run_2='MLP_MODELS', models_to_run=None, ensembles=None)
    # start_b("r_ppb", models_to_run_2='MLP_MODELS', models_to_run=None, ensembles=None)
    # start_b("r_bio", models_to_run_2='MLP_MODELS', models_to_run=None, ensembles=None)
    # start_b("c_diabetes", models_to_run_2='MLP_MODELS', models_to_run=None, ensembles=None)
    # start_b("c_credit", models_to_run_2='MLP_MODELS', models_to_run=None, ensembles=None)
    
    start_b("r_music", models_to_run_2='MLP_MODELS', models_to_run=None, ensembles=None)
    start_b("r_parkinsons", models_to_run_2='MLP_MODELS', models_to_run=None, ensembles=None)
    start_b("c_sonar", models_to_run_2='MLP_MODELS', models_to_run=None, ensembles=None)
    start_b("c_cancer", models_to_run_2='MLP_MODELS', models_to_run=None, ensembles=None)
    start_b("r_concrete", models_to_run_2='MLP_MODELS', models_to_run=None, ensembles=None)
    """

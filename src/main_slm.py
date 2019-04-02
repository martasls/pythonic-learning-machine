from benchmark.benchmarker import Benchmarker, continue_benchmark, pickup_benchmark, SLM_MODELS_NO_SSC, ENSEMBLES


def start_b(data_set_name, file_name=None, models_to_run=None, models_to_run_2=None, ensembles=ENSEMBLES):
    """ starts benchmark """
    if models_to_run == 'SLM_MODELS_NO_SSC':
        models_to_run = SLM_MODELS_NO_SSC
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


if __name__ == '__main__':
    
    # start_b("r_student", models_to_run='SLM_MODELS_NO_SSC', models_to_run_2=None, ensembles=None)
    # start_b("r_ppb", models_to_run='SLM_MODELS', models_to_run_2=None, ensembles=None)
    # start_b("r_bio", models_to_run='SLM_MODELS', models_to_run_2=None, ensembles=None)
    start_b("c_diabetes", models_to_run='SLM_MODELS_NO_SSC', models_to_run_2=None, ensembles=None)
    start_b("c_credit", models_to_run='SLM_MODELS', models_to_run_2=None, ensembles=None)
    
    start_b("r_music", models_to_run='SLM_MODELS', models_to_run_2=None, ensembles=None)
    start_b("r_parkinsons", models_to_run='SLM_MODELS', models_to_run_2=None, ensembles=None)
    start_b("c_sonar", models_to_run='SLM_MODELS', models_to_run_2=None, ensembles=None)
    start_b("c_cancer", models_to_run='SLM_MODELS', models_to_run_2=None, ensembles=None)
    start_b("r_concrete", models_to_run='SLM_MODELS', models_to_run_2=None, ensembles=None)

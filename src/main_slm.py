from benchmark.benchmarker import Benchmarker, continue_benchmark, pickup_benchmark, SLM_MODELS, MLP_MODELS, ENSEMBLES

def continue_b(data_set_name, file_name):
    """ continues benchmark """
    continue_benchmark(data_set_name, file_name)


def pickup_b(data_set_name, file_name):
    """continues benchmark after parameter tuning""" 
    pickup_benchmark(data_set_name, file_name)

def start_slm(data_set_name, ensembles=ENSEMBLES, benchmark_id='slm'):
    benchmarker = Benchmarker(data_set_name, models=SLM_MODELS, ensembles=ensembles, benchmark_id=benchmark_id)
    benchmarker.run_nested_cv()


if __name__ == '__main__':
    
    start_slm("c_credit", ensembles=None)

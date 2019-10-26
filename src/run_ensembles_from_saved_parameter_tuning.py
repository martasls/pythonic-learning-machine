#from benchmark.benchmarker import pickup_benchmark
from benchmark.pmlb_benchmarker import pickup_benchmark

def run_ensembles(dataset_name, benchmark_file_name):
    pickup_benchmark(dataset_name, benchmark_file_name)


if __name__ == '__main__':
    
    run_ensembles('agaricus-lepiota', 'c_agaricus-lepiota_mlp__2019_09_17__19_56_46.pkl')
    run_ensembles('agaricus-lepiota', 'c_agaricus-lepiota_slm__2019_09_17__19_56_46.pkl')
    
    run_ensembles('backache', 'c_backache_mlp__2019_09_17__19_56_46.pkl')
    run_ensembles('backache', 'c_backache_slm__2019_09_17__19_56_46.pkl')

    run_ensembles('breast-cancer-wisconsin', 'c_breast-cancer-wisconsin_mlp__2019_09_19__17_42_05.pkl')
    run_ensembles('breast-cancer-wisconsin', 'c_breast-cancer-wisconsin_slm__2019_09_19__17_42_05.pkl')
    
    run_ensembles('clean1', 'c_clean1_mlp__2019_09_17__19_56_46.pkl')
    run_ensembles('clean1', 'c_clean1_slm__2019_09_17__19_56_46.pkl')
    
    run_ensembles('clean2', 'c_clean2_mlp__2019_09_17__19_56_46.pkl')
    run_ensembles('clean2', 'c_clean2_slm__2019_09_17__19_56_46.pkl')
    
    run_ensembles('coil2000', 'c_coil2000_mlp__2019_09_17__19_56_46.pkl')
    run_ensembles('coil2000', 'c_coil2000_slm__2019_09_17__19_56_46.pkl')

    run_ensembles('credit-g', 'c_credit-g_mlp__2019_09_17__19_56_46.pkl')
    run_ensembles('credit-g', 'c_credit-g_slm__2019_09_17__19_56_46.pkl')

    run_ensembles('diabetes', 'c_diabetes_mlp__2019_09_19__17_42_05.pkl')
    run_ensembles('diabetes', 'c_diabetes_slm__2019_09_19__17_42_05.pkl')

    run_ensembles('dis', 'c_dis_mlp__2019_09_17__19_56_46.pkl')
    run_ensembles('dis', 'c_dis_slm__2019_09_17__19_56_46.pkl')

    run_ensembles('Hill_Valley_with_noise', 'c_Hill_Valley_with_noise_mlp__2019_09_17__19_56_46.pkl')
    run_ensembles('Hill_Valley_with_noise', 'c_Hill_Valley_with_noise_slm__2019_09_17__19_56_46.pkl')

    run_ensembles('Hill_Valley_without_noise', 'c_Hill_Valley_without_noise_mlp__2019_09_17__19_56_46.pkl')
    run_ensembles('Hill_Valley_without_noise', 'c_Hill_Valley_without_noise_slm__2019_09_17__19_56_46.pkl')

    run_ensembles('hypothyroid', 'c_hypothyroid_mlp__2019_09_17__19_56_46.pkl')
    run_ensembles('hypothyroid', 'c_hypothyroid_slm__2019_09_17__19_56_46.pkl')

    run_ensembles('ionosphere', 'c_ionosphere_mlp__2019_09_17__19_56_46.pkl')
    run_ensembles('ionosphere', 'c_ionosphere_slm__2019_09_17__19_56_46.pkl')

    run_ensembles('kr-vs-kp', 'c_kr-vs-kp_mlp__2019_09_17__19_56_46.pkl')
    run_ensembles('kr-vs-kp', 'c_kr-vs-kp_slm__2019_09_17__19_56_46.pkl')

    run_ensembles('molecular-biology_promoters', 'c_molecular-biology_promoters_mlp__2019_09_17__19_56_46.pkl')
    run_ensembles('molecular-biology_promoters', 'c_molecular-biology_promoters_slm__2019_09_17__19_56_46.pkl')

    run_ensembles('sonar', 'c_sonar_mlp__2019_09_17__19_56_46.pkl')
    run_ensembles('sonar', 'c_sonar_slm__2019_09_17__19_56_46.pkl')

    run_ensembles('spambase', 'c_spambase_mlp__2019_09_17__19_56_46.pkl')
    run_ensembles('spambase', 'c_spambase_slm__2019_09_17__19_56_46.pkl')

    run_ensembles('spectf', 'c_spectf_mlp__2019_09_17__19_56_46.pkl')
    run_ensembles('spectf', 'c_spectf_slm__2019_09_17__19_56_46.pkl')

    run_ensembles('tokyo1', 'c_tokyo1_mlp__2019_09_17__19_56_46.pkl')
    run_ensembles('tokyo1', 'c_tokyo1_slm__2019_09_17__19_56_46.pkl')


    



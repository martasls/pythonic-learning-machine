from benchmark.benchmarker import pickup_benchmark


def run_ensembles(dataset_name, benchmark_file_name):
    pickup_benchmark(dataset_name, benchmark_file_name)


if __name__ == '__main__':
    
    # run_ensembles('r_concrete', 'r_concrete_slm__2019_02_05__06_13_44.pkl')
    run_ensembles('r_concrete', 'r_concrete_mlp-sgd-adam__2019_02_05__21_14_31.pkl')
    #  run_ensembles('r_concrete', 'r_concrete_mlp__2019_02_05__19_26_39.pkl')
    
    # run_ensembles('c_cancer', 'c_cancer_slm__2019_02_05__06_13_44.pkl')
    run_ensembles('c_cancer', 'c_cancer_mlp-sgd-adam__2019_02_05__22_12_15.pkl')
    #  run_ensembles('c_cancer', 'c_cancer_mlp__2019_02_05__19_12_26.pkl')
    
    # run_ensembles('c_sonar', 'c_sonar_slm__2019_02_05__06_13_44.pkl')
    run_ensembles('c_sonar', 'c_sonar_mlp-sgd-adam__2019_02_05__22_12_15.pkl')
    #  run_ensembles('c_sonar', 'c_sonar_mlp__2019_02_05__19_12_26.pkl')
    
    # run_ensembles('r_parkinsons', 'r_parkinsons_slm__2019_02_05__06_13_44.pkl')
    run_ensembles('r_parkinsons', 'r_parkinsons_mlp-sgd-adam__2019_02_05__21_14_31.pkl')
    #  run_ensembles('r_parkinsons', 'r_parkinsons_mlp__2019_02_05__18_43_45.pkl')
    
    # run_ensembles('r_music', 'r_music_slm__2019_02_05__06_13_44.pkl')
    run_ensembles('r_music', 'r_music_mlp-sgd-adam__2019_02_05__21_14_31.pkl')
    #  run_ensembles('r_music', 'r_music_mlp__2019_02_05__18_43_45.pkl')
    
    # run_ensembles('c_credit', 'c_credit_slm__2019_02_05__06_13_44.pkl')
    run_ensembles('c_credit', 'c_credit_mlp-sgd-adam__2019_02_05__22_12_15.pkl')
    #  run_ensembles('c_credit', 'c_credit_mlp__2019_02_05__18_10_54.pkl')
    
    # run_ensembles('c_diabetes', 'c_diabetes_slm__2019_02_05__06_13_44.pkl')
    run_ensembles('c_diabetes', 'c_diabetes_mlp-sgd-adam__2019_02_05__22_12_15.pkl')
    #  run_ensembles('c_diabetes', 'c_diabetes_mlp__2019_02_05__18_10_54.pkl')
    
    # run_ensembles('r_bio', 'r_bio_slm__2019_02_05__06_13_44.pkl')
    run_ensembles('r_bio', 'r_bio_mlp-sgd-adam__2019_02_05__21_14_31.pkl')
    #  run_ensembles('r_bio', 'r_bio_mlp__2019_02_05__06_13_51.pkl')
    
    # run_ensembles('r_ppb', 'r_ppb_slm__2019_02_05__06_13_44.pkl')
    run_ensembles('r_ppb', 'r_ppb_mlp-sgd-adam__2019_02_05__21_14_31.pkl')
    #  run_ensembles('r_ppb', 'r_ppb_mlp__2019_02_05__06_13_51.pkl')
    
    # run_ensembles('r_student', 'r_student_slm__2019_02_05__06_13_44.pkl')
    run_ensembles('r_student', 'r_student_mlp-sgd-adam__2019_02_05__21_14_31.pkl')
    #  run_ensembles('r_student', 'r_student_mlp__2019_02_05__06_13_51.pkl')]  
    
    """ not updated: SLM """
    # run_ensembles("c_credit", "c_credit_slm__2019_01_27__23_03_36.pkl")
    # run_ensembles("c_diabetes", "c_diabetes_slm__2019_01_27__23_03_36.pkl")
    # run_ensembles("r_bio", "r_bio_slm__2019_01_27__23_03_36.pkl")
    # run_ensembles("r_ppb", "r_ppb_slm__2019_01_27__23_03_36.pkl")
    # run_ensembles("r_student", "r_student_slm__2019_01_27__23_03_36.pkl")
    
    """ not updated: MLP """
    # run_ensembles("c_credit", "c_credit_mlp__2019_01_29__13_57_03.pkl")
    # run_ensembles("c_diabetes", "c_diabetes_mlp__2019_01_29__13_57_03.pkl")
    # run_ensembles("r_bio", "r_bio_mlp__2019_01_27__23_03_38.pkl")
    # run_ensembles("r_ppb", "r_ppb_mlp__2019_01_27__23_03_38.pkl")
    # run_ensembles("r_student", "r_student_mlp__2019_01_27__23_03_38.pkl")

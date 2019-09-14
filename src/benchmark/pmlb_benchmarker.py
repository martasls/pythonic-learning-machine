import datetime
from random import shuffle, uniform, randint

from numpy import mean
from sklearn.model_selection import KFold, StratifiedKFold

from algorithms.common.metric import RootMeanSquaredError, is_better, Accuracy, AUROC
from benchmark.configuration import generate_random_slm_bls_tie_edv_configuration, generate_random_slm_ols_configuration, \
    generate_random_slm_bls_configuration, generate_random_slm_ols_edv_configuration, get_config_simple_bagging_ensemble, \
    get_config_riw_ensemble, get_config_boosting_ensemble, generate_random_adam_configuration, generate_random_sgd_configuration
from benchmark.evaluator import EvaluatorSLM, EvaluatorNEAT, EvaluatorFTNE, EvaluatorSLM_RST, EvaluatorSLM_RWT, \
    EvaluatorEnsemble, \
    EvaluatorEnsembleBagging, EvaluatorEnsembleRandomIndependentWeighting, EvaluatorEnsembleBoosting, \
    EvaluatorMLPC, EvaluatorMLPR
from benchmark.formatter import _format_static_table
from data.extract import is_classification, get_input_variables, get_target_variable
from data.io_plm import load_samples, load_samples_no_val, benchmark_to_pickle, benchmark_from_pickle, load_standardized_samples,  \
    load_pmlb_samples
import pandas as pd
from pmlb import fetch_data

# Returns the current date and time.
_now = datetime.datetime.now()


_SLM_MAX_COMBINATIONS_PER_VARIANT = 1
_MLP_MAX_COMBINATIONS_PER_VARIANT = 1

_OUTER_FOLDS = 2 # changed from 30
_INNER_FOLDS = 2

#_MAX_COMBINATIONS = 50
#_MAX_COMBINATIONS_SLM_OLS_EDV = 5


#_MAX_COMBINATIONS_ADAM = 52
#_MAX_COMBINATIONS_SGD = 52

#==============================================================================
# SLM_MODELS_NO_SSC = {

#     'slm_ols_group': {
#         'name_long': 'Semantic Learning Machine (Optimized Learning Step) Group',
#         'name_short': 'SLM (OLS), SLM (OLS) + RST, SLM (OLS) + RWT',
#         'algorithms': [EvaluatorSLM],
#         'configuration_method': get_random_config_slm_ols_grouped,
#         'max_combinations': _SLM_MAX_COMBINATIONS_PER_VARIANT},
    
#     'slm_fls_group': {
#         'name_long': 'Semantic Learning Machine (Fixed Learning Step) Group',
#         'name_short': 'SLM (FLS), SLM (FLS) + RST, SLM (FLS) + RWT',
#         'algorithms':  [EvaluatorSLM],
#         'configuration_method': get_random_config_slm_fls_grouped,
#         'max_combinations': _SLM_MAX_COMBINATIONS_PER_VARIANT},
# }
#===============================================================================

# Default models to be compared.
#===============================================================================
SLM_MODELS = {

    'slm_bls_tie_edv_group': {
        'name_long': 'Semantic Learning Machine (Bounded Learning Step) Stopping Criteria Group',
        'name_short': 'SLM (BLS) + TIE, SLM (BLS) + EDV',
        'algorithms': [EvaluatorSLM],
        'configuration_method': generate_random_slm_bls_tie_edv_configuration, 
        'max_combinations': _SLM_MAX_COMBINATIONS_PER_VARIANT},

    'slm_ols_group': {
        'name_long': 'Semantic Learning Machine (Optimized Learning Step) Group',
        'name_short': 'SLM (OLS), SLM (OLS) + RST, SLM (OLS) + RWT',
        'algorithms': [EvaluatorSLM],
        'configuration_method': generate_random_slm_ols_configuration,
        'max_combinations': _SLM_MAX_COMBINATIONS_PER_VARIANT},
    
    'slm_bls_group': {
        'name_long': 'Semantic Learning Machine (Bounded Learning Step) Group',
        'name_short': 'SLM (BLS), SLM (BLS) + RST, SLM (BLS) + RWT',
        'algorithms':  [EvaluatorSLM],
        'configuration_method': generate_random_slm_bls_configuration, 
        'max_combinations': _SLM_MAX_COMBINATIONS_PER_VARIANT},

    'slm_ols_edv': {
        'name_long': 'Semantic Learning Machine (Optimized Learning Step) + Error Deviation Variation Criterion',
        'name_short': 'SLM (OLS) + EDV',
        'algorithms': [EvaluatorSLM],
        'configuration_method': generate_random_slm_ols_edv_configuration,
        'max_combinations': _SLM_MAX_COMBINATIONS_PER_VARIANT},
    
    #===========================================================================
    # 'slm_fls_group': {
    #     'name_long': 'Semantic Learning Machine (Fixed Learning Step) Group',
    #     'name_short': 'SLM (FLS), SLM (FLS) + RST, SLM (FLS) + RWT',
    #     'algorithms':  [EvaluatorSLM, EvaluatorSLM_RST, EvaluatorSLM_RWT],
    #     'configuration_method': get_random_config_slm_fls_grouped,
    #     'max_combinations': _MAX_COMBINATIONS},
    # 'slm_ols_group': {
    #     'name_long': 'Semantic Learning Machine (Optimized Learning Step) Group',
    #     'name_short': 'SLM (OLS), SLM (OLS) + RST, SLM (OLS) + RWT',
    #     'algorithms': [EvaluatorSLM, EvaluatorSLM_RST, EvaluatorSLM_RWT],
    #     'configuration_method': get_random_config_slm_ols_grouped,
    #     'max_combinations': _MAX_COMBINATIONS},
    # 'slm_fls_tie_edv_group': {
    #     'name_long': 'Semantic Learning Machine (Fixed Learning Step) Stopping Criteria Group',
    #     'name_short': 'SLM (FLS) + TIE, SLM (FLS) + EDV',
    #     'algorithms': [EvaluatorSLM],
    #     'configuration_method': get_random_config_slm_fls_tie_edv,
    #     'max_combinations': _MAX_COMBINATIONS},
    # 'slm_ols_edv': {
    #     'name_long': 'Semantic Learning Machine (Optimized Learning Step) + Error Deviation Variation Crit',
    #     'name_short': 'SLM (OLS) + EDV',
    #     'algorithms': [EvaluatorSLM],
    #     'configuration_method': get_random_config_slm_ols_edv,
    #     'max_combinations': _MAX_COMBINATIONS},
    #===========================================================================
    
    # 'slm_ensemble': {
    #     'name_long': 'Semantic Learning Machine Ensemble',
    #     'name_short': 'SLM (Ensemble)',
    #     'algorithms': EvaluatorEnsemble,
    #     'configurations': ENSEMBLE_CONFIGURATIONS},
    # 'slm_fls_ensemble':{
    #     'name_long': 'Semantic Learning Machine(FLS) Ensemble',
    #     'name_short': 'SLM-FLS (Ensemble)',
    #     'algorithms': EvaluatorEnsemble,
    #     'configurations': ENSEMBLE_FLS_CONFIGURATIONS},
    # 'slm_ensemble_rst': {
    #     'name_long': 'Semantic Learning Machine Ensemble with Random Sampling Technique',
    #     'name_short': 'SLM (Ensemble) + RST',
    #     'algorithms': EvaluatorEnsemble,
    #     'configurations': ENSEMBLE_RST_CONFIGURATIONS},
    # 'slm_ensemble_rwt': {
    #     'name_long': 'Semantic Learning Machine Ensemble with Random Weighting Technique',
    #     'name_short': 'SLM (Ensemble) + RWT',
    #     'algorithms': EvaluatorEnsemble,
    #     'configurations': ENSEMBLE_RWT_CONFIGURATIONS},    
    # 'slm_ensemble_bagging': {
    #     'name_long': 'Semantic Learning Machine Ensemble with Bagging',
    #     'name_short': 'SLM (Ensemble-Bagging)',
    #     'algorithms': EvaluatorEnsembleBagging,
    #     'configurations': ENSEMBLE_BAGGING_CONFIGURATIONS}, 
    # 'slm_fls_ensemble_bagging': {
    #     'name_long': 'Semantic Learning Machine (FLS) Ensemble with Bagging',
    #     'name_short': 'SLM-FLS (Ensemble-Bagging)',
    #     'algorithms': EvaluatorEnsembleBagging,
    #     'configurations': ENSEMBLE_BAGGING_FLS_CONFIGURATIONS}, 
    # 'slm_ensemble_bagging_rst': {
    #     'name_long': 'Semantic Learning Machine Ensemble with Bagging and Random Sampling Technique',
    #     'name_short': 'SLM (Ensemble-Bagging) + RST',
    #     'algorithms': EvaluatorEnsembleBagging,
    #     'configurations': ENSEMBLE_BAGGING_RST_CONFIGURATIONS},
    # 'slm_ensemble_bagging_rwt': {
    #     'name_long': 'Semantic Learning Machine Ensemble with Bagging and Random Weighting Technique',
    #     'name_short': 'SLM (Ensemble-Bagging) + RWT',
    #     'algorithms': EvaluatorEnsembleBagging,
    #     'configurations': ENSEMBLE_BAGGING_RWT_CONFIGURATIONS},   
    # 'slm_random_independent_weighting': {
    #     'name_long': 'Semantic Learning Machine Ensemble with Random Independent Weighting',
    #     'name_short': 'SLM (Ensemble-RIW)',
    #     'algorithms': EvaluatorEnsembleRandomIndependentWeighting,
    #     'configurations': ENSEMBLE_RANDOM_INDEPENDENT_WEIGHTING_CONFIGURATIONS}, 
    # 'slm_fls_random_independent_weighting': {
    #     'name_long': 'Semantic Learning Machine (FLS) Ensemble with Random Independent Weighting',
    #     'name_short': 'SLM-FLS (Ensemble-RIW)',
    #     'algorithms': EvaluatorEnsembleRandomIndependentWeighting,
    #     'configurations': ENSEMBLE_RANDOM_INDEPENDENT_WEIGHTING_FLS_CONFIGURATIONS},
    # 'slm_ensemble_boosting': {
    #     'name_long': 'Semantic Learning Machine Ensemble with Boosting',
    #     'name_short': 'SLM (Ensemble-Boosting)',
    #     'algorithms': EvaluatorEnsembleBoosting,
    #     'configurations': ENSEMBLE_BOOSTING_CONFIGURATIONS},  
    # 'slm_fls_ensemble_boosting': {
    #     'name_long': 'Semantic Learning Machine (FLS) Ensemble with Boosting',
    #     'name_short': 'SLM-FLS (Ensemble-Boosting)',
    #     'algorithms': EvaluatorEnsembleBoosting,
    #     'configurations': ENSEMBLE_BOOSTING_FLS_CONFIGURATIONS},
}
#===============================================================================

MLP_MODELS_SGD_ADAM = {
    'mlpc_adam': {
        'name_long': 'Multilayer Perceptron (ADAM Solver)',
        'name_short': 'MLP (ADAM)',
        'algorithms': [EvaluatorMLPC],
        'configuration_method': generate_random_adam_configuration,
        'max_combinations': _MLP_MAX_COMBINATIONS_PER_VARIANT},
    'mlpr_adam': {
        'name_long': 'Multilayer Perceptron (ADAM Solver)',
        'name_short': 'MLP (ADAM)',
        'algorithms': [EvaluatorMLPR],
        'configuration_method': generate_random_adam_configuration,
        'max_combinations': _MLP_MAX_COMBINATIONS_PER_VARIANT},
    'mlpc_sgd': {
        'name_long': 'Multilayer Perceptron (SGD Solver)',
        'name_short': 'MLP (SGD)',
        'algorithms': [EvaluatorMLPC],
        'configuration_method': generate_random_sgd_configuration,
        'max_combinations': _MLP_MAX_COMBINATIONS_PER_VARIANT},
    'mlpr_sgd': {
        'name_long': 'Multilayer Perceptron (SGD Solver)',
        'name_short': 'MLP (SGD)',
        'algorithms': [EvaluatorMLPR],
        'configuration_method': generate_random_sgd_configuration,
        'max_combinations': _MLP_MAX_COMBINATIONS_PER_VARIANT},
}

ENSEMBLES = { 
    'simple': {
        'name_long': 'Simple Ensemble',
        'name_short': 'Simple Ensemble',
        'algorithms': EvaluatorEnsemble,
        'configuration_method': get_config_simple_bagging_ensemble},
    'bagging': {
        'name_long': 'Bagging Ensemble',
        'name_short': 'Bagging Ensemble',
        'algorithms': EvaluatorEnsembleBagging,
        'configuration_method': get_config_simple_bagging_ensemble},
    'riw': { 
        'name_long': 'Random Independent Weighting Ensemble',
        'name_short': 'RIW Ensemble',
        'algorithms': EvaluatorEnsembleRandomIndependentWeighting,
        'configuration_method': get_config_riw_ensemble},
    'boosting_1': {
        'name_long': 'Boosting Ensemble with Median and Fixed Learning Rate',
        'name_short': 'Boosting Ensemble (Median + FLR)',
        'algorithms': EvaluatorEnsembleBoosting,
        'configuration_method': get_config_boosting_ensemble},
    'boosting_2': {
        'name_long': 'Boosting Ensemble with Median and Random Learning Rate',
        'name_short': 'Boosting Ensemble (Median + RLR)',
        'algorithms': EvaluatorEnsembleBoosting,
        'configuration_method': get_config_boosting_ensemble},
    'boosting_3': {
        'name_long': 'Boosting Ensemble with Mean and Fixed Learning Rate',
        'name_short': 'Boosting Ensemble (Mean + FLR)',
        'algorithms': EvaluatorEnsembleBoosting,
        'configuration_method': get_config_boosting_ensemble},
    'boosting_4': {
        'name_long': 'Boosting Ensemble with Mean and Random Learning Rate',
        'name_short': 'Boosting Ensemble (Mean + RLR)',
        'algorithms': EvaluatorEnsembleBoosting,
        'configuration_method': get_config_boosting_ensemble}
}

SIMPLE_ENSEMBLE = {
    'simple': {
        'name_long': 'Simple Ensemble',
        'name_short': 'Simple Ensemble',
        'algorithms': EvaluatorEnsemble,
        'configuration_method': get_config_simple_bagging_ensemble
    },
}


class Benchmarker():

    def __init__(self, dataset_name, learning_metric=None, selection_metric=None, models=None, ensembles=None, benchmark_id=None):
        """Initializes benchmark environment."""
        
        self.benchmark_id = benchmark_id
        self.data_set_name = dataset_name
        # Creates file name as combination of dataset name and and date # this is the benchmark name
        self.file_name = 'c_' + self.data_set_name + "_" + self.benchmark_id + "__" + _now.strftime("%Y_%m_%d__%H_%M_%S")

        self.samples = load_pmlb_samples(dataset_name)
        self.ensembles = ensembles
        self.models = models
        
        # If dataset is classification problem, remove regression models. Else, vice versa.
        if is_classification(self.samples):  # original self.samples[0][0] new self.samples
            self.classification = True
            
            if learning_metric != None:
                self.learning_metric = learning_metric
            else:
                self.learning_metric = RootMeanSquaredError
            
            if selection_metric != None:
                self.selection_metric = selection_metric
            else:
                self.selection_metric = AUROC
            
            if 'mlpr_lbfgs' in self.models.keys():
                del self.models['mlpr_lbfgs']
            if 'mlpr_adam' in self.models.keys():
                del self.models['mlpr_adam']
            if 'mlpr_sgd' in self.models.keys():
                del self.models['mlpr_sgd']
        else:
            self.classification = False
            
            if learning_metric != None:
                self.learning_metric = learning_metric
            else:
                self.learning_metric = RootMeanSquaredError
            
            if selection_metric != None:
                self.selection_metric = selection_metric
            else:
                self.selection_metric = RootMeanSquaredError

            if 'mlpc_lbfgs' in self.models.keys():
                del self.models['mlpc_lbfgs']
            if 'mlpc_adam' in self.models.keys():
                del self.models['mlpc_adam']
            if 'mlpc_sgd' in self.models.keys(): 
                del self.models['mlpc_sgd']
        
        # if models = MLP, remove Random Independent Weighting 
        if self.ensembles != None:
            if 'mlpc_lbfgs' in self.models.keys() or 'mlpr_lbfgs' in self.models.keys(): 
                if 'riw' in self.ensembles.keys(): 
                    del self.ensembles['riw']
        
        # Create results dictionary with models under study.
        self.results = {k: [None for i in range(_OUTER_FOLDS)] for k in self.models.keys()}
        
        if self.ensembles != None:
            self.results_ensemble = {ensemble: [None for i in range(_OUTER_FOLDS)] for ensemble in self.ensembles.keys()}
        
        self.best_result = [None for i in range(_OUTER_FOLDS)]
        
        # Serialize benchmark environment.
        benchmark_to_pickle(self)


    def _evaluate_algorithm(self, algorithm, configurations, training_set, validation_set, testing_set, metric):
        """Creates evaluator, based on algorithms and configurations."""
        evaluator = algorithm(configurations, training_set, validation_set, testing_set, metric)
        return evaluator.run_nested_cv()

    def get_dataset_size(self, dataset): 
        return dataset.shape[0]

    def _get_inner_folds(self, outer_iteration):
        if self.classification:
            return StratifiedKFold(n_splits=_INNER_FOLDS, random_state=outer_iteration, shuffle=True)
        return KFold(n_splits=_INNER_FOLDS, random_state=outer_iteration, shuffle=True)

    def _get_outer_folds(self, outer_iteration):
        if self.classification:
            return StratifiedKFold(n_splits=_OUTER_FOLDS, random_state=outer_iteration, shuffle=True)
        return KFold(n_splits=_OUTER_FOLDS, random_state=outer_iteration, shuffle=True)

    def run_nested_cv(self):
        """ runs benchmark study on a nested cross-validation environment """
        
        #=======================================================================
        # print('self.learning_metric =', self.learning_metric)
        # print('self.selection_metric =', self.selection_metric)
        #=======================================================================
        
        print('Entering run_nested_cv for dataset:', self.data_set_name)
    
        
        outer_cv = 0
        outer_folds = self._get_outer_folds(outer_cv)
        for training_outer_index, testing_index in outer_folds.split(get_input_variables(self.samples).values, get_target_variable(self.samples).values):
            
            print('\n\tIndex of outer fold:', outer_cv)
            
            training_outer, testing = pd.DataFrame(self.samples.values[training_outer_index]), pd.DataFrame(self.samples.values[testing_index])
            
            if self.classification:
                best_overall_validation_value = float('-Inf')
            else:
                best_overall_validation_value = float('-Inf') if self.selection_metric.greater_is_better else float('Inf')
            
            for key in self.models.keys():
                
                print('\t\tAlgorithm with key:', key)
                
                if not self.results[key][outer_cv]:
                    
                    if self.classification:
                        best_validation_value = float('-Inf')
                    else:
                        best_validation_value = float('-Inf') if self.selection_metric.greater_is_better else float('Inf')
                    
                    validation_value_list = list()
                    for configuration in range(self.models[key]['max_combinations']):
                        
                        print('\n\t\t\tIndex of algorithm configuration:', len(validation_value_list))
                        
                        if(len(self.models[key]['algorithms'])) > 1:
                            option = randint(0, 2)
                            algorithm = self.models[key]['algorithms'][option]
                            config = self.models[key]['configuration_method'](option)
                        else:
                            algorithm = self.models[key]['algorithms'][0]
                            #===================================================
                            # if (key == 'mlpc_sgd' or key == 'mlpc_adam' or key == 'mlpr_sgd' or key == 'mlpr_adam'):
                            #===================================================
                            if key.startswith('mlp'):
                                # version from 01-22
                                # config = self.models[key]['configuration_method'](self.get_dataset_size(training_outer))
                                # version from 01-25
                                batch_size = int(training_outer.shape[0] / _INNER_FOLDS)
                                # batch_size = int(training_outer.shape[0] / _INNER_FOLDS) * 2
                                config = self.models[key]['configuration_method'](batch_size)
                            else:
                                config = self.models[key]['configuration_method']()
                        
                        inner_folds = self._get_inner_folds(outer_cv)
                        tmp_valid_training_values_list = list()
                        for training_inner_index, validation_index in inner_folds.split(get_input_variables(training_outer).values, get_target_variable(training_outer).values):
                            
                            print('\t\t\t\tIndex of inner fold:', len(tmp_valid_training_values_list))
                            
                            training_inner, validation = pd.DataFrame(training_outer.values[training_inner_index]), pd.DataFrame(training_outer.values[validation_index])
                            
                            results = self._evaluate_algorithm(algorithm=algorithm, configurations=config,
                                                               training_set=training_inner, validation_set=None, testing_set=validation, metric=self.learning_metric)
                            
                            # print('results[testing_value] =', results['testing_value'], ', results[training_value] =', results['training_value'])
                            
                            if self.classification:
                                tmp_valid_training_values_list.append((results['testing_accuracy'], results['training_accuracy']))
                            else:
                                tmp_valid_training_values_list.append((results['testing_value'], results['training_value']))
                        
                        # Calculate average validation value and check if the current value is better than the best one 
                        average_validation_value = mean(tmp_valid_training_values_list, axis=0)[0]
                        average_training_value = mean(tmp_valid_training_values_list, axis=0)[1]
                        
                        if self.classification:
                            print("\t\t\tAverage AUROC training vs. validation: %.3f vs. %.3f" % (average_training_value, average_validation_value))
                        else:
                            print("\t\t\tAverage RMSE training vs. validation: %.3f vs. %.3f" % (average_training_value, average_validation_value))
                        
                        if self.classification:
                            if average_validation_value > best_validation_value:
                                #===============================================
                                # print('\n\t\t\t\t\tClassification: %.3f is better than %.3f\n' % (average_validation_value, best_validation_value))
                                #===============================================
                                best_algorithm = algorithm
                                best_key = key
                                best_configuration = config
                                best_validation_value = average_validation_value
                                best_training_value = average_training_value
                            #===================================================
                            # else:
                            #     print('\n\t\t\t\t\tClassification: %.3f is worse (!) than %.3f\n' % (average_validation_value, best_validation_value))
                            #===================================================
                        else:
                            if is_better(average_validation_value, best_validation_value, self.selection_metric):
                                #===============================================
                                # print('\n\t\t\t\t\tRegression: %.3f is better than %.3f\n' % (average_validation_value, best_validation_value))
                                #===============================================
                                best_algorithm = algorithm
                                best_key = key
                                best_configuration = config
                                best_validation_value = average_validation_value
                                best_training_value = average_training_value
                            #===================================================
                            # else:
                            #     print('\n\t\t\t\t\tRegression: %.3f is worse (!) than %.3f\n' % (average_validation_value, best_validation_value))
                            #===================================================
                        
                        # Add configuration and validation error to validation error list.
                        validation_value_list.append((configuration, average_validation_value))
                    
                    """ all allowed configurations assessed of a given variant/algorithm/method (key) """
                    print('\n\t\tEvaluating best configuration in outer fold with index', outer_cv)
                    self.results[key][outer_cv] = self._evaluate_algorithm(algorithm=best_algorithm, configurations=best_configuration,
                                                                    training_set=training_outer, validation_set=None, testing_set=testing, metric=self.learning_metric)
                    self.results[key][outer_cv]['best_configuration'] = best_configuration
                    self.results[key][outer_cv]['avg_inner_validation_error'] = best_validation_value
                    self.results[key][outer_cv]['avg_inner_training_error'] = best_training_value
                    if self.classification:
                        self.results[key][outer_cv]['avg_inner_validation_accuracy'] = best_validation_value
                        self.results[key][outer_cv]['avg_inner_training_accuracy'] = best_training_value                    
                    
                    if self.classification:
                        print("\n\t\tAUROC training vs. test: %.3f vs. %.3f" % (self.results[key][outer_cv]['training_accuracy'], self.results[key][outer_cv]['testing_accuracy']))
                        #=======================================================
                        # print("\n\t\tAlgorithm %s, AUROC training vs. test: %.3f vs. %.3f" % (key, self.results[key][outer_cv]['training_accuracy'], self.results[key][outer_cv]['testing_accuracy']))
                        #=======================================================
                    else:
                        print("\n\t\tRMSE training vs. test: %.3f vs. %.3f" % (self.results[key][outer_cv]['training_value'], self.results[key][outer_cv]['testing_value']))
                        #=======================================================
                        # print("\n\t\tAlgorithm %s, RMSE training vs. test: %.3f vs. %.3f" % (key, self.results[key][outer_cv]['training_value'], self.results[key][outer_cv]['testing_value']))
                        #=======================================================
                    
                    best_overall_algorithm = best_algorithm
                    best_overall_configuration = best_configuration
                    best_overall_key = best_key
                    
                    self.best_result[outer_cv] = self.results[key][outer_cv]
                    self.best_result[outer_cv]['best_overall_algorithm'] = best_overall_algorithm
                    self.best_result[outer_cv]['best_overall_configuration'] = best_overall_configuration
                    self.best_result[outer_cv]['best_overall_key'] = best_overall_key
                    
                    # # Serialize benchmark 
                    # benchmark_to_pickle(self)
            
            outer_cv += 1
        
        # Serialize benchmark 
        benchmark_to_pickle(self)
        
        print('Leaving run_nested_cv for dataset:', self.data_set_name)

    def _run_ensembles(self, iteration, best_algorithm, best_configuration, training_outer, testing, metric):
        for key, value in self.ensembles.items():
            
            print('\t\tEnsemble method with key:', key)
            
            if not self.results_ensemble[key][iteration]:
                if '_' in key:  # boosting
                    nr_ensemble = key.split('_')[1]
                    config = self.ensembles[key]['configuration_method'](best_algorithm, best_configuration, nr_ensemble, training_outer, testing, metric)
                else:  # simple, bagging, riw 
                    config = self.ensembles[key]['configuration_method'](best_algorithm, best_configuration, training_outer, testing, metric)
                
                self.results_ensemble[key][iteration] = self._evaluate_algorithm(algorithm=value['algorithms'], configurations=config, training_set=training_outer,
                                                validation_set=None, testing_set=testing, metric=self.metric)
                self.results_ensemble[key][iteration]['algorithm'] = best_algorithm
                self.results_ensemble[key][iteration]['configuration'] = best_configuration
    
    def pickup(self):
        
        print('Entering ensemble pickup for dataset:', self.data_set_name)
        
        outer_cv = 0
        
        outer_folds = self._get_outer_folds(outer_cv)
        for training_outer_index, testing_index in outer_folds.split(get_input_variables(self.samples).values, get_target_variable(self.samples).values):
            
            print('\tIndex of outer fold:', outer_cv)
            
            training_outer, testing = pd.DataFrame(self.samples.values[training_outer_index]), pd.DataFrame(self.samples.values[testing_index])
            
            algorithm = self.best_result[outer_cv]['best_overall_algorithm']
            configuration = self.best_result[outer_cv]['best_overall_configuration']
            self._run_ensembles(outer_cv, algorithm.get_corresponding_algo(), configuration, training_outer, testing, self.metric)
            
            outer_cv += 1
        
        benchmark_to_pickle(self)
        
        print('Leaving ensemble pickup for dataset:', self.data_set_name)


def pickup_benchmark(data_set_name, file_name):
    benchmark = benchmark_from_pickle(data_set_name, file_name)
    
    # -TODO- -TEMP- ! remove afterwards
    # benchmark.ensembles = SIMPLE_ENSEMBLE
    #===========================================================================
    benchmark.ensembles = ENSEMBLES
    if 'riw' in benchmark.ensembles.keys():
        del benchmark.ensembles['riw']
    #===========================================================================
    # -TODO- -TEMP- ! remove afterwards
    
    benchmark.results_ensemble = {ensemble: [None for i in range(_OUTER_FOLDS)] for ensemble in benchmark.ensembles.keys()}
    benchmark.file_name = benchmark.data_set_name + "_" + benchmark.benchmark_id + "__" + _now.strftime("%Y_%m_%d__%H_%M_%S")
    benchmark.pickup()


def continue_benchmark(data_set_name, file_name):
    """Loads a benchmark from .pkl file and continues run."""

    benchmark = benchmark_from_pickle(data_set_name, file_name)
    benchmark.run()
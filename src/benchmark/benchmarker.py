from benchmark.evaluator import EvaluatorSLM, EvaluatorNEAT, EvaluatorSGA, EvaluatorSLM_RST, EvaluatorSLM_RWT, \
    EvaluatorEnsemble, \
    EvaluatorEnsembleBagging, EvaluatorEnsembleRandomIndependentWeighting, EvaluatorEnsembleBoosting, \
    EvaluatorMLPC, EvaluatorMLPR
    # EvaluatorSVC, EvaluatorSVR, , EvaluatorRFC, EvaluatorRFR,
from benchmark.configuration import get_random_config_slm_fls_grouped, get_random_config_slm_ols_grouped, \
    get_random_config_slm_fls_tie_edv, get_random_config_slm_ols_edv, get_config_simple_bagging_ensemble, \
    get_config_riw_ensemble, get_config_boosting_ensemble, get_config_mlp_lbfgs, \
    get_config_mlp_adam, get_config_mlp_sgd, \
    analyze_slm_fls_group_config, analyze_slm_ols_group_config, analyze_slm_fls_tie_edv_group_config, \
    analyze_slm_ols_edv_config
    # , ENSEMBLE_RST_CONFIGURATIONS, ENSEMBLE_CONFIGURATIONS
#   SLM_FLS_CONFIGURATIONS, SLM_OLS_CONFIGURATIONS, \
#     SLM_OLS_RST_CONFIGURATIONS, SLM_OLS_RWT_CONFIGURATIONS, SLM_FLS_RST_CONFIGURATIONS, SLM_FLS_RWT_CONFIGURATIONS, \
    # SLM_OLS_EDV_CONFIGURATIONS, SLM_FLS_TIE_CONFIGURATIONS, SLM_FLS_EDV_CONFIGURATIONS
    # NEAT_CONFIGURATIONS, SGA_CONFIGURATIONS, SVC_CONFIGURATIONS, SVR_CONFIGURATIONS, MLP_CONFIGURATIONS, # RF_CONFIGURATIONS, \
    # ENSEMBLE_CONFIGURATIONS, ENSEMBLE_BAGGING_CONFIGURATIONS, ENSEMBLE_RANDOM_INDEPENDENT_WEIGHTING_CONFIGURATIONS, \
    # ENSEMBLE_BOOSTING_CONFIGURATIONS, ENSEMBLE_RST_CONFIGURATIONS, ENSEMBLE_RWT_CONFIGURATIONS, ENSEMBLE_BAGGING_RST_CONFIGURATIONS, \
    # ENSEMBLE_BAGGING_RWT_CONFIGURATIONS, \
    # ENSEMBLE_FLS_CONFIGURATIONS, ENSEMBLE_BAGGING_FLS_CONFIGURATIONS, ENSEMBLE_RANDOM_INDEPENDENT_WEIGHTING_FLS_CONFIGURATIONS, \
    # ENSEMBLE_BOOSTING_FLS_CONFIGURATIONS
from benchmark.formatter import _format_static_table
from algorithms.common.metric import RootMeanSquaredError, is_better
from data.extract import is_classification, get_input_variables, get_target_variable
from data.io_plm import load_samples, load_samples_no_val, benchmark_to_pickle, benchmark_from_pickle, load_standardized_samples
from tqdm import tqdm
from random import shuffle, uniform, randint
import datetime
import pandas as pd
from numpy import mean
from sklearn.model_selection import KFold, StratifiedKFold

# Disable the monitor thread. (https://github.com/tqdm/tqdm/issues/481)
tqdm.monitor_interval = 0

# Returns the current date and time.
_now = datetime.datetime.now()

_MAX_COMBINATIONS = 50 # to be 50
_MAX_COMBINATIONS_SLM_OLS_EDV = 5 # to be 5
_OUTER_FOLDS = 30 # to be 30
_INNER_FOLDS = 3 # to be 3

# Default models to be compared.
SLM_MODELS = {  
    'slm_fls_group': {
        'name_long': 'Semantic Learning Machine (Fixed Learning Step) Group',
        'name_short': 'SLM (FLS), SLM (FLS) + RST, SLM (FLS) + RWT',
        'algorithms':  [EvaluatorSLM, EvaluatorSLM_RST, EvaluatorSLM_RWT],
        'configuration_method': get_random_config_slm_fls_grouped,
        'max_combinations': _MAX_COMBINATIONS},      
    'slm_ols_group': {
        'name_long': 'Semantic Learning Machine (Optimized Learning Step) Group',
        'name_short': 'SLM (OLS), SLM (OLS) + RST, SLM (OLS) + RWT',
        'algorithms': [EvaluatorSLM, EvaluatorSLM_RST, EvaluatorSLM_RWT],
        'configuration_method': get_random_config_slm_ols_grouped,
        'max_combinations': _MAX_COMBINATIONS},
    'slm_fls_tie_edv_group': {
        'name_long': 'Semantic Learning Machine (Fixed Learning Step) Stopping Criteria Group',
        'name_short': 'SLM (FLS) + TIE, SLM (FLS) + EDV',
        'algorithms': [EvaluatorSLM],
        'configuration_method': get_random_config_slm_fls_tie_edv,
        'max_combinations': _MAX_COMBINATIONS},
    'slm_ols_edv': {
        'name_long': 'Semantic Learning Machine (Optimized Learning Step) + Error Deviation Variation Crit',
        'name_short': 'SLM (OLS) + EDV',
        'algorithms': [EvaluatorSLM],
        'configuration_method': get_random_config_slm_ols_edv,
        'max_combinations': _MAX_COMBINATIONS_SLM_OLS_EDV},
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

MLP_MODELS = {
    'mlpc_lbfgs': {
        'name_long': 'Multilayer Perceptron (LBFGS Solver)',
        'name_short': 'MLP (LBFGS)',
        'algorithms': [EvaluatorMLPC],
        # 'configurations': MLP_CONFIGURATIONS,
        'configuration_method': get_config_mlp_lbfgs,
        'max_combinations': _MAX_COMBINATIONS},
    'mlpr_lbfgs': {
        'name_long': 'Multilayer Perceptron (LBFGS Solver)',
        'name_short': 'MLP (LBFGS)',
        'algorithms': [EvaluatorMLPR],
        # 'configurations': MLP_CONFIGURATIONS,
        'configuration-method': get_config_mlp_lbfgs,
        'max_combinations': _MAX_COMBINATIONS},
    'mlpc_adam': {
        'name_long': 'Multilayer Perceptron (ADAM Solver)',
        'name_short': 'MLP (ADAM)',
        'algorithms': [EvaluatorMLPC],
        # 'configurations': MLP_CONFIGURATIONS,
        'configuration_method': get_config_mlp_adam,
        'max_combinations': _MAX_COMBINATIONS},
    'mlpr_adam': {
        'name_long': 'Multilayer Perceptron (ADAM Solver)',
        'name_short': 'MLP (ADAM)',
        'algorithms': [EvaluatorMLPR],
        # 'configurations': MLP_CONFIGURATIONS,
        'configuration-method': get_config_mlp_adam,
        'max_combinations': _MAX_COMBINATIONS},
    'mlpc_sgd': {
        'name_long': 'Multilayer Perceptron (SGD Solver)',
        'name_short': 'MLP (SGD)',
        'algorithms': [EvaluatorMLPC],
        # 'configurations': MLP_CONFIGURATIONS,
        'configuration_method': get_config_mlp_sgd,
        'max_combinations': _MAX_COMBINATIONS},
    'mlpr_sgd': {
        'name_long': 'Multilayer Perceptron (SGD Solver)',
        'name_short': 'MLP (SGD)',
        'algorithms': [EvaluatorMLPR],
        # 'configurations': MLP_CONFIGURATIONS,
        'configuration-method': get_config_mlp_sgd,
        'max_combinations': _MAX_COMBINATIONS},
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

class Benchmarker():
    """
    Class represents benchmark environment to compare different algorithms in various parameter configurations
    on a given data set and a defined performed metric.

    Attributes:
        data_set_name: Name of data set to study.
        metric: Performance measure to compare with.
        models: Dictionary of models and their corresponding parameter configurations.
    """

    def __init__(self, data_set_name, metric=RootMeanSquaredError, models=None, ensembles=None):
        """Initializes benchmark environment."""

        self.data_set_name = data_set_name
        # Creates file name as combination of data set name and and date.
        self.file_name = self.data_set_name + "__" + _now.strftime("%Y_%m_%d__%H_%M_%S")
        # Loads samples into object.
        # self.samples = [load_samples(data_set_name, index) for index in range(10)]
        # self.samples = [load_samples_no_val(data_set_name, index) for index in range(10)] # changed from 30 , change back at the end 
        self.samples = load_standardized_samples(data_set_name)
        self.metric = metric
        self.ensembles = ensembles
        self.models = models
        # If data set is classification problem, remove regression models. Else, vice versa.
        if is_classification(self.samples): #original self.samples[0][0] new self.samples
            self.classification = True
            if 'mlpr_lbfgs' in self.models.keys():
                del self.models['mlpr_lbfgs']
            if 'mlpr_adam' in self.models.keys():
                del self.models['mlpr_adam']
            if 'mlpr_sgd' in self.models.keys():
                del self.models['mlpr_sgd']
        else:
            self.classification = False
            if 'mlpc_lbfgs' in self.models.keys():
                del self.models['mlpc_lbfgs']
            if 'mlpc_adam' in self.models.keys():
                del self.models['mlpc_adam']
            if 'mlpc_sgd' in self.models.keys(): 
                del self.models['mlpc_sgd']
        # if models = MLP, remove Random Independent Weighting 
        if 'mlpc_lbfgs' in self.models.keys() or 'mlpr_lbfgs' in self.models.keys(): 
            if 'riw' in self.ensembles.keys(): 
                del self.ensembles['riw']
        # Create results dictionary with models under study.
        self.results = {k: [None for i in range(_OUTER_FOLDS)] for k in self.models.keys()}
        self.results_ensemble = {ensemble: [None for i in range(_OUTER_FOLDS)] for ensemble in self.ensembles.keys()}
        # Serialize benchmark environment.
        benchmark_to_pickle(self)

    def _evaluate_algorithm(self, algorithm, configurations, training_set, validation_set, testing_set, metric):
        """Creates evaluator, based on algorithms and configurations."""

        evaluator = algorithm(configurations, training_set, validation_set, testing_set, metric)
        return evaluator.run_nested_cv()
        # return evaluator.run()

    def _evaluate_outer(self, algorithm, configurations, training_set, validation_set, testing_set, metric):
        evaluator = algorithm(configurations, training_set, validation_set, testing_set, metric)
        return evaluator.run_outer()

    def run(self):
        """Runs benchmark study, where it evaluates every algorithms on every sample set."""

        i = 0
        for training, validation, testing in tqdm(self.samples):
            for key, value in tqdm(self.models.items()):
                # If evaluation for key, iteration pair already exists, skip this pair.
                if not self.results[key][i]:          
                    self.results[key][i] = self._evaluate_algorithm(
                        algorithm=value['algorithms'], configurations=value['configurations'], training_set=training,
                        validation_set=validation, testing_set=testing, metric=self.metric)
                    # Serialize benchmark.
                    benchmark_to_pickle(self)
            i += 1
    
    def get_data_set_size(self, data_set): 
        return data_set.shape[0]


    def _get_inner_folds(self, outer_iteration):
        if(self.classification):
            return StratifiedKFold(n_splits=_INNER_FOLDS, random_state=outer_iteration, shuffle=True)
        return KFold(n_splits=_INNER_FOLDS, random_state=outer_iteration, shuffle=True)

    def _get_outer_folds(self, outer_iteration):
        if(self.classification):
            return StratifiedKFold(n_splits=_OUTER_FOLDS, random_state=outer_iteration, shuffle=True)
        return KFold(n_splits=_OUTER_FOLDS, random_state=outer_iteration, shuffle=True)

    def run_nested_cv(self):
        """ runs benchmark study on a nested cross-validation environment for a regression prob"""
        outer_cv = 0
        outer_folds = self._get_outer_folds(outer_cv)
        for training_outer_index, testing_index in tqdm(outer_folds.split(get_input_variables(self.samples).values, get_target_variable(self.samples).values)):
            training_outer, testing = pd.DataFrame(self.samples.values[training_outer_index]), pd.DataFrame(self.samples.values[testing_index])

            best_overall_validation_value = float('-Inf') if self.metric.greater_is_better else float('Inf')
            
            for key, value in tqdm(self.models.items()):
                if not self.results[key][outer_cv]:
                    best_validation_value = float('-Inf') if self.metric.greater_is_better else float('Inf')
                    validation_value_list = list()
                    for configuration in tqdm(range(self.models[key]['max_combinations'])): #changed from tqdm(value['configurations'])
                        if(len(self.models[key]['algorithms'])) > 1:
                            option = randint(0, 2)
                            algorithm = self.models[key]['algorithms'][option]
                            config = self.models[key]['configuration_method'](option)
                        else: 
                            algorithm = self.models[key]['algorithms'][0]
                            if (key == 'mlpc_sgd' or key == 'mlpc_adam' or key == 'mlpr_sgd' or key == 'mlpr_adam'):
                                config = self.models[key]['configuration_method'](self.get_data_set_size(training_outer))
                            else:
                                config = self.models[key]['configuration_method']()
                        
                        inner_folds = self._get_inner_folds(outer_cv)
                        tmp_valid_training_values_list = list()
                        for training_inner_index, validation_index in inner_folds.split(get_input_variables(training_outer).values, get_target_variable(training_outer).values):
                            training_inner, validation = pd.DataFrame(training_outer.values[training_inner_index]), pd.DataFrame(training_outer.values[validation_index])
                            
                            results = self._evaluate_algorithm(algorithm=algorithm, configurations=config, 
                                                               training_set=training_inner, validation_set=None, testing_set=validation, metric=self.metric)
                            
                            tmp_valid_training_values_list.append((results['testing_value'], results['training_value']))
         
                        # Calculate average validation valueand check if the current value is better than the best one 
                        average_validation_value = mean(tmp_valid_training_values_list, axis=0)[0]
                        average_training_value = mean(tmp_valid_training_values_list, axis=0)[1]
                        if is_better(average_validation_value, best_validation_value, self.metric):
                            best_algorithm = algorithm
                            best_configuration = config
                            best_validation_value = average_validation_value
                            best_training_value = average_training_value
                        # Add configuration and validation error to validation error list.
                        validation_value_list.append((configuration, average_validation_value))
                    
                    self.results[key][outer_cv] = self._evaluate_algorithm(algorithm=best_algorithm, configurations=best_configuration, 
                                                                    training_set=training_outer, validation_set=None, testing_set=testing, metric=self.metric)

                    self.results[key][outer_cv]['best_configuration'] = best_configuration
                    self.results[key][outer_cv]['avg_inner_validation_error'] = best_validation_value
                    self.results[key][outer_cv]['avg_inner_training_error'] = best_training_value

                    # Serialize benchmark 
                    benchmark_to_pickle(self)
                    
                    if is_better(best_validation_value, best_overall_validation_value, self.metric):
                        best_overall_algorithm = best_algorithm
                        best_overall_configuration = best_configuration
                        best_overall_validation_value = best_validation_value

            self._run_ensembles(outer_cv, best_overall_algorithm.get_corresponding_algo(), best_overall_configuration, training_outer, testing, self.metric)

            outer_cv += 1            

    def _run_ensembles(self, iteration, best_algorithm, best_configuration, training_outer, testing, metric):
        for key, value in tqdm(self.ensembles.items()):
            if not self.results_ensemble[key][iteration]:
                if '_' in key: # boosting
                    nr_ensemble = key.split('_')[1]
                    config = self.ensembles[key]['configuration_method'](best_algorithm, best_configuration, nr_ensemble, training_outer, testing, metric)
                else: # simple, bagging, riw 
                    config = self.ensembles[key]['configuration_method'](best_algorithm, best_configuration, training_outer, testing, metric)
                self.results_ensemble[key][iteration] = self._evaluate_algorithm(algorithm=value['algorithms'], configurations=config, training_set=training_outer,
                                                validation_set=None, testing_set=testing, metric=self.metric)
                self.results_ensemble[key][iteration]['algorithm'] = best_algorithm
                self.results_ensemble[key][iteration]['configuration'] = best_configuration

    def _format_tables(self):
        pass


def continue_benchmark(data_set_name, file_name):
    """Loads a benchmark from .pkl file and continues run."""

    benchmark = benchmark_from_pickle(data_set_name, file_name)
    benchmark.run()

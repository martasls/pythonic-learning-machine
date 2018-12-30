from benchmark.evaluator import EvaluatorSLM, EvaluatorNEAT, EvaluatorSGA, \
    EvaluatorSVC, EvaluatorSVR, EvaluatorMLPC, EvaluatorMLPR, EvaluatorRFC, EvaluatorRFR, EvaluatorEnsemble, \
    EvaluatorEnsembleBagging, EvaluatorEnsembleRandomIndependentWeighting, EvaluatorEnsembleBoosting, \
    EvaluatorSLM_RST, EvaluatorSLM_RWT
from benchmark.configuration import SLM_FLS_CONFIGURATIONS, SLM_OLS_CONFIGURATIONS, \
    NEAT_CONFIGURATIONS, SGA_CONFIGURATIONS, SVC_CONFIGURATIONS, SVR_CONFIGURATIONS, MLP_CONFIGURATIONS, \
    RF_CONFIGURATIONS, ENSEMBLE_CONFIGURATIONS, ENSEMBLE_BAGGING_CONFIGURATIONS, ENSEMBLE_RANDOM_INDEPENDENT_WEIGHTING_CONFIGURATIONS, \
    ENSEMBLE_BOOSTING_CONFIGURATIONS, SLM_OLS_RST_CONFIGURATIONS, SLM_OLS_RWT_CONFIGURATIONS, \
    ENSEMBLE_RST_CONFIGURATIONS, ENSEMBLE_RWT_CONFIGURATIONS, ENSEMBLE_BAGGING_RST_CONFIGURATIONS, ENSEMBLE_BAGGING_RWT_CONFIGURATIONS, \
    ENSEMBLE_FLS_CONFIGURATIONS, ENSEMBLE_BAGGING_FLS_CONFIGURATIONS, ENSEMBLE_RANDOM_INDEPENDENT_WEIGHTING_FLS_CONFIGURATIONS, \
    ENSEMBLE_BOOSTING_FLS_CONFIGURATIONS, SLM_FLS_RST_CONFIGURATIONS, SLM_FLS_RWT_CONFIGURATIONS
from benchmark.formatter import _format_static_table
from algorithms.common.metric import RootMeanSquaredError, is_better
from data.extract import is_classification, get_input_variables, get_target_variable
from data.io_plm import load_samples, load_samples_no_val, benchmark_to_pickle, benchmark_from_pickle, load_standardized_samples
from tqdm import tqdm
from random import shuffle
import datetime
import pandas as pd
from numpy import mean 
from sklearn.model_selection import KFold, StratifiedKFold

# Disable the monitor thread. (https://github.com/tqdm/tqdm/issues/481)
tqdm.monitor_interval = 0

# Returns the current date and time.
_now = datetime.datetime.now()

# Default models to be compared.
_MODELS = {  
    'slm_fls': {
        'name_long': 'Semantic Learning Machine (Fixed Learning Step)',
        'name_short': 'SLM (FLS)',
        'algorithms': EvaluatorSLM,
        'configurations': SLM_FLS_CONFIGURATIONS},      
    'slm_ols': {
        'name_long': 'Semantic Learning Machine (Optimized Learning Step)',
        'name_short': 'SLM (OLS)',
        'algorithms': EvaluatorSLM,
        'configurations': SLM_OLS_CONFIGURATIONS},
    'slm-ols-rst': {
        'name_long': 'Semantic Learning Machine (Optimized Learning Step) + Random Sampling Technique',
        'name_short': 'SLM (OLS) + RST',
        'algorithms': EvaluatorSLM_RST, 
        'configurations': SLM_OLS_RST_CONFIGURATIONS},
    'slm-ols-rwt': {
        'name_long': 'Semantic Learning Machine (Optimized Learning Step) + Random Weighting Technique',
        'name_short': 'SLM (OLS) + RWT',
        'algorithms': EvaluatorSLM_RWT, 
        'configurations': SLM_OLS_RWT_CONFIGURATIONS},
    'slm-fls-rst': {
        'name_long': 'Semantic Learning Machine (Fixed Learning Step) + Random Sampling Technique',
        'name_short': 'SLM (FLS) + RST',
        'algorithms': EvaluatorSLM_RST, 
        'configurations': SLM_FLS_RST_CONFIGURATIONS},
    'slm-fls-rwt': {
        'name_long': 'Semantic Learning Machine (Fixed Learning Step) + Random Weighting Technique',
        'name_short': 'SLM (FLS) + RWT',
        'algorithms': EvaluatorSLM_RWT, 
        'configurations': SLM_FLS_RWT_CONFIGURATIONS},
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
    # 'neat': {
    #     'name_long': 'Neuroevolution of Augmenting Topologies',
    #     'name_short': 'NEAT',
    #     'algorithms': EvaluatorNEAT,
    #     'configurations': NEAT_CONFIGURATIONS},
    # 'sga': {
    #     'name_long': 'Simple Genetic Algorithm',
    #     'name_short': 'SGA',
    #     'algorithms': EvaluatorSGA,
    #     'configurations': SGA_CONFIGURATIONS},
    # 'svc': {
    #     'name_long': 'Support Vector Machine',
    #     'name_short': 'SVM',
    #     'algorithms': EvaluatorSVC,
    #     'configurations': SVC_CONFIGURATIONS},
    # 'svr': {
    #     'name_long': 'Support Vector Machine',
    #     'name_short': 'SVM',
    #     'algorithms': EvaluatorSVR,
    #     'configurations': SVR_CONFIGURATIONS},
    # 'mlpc': {
    #     'name_long': 'Multilayer Perceptron',
    #     'name_short': 'MLP',
    #     'algorithms': EvaluatorMLPC,
    #     'configurations': MLP_CONFIGURATIONS},
    # 'mlpr': {
    #     'name_long': 'Multilayer Perceptron',
    #     'name_short': 'MLP',
    #     'algorithms': EvaluatorMLPR,
    #     'configurations': MLP_CONFIGURATIONS},
    # 'rfc': {
    #     'name_long': 'Random Forest',
    #     'name_short': 'RF',
    #     'algorithms': EvaluatorRFC,
    #     'configurations': RF_CONFIGURATIONS},
    # 'rfr': {
    #     'name_long': 'Random Forest',
    #     'name_short': 'RF',
    #     'algorithms': EvaluatorRFR,
    #     'configurations': RF_CONFIGURATIONS}
}

MAX_COMBINATIONS = 30 # to be 30
OUTER_FOLDS = 30 # to be 30
INNER_FOLDS = 5 # to be 5


class Benchmarker():
    """
    Class represents benchmark environment to compare different algorithms in various parameter configurations
    on a given data set and a defined performed metric.

    Attributes:
        data_set_name: Name of data set to study.
        metric: Performance measure to compare with.
        models: Dictionary of models and their corresponding parameter configurations.
    """

    def __init__(self, data_set_name, metric=RootMeanSquaredError, models=_MODELS):
        """Initializes benchmark environment."""

        self.data_set_name = data_set_name
        # Creates file name as combination of data set name and and date.
        self.file_name = self.data_set_name + "__" + _now.strftime("%Y_%m_%d__%H_%M_%S")
        # Loads samples into object.
        self.samples = [load_samples_no_val(data_set_name, index) for index in range(10)] # changed from 30 , change back at the end 
        self.samples = load_standardized_samples(data_set_name)
        self.metric = metric
        self.models = models
        # If data set is classification problem, remove regression models. Else, vice versa.
        if is_classification(self.samples):
            if 'svr' in self.models.keys():
                del self.models['svr']
            if 'mlpr' in self.models.keys():
                del self.models['mlpr']
            if 'rfr' in self.models.keys():
                del self.models['rfr']
        else:
            if 'svc' in self.models.keys():
                del self.models['svc']
            if 'mlpc' in self.models.keys():
                del self.models['mlpc']
            if 'rfc' in self.models.keys():
                del self.models['rfc']
        # Create results dictionary with models under study.
        self.results = {k: [None for i in range(OUTER_FOLDS)] for k in self.models.keys()}
        # Serialize benchmark environment.
        benchmark_to_pickle(self)

    def _evaluate_algorithm(self, algorithm, configurations, training_set, validation_set, testing_set, metric):
        """Creates evaluator, based on algorithms and configurations."""

        evaluator = algorithm(configurations, training_set, validation_set, testing_set, metric)
        return evaluator.run_nested_cv()

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
                    
    def _get_folds(self, outer_iteration):
        if(is_classification(self.samples)):
            return StratifiedKFold(n_splits=INNER_FOLDS, random_state=outer_iteration, shuffle=True)
        return KFold(n_splits=INNER_FOLDS, random_state=outer_iteration, shuffle=True)

    def _get_outer_folds(self, outer_iteration):
        if(is_classification(self.samples)):
            return StratifiedKFold(n_splits=OUTER_FOLDS, random_state=outer_iteration, shuffle=True)
        return KFold(n_splits=OUTER_FOLDS, random_state=outer_iteration, shuffle=True)

    def run_nested_cv(self):
        """ runs benchmark study on a nested cross-validation environment for a regression prob"""
        
        outer_cv = 0
        outer_folds = self._get_outer_folds(outer_cv)
        for training_outer_index, testing_index in tqdm(outer_folds.split(get_input_variables(self.samples).values, get_target_variable(self.samples).values)):
            training_outer, testing = pd.DataFrame(self.samples.values[training_outer_index]), pd.DataFrame(self.samples.values[testing_index])
            for key, value in tqdm(self.models.items()):

                if not self.results[key][outer_cv]:
                    shuffle(value['configurations'])
                    nr_configurations = 0
                    best_validation_value = float('-Inf') if self.metric.greater_is_better else float('Inf')
                    validation_value_list = list()
                    for configuration in tqdm(value['configurations']):
                        folds = self._get_folds(outer_cv)
                        tmp_validation_value_list = list()
                        for training_inner_index, validation_index in folds.split(get_input_variables(training_outer).values, get_target_variable(training_outer).values):
                            training_inner, validation = pd.DataFrame(training_outer.values[training_inner_index]), pd.DataFrame(training_outer.values[validation_index])
                            
                            results = self._evaluate_algorithm(algorithm=value['algorithms'], configurations=configuration, 
                                                    training_set=training_inner, validation_set=None, testing_set=validation, metric=self.metric)
                            
                            tmp_validation_value_list.append(results['testing_value'])

                        nr_configurations += 1
                        
                        # Calculate average validation valueand check if the current value is better than the best one 
                        average_validation_value = mean(tmp_validation_value_list)
                        if is_better(average_validation_value, best_validation_value, self.metric):
                            best_configuration = configuration
                            best_validation_value = average_validation_value
                        # Add configuration and validation error to validation error list.
                        validation_value_list.append((configuration, average_validation_value))

                        if nr_configurations == MAX_COMBINATIONS:
                            break

                    results_best_learner = self._evaluate_algorithm(algorithm=value['algorithms'], configurations=best_configuration, 
                                                training_set=training_outer, validation_set=None, testing_set=testing, metric=self.metric)
                    self.results[key][outer_cv] = results_best_learner
                    self.results[key][outer_cv]['best_configuration'] = best_configuration
                    # Serialize benchmark 
                    benchmark_to_pickle(self)
                    
            outer_cv += 1            

    def _format_tables(self):
        pass


def continue_benchmark(data_set_name, file_name):
    """Loads a benchmark from .pkl file and continues run."""

    benchmark = benchmark_from_pickle(data_set_name, file_name)
    benchmark.run()

import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from os.path import join, dirname, exists
from data.io_plm import read_csv_, get_results_folder


def extract_results(path): 
    """ given a path, the method extracts the results inside that path""" 
    generate_boxplot_error(path, 'testing_value')
    generate_boxplot_error(path, 'training_value')
    generate_boxplot_error(path, 'avg_inner_training_error')
    generate_boxplot_error(path, 'avg_inner_validation_error')
    generate_comparing_boxplot(path, 'avg_inner_validation_error', 'testing_value')


def generate_boxplot_error(path, metric_name):
    """generates a boxplot for a certain metric""" 
    data_set_name = path.split('\\')[-1]
    to_read = join(path, metric_name + '.csv')
    value = read_csv_(to_read)
    fig, ax = plt.subplots()
    boxplot = sns.boxplot(data=value, palette="PuBuGn_d")
    boxplot.set(xlabel='Algorithms', ylabel='RMSE')
    sns.set_context("paper", font_scale=1.5)
    sns.set_style("white")
    sns.despine(left=True)
    boxplot.set_xticklabels(['SLM (FLS) Group', 'SLM (OLS) Group', 'SLM (FLS) + TIE/EDV', 'SLM (OLS) + EDV'], rotation=90) #changed from boxplot.get_xticklabels()-BE CAREFUL, SPECIFIC VALUES
    fig.set_size_inches(11.69, 8.27)
    results_folder_path = join(get_results_folder(), data_set_name)
    fig.savefig(join(results_folder_path, metric_name + '.svg'), bbox_inches='tight')
    fig.savefig(join(results_folder_path, metric_name + '.pdf'), bbox_inches='tight')

def generate_comparing_boxplot(path, metric_one, metric_two):
    """generates a grouped boxplot taking into account two metrics"""
    data_set_name = path.split('\\')[-1]
    metric_one_path = join(path, metric_one + '.csv')
    metric_one_data_set = read_csv_(metric_one_path)
    metric_two_path = join(path, metric_two + '.csv')
    metric_two_data_set = read_csv_(metric_two_path)

    # melt data sets
    metric_one_data_set_long = metric_one_data_set.melt(var_name='algorithm', value_name=metric_one)
    metric_two_data_set_long = metric_two_data_set.melt(var_name='algorithm', value_name=metric_two)
    metric_two_data_set_long = metric_two_data_set_long.drop(metric_two_data_set_long.columns[0], axis=1)
    #concatenate data sets 
    concatenated_metrics = pd.concat([metric_one_data_set_long, metric_two_data_set_long], sort=False, axis=1)

    # melt concatenated data set
    melted = pd.melt(concatenated_metrics, id_vars=['algorithm'], value_vars=[metric_one, metric_two], var_name='Metric')

    catplot = sns.catplot('algorithm', hue='Metric', y='value', data=melted, kind="box", legend=False, palette="PuBuGn_d", 
                     height=5, aspect=1.75)
    catplot.set(xlabel='Algorithms', ylabel="RMSE")
    sns.set_context("paper", font_scale=1.5)
    sns.set_style("white")
    sns.despine(left=True)
    catplot.set_xticklabels(['SLM (FLS) Group', 'SLM (OLS) Group', 'SLM (FLS) + TIE/EDV', 'SLM (OLS) + EDV'], rotation=90) # BE CAREFUL, SPECIFIC VALUES
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title="Metric", frameon=False) #this changes the legend to outside the plot
    #plt.figure()
    catplot.fig.set_size_inches(11.69, 8.27)
    results_folder_path = join(get_results_folder(), data_set_name)
    catplot.fig.savefig(join(results_folder_path, metric_one + '__' + metric_two + '.svg'), bbox_inches='tight')
    catplot.fig.savefig(join(results_folder_path, metric_one + '__' + metric_two + '.pdf'), bbox_inches='tight')

def extract_avg_inner_validation_value(path):
    pass

def exctrat_avg_inner_training_value(path):
    pass

def extract_outer_training_time(path):
    pass

"""
#change working directory
os.chdir("C:\\Users\\Marta\\Documents\\GitHub\\pythonic-learning-machine\\data_sets\\06_formatted\\c_ionosphere\\")

#testing value 
testing_value = pd.read_csv("testing_value.csv")
testing_value = testing_value[['SLM (Ensemble)', 'SLM (Ensemble) + RST', 'SLM (Ensemble) + RWT', 'SLM (Ensemble-Bagging)', 
                               'SLM (Ensemble-Bagging) + RST', 'SLM (Ensemble-Bagging) + RWT', 'SLM (Ensemble-RIW)',
                                'SLM (Ensemble-Boosting)', 'SLM (OLS)', 'SLM (OLS) + RST', 'SLM (OLS) + RWT']]
testing_value.head()

fig, ax = plt.subplots()
boxplot = sns.boxplot(data=testing_value, palette="PuBuGn_d")
boxplot.set(xlabel='Algorithms', ylabel='RMSE')
sns.set_context("paper", font_scale=1.5)
sns.set_style("white")
sns.despine(left=True)
boxplot.set_xticklabels(boxplot.get_xticklabels(), rotation=90)
fig.set_size_inches(11.69, 8.27)
fig.savefig("testing_error.svg", bbox_inches='tight')
fig.savefig("testing_error.pdf", bbox_inches='tight')
plt.show()
"""
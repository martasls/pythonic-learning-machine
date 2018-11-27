import os
import sys 
import pandas as pd 

#get working directory 
#wd = os.getcwd() 

#print(wd)
path = 'C:\\Users\\Marta\\Documents\\GitHub\\pythonic-learning-machine\\data_sets\\'
wd = os.chdir(path)

#print(os.listdir())

c_diabetes = pd.read_csv('01_raw\\c_diabetes\\data_set.csv', header=None)
#print(c_diabetes.head())

outname = 'c_diabetes.csv'
outdir = '02_cleaned'
if not os.path.exists(outdir):
    os.mkdir(outdir)

fullname = os.path.join(outdir, outname)

#save data set to csv
c_diabetes.to_csv(fullname)
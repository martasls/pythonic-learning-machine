import os
import sys 
import pandas as pd 

#get working directory 
wd = os.getcwd() 
wd = os.path.realpath('..')
print(wd)
folder = os.path.join(wd, 'data_sets')
print(folder)
#wd = os.chdir(path)

#print(os.listdir())

data_set = os.path.join(folder, '01_raw', 'c_diabetes', 'data_set.csv')
c_diabetes = pd.read_csv(data_set, header=None)
print(c_diabetes.head())

outname = 'c_diabetes.csv'
outdir = '02_cleaned'
if not os.path.join(folder, outdir):
    os.mkdir(outdir)

fullname = os.path.join(folder, outdir, outname)
print(fullname)
#save data set to csv
c_diabetes.to_csv(fullname)
import os
import sys 
import pandas as pd 

#get working directory 
#wd = os.getcwd() 

#print(wd)
path = 'C:\\Users\\Marta\\Documents\\GitHub\\pythonic-learning-machine\\data_sets\\'
wd = os.chdir(path)

#print(os.listdir())

r_ld50 = pd.read_csv('01_raw\\r_ld50\\data_set.csv', header=None)


#### cleaning ####




outname = 'r_ld50.csv'
outdir = '02_cleaned'
if not os.path.exists(outdir):
    os.mkdir(outdir)

fullname = os.path.join(outdir, outname)

#save data set to csv
r_ld50.to_csv(fullname)
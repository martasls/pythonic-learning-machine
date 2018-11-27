import os
import sys 
import pandas as pd 

#get working directory 
#wd = os.getcwd() 

#print(wd)
path = 'C:\\Users\\Marta\\Documents\\GitHub\\pythonic-learning-machine\\data_sets\\'
wd = os.chdir(path)

#print(os.listdir())

r_bio = pd.read_csv('01_raw\\r_bio\\data_set.csv', header=None)


#### cleaning ####




outname = 'r_bio.csv'
outdir = '02_cleaned'
if not os.path.exists(outdir):
    os.mkdir(outdir)

fullname = os.path.join(outdir, outname)

#save data set to csv
r_bio.to_csv(fullname)
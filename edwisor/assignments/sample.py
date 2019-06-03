import os
import pandas as pd
#Set working directory
os.chdir("../Documents")

#check for the current working directory
pwd=os.getcwd()
print pwd

#read CSV file using pandas library

df_csv=pd.read_csv("asample.csv",Sep=",")
print df_csv

import numpy as np, time
import pandas as pd, keras
import os.path
from pathlib import Path
from pytorch import run_final_model, run_cross_valid, plot_saved_model, plot_saved_model2, check_interpolation
from pytorch_utilities import hyper_param
from read_in_out import initiate_data
import matplotlib.pyplot as plt
import sys

hyper_arg =  int(sys.argv[1])
path = '/Users/rsharma/Dropbox/Musculoskeletal_Modeling/MSK_ML_beta/'
path = '/work/lcvmm/rsharma/MSK/MSK_ML_beta/'

data = initiate_data(path)
hyper =  pd.read_csv(path+'hyperparam.txt',delimiter='\s+')
#hyper =  pd.read_csv(path+'hyperparam_linear.txt',delimiter='\s+')
hyper_val =  hyper.iloc[hyper_arg]

for feat in ['JA','JM','JRF','MA','MF']:
#    tmp_data = data.subject_naive(feat)
    tmp_data = data.subject_exposed(feat)
    print(feat)
    run_cross_valid(tmp_data,hyper_arg,hyper_val,'NN')
    run_final_model(tmp_data,hyper_arg,hyper_val,'NN')

#check_interpolation(data)


# MSK_ML_beta
# #This file contains the code corresponding to the article "Machine Learning to Map Inertial Motion Capture Inputs to Optical Motion Capture-driven Musculoskeletal Model Outputs".

#A cluster was used to run cross-validation, so the code may require some changes to run on a laptop. Final model training can be easily done on a normal laptop using the function "train_final_model" in plot_results.py.

#Lastly, all the plots and statistics provided in the paper can be reproduced by simply running plot_results.py.

#Test data for reproducing the results are provided here. The full training dataset can be obtained on request to vikranth.harthikotenagaraja@eng.ox.ac.uk

#The codes are developed using Python 3 with keras for ML.
#MSK_ML_codes #This file contains the code description used in the article: "Machine Learning for Musculoskeletal Modeling of Upper Extremity" #The codes are developed in python3 using standard python modules and keras for ML.

#The code here contains: #a) Pipeline to run ML methods such NN, LM which can be easily extended for other methods. #b) We have used this pipeline to run cross-validation on cluster. Therefore, to run on laptop may require slight tweak in the code. #c) Final training and testing can be easily done on any laptop/computer. #d) Code for analysis and plotting is also provided.

#Code description #pytorch_utilities.py -- contains function for various models (Linear, Neural Network, ....) and generate a file with hyperparameters choices #pytorch.py -- several functions to handle data, perform cross-validation, train model, forward pass, plot and analyse results #analysis*py -- perform cross-validation #plot_results.py -- contains function to estimate validation accuracy, find best hyper-parameters, train final model, plot results and compute statistics

#final trained NN model #Files are heavy to upload, can be downloaded from here: https://www.dropbox.com/sh/svuqdy597d6pg60/AACiWr-kVx_W0bU0AD3HWPNha?dl=0
#Test data for reproducing the results are provided here, 
Input data --  https://www.dropbox.com/sh/8isp6yl29np6ngo/AAAqRWIc8lTOtYieehUSEQN1a?dl=0
Output data -- https://www.dropbox.com/sh/7h1oncpyru9vupl/AACm0YPmlmdu2rlabUpynCW4a?dl=0
The full training can be obtained on request to vikranth.harthikotenagaraja@eng.ox.ac.uk


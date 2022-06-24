from pytorch_utilities import *
import pandas as pd
from torchvision import transforms, datasets
from torch.utils.data import TensorDataset, DataLoader
import torch as tr, time
import numpy as np
import statsmodels.api as sm
from tensorflow import keras
import seaborn as sns
import matplotlib.pyplot as plt
from fractions import Fraction
import sys, copy
import scipy
from scipy import signal
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
import random
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from matplotlib import gridspec
from scipy.interpolate import interp1d



    
    
def combined_plot_2(model1,model2,X_Test1,Y_Test1,X_Test2,Y_Test2,label,scale_out,model_class):
    ## Need to think about computing each trial separately and how that affects the output
    print("\n plotting results -----------------------")
    RMSE_list, PC_list = [],[]
    YP1, YP2 = model1.predict(X_Test1), model2.predict(X_Test2)
    YT1, YT2 = np.array(Y_Test1),np.array(Y_Test2) 
    a,b = np.shape(YT1)
    #### the below loop is to set the time in terms of percentage of task
    count,aa = -1,[]
    df = X_Test1['time']
    zero_entries = np.where(df==0)
    zero_entries = np.concatenate([zero_entries[0],np.array([a])])   #### adding last element
    for u in df:
        if u == 0:
            count = count + 1
        aa.append(u+count)
    count=0
    if 'JRF' in label:
        fig = plt.figure(figsize=(8,10.5))
        gs1 = gridspec.GridSpec(700, 560)
        gs1.update(left=0.065, right=0.98,top=0.945, bottom=0.06)
        d1, d2 =10, 10
        ax00 = plt.subplot(gs1[  0+d2:100  ,   0+d1:100 ])
        ax01 = plt.subplot(gs1[  0+d2:100  , 150+d1:250 ])
        ax10 = plt.subplot(gs1[120+d2:220  ,   0+d1:100 ])
        ax11 = plt.subplot(gs1[120+d2:220  , 150+d1:250 ])
        ax20 = plt.subplot(gs1[240+d2:340  ,   0+d1:100 ])
        ax21 = plt.subplot(gs1[240+d2:340  , 150+d1:250 ])
        ax30 = plt.subplot(gs1[360+d2:460  ,   0+d1:100 ])
        ax31 = plt.subplot(gs1[360+d2:460  , 150+d1:250 ])
        ax40 = plt.subplot(gs1[480+d2:580  ,   0+d1:100 ])
        ax41 = plt.subplot(gs1[480+d2:580  , 150+d1:250 ])
        ax50 = plt.subplot(gs1[600+d2:700  ,   0+d1:100 ])
        ax51 = plt.subplot(gs1[600+d2:700  , 150+d1:250 ])

        ax02 = plt.subplot(gs1[  0+d2:100  , 310+d1:410 ])
        ax03 = plt.subplot(gs1[  0+d2:100  , 460+d1:560 ])
        ax12 = plt.subplot(gs1[120+d2:220  , 310+d1:410 ])
        ax13 = plt.subplot(gs1[120+d2:220  , 460+d1:560 ])
        ax22 = plt.subplot(gs1[240+d2:340  , 310+d1:410 ])
        ax23 = plt.subplot(gs1[240+d2:340  , 460+d1:560 ])
        ax32 = plt.subplot(gs1[360+d2:460  , 310+d1:410 ])
        ax33 = plt.subplot(gs1[360+d2:460  , 460+d1:560 ])
        ax42 = plt.subplot(gs1[480+d2:580  , 310+d1:410 ])
        ax43 = plt.subplot(gs1[480+d2:580  , 460+d1:560 ])
        ax52 = plt.subplot(gs1[600+d2:700  , 310+d1:410 ])
        ax53 = plt.subplot(gs1[600+d2:700  , 460+d1:560 ])

        ax_list  = [ax00, ax01, ax10, ax11, ax20, ax21, ax30, ax31, ax40, ax41, ax50, ax51]
        ax_list2 = [ax02, ax03, ax12, ax13, ax22, ax23, ax32, ax33, ax42, ax43, ax52, ax53]

        ss,b_xlabel = 8,9
        ylabel = [ 'Trunk \n Mediolateral', 'Trunk \n Proximodistal', 'Trunk \n Anteroposterior', 'Shoulder \n Mediolateral',
                  'Shoulder \n Proximodistal', 'Shoulder \n Anteroposterior', 'Elbow \n Mediolateral', 'Elbow \n Proximodistal',
                  'Elbow \n Anteroposterior', 'Wrist \n Mediolateral', 'Wrist \n Proximodistal', 'Wrist \n Anteroposterior']
        plot_list = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)','(j)','(k)','(l)']

    elif 'Muscle' in label:
        fig = plt.figure(figsize=(8,4))
        gs1 = gridspec.GridSpec(215, 560)
        gs1.update(left=0.05, right=0.98,top=0.84, bottom=0.15)
        d1, d2 =10, 10
        ax00 = plt.subplot(gs1[0:100 -d2    , 0+d1:100  ])
        ax01 = plt.subplot(gs1[0:100 -d2   , 150+d1:250 ])
        ax10 = plt.subplot(gs1[115+d2:215  , 0+d1:100 ])
        ax11 = plt.subplot(gs1[115+d2:215  , 150+d1:250 ])

        ax02 = plt.subplot(gs1[0:100 -d2   , 310+d1:410 ])
        ax03 = plt.subplot(gs1[0:100 -d2   , 460+d1:560 ])
        ax12 = plt.subplot(gs1[115+d2:215  , 310+d1:410 ])
        ax13 = plt.subplot(gs1[115+d2:215  , 460+d1:560 ])

        ax_list = [ax00,ax01,ax10 ,ax11]
        ax_list2= [ ax02,ax03, ax12 ,ax13 ]
        ss,b_xlabel = 8,1
        ylabel = ['Pectoralis major \n (Clavicle)','Biceps Brachii','Deltoid (Medial)','Brachioradialis']
        plot_list = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)','(j)','(k)','(l)']


    elif 'JM' in label:
        fig = plt.figure(figsize=(8,8.25))
        gs1 = gridspec.GridSpec(580, 560)
        gs1.update(left=0.065, right=0.98,top=0.945, bottom=0.08)
        d1, d2 =10, 10
        ax00 = plt.subplot(gs1[  0+d2:100  ,   0+d1:100 ])
        ax01 = plt.subplot(gs1[  0+d2:100  , 150+d1:250 ])
        ax10 = plt.subplot(gs1[120+d2:220  ,   0+d1:100 ])
        ax11 = plt.subplot(gs1[120+d2:220  , 150+d1:250 ])
        ax20 = plt.subplot(gs1[240+d2:340  ,   0+d1:100 ])
        ax21 = plt.subplot(gs1[240+d2:340  , 150+d1:250 ])
        ax30 = plt.subplot(gs1[360+d2:460  ,   0+d1:100 ])
        ax31 = plt.subplot(gs1[360+d2:460  , 150+d1:250 ])
        ax40 = plt.subplot(gs1[480+d2:580  ,   0+d1:100 ])
        ax41 = plt.subplot(gs1[480+d2:580  , 150+d1:250 ])

        ax02 = plt.subplot(gs1[  0+d2:100  , 310+d1:410 ])
        ax03 = plt.subplot(gs1[  0+d2:100  , 460+d1:560 ])
        ax12 = plt.subplot(gs1[120+d2:220  , 310+d1:410 ])
        ax13 = plt.subplot(gs1[120+d2:220  , 460+d1:560 ])
        ax22 = plt.subplot(gs1[240+d2:340  , 310+d1:410 ])
        ax23 = plt.subplot(gs1[240+d2:340  , 460+d1:560 ])
        ax32 = plt.subplot(gs1[360+d2:460  , 310+d1:410 ])
        ax33 = plt.subplot(gs1[360+d2:460  , 460+d1:560 ])
        ax42 = plt.subplot(gs1[480+d2:580  , 310+d1:410 ])
        ax43 = plt.subplot(gs1[480+d2:580  , 460+d1:560 ])

        ax_list  = [ax00, ax10, ax01, ax11, ax20, ax21, ax30, ax31, ax40, ax41]
        ax_list2 = [ax02, ax12, ax03, ax13, ax22, ax23, ax32, ax33, ax42, ax43]

        ss,b_xlabel = 8,7


        ylabel = [ 'Trunk Flexion / \n Extension', 'Trunk Internal / \n External Rotation', 'Trunk Right / \n Left Bending',
                  'Shoulder Flexion / \n Extension', 'Shoulder Abduction / \n Adduction', 'Shoulder Internal / \n External Rotation',
                  'Elbow Flexion / \n Extension', 'Elbow Pronation / \n Supination', 'Wrist Flexion / \n Extension', 'Wrist Radial / \n Ulnar Deviation']
        plot_list = ['(a)','(c)','(b)','(d)','(e)','(f)','(g)','(h)','(i)','(j)','(k)','(l)']

    elif 'Angles' in label:
        new_order = [7,8,9,0,1,2,3,4,5,6] 
        YP1 = YP1[:,new_order]
        YP2 = YP2[:,new_order]
        YT1 = YT1[:,new_order]
        YT2 = YT2[:,new_order]
        fig = plt.figure(figsize=(8,8.25))
        gs1 = gridspec.GridSpec(580, 560)
        gs1.update(left=0.065, right=0.98,top=0.945, bottom=0.07)
        d1, d2 =10, 10
        ax00 = plt.subplot(gs1[  0+d2:100  ,   0+d1:100 ])
        ax01 = plt.subplot(gs1[  0+d2:100  , 150+d1:250 ])
        ax10 = plt.subplot(gs1[120+d2:220  ,   0+d1:100 ])
        ax11 = plt.subplot(gs1[120+d2:220  , 150+d1:250 ])
        ax20 = plt.subplot(gs1[240+d2:340  ,   0+d1:100 ])
        ax21 = plt.subplot(gs1[240+d2:340  , 150+d1:250 ])
        ax30 = plt.subplot(gs1[360+d2:460  ,   0+d1:100 ])
        ax31 = plt.subplot(gs1[360+d2:460  , 150+d1:250 ])
        ax40 = plt.subplot(gs1[480+d2:580  ,   0+d1:100 ])
        ax41 = plt.subplot(gs1[480+d2:580  , 150+d1:250 ])

        ax02 = plt.subplot(gs1[  0+d2:100  , 310+d1:410 ])
        ax03 = plt.subplot(gs1[  0+d2:100  , 460+d1:560 ])
        ax12 = plt.subplot(gs1[120+d2:220  , 310+d1:410 ])
        ax13 = plt.subplot(gs1[120+d2:220  , 460+d1:560 ])
        ax22 = plt.subplot(gs1[240+d2:340  , 310+d1:410 ])
        ax23 = plt.subplot(gs1[240+d2:340  , 460+d1:560 ])
        ax32 = plt.subplot(gs1[360+d2:460  , 310+d1:410 ])
        ax33 = plt.subplot(gs1[360+d2:460  , 460+d1:560 ])
        ax42 = plt.subplot(gs1[480+d2:580  , 310+d1:410 ])
        ax43 = plt.subplot(gs1[480+d2:580  , 460+d1:560 ])

        ax_list  = [ax00, ax01, ax10, ax11, ax20, ax21, ax30, ax31, ax40, ax41]
        ax_list2 = [ax02, ax03, ax12, ax13, ax22, ax23, ax32, ax33, ax42, ax43]

        ss,b_xlabel = 8,7

        plot_list = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)','(j)']

        ylabel = ['Trunk Forward / \n Backward Bending', 'Trunk Right / \n Left Bending', 'Trunk Internal / \n External Rotation', 
                  'Shoulder Flexion / \n Extension', 'Shoulder Abduction / \n Adduction', 'Shoulder Internal / \n External Rotation', 
                  'Elbow Flexion / \n Extension', 'Elbow Pronation / \n Supination', 'Wrist Flexion / \n Extension', 'Wrist Radial / \n Ulnar Deviation']

    sparse_plot=5
    for i in range(b):
        Title,NRMSE,push_plot = 0,0,0
        Title2,NRMSE2,push_plot = 0,0,0
        for c in range(count+1):    ########## this loop is required to separate trials
            ttmmpp = np.arange(zero_entries[c],zero_entries[c+1])

            if ax_list[i] == ax00:
                label1, label2 = 'NN prediction', 'MSK model output'
            else:
                label1, label2 = '_no_legend_', '_no_legend_'
                
            ax_list[i].plot([aa[q] + push_plot for q in ttmmpp][::sparse_plot] ,YP1[ttmmpp,i][::sparse_plot],color='red', lw=0.8,label=label1)   ### np.arange(a)
            ax_list[i].plot([aa[q] + push_plot for q in ttmmpp][::sparse_plot] ,YT1[ttmmpp,i][::sparse_plot],color='blue',lw=0.8,label=label2)

            ax_list2[i].plot([aa[q] + push_plot for q in ttmmpp][::sparse_plot] ,YP2[ttmmpp,i][::sparse_plot],color='red', lw=0.8,label ='_no_legend_')#,label=label1)   ### np.arange(a)
            ax_list2[i].plot([aa[q] + push_plot for q in ttmmpp][::sparse_plot] ,YT2[ttmmpp,i][::sparse_plot],color='blue',lw=0.8,label ='_no_legend_')#,label=label2)
            Title = Title + scipy.stats.pearsonr(YP1[ttmmpp,i],YT1[ttmmpp,i])[0]
            NRMSE  = NRMSE + mean_squared_error(YP1[ttmmpp,i], YT1[ttmmpp,i],squared=False)

            Title2  = Title2 + scipy.stats.pearsonr(YP2[ttmmpp,i],YT2[ttmmpp,i])[0]
            NRMSE2  = NRMSE2 + mean_squared_error(YP2[ttmmpp,i], YT2[ttmmpp,i],squared=False)

            push_plot = push_plot + 0.1
        push2 = 0.05
        ax_list[i].set_xlim(0,count+1)
        ax_list2[i].set_xlim(0,count+1)
        ind = ['Trial '+str(i+1) for i in range(count+1)]
        Title = str(np.around(Title/(count+1),2))
        NRMSE = str(np.around(NRMSE/(count+1),2))
        Title2 = str(np.around(Title2/(count+1),2))
        NRMSE2 = str(np.around(NRMSE2/(count+1),2))
        if len(Title) == 3:
            Title = Title+'0'
        if len(Title2) == 3:
            Title2 = Title2+'0'
        if len(NRMSE) == 3:
            NRMSE = NRMSE+'0'
        if len(NRMSE2) == 3:
            NRMSE2 = NRMSE2+'0'

        Title = plot_list[i] + "  r = " + Title + ", RMSE = " + NRMSE
        Title2 = plot_list[i] + "  r = " + Title2 + ", RMSE = " + NRMSE2
        ax_list[i].text(-0.25, 1.1, Title, transform=ax_list[i].transAxes, size=ss)#,fontweight='bold')
        ax_list2[i].text(-0.25, 1.1, Title2, transform=ax_list2[i].transAxes, size=ss)#,fontweight='bold')
        minor_ticks = [] 
        percent = ['0%','25%','50%','75%','100%']
        push3 = 0
        for sn in range(count+1):
            for sn1 in np.arange(sn,sn+1.25,0.25):
                minor_ticks.append(sn1+push3)
            push3=push3+0.1

        ax_list[i].set_xticks(minor_ticks ,minor=True)
        ax_list[i].set_xticks(np.array(minor_ticks[2::5])+0.0005,minor=False)

        ax_list[i].set_ylabel(ylabel[i],fontsize=ss)
        # ax_list[i].yaxis.set_label_coords(-0.28,0.5)

        ax_list2[i].set_xticks(minor_ticks ,minor=True)
        ax_list2[i].set_xticks(np.array(minor_ticks[2::5])+0.0005,minor=False)
        ax_list2[i].set_ylabel(ylabel[i],fontsize=ss)
        # ax_list2[i].yaxis.set_label_coords(-0.28,0.5)

        for axx1,axx2 in zip(ax_list[0:len(ax_list)-2], ax_list2[0:len(ax_list2)-2]):
            axx1.set_xticklabels([],fontsize=ss,minor=False)
            axx2.set_xticklabels([],fontsize=ss,minor=False)

        for axx1,axx2 in zip(ax_list[-2:], ax_list2[-2:]):
            axx1.set_xticklabels([],fontsize=ss,minor=False)
            axx1.set_xticklabels(percent*(count+1),fontsize=ss,minor=True,rotation=45)
            axx1.set_xlabel("% of task completion",fontsize=ss)

            axx2.set_xticklabels([],fontsize=ss,minor=False)
            axx2.set_xticklabels(percent*(count+1),fontsize=ss,minor=True,rotation=45)
            axx2.set_xlabel("% of task completion",fontsize=ss)

        ax_list[i].tick_params(axis='x', labelsize=ss,   pad=14,length=3,width=0.5,direction= 'inout',which='major')
        ax_list[i].tick_params(axis='x', labelsize=ss-1, pad=2, length=3,width=0.5,direction= 'inout',which='minor')
        ax_list[i].tick_params(axis='y', labelsize=ss,   pad=3, length=3,width=0.5,direction= 'inout')

        ax_list2[i].tick_params(axis='x', labelsize=ss,   pad=14,length=3,width=0.5,direction= 'inout',which='major')
        ax_list2[i].tick_params(axis='x', labelsize=ss-1, pad=2, length=3,width=0.5,direction= 'inout',which='minor')
        ax_list2[i].tick_params(axis='y', labelsize=ss,   pad=3, length=3,width=0.5,direction= 'inout')


    if 'JM' in label:
        ax00.legend(fontsize=ss-1,loc='upper center',fancybox=True,ncol=1, frameon=True,framealpha=1, bbox_to_anchor=(3, 1.54))
        ax00.text(0.9, 1.35, "(I) Subject-exposed", transform=ax00.transAxes, size=ss+0.5,fontweight='bold')
        ax00.text(4.35, 1.35, "(II) Subject-naive", transform=ax00.transAxes, size=ss+0.5,fontweight='bold')
        
    elif 'Angles' in label:
        ax00.legend(fontsize=ss-1,loc='upper center',fancybox=True,ncol=1, frameon=True,framealpha=1, bbox_to_anchor=(3, 1.54))
        ax00.text(0.9, 1.35, "(I) Subject-exposed", transform=ax00.transAxes, size=ss+0.5,fontweight='bold')
        ax00.text(4.35, 1.35, "(II) Subject-naive", transform=ax00.transAxes, size=ss+0.5,fontweight='bold')

    elif 'Muscle' in label:
        ax00.legend(fontsize=ss-1,loc='upper center',fancybox=True,ncol=1, frameon=True,framealpha=1, bbox_to_anchor=(2.95, 1.55))
        ax00.text(0.94, 1.35, "(I) Subject-exposed", transform=ax00.transAxes, size=ss+0.5,fontweight='bold')
        ax00.text(4.35, 1.35, "(II) Subject-naive", transform=ax00.transAxes, size=ss+0.5,fontweight='bold')
        
    elif 'JRF' in label:
        ax00.legend(fontsize=ss-1,loc='upper center',fancybox=True,ncol=1, frameon=True,framealpha=1, bbox_to_anchor=(2.9, 1.55))
        ax00.text(0.9, 1.35, "(I) Subject-exposed", transform=ax00.transAxes, size=ss+0.5,fontweight='bold')
        ax00.text(4.35, 1.35, "(II) Subject-naive", transform=ax00.transAxes, size=ss+0.5,fontweight='bold')
    
    if scale_out == True:
        label = label + '_scaled_out'
    fig.savefig('./plots_out/Both_sub'+'_'+model_class+'_'+label+'_combine'+'.pdf',dpi=600)
    plt.close()

def create_PC_data(model,X1,Y2):    ###obselete now
    Y1 = model.predict(X1)
    Y1,Y2 = np.array(Y1),np.array(Y2) 
    Y1,Y2 = np.nan_to_num(Y1, nan=0),np.nan_to_num(Y2, nan=0) 
    a,b = np.shape(Y1)
    PC = np.zeros(b)
    for i in range(b):
        PC[i] = np.around(scipy.stats.pearsonr(Y1[:,i],Y2[:,i])[0],3)
    return PC

def save_outputs(model,hyper_val, X_Train, Y_Train, X_val, Y_val, feature, subject, enum, save_model):
    #   save_outputs(model, hyper_val, X_Train, Y_Train, X_Test, Y_Test, data.feature,data.subject, enum)
    train_error = create_PC_data(model,X_Train, Y_Train)
    val_error = create_PC_data(model,X_val, Y_val)   ## it is test error in case of final model
    mse = np.zeros(np.shape(train_error)[0])
    mse[0] = model.evaluate(X_Train, Y_Train,verbose=0)[0]
    mse[1] = model.evaluate(X_val, Y_val,verbose=0)[0]
    out = np.vstack([mse,train_error, val_error])
    out = np.nan_to_num(out, nan=0, posinf=2222)
    np.savetxt('./text_out/stat_'+ feature + '_' + subject +'.'+ 'hv_'+ str(hyper_val)   +'.CV_'+str(enum)+'.txt',out,fmt='%1.6f')
    if save_model == True:
        model.save('./model_out/model_'+ feature + '_' + subject +'.'+ 'hv_'+ str(hyper_val) + '.h5')
    return None

def run_NN(X_Train, Y_Train, X_val, Y_val,hyper_val,model_class):
    inp_dim = X_Train.shape[1]
    out_dim = Y_Train.shape[1]
    opt, kinit, batch_size, epoch, act, num_nodes, H_layer, metric, loss, lr, p , regularizer_val =   hyper_val
    if opt == 'Adam':
        optim = keras.optimizers.Adam
    elif opt == 'RMSprop':
        optim = keras.optimizers.RMSprop
    elif opt == 'SGD':
        optim = keras.optimizers.SGD
    #inp_dim, out_dim, nbr_Hlayer, Neu_layer, activation, p_drop, lr, optim,loss,metric,kinit
    final_act = None
    loss = keras.losses.mean_squared_error
    if model_class == 'NN':
        model = initiate_NN_model(inp_dim, out_dim, H_layer, num_nodes, act, p, lr, optim, loss, [metric], kinit,final_act,regularizer_val)
    elif model_class == 'LM':
        model = initiate_Linear_model(inp_dim, out_dim, H_layer, num_nodes, act, p, lr, optim, loss, [metric], kinit,final_act,regularizer_val)
    elif model_class == 'LR':
        model = initiate_LR_model(inp_dim, out_dim, H_layer, num_nodes, act, p, lr, optim, loss, [metric], kinit,final_act,regularizer_val)
    history = model.fit(X_Train, Y_Train, validation_data = (X_val,Y_val),epochs=epoch, batch_size=batch_size, verbose=2,shuffle=True)
    return model

def run_cross_valid(data,hyper_arg,hyper_val,model_class):
    save_model= False
    Da = [data.cv1, data.cv2, data.cv3, data.cv4]
    for enum,d in enumerate(Da):
        X_Train, Y_Train, X_Test, Y_Test = d['train_in'], d['train_out'], d['val_in'], d['val_out']
        model = run_NN(X_Train, Y_Train, X_Test, Y_Test, hyper_val,  model_class)
        save_outputs(model, hyper_arg, X_Train, Y_Train, X_Test, Y_Test, data.feature, data.subject, enum, save_model)
    return model

def run_final_model(data,hyper_arg,hyper_val,model_class):
	X_Train, Y_Train, X_Test, Y_Test = data.train_in, data.train_out, data.test_in, data.test_out
	model = run_NN(X_Train, Y_Train, X_Test, Y_Test, hyper_val,  model_class)
#	save_outputs(subject_condition,model, hyper_val, X_Train, Y_Train,X_Test, Y_Test, X_Test, Y_Test,label='comp'+which+'_hyper_'+str(hyper_arg)+'_')
	try:
		print("Plot created for the following index --- ",hyper_arg)#,hyper_val)
		save_outputs(data.subject,model,hyper_val, X_Train, Y_Train, X_Test, Y_Test, X_test, Y_test,'JRF')
		combined_plot_2(model,model,X_Test,Y_Test,X_Test,Y_Test,'JRF',False,model_class)	
		print("-----------------------------------","\n")
	except:
		print("this index is creating problem in printing --- ",hyper_arg,hyper_val )
		print("-----------------------------------","\n")
	return model

def create_final_model(hyper_arg,hyper_val,which,pca,scale_out, model_class):
	model = run_final_model(which,hyper_arg,hyper_val,pca,scale_out, model_class)
	return model


def load_model(subject_condition,hyper_arg,which):
    path = './model_out/'+subject_condition+'_comp'+which+'_hyper_'+str(hyper_arg)+'_.h5'  
    model = keras.models.load_model(path)
    return model

def plot_saved_model(subject_condition,which,hyper_arg,hyper_val,pca,scale_out,model_class):
	X_Train, Y_Train, X_Test, Y_Test = read_total_data(subject_condition,which,pca,scale_out,model_class)
	model = load_model(subject_condition,hyper_arg,which)
	try:
		print("Plot created for the following index --- ")#,which,hyper_val,hyper_arg)
# 		combined_plot(subject_condition,model,X_Train,Y_Train,"Train_"+which+"_"+str(hyper_arg),scale_out)
		combined_plot(subject_condition,model,X_Test,Y_Test,"Test_"+which+"_"+str(hyper_arg),scale_out,model_class)
		print("-----------------------------------","\n")
	except:
		print("this index is creating problem in printing --- ",which,hyper_arg,hyper_val )
		print("-----------------------------------","\n")

def plot_saved_model2(which, hyper_arg1,hyper_val1, hyper_arg2,hyper_val2, pca,scale_out,model_class):
	_, _, X_Test1, Y_Test1 = read_total_data('subject_exposed',which,pca,scale_out)
	_, _, X_Test2, Y_Test2 = read_total_data('subject_naive',which,pca,scale_out)
	model1 = load_model('subject_exposed',hyper_arg1,which)
	model2 = load_model('subject_naive' ,hyper_arg2,which)
	try:
		print("Plot created for the following index --- ",which,hyper_arg1,hyper_val1)
		combined_plot_2(model1,model2,X_Test1,Y_Test1,X_Test2,Y_Test2,"Test_"+which+"_"+str(hyper_arg1)+"_"+str(hyper_arg2),scale_out,model_class)
		print("-----------------------------------","\n")
	except:
		print("this index is creating problem in printing --- ",which,hyper_arg1,hyper_val1 )
		print("-----------------------------------","\n")


def interpolate(xnew,x,y):
    f1 = interp1d(x, y, kind='cubic')
    ynew = f1(xnew)
    return ynew

def check_interpolation(data):
    # this plot the input IMU data before and after interpolation and help visualize the interpolation
    xnew = np.linspace(0, 1, num=data.o1.T1.shape[0], endpoint=True)
    x = data.i1.T1['time']
    columns = data.i1.T1.columns.to_list()
    for enum,fea in enumerate(columns):
        y = data.i1.T1[fea]
        ynew = interpolate(xnew, x, y)
        fig,ax = plt.subplots(2)
        lw = 0.4
        ax[1].plot(x,y,color='r',lw=lw,label = 'data')
        ax[0].plot(x,y,color='r',lw=lw,label = 'data')
        ax[0].plot(xnew,ynew,color='b',lw=lw,label = 'cubic')
        ax[0].set_title(str(enum)+ '  '+ fea)
        ax[0].legend()
        ax[1].legend()
        plt.show()
        plt.close()
        time.sleep(1)
    return None
    
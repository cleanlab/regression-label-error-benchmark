import numpy as np
import random
import os
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from autogluon.tabular  import TabularPredictor
from statsmodels.nonparametric.kernel_regression import KernelReg as kr
from statsmodels.distributions.empirical_distribution import ECDF
from sklearn.metrics import roc_curve, auc, precision_recall_curve



def Get_data5D(n_train,n_test,run=50,outlier=[0,0.1],signal=5):
    X_train=[];X_cal=[];X_test=[]
    def f(x):
        return (x-1)**2*(x+1)
    def g(x):
        if (x>=0.5):
            return 2*np.sqrt(x-0.5)
        else:
            return 0
    for j in range(run):
        temp_x=np.zeros([n_train,5])
        temp_y=np.zeros(n_train)
        temp_x[:,0]=np.concatenate((np.random.uniform(-1.5,-0.5, int(0.1*n_train)), np.random.uniform(-0.5,1.5, int(0.9*n_train))))
        temp_x[:,[1,2,3,4]]=np.random.uniform(-1.5,1.5,[n_train,4])
        for i in range(n_train):
            a=np.random.binomial(1, 0.5)
            if (a==0):
                temp_y[i]=f(temp_x[i,0])-g(temp_x[i,0])+np.random.normal(0,1,1)*0.5
            if (a==1):
                temp_y[i]=f(temp_x[i,0])+g(temp_x[i,0])+np.random.normal(0,1,1)*0.5
        X_train.append({'X':temp_x,'Y':temp_y })
        temp_x=np.zeros([n_train,5])
        temp_y=np.zeros(n_train)
        temp_x[:,0]=np.concatenate((np.random.uniform(-1.5,-0.5, int(0.1*n_train)), np.random.uniform(-0.5,1.5, int(0.9*n_train))))
        temp_x[:,[1,2,3,4]]=np.random.uniform(-1.5,1.5,[n_train,4])
        for i in range(n_train):
            a=np.random.binomial(1, 0.5)
            if (a==0):
                temp_y[i]=f(temp_x[i,0])-g(temp_x[i,0])+np.random.normal(0,1,1)*0.5
            if (a==1):
                temp_y[i]=f(temp_x[i,0])+g(temp_x[i,0])+np.random.normal(0,1,1)*0.5
        X_cal.append({'X':temp_x,'Y':temp_y })
        temp_x=np.zeros([n_train,5])
        temp_y=np.zeros(n_train)
        temp_x[:,0]=np.concatenate((np.random.uniform(-1.5,-0.5, int(0.1*n_train)), np.random.uniform(-0.5,1.5, int(0.9*n_train))))
        temp_x[:,[1,2,3,4]]=np.random.uniform(-1.5,1.5,[n_train,4])
        for i in range(n_train):
            a=np.random.binomial(1, 0.5)
            if (a==0):
                temp_y[i]=f(temp_x[i,0])-g(temp_x[i,0])+np.random.normal(0,1,1)*0.5
            if (a==1):
                temp_y[i]=f(temp_x[i,0])+g(temp_x[i,0])+np.random.normal(0,1,1)*0.5
        X_test.append({'X':temp_x,'Y':temp_y })
    if outlier[0]!=0:
        for l in range(run):
            out_list=random.sample(range(n_train),int(outlier[0]*n_train))
            out_list.sort()
            out_id=np.array([0]*n_train) 
            out_id[out_list]=1
            for i in out_list:
                X_train[l]['Y'][i]=X_train[l]['Y'][i]+signal
            out_list=random.sample(range(n_train),int(outlier[0]*n_train))
            out_list.sort()
            out_id=np.array([0]*n_train) 
            out_id[out_list]=1
            for i in out_list:
                X_cal[l]['Y'][i]=X_cal[l]['Y'][i]+signal
    if outlier[1]!=0:
        for l in range(run):
            out_list=random.sample(range(n_test),int(outlier[1]*n_test))
            out_list.sort()
            out_id=np.array([0]*n_test) 
            out_id[out_list]=1
            for i in out_list:
                X_test[l]['Y'][i]=X_test[l]['Y'][i]+signal
            X_test[l]['Outlier']=out_id
    return {'train':X_train,'calibration':X_cal,'test':X_test}

def Get_data5DLM(n_train,n_test,p=5,run=50,outlier=[0,0.1],signal=5):
    X_train=[];X_cal=[];X_test=[]
    for j in range(run):
        random.seed(j+2023)
        sgn=np.random.choice([-1,1],int(p))
        def f(x):
            return sum(x*sgn)
        temp_x=np.zeros([n_train,p])
        temp_y=np.zeros(n_train)
        temp_x=np.random.uniform(-1.5,1.5,[n_train,p])
        for i in range(n_train):
            temp_y[i]=f(temp_x[i,:])+0.5*np.random.normal(0,1,1)
        X_train.append({'X':temp_x,'Y':temp_y })
        temp_x=np.zeros([n_train,p])
        temp_y=np.zeros(n_train)
        temp_x=np.random.uniform(-1.5,1.5,[n_train,p])
        for i in range(n_train):
            temp_y[i]=f(temp_x[i,:])+0.5*np.random.normal(0,1,1)
        X_cal.append({'X':temp_x,'Y':temp_y })
        temp_x=np.zeros([n_train,p])
        temp_y=np.zeros(n_train)
        temp_x=np.random.uniform(-1.5,1.5,[n_train,p])
        for i in range(n_train):
            temp_y[i]=f(temp_x[i,:])+0.5*np.random.normal(0,1,1)
        X_test.append({'X':temp_x,'Y':temp_y })
    if outlier[0]!=0:
        for l in range(run):
            out_list=random.sample(range(n_train),int(outlier[0]*n_train))
            out_list.sort()
            out_id=np.array([0]*n_train) 
            out_id[out_list]=1
            for i in out_list:
                X_train[l]['Y'][i]=X_train[l]['Y'][i]+signal
            X_train[l]['Outlier']=out_id
            out_list=random.sample(range(n_train),int(outlier[0]*n_train))
            out_list.sort()
            out_id=np.array([0]*n_train) 
            out_id[out_list]=1
            for i in out_list:
                X_cal[l]['Y'][i]=X_cal[l]['Y'][i]+signal
            X_cal[l]['Outlier']=out_id
    if outlier[1]!=0:
        for l in range(run):
            out_list=random.sample(range(n_test),int(outlier[1]*n_test))
            out_list.sort()
            out_id=np.array([0]*n_test) 
            out_id[out_list]=1
            for i in out_list:
                X_test[l]['Y'][i]=X_test[l]['Y'][i]+signal
            X_test[l]['Outlier']=out_id
    return {'train':X_train,'calibration':X_cal,'test':X_test}

def intersection(list_a, list_b):
    return [ e for e in list_a if e in list_b ]    

def CV_bw(Data_train,k=5):
    n_train=len(Data_train['X'])
    L=int(n_train/k)
    def obj(bw):
        if bw<=0:
            return 100
        else:
            S=0
            for fold in range(k):
                    test_id=(np.linspace(fold*L,(1+fold)*L-1,L) ).astype(int)
                    train_id=np.setdiff1d(range(n_train),test_id)
                    X_in=Data_train['X'][train_id];Y_in=Data_train['Y'][train_id]
                    X_out=Data_train['X'][test_id];Y_out=Data_train['Y'][test_id]
                    def F_hat(x,y):
                        temp_1=(X_in>=x-bw).nonzero()[0]
                        temp_2=(X_in<=x+bw).nonzero()[0]
                        id_in=intersection(temp_1,temp_2)
                        if (len(id_in)==0):
                            return 100
                        else:
                            F_y=ECDF(Y_in[id_in])
                            return F_y(y)    
                    F_xy=np.zeros(L)
                    for j in range(L):
                        F_xy[j]=F_hat(X_out[j],Y_out[j] )
                    S=S+(np.mean(F_xy)-0.5)**2 
        return(S)
    return minimize(obj, 0.2,method='Nelder-Mead').x    

    
def Fdr_power(Data,dect,l):
    n_test=len(Data['test'][l]['X'] )
    if ((dect==[0]*n_test).all()):
        return {'FDR':0,'Power':0}
    else:
        temp_fdr=0;temp_power=0
        for i in range(n_test):
            if Data['test'][l]['Outlier'][i]==1:
                if dect[i]==1:
                    temp_power=temp_power+1
            if dect[i]==1:
                if Data['test'][l]['Outlier'][i]==0:
                    temp_fdr=temp_fdr+1
        Fdr=temp_fdr/sum(dect);power=temp_power/sum(Data['test'][l]['Outlier'])
        return {'FDR':Fdr,'Power':power}
                
def BH(v,alpha=0.1):
    m=len(v)
    v_temp=sorted(v)
    if ((v_temp>np.linspace(1, m,m)*alpha/m).all() ):
        return np.array([0]*m)
    else:
        temp=(v_temp<=np.linspace(1, m,m)*alpha/m )
        return v<=v_temp[max(max((temp).nonzero()))]


def Conformal_train(Data_train,l,n_boot=20,run=50):
    model_ll=RandomForestRegressor()
    model_lx=RandomForestRegressor()
    ux=RandomForestRegressor()
    n_train=len(Data_train[l]['X'] )
    model_ll.fit(Data_train[l]['X'],Data_train[l]['Y']);
    model_lx.fit(Data_train[l]['X'], (Data_train[l]['Y']-model_ll.predict(Data_train[l]['X']) )**2 );
    tempx=np.zeros([50,5])
    for i in range(5):
        tempx[:,i]=np.linspace(-1.5, 1.5)   
    hatf=np.zeros([n_boot,run])
    for bt in range(n_boot):
        idx=random.sample(range(n_train),k=int(0.8*n_train))
        model_temp=RandomForestRegressor()
        model_temp.fit(Data_train[l]['X'][idx], Data_train[l]['Y'][idx]);        
        hatf[bt,:]=model_temp.predict(tempx)
    varf=np.sqrt( np.nanvar(hatf,axis=0))
    ux.fit(tempx,varf);
    return {'hat_mu':model_ll,'hat_sigma':model_lx,'hat_u':ux}
        
    
def Conformal_eva(Data,trained_model,l):
    res_cal=(abs(trained_model['hat_mu'].predict(Data['calibration'][l]['X'])- Data['calibration'][l]['Y']))
    scores_res_cal=res_cal
    scores_ari_cal=res_cal/(trained_model['hat_u'].predict(Data['calibration'][l]['X'])+np.sqrt(abs(trained_model['hat_sigma'].predict(Data['calibration'][l]['X']))))
    scores_geo_cal=res_cal/np.sqrt(trained_model['hat_u'].predict(Data['calibration'][l]['X'])*np.sqrt(abs(trained_model['hat_sigma'].predict(Data['calibration'][l]['X']))))    
    F_scores_res=ECDF(scores_res_cal)
    F_scores_ari=ECDF(scores_ari_cal)
    F_scores_geo=ECDF(scores_geo_cal)
    res_test=(abs(trained_model['hat_mu'].predict(Data['test'][l]['X'])- Data['test'][l]['Y']))
    scores_res_test=res_test
    scores_ari_test=res_test/(trained_model['hat_u'].predict(Data['test'][l]['X'])+np.sqrt(abs(trained_model['hat_sigma'].predict(Data['test'][l]['X']))))
    scores_geo_test=res_test/np.sqrt(trained_model['hat_u'].predict(Data['test'][l]['X'])*np.sqrt(abs(trained_model['hat_sigma'].predict(Data['test'][l]['X']))))  
    pvalue_res=1-F_scores_res(scores_res_test-1E-8)
    pvalue_ari=1-F_scores_ari(scores_ari_test-1E-8)
    pvalue_geo=1-F_scores_geo(scores_geo_test-1E-8)
    FP_res=Fdr_power(Data,BH(pvalue_res,0.1),l)
    FP_ari=Fdr_power(Data,BH(pvalue_ari,0.1),l)
    FP_geo=Fdr_power(Data,BH(pvalue_geo,0.1),l)
    return {'res':FP_res,'ari':FP_ari,'geo':FP_geo}

def Conformal_train_atg2(Data_train,l,n_boot=20,run=50):
    X=Data_train[l]['X']
    y=Data_train[l]['Y']
    X = pd.DataFrame(X) 
    y = pd.DataFrame(y)
    y.columns = ['label']
    training_data = pd.concat([X, y], axis=1)
    savepath='./autogluon'
    if not os.path.isdir(savepath+"/training_y/"):
        os.mkdir(savepath+"/training_y/")
    predictorArgs={'path': savepath+"/training_y/",
        'label':  "label", 
        'eval_metric':  "r2", 
        'problem_type': "regression",
        'verbosity': 0}
    hyperparameters: dict = {
    'GBM': [
        {'ag_args_fit': {'num_gpus': 10}},  # Train with CPU
        {'ag_args_fit': {'num_gpus': 1}}   # Train with GPU. This amount needs to be <= total num_gpus granted to TabularPredictor
    ]
}
    predictor_y=TabularPredictor(**predictorArgs)
    predictor_y.fit(training_data,presets="best_quality", 
                    num_stack_levels = 0, 
                    hyperparameters=hyperparameters)
    ##train sigma
    model_name=predictor_y.get_model_names()[0]
    predict_y=predictor_y.get_oof_pred(model=model_name)
    training_sigma=training_data.drop(['label'], axis=1)
    training_sigma['sigma_square']=(predict_y-training_data['label'] )**2 
    if not os.path.isdir(savepath+"/training_sigma/"):
        os.mkdir(savepath+"/training_sigma/")
    predictorArgs={'path': savepath+"/training_sigma/",
        'label':  "sigma_square", 
        'eval_metric':  "r2", 
        'problem_type': "regression",
        'verbosity': 0}
    predictor_sigma=TabularPredictor(**predictorArgs)
    predictor_sigma.fit(training_sigma, presets="best_quality", num_stack_levels= 0,hyperparameters=hyperparameters)
    ## train u
    if not os.path.isdir(savepath+"/training_u/"):
        os.mkdir(savepath+"/training_u/")
    p=len(Data_train[0]['X'][0,:] )
    tempx=np.zeros([50,p])
    for i in range(5):
        tempx[:,i]=np.linspace(-1.5, 1.5)   
    hatf=np.zeros([n_boot,50])
    for i in range(n_boot):
        if not os.path.isdir(savepath+"/training_u/"+str(i)+'/'):
            os.mkdir(savepath+"/training_u/"+str(i)+'/')
        predictorArgs={'path': savepath+"/training_u/"+str(i)+'/',
             'label':  "label", 
             'eval_metric':  "r2", 
             'problem_type': "regression",
             'verbosity': 0}
        random.seed(i)
        train_idx=sorted(random.sample(sorted(training_data.index) , int(len(training_data )*0.8 ) ) )
        data_bootstrap=training_data[  training_data.index.isin(train_idx)]
        predictor_u=TabularPredictor(**predictorArgs)
        predictor_u.fit(data_bootstrap, presets="best_quality", num_stack_levels= 0,hyperparameters=hyperparameters)
        hatf[i,:]=predictor_u.predict(pd.DataFrame (tempx))
    varf=np.sqrt( np.nanvar(hatf,axis=0))
    ux=RandomForestRegressor()
    ux.fit(tempx,varf);
    return {'hat_u':ux}
    

    
def Conformal_eva_atg2(Data,trained_model,l):
    savepath='./autogluon'
    predictor_y=TabularPredictor.load(savepath+"/training_y/") 
    model_name=predictor_y.get_model_names()[0]
    y_cal_pred=predictor_y.predict(pd.DataFrame(Data['calibration'][l]['X']),model=model_name )
    predictor_sigma=TabularPredictor.load(savepath+"/training_sigma/") 
    sigma_cal_pred=np.sqrt(abs(predictor_sigma.predict(pd.DataFrame(Data['calibration'][l]['X']),model=model_name )))
    u_cal_pred=trained_model['hat_u'].predict(Data['calibration'][l]['X'])
    res_cal=abs(y_cal_pred- Data['calibration'][l]['Y'] )
    scores_res_cal=res_cal
    scores_ari_cal=res_cal/(sigma_cal_pred+u_cal_pred )
    scores_geo_cal=res_cal/np.sqrt(sigma_cal_pred*u_cal_pred )
    F_scores_res=ECDF(scores_res_cal)
    F_scores_ari=ECDF(scores_ari_cal)
    F_scores_geo=ECDF(scores_geo_cal)
    ##testing set
    y_test_pred=predictor_y.predict(pd.DataFrame(Data['test'][l]['X']),model=model_name )
    sigma_test_pred=np.sqrt(abs( predictor_sigma.predict(pd.DataFrame(Data['test'][l]['X']),model=model_name )))
    u_test_pred=trained_model['hat_u'].predict(Data['test'][l]['X'])
    res_test=abs(y_test_pred- Data['test'][l]['Y'] )
    scores_res_test=res_test
    scores_ari_test=res_test/(sigma_test_pred+u_test_pred )
    scores_geo_test=res_test/np.sqrt(sigma_test_pred*u_test_pred )
    pvalue_res=1-F_scores_res(scores_res_test-1E-8)
    pvalue_ari=1-F_scores_ari(scores_ari_test-1E-8)
    pvalue_geo=1-F_scores_geo(scores_geo_test-1E-8)
    FP_res=Fdr_power(Data,BH(pvalue_res,0.1),l)
    FP_ari=Fdr_power(Data,BH(pvalue_ari,0.1),l)
    FP_geo=Fdr_power(Data,BH(pvalue_geo,0.1),l)
    return {'res':FP_res,'ari':FP_ari,'geo':FP_geo}    
    
    
def Eva_scores(Data,l):
    X=Data['train'][l]['X']
    y=Data['train'][l]['Y']
    X = pd.DataFrame(X) 
    y = pd.DataFrame(y)
    y.columns = ['label']
    training_data = pd.concat([X, y], axis=1)
    savepath='./autogluon'
    if not os.path.isdir(savepath):
        os.mkdir(savepath)
    if not os.path.isdir(savepath+"/training_y/"):
        os.mkdir(savepath+"/training_y/")
    predictorArgs={'path': savepath+"/training_y/",
        'label':  "label", 
        'eval_metric':  "r2", 
        'problem_type': "regression",
        'verbosity': 0}
    hyperparameters={  'GBM'   : {}}
    predictor_y=TabularPredictor(**predictorArgs)
    predictor_y.fit(training_data,presets="best_quality", 
                    num_stack_levels = 0, 
                    hyperparameters=hyperparameters)
    # ##train sigma
    model_name=predictor_y.get_model_names()[0]
    predict_y=predictor_y.get_oof_pred(model=model_name)
    training_sigma=training_data.drop(['label'], axis=1)
    training_sigma['sigma_square']=(predict_y-training_data['label'] )**2 
    if not os.path.isdir(savepath+"/training_sigma/"):
        os.mkdir(savepath+"/training_sigma/")
    predictorArgs={'path': savepath+"/training_sigma/",
        'label':  "sigma_square", 
        'eval_metric':  "r2", 
        'problem_type': "regression",
        'verbosity': 0}
    predictor_sigma=TabularPredictor(**predictorArgs)
    predictor_sigma.fit(training_sigma, presets="best_quality", num_stack_levels= 0,hyperparameters=hyperparameters)
    ## get predictions
    testing_data=pd.DataFrame(Data['test'][l]['X']) 
    predictions_y_test=predictor_y.predict(testing_data,model=model_name)
    predictions_sigma_test=np.sqrt(abs( predictor_sigma.predict(testing_data,model=model_name) ))
    predictions_y_oof=predictor_y.get_oof_pred(model=model_name)
    predictions_sigma_oof=np.sqrt(abs( predictor_sigma.get_oof_pred(model=model_name) ))
    predictions_y_train=predictor_y.predict(X,model=model_name)
    predictions_sigma_train=np.sqrt(abs( predictor_sigma.predict(X,model=model_name) ))
    ## train u
    n_boot=20
    if not os.path.isdir(savepath+"/training_u/"):
        os.mkdir(savepath+"/training_u/")
    hatf_test=np.zeros([n_boot,len(Data['test'][0]['X'][:,0] )  ])
    hatf_train=np.zeros([n_boot,len(Data['train'][0]['X'][:,0] )  ])
    hatf_oof=np.zeros([n_boot,len(Data['train'][0]['X'][:,0] )  ])
    for i in range(n_boot):
        if not os.path.isdir(savepath+"/training_u/"+str(i)+'/'):
            os.mkdir(savepath+"/training_u/"+str(i)+'/')
        predictorArgs={'path': savepath+"/training_u/"+str(i)+'/',
             'label':  "label", 
             'eval_metric':  "r2", 
             'problem_type': "regression",
             'verbosity': 0}
        random.seed(i)
        train_idx=sorted(random.sample(sorted(training_data.index) , int(len(training_data )*0.8 ) ) )
        data_bootstrap=training_data[  training_data.index.isin(train_idx)]
        data_out=X[ ~X.index.isin(train_idx)]
        predictor_u=TabularPredictor(**predictorArgs)
        predictor_u.fit(data_bootstrap, presets="best_quality", num_stack_levels= 0,hyperparameters=hyperparameters)
        hatf_test[i,:]=predictor_u.predict(testing_data,model=model_name)
        hatf_train[i,:]=predictor_u.predict(X,model=model_name)
        pred_u_in=predictor_u.get_oof_pred()
        pred_u_out=predictor_u.predict(data_out,model=model_name)
        pred_u=pd.concat([pred_u_in,pred_u_out])
        hatf_oof[i,:]=pred_u.sort_index()
    predictions_u_test=np.sqrt( np.nanvar(hatf_test,axis=0))
    predictions_u_train=np.sqrt( np.nanvar(hatf_train,axis=0))
    predictions_u_oof=np.sqrt( np.nanvar(hatf_oof,axis=0))
    ## get scores
    Scores={}  
    S_res_test=abs( predictions_y_test-Data['test'][l]['Y'])
    S_ari_test=S_res_test/(predictions_sigma_test+predictions_u_test)
    S_geo_test=S_res_test/np.sqrt(predictions_sigma_test*predictions_u_test)
    S_res_train=abs( predictions_y_train-Data['train'][l]['Y'])
    S_ari_train=S_res_train/(predictions_sigma_train+predictions_u_train)
    S_geo_train=S_res_train/np.sqrt(predictions_sigma_train*predictions_u_train)
    S_res_oof=abs( predictions_y_oof-Data['train'][l]['Y'])
    S_ari_oof=S_res_oof/(predictions_sigma_oof+predictions_u_oof)
    S_geo_oof=S_res_oof/np.sqrt(predictions_sigma_oof*predictions_u_oof)
    Scores['test']={'res':S_res_test,'ari':S_ari_test,'geo':S_geo_test}
    Scores['train']={'res':S_res_train,'ari':S_ari_train,'geo':S_geo_train}
    Scores['oof']={'res':S_res_oof,'ari':S_ari_oof,'geo':S_geo_oof}
    return Scores

def Get_au(Data,l,scores,type_p):
    if (type_p=="test"):
        FPR_temp, TPR_temp, _ = roc_curve(Data['test'][l]['Outlier'] , scores)
        auR= auc(FPR_temp,TPR_temp)
        PRECISION_temp, RECALL_temp, _ = precision_recall_curve(Data['test'][l]['Outlier'] , scores )
        auP = auc(RECALL_temp, PRECISION_temp)
    if (type_p=="train" or type_p=="oof"):
        FPR_temp, TPR_temp, _ = roc_curve(Data['train'][l]['Outlier'] , scores)
        auR = auc(FPR_temp,TPR_temp)
        PRECISION_temp, RECALL_temp, _ = precision_recall_curve(Data['train'][l]['Outlier'] , scores )
        auP= auc(RECALL_temp, PRECISION_temp)
    return auR,auP;

def removed_list(Data,l,Scores):
    X=Data['train'][l]['X']
    y=Data['train'][l]['Y']
    X = pd.DataFrame(X) 
    y = pd.DataFrame(y)
    y.columns = ['label']
    training_data = pd.concat([X, y], axis=1)
    rem_p_p1=[0.05,0.1,0.15,0.2]
    savepath='./auto_y'
    if not os.path.isdir(savepath):
        os.mkdir(savepath)
    predictorArgs={'path': savepath,
        'label':  "label", 
        'eval_metric':  "r2", 
        'problem_type': "regression",
        'verbosity': 0}
    hyperparameters={  'GBM'   : {}}
    removed_list={}
    for s_type in ['res','ari','geo']:
        rsquare_p1=np.zeros(4)
        s_current=Scores['oof'][s_type]
        removed_id_p1={}
        for i in range(4):
            removed_id_p1[str(rem_p_p1[i])]=np.where(s_current>=np.quantile(s_current, 1-rem_p_p1[i]))[0]
            data_removed=training_data[ ~training_data.index.isin(removed_id_p1[str(rem_p_p1[i])])]
            data_removed_out=training_data[ training_data.index.isin(removed_id_p1[str(rem_p_p1[i])])]
            data_removed_out=data_removed_out.drop(['label'], axis=1)
            predictor_y=TabularPredictor(**predictorArgs)
            predictor_y.fit(data_removed,presets="best_quality", 
                            num_stack_levels = 0, 
                            hyperparameters=hyperparameters)
            model_name=predictor_y.get_model_names()[0]
            pred_y_in=predictor_y.get_oof_pred(model=model_name)
            pred_y_out=predictor_y.predict(data_removed_out,model=model_name)
            pred_y=pd.concat([pred_y_in,pred_y_out])
            pred_y=pred_y.sort_index()
            rsquare_p1[i]=1-sum((pred_y-training_data['label'] )**2 )/sum((training_data['label']-np.mean(training_data['label']) )**2 )
        p_p1=rem_p_p1[np.where(rsquare_p1==max(rsquare_p1))[0][0]]
        rem_p_p2=[p_p1-0.04,p_p1-0.03,p_p1-0.02,p_p1-0.01,p_p1,p_p1+0.01,p_p1+0.02,p_p1+0.03,p_p1+0.04 ]
        removed_id_p2={}
        rsquare_p2=np.zeros(9)
        for i in range(9):
            removed_id_p2[i]=np.where(s_current>=np.quantile(s_current, 1-rem_p_p2[i]))[0]
            data_removed=training_data[ ~training_data.index.isin(removed_id_p2[i])]
            data_removed_out=training_data[ training_data.index.isin(removed_id_p2[i])]
            data_removed_out=data_removed_out.drop(['label'], axis=1)
            predictor_y=TabularPredictor(**predictorArgs)
            predictor_y.fit(data_removed,presets="best_quality", 
                            num_stack_levels = 0, 
                            hyperparameters=hyperparameters)
            model_name=predictor_y.get_model_names()[0]
            pred_y_in=predictor_y.get_oof_pred(model=model_name)
            pred_y_out=predictor_y.predict(data_removed_out,model=model_name)
            pred_y=pd.concat([pred_y_in,pred_y_out])
            pred_y=pred_y.sort_index()
            rsquare_p2[i]=1-sum((pred_y-training_data['label'] )**2 )/sum((training_data['label']-np.mean(training_data['label']) )**2 )
        removed_list[s_type]=removed_id_p2[np.where(rsquare_p2==max(rsquare_p2))[0][0]]
    return removed_list


def Eva_a_score(Data,l,removed_list):
    X=Data['train'][l]['X']
    y=Data['train'][l]['Y']
    Outlier=Data['train'][l]['Outlier']
    n=len(y)
    X = pd.DataFrame(X) 
    y = pd.DataFrame(y)
    Outlier=pd.DataFrame(Outlier)
    y.columns = ['label']
    Outlier.columns = ['Outlier']
    Data_all = pd.concat([X, y,Outlier], axis=1)
    training_data=Data_all[ ~Data_all.index.isin(removed_list )]
    training_data_out=Data_all[ Data_all.index.isin(removed_list )]
    removed_prop=1-len(training_data)/n
    out_prop=sum(training_data['Outlier'])/len(training_data)
    training_data=training_data.drop(['Outlier'], axis=1)
    training_data_out=training_data_out.drop(['Outlier','label'], axis=1)
    savepath='./autogluon'
    if not os.path.isdir(savepath):
        os.mkdir(savepath)
    if not os.path.isdir(savepath+"/training_y/"):
        os.mkdir(savepath+"/training_y/")
    predictorArgs={'path': savepath+"/training_y/",
        'label':  "label", 
        'eval_metric':  "r2", 
        'problem_type': "regression",
        'verbosity': 0}
    hyperparameters={  'GBM'   : {}}
    predictor_y=TabularPredictor(**predictorArgs)
    predictor_y.fit(training_data,presets="best_quality", 
                    num_stack_levels = 0, 
                    hyperparameters=hyperparameters)
    # ##train sigma
    model_name=predictor_y.get_model_names()[0]
    predict_y=predictor_y.get_oof_pred(model=model_name)
    training_sigma=training_data.drop(['label'], axis=1)
    training_sigma['sigma_square']=(predict_y-training_data['label'] )**2 
    if not os.path.isdir(savepath+"/training_sigma/"):
        os.mkdir(savepath+"/training_sigma/")
    predictorArgs={'path': savepath+"/training_sigma/",
        'label':  "sigma_square", 
        'eval_metric':  "r2", 
        'problem_type': "regression",
        'verbosity': 0}
    predictor_sigma=TabularPredictor(**predictorArgs)
    predictor_sigma.fit(training_sigma, presets="best_quality", num_stack_levels= 0,hyperparameters=hyperparameters)
    ## get predictions
    predictions_y_in=predictor_y.get_oof_pred(model=model_name)
    predictions_y_out=predictor_y.predict(training_data_out,model=model_name)
    predictions_y=pd.concat([predictions_y_in,predictions_y_out])
    predictions_y=predictions_y.sort_index()
    predictions_sigma_in=np.sqrt(abs( predictor_sigma.get_oof_pred(model=model_name) ))
    predictions_sigma_out=np.sqrt(abs( predictor_sigma.predict(training_data_out,model=model_name) ))
    predictions_sigma=pd.concat([predictions_sigma_in,predictions_sigma_out])
    predictions_sigma=predictions_sigma.sort_index()
    ## train u
    n_boot=20
    if not os.path.isdir(savepath+"/training_u/"):
        os.mkdir(savepath+"/training_u/")
    hatf_test=np.zeros([n_boot,len(Data['train'][0]['X'][:,0] )  ])
    for i in range(n_boot):
        if not os.path.isdir(savepath+"/training_u/"+str(i)+'/'):
            os.mkdir(savepath+"/training_u/"+str(i)+'/')
        predictorArgs={'path': savepath+"/training_u/"+str(i)+'/',
             'label':  "label", 
             'eval_metric':  "r2", 
             'problem_type': "regression",
             'verbosity': 0}
        random.seed(i)
        train_idx=sorted(random.sample(sorted(training_data.index) , int(len(training_data )*0.8 ) ) )
        data_bootstrap=training_data[  training_data.index.isin(train_idx)]
        data_bootstrap_out=training_data[  ~training_data.index.isin(train_idx)]
        data_bootstrap_out=data_bootstrap_out.drop(['label'], axis=1)
        predictor_u=TabularPredictor(**predictorArgs)
        predictor_u.fit(data_bootstrap, presets="best_quality", num_stack_levels= 0,hyperparameters=hyperparameters)
        predictions_u_in=predictor_u.get_oof_pred(model=model_name)
        predictions_u_out=predictor_u.predict(training_data_out,model=model_name)
        predictions_u_out_bt=predictor_u.predict(data_bootstrap_out,model=model_name)
        predictions_u=pd.concat([predictions_u_in,predictions_u_out,predictions_u_out_bt])
        hatf_test[i,:]=predictions_u.sort_index()
    predictions_u=np.sqrt( np.nanvar(hatf_test,axis=0))
    ## get scores
    Scores={}  
    S_res_test=abs( predictions_y-Data['train'][l]['Y'])
    S_ari_test=S_res_test/(predictions_sigma+predictions_u)
    S_geo_test=S_res_test/np.sqrt(predictions_sigma*predictions_u)
    Scores={'res':S_res_test,'ari':S_ari_test,'geo':S_geo_test}
    return Scores,removed_prop,out_prop


def Eva_b_score(Data,l):
    X=Data['train'][l]['X']
    y=Data['train'][l]['Y']
    Outlier=Data['train'][l]['Outlier']
    n=len(y)
    X = pd.DataFrame(X) 
    y = pd.DataFrame(y)
    Outlier=pd.DataFrame(Outlier)
    y.columns = ['label']
    Outlier.columns = ['Outlier']
    training_data = pd.concat([X, y,Outlier], axis=1)
    training_data=training_data.drop(['Outlier'], axis=1)
    savepath='./autogluon'
    if not os.path.isdir(savepath):
        os.mkdir(savepath)
    if not os.path.isdir(savepath+"/training_y/"):
        os.mkdir(savepath+"/training_y/")
    predictorArgs={'path': savepath+"/training_y/",
        'label':  "label", 
        'eval_metric':  "r2", 
        'problem_type': "regression",
        'verbosity': 0}
    hyperparameters={  'GBM'   : {}}
    predictor_y=TabularPredictor(**predictorArgs)
    predictor_y.fit(training_data,presets="best_quality", 
                    num_stack_levels = 0, 
                    hyperparameters=hyperparameters)
    # ##train sigma
    model_name=predictor_y.get_model_names()[0]
    predict_y=predictor_y.get_oof_pred(model=model_name)
    training_sigma=training_data.drop(['label'], axis=1)
    training_sigma['sigma_square']=(predict_y-training_data['label'] )**2 
    if not os.path.isdir(savepath+"/training_sigma/"):
        os.mkdir(savepath+"/training_sigma/")
    predictorArgs={'path': savepath+"/training_sigma/",
        'label':  "sigma_square", 
        'eval_metric':  "r2", 
        'problem_type': "regression",
        'verbosity': 0}
    predictor_sigma=TabularPredictor(**predictorArgs)
    predictor_sigma.fit(training_sigma, presets="best_quality", num_stack_levels= 0,hyperparameters=hyperparameters)
    ## get predictions
    predictions_y=predictor_y.get_oof_pred(model=model_name)
    predictions_sigma=np.sqrt(abs( predictor_sigma.get_oof_pred(model=model_name) ))
    ## train u
    n_boot=20
    if not os.path.isdir(savepath+"/training_u/"):
        os.mkdir(savepath+"/training_u/")
    hatf_test=np.zeros([n_boot,len(Data['train'][0]['X'][:,0] )  ])
    for i in range(n_boot):
        if not os.path.isdir(savepath+"/training_u/"+str(i)+'/'):
            os.mkdir(savepath+"/training_u/"+str(i)+'/')
        predictorArgs={'path': savepath+"/training_u/"+str(i)+'/',
             'label':  "label", 
             'eval_metric':  "r2", 
             'problem_type': "regression",
             'verbosity': 0}
        random.seed(i)
        train_idx=sorted(random.sample(sorted(training_data.index) , int(len(training_data )*0.8 ) ) )
        data_bootstrap=training_data[  training_data.index.isin(train_idx)]
        data_bootstrap_out=training_data[  ~training_data.index.isin(train_idx)]
        data_bootstrap_out=data_bootstrap_out.drop(['label'], axis=1)
        predictor_u=TabularPredictor(**predictorArgs)
        predictor_u.fit(data_bootstrap, presets="best_quality", num_stack_levels= 0,hyperparameters=hyperparameters)
        predictions_u_in=predictor_u.get_oof_pred(model=model_name)
        predictions_u_out_bt=predictor_u.predict(data_bootstrap_out,model=model_name)
        predictions_u=pd.concat([predictions_u_in,predictions_u_out_bt])
        hatf_test[i,:]=predictions_u.sort_index()
    predictions_u=np.sqrt( np.nanvar(hatf_test,axis=0))
    ## get scores
    Scores={}  
    S_res_test=abs( predictions_y-Data['train'][l]['Y'])
    S_ari_test=S_res_test/(predictions_sigma+predictions_u)
    S_geo_test=S_res_test/np.sqrt(predictions_sigma*predictions_u)
    Scores={'res':S_res_test,'ari':S_ari_test,'geo':S_geo_test}
    return Scores






 

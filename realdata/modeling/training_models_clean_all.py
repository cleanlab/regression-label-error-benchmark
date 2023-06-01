import os
import glob
import numpy as np 
import pandas as pd 
from autogluon.tabular import  TabularPredictor
import random 
import pickle


data_path = "./../dataset/"
all_datasets=os.listdir(data_path)
dataset_names=[]
for path in all_datasets:
    dataset_names.append(os.path.splitext(os.path.basename(path))[0])

for data_name in dataset_names:
    n_boot=20
    testing_data=pd.read_csv(data_path+data_name+".csv", index_col=0)
    training_data=testing_data[testing_data['true_error']==0 ]
    training_data = training_data.drop(['true_label', 'true_error'], axis=1)  
    ##train hat{f}
    savepath = "./../modeling/"+data_name+"/"
    if not os.path.isdir(savepath):
        os.mkdir(savepath) 
    savepath = "./../modeling/"+data_name+"/trained_models/"
    if not os.path.isdir(savepath):
        os.mkdir(savepath) 
    savepath = "./../modeling/"+data_name+"/trained_models/hatf/"
    if not os.path.isdir(savepath):
        os.mkdir(savepath) 
    predictorArgs = {
                        "path"             : savepath, 
                        "label"            : 'given_label',
                        "eval_metric"      : 'r2',
                        "problem_type"     : 'regression'
                    }
    hyperparameters = { 
                        'GBM'   : {}, 
                        'FASTAI': {}, 
                        'RF'    : {'criterion': 'squared_error', 
                                    'ag_args': {'name_suffix': 'MSE', 
                                    'problem_types': ['regression', 'quantile']}}
                        }
    predictor = TabularPredictor(**predictorArgs)
    predictor.fit(training_data, hyperparameters=hyperparameters)
    leaderboard = predictor.leaderboard(training_data, silent=True)
    predictor_cv = TabularPredictor(**predictorArgs) 
    predictor_cv.fit(training_data, presets="best_quality", num_stack_levels= 0, hyperparameters=hyperparameters)
    leaderboard_cv = predictor_cv.leaderboard(training_data, silent=True) 
    
    models = {  'normal'            : list(leaderboard['model']), 
                'crossValidation'   : list(np.append(leaderboard_cv['model'], ['oof']))}

    predictors = { 'normal'          : predictor, 
                   'crossValidation' : predictor_cv}
    #predictions on the testing set 
    data_without_label = testing_data.drop(['given_label','true_label', 'true_error'], axis=1)
    pred_path =  "./../modeling/"+data_name+"/predictions/testing_set/"
    if not os.path.isdir(pred_path):
        os.mkdir(pred_path)

    for model_type in models.keys():
        if model_type == "normal": 
            for model in models[model_type]:
                savepath = os.path.join(pred_path, model+".npy")
                predictions = predictors[model_type].predict(data_without_label, model=model)
                np.save(savepath, np.array(predictions))
        elif model_type == "crossValidation":
            for model in models[model_type]:
                if model == "oof":
                    predictions = predictors[model_type].get_oof_pred()
                    savepath = os.path.join(pred_path, model+".npy")
                elif model in models['normal']:
                    predictions = predictors[model_type].predict(data_without_label, model=model)
                    savepath = os.path.join(pred_path, model+"_CV.npy")
                else: 
                    predictions = predictors[model_type].predict(data_without_label, model=model)
                    savepath = os.path.join(pred_path, model+".npy")

                np.save(savepath, np.array(predictions))
    #predictions on the training set
    data_without_label = training_data.drop(['given_label'], axis=1)
    pred_path = "./../modeling/"+data_name+"/predictions/training_set/"
    if not os.path.isdir(pred_path):
        os.mkdir(pred_path)

    for model_type in models.keys():
        if model_type == "normal": 
            for model in models[model_type]:
                savepath = os.path.join(pred_path, model+".npy")
                predictions = predictors[model_type].predict(data_without_label, model=model)
                np.save(savepath, np.array(predictions))
        elif model_type == "crossValidation":
            for model in models[model_type]:
                if model == "oof":
                    predictions = predictors[model_type].get_oof_pred()
                    savepath = os.path.join(pred_path, model+".npy")
                elif model in models['normal']:
                    predictions = predictors[model_type].predict(data_without_label, model=model)
                    savepath = os.path.join(pred_path, model+"_CV.npy")
                else: 
                    predictions = predictors[model_type].predict(data_without_label, model=model)
                    savepath = os.path.join(pred_path, model+".npy")

                np.save(savepath, np.array(predictions))
    ##train sigma
    savepath = "./../modeling/"+data_name+"/trained_models/sigma/"
    if not os.path.isdir(savepath):
        os.mkdir(savepath)

    predictions_path =  "./../modeling/"+data_name+"/predictions/training_set/"
    all_predictions_path = glob.glob(predictions_path + "/*.npy")
    for path in all_predictions_path:
        model_name = os.path.splitext(os.path.basename(path))[0]
        Y_pred= np.load(path)
        training_sigma=training_data.drop(['given_label'], axis=1)
        training_sigma['sigma_square']=(Y_pred-training_data['given_label'] )**2
        predictorArgs = {
                            "path"             : savepath, 
                            "label"            : 'sigma_square',
                            "eval_metric"      : 'r2',
                            "problem_type"     : 'regression'
                        }
        hyperparameters = { 
                            'GBM'   : {}, 
                            'FASTAI': {}, 
                            'RF'    : {'criterion': 'squared_error', 
                                        'ag_args': {'name_suffix': 'MSE', 
                                        'problem_types': ['regression', 'quantile']}}
                            }
        predictor = TabularPredictor(**predictorArgs)
        predictor.fit(training_sigma, hyperparameters=hyperparameters)
        leaderboard = predictor.leaderboard(training_sigma, silent=True)
        
        predictor_cv = TabularPredictor(**predictorArgs) 
        predictor_cv.fit(training_sigma, presets="best_quality", num_stack_levels= 0, hyperparameters=hyperparameters)
        leaderboard_cv = predictor_cv.leaderboard(training_sigma, silent=True)
        
        models = {  'normal'            : list(leaderboard['model']), 
                    'crossValidation'   : list(np.append(leaderboard_cv['model'], ['oof']))}

        predictors = { 'normal'          : predictor, 
                       'crossValidation' : predictor_cv}

        # check if path is available, else create it. 
        data_without_label = testing_data.drop(['given_label','true_label', 'true_error'], axis=1)
        pred_path = "./../modeling/"+data_name+"/predictions/sigma/"
        if not os.path.isdir(pred_path):
            os.mkdir(pred_path)
    
        if models['normal'].count(model_name)>0:
            predictions = predictors['normal'].predict(data_without_label, model=model_name)
            np.save(os.path.join(pred_path, model_name+".npy"), np.array(predictions))
        elif model_name=="oof":
            predictions = predictors['crossValidation'].get_oof_pred()
            np.save(os.path.join(pred_path, model_name+".npy"), np.array(predictions))
        elif model_name=="WeightedEnsemble_L2_CV":
            predictions = predictors['crossValidation'].predict(data_without_label, model="WeightedEnsemble_L2")
            np.save(os.path.join(pred_path, model_name+".npy"), np.array(predictions))
        else:
            predictions = predictors['crossValidation'].predict(data_without_label, model=model_name)
            np.save(os.path.join(pred_path, model_name+".npy"), np.array(predictions))
    ##train u (boot strap)
    data_without_label = testing_data.drop(['given_label','true_label', 'true_error'], axis=1)
    savepath ="./../modeling/"+data_name+"/trained_models/bootstrap/"
    if not os.path.isdir(savepath):
        os.mkdir(savepath)
    pred_path = "./../modeling/"+data_name+"/predictions/bootstrap/"
    if not os.path.isdir(pred_path):
        os.mkdir(pred_path)
    for i in range(n_boot):
        random.seed(i)
        if not os.path.isdir(savepath+str(i)+"/"):
            os.mkdir(savepath+str(i)+"/")
        predictorArgs = {
                            "path"             : savepath+str(i)+"/", 
                            "label"            : 'given_label',
                            "eval_metric"      : 'r2',
                            "problem_type"     : 'regression'
                        }

        hyperparameters = { 
                            'GBM'   : {}, 
                            'FASTAI': {}, 
                            'RF'    : {'criterion': 'squared_error', 
                                        'ag_args': {'name_suffix': 'MSE', 
                                        'problem_types': ['regression', 'quantile']}}
                            }
        train_idx=sorted(random.sample(sorted(training_data.index) , int(len(training_data )*0.8 ) ) )
        data_bootstrap=training_data[  training_data.index.isin(train_idx)]
        predictor = TabularPredictor(**predictorArgs)
        predictor.fit(data_bootstrap, hyperparameters=hyperparameters)
        leaderboard = predictor.leaderboard(data_bootstrap, silent=True)
        predictor_cv = TabularPredictor(**predictorArgs) 
        predictor_cv.fit(data_bootstrap, presets="best_quality", num_stack_levels= 0, hyperparameters=hyperparameters)
        leaderboard_cv = predictor_cv.leaderboard(data_bootstrap, silent=True)
        ##
        models = {  'normal'            : list(leaderboard['model']), 
                    'crossValidation'   : list(np.append(leaderboard_cv['model'], ['oof']))}

        predictors = { 'normal'          : predictor, 
                       'crossValidation' : predictor_cv}
        ##
        pred_path = "./../modeling/"+data_name+"/predictions/bootstrap/"+str(i)+"/"
        if not os.path.isdir(pred_path):
            os.mkdir(pred_path)
        for model_type in models.keys():
            if model_type == "normal": 
                for model in models[model_type]:
                    predictions = predictors[model_type].predict(data_without_label, model=model)
                    np.save(os.path.join(pred_path, model+".npy"), np.array(predictions))
            elif model_type == "crossValidation":
                for model in models[model_type]:
                    if model == "oof":
                        predictions = predictors[model_type].get_oof_pred()
                        savepath_pred = os.path.join(pred_path, model+".npy")
                        np.save(savepath_pred, np.array(predictions))
                    elif model in models['normal']:
                        predictions = predictors[model_type].predict(data_without_label, model=model)
                        savepath_pred = os.path.join(pred_path, model+"_CV.npy")
                        np.save(savepath_pred, np.array(predictions))
                    else: 
                        predictions = predictors[model_type].predict(data_without_label, model=model)
                        savepath_pred = os.path.join(pred_path, model+".npy")
                        np.save(savepath_pred, np.array(predictions))
    
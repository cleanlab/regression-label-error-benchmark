import numpy as np
import pandas as pd
from autogluon.tabular  import TabularPredictor
import random
import os



class MyAutoGluonPredictor(TabularPredictor):
    def __init__(self, 
                label= "", 
                presets = "best_quality", 
                get_oof = True,
                num_stack_levels = 0, 
                problem_type="regression", 
                eval_metric="r2", 
                path=None, 
                num_bag_folds=5,
                verbosity=2,
                n_boot=20,
                hyperparameter = { 
                                    'GBM'   : {}, 
                                    'FASTAI': {}, 
                                    'RF'    : {'criterion': 'squared_error', 
                                                'ag_args': {'name_suffix': 'MSE', 
                                                'problem_types': ['regression', 'quantile']}}
                                    }, 
                ):
        super().__init__(label = label,
                        problem_type = problem_type, 
                        eval_metric=eval_metric, 
                        path= path, 
                        verbosity =verbosity
                        )
     
        self.preset = presets
        self.get_oof = get_oof
        self.num_stack_levels = num_stack_levels
        self.hyperparameter = hyperparameter
        self.num_bag_folds=num_bag_folds
        self.n_boot=n_boot
    
    def fit(self, X, y=None):
        x_df = pd.DataFrame(X)
        y_df = pd.DataFrame(y)        
        x_df.columns = ["col"+str(i) for i in range(x_df.shape[1])]  # type: ignore
        y_df.columns = ['label']  # type: ignore
        ##train y
        training_data = pd.concat([x_df, y_df], axis=1)
        self.__init__(label='label', 
                get_oof=self.get_oof, 
                presets=self.preset, 
                num_stack_levels=self.num_stack_levels, 
                path=self.path,
                hyperparameter=self.hyperparameter)
    
        super().fit(training_data, 
                    presets=self.preset, 
                    num_stack_levels=self.num_stack_levels, 
                    hyperparameters = self.hyperparameter)
        ##train sigma
        if not os.path.isdir(self.path+"/training_sigma"):
            os.mkdir(self.path+"/training_sigma")
        models_to_be_considered=list(super().leaderboard(silent=True)['model'])
        for model_name in models_to_be_considered:
            Y_pred=super().get_oof_pred(model=model_name)
            training_sigma=training_data.drop(['label'], axis=1)
            training_sigma['sigma_square']=(Y_pred-training_data['label'] )**2
            predictorArgs = {
                                "path"             : self.path+"/training_sigma/"+model_name, 
                                "label"            : 'sigma_square',
                                "eval_metric"      : 'r2',
                                "problem_type"     : 'regression'
                            }
            predictor_sigma=TabularPredictor(**predictorArgs)
            predictor_sigma.fit(training_sigma, presets=self.preset, num_stack_levels= self.num_stack_levels, 
                                num_bag_folds=self.num_bag_folds,hyperparameters=self.hyperparameter)
            
        ##train u
        if not os.path.isdir(self.path+"/training_u"):
            os.mkdir(self.path+"/training_u")
        for i in range(self.n_boot):
            
            predictorArgs = {
                                "path"             : self.path+"/training_u/"+str(i)+'/', 
                                "label"            : 'label',
                                "eval_metric"      : 'r2',
                                "problem_type"     : 'regression'
                            }   
            random.seed(i)
            train_idx=sorted(random.sample(sorted(training_data.index) , int(len(training_data )*0.8 ) ) )
            data_bootstrap=training_data[  training_data.index.isin(train_idx)]
            predictor_u=TabularPredictor(**predictorArgs)
            predictor_u.fit(data_bootstrap, presets=self.preset, num_stack_levels= self.num_stack_levels, 
                                num_bag_folds=self.num_bag_folds,hyperparameters=self.hyperparameter)
        return self
    
    
    def predict(self, data=None, model=None, as_pandas=True, transform_features=True):
        data = pd.DataFrame(data)
        
        if self.get_oof:
            return super().get_oof_pred(model=model)
        else:
            return super().predict(data, model, as_pandas, transform_features)
    
    def predict_sigma(self,data=None,model=None,as_pandas=True, transform_features=True):
        predictor_sigma=TabularPredictor.load("./MyAutogluonModels\/training_sigma/"+model+"/" )
        if self.get_oof:
            return np.sqrt( predictor_sigma.get_oof_pred(model=model))
        else:
            return np.sqrt( predictor_sigma.predict(data,model=model))
    
    def predict_u(self,data=None,model=None,as_pandas=True, transform_features=True):
        bootstrap_result=np.zeros([self.n_boot,int(len(data))])
        if self.get_oof:
            for i in range(self.n_boot):
                random.seed(i)
                train_idx=sorted(random.sample(sorted(data.index) , int(len(data )*0.8 ) ) )
                out_idx=sorted( set(data.index )-set(train_idx) )
                data_out=data[  data.index.isin(out_idx)]
                predictor_u=TabularPredictor.load("./MyAutogluonModels\/training_u/"+str(i)+"/" )
                predictions_u_train = predictor_u.get_oof_pred(model=model)
                predictions_u_out = predictor_u.predict(data=data_out,model=model)
                predictions=pd.concat([predictions_u_train,predictions_u_out])
                bootstrap_result[i,:] =predictions.sort_index()
            return np.sqrt( np.nanvar(bootstrap_result,axis=0) )
        else:
            for i in range(self.n_boot):
                predictor_u=TabularPredictor.load("./MyAutogluonModels\/training_u/"+str(i)+"/" )
                bootstrap_result[i,:] =predictor_u.predict(data,model=model) 
            return np.sqrt( np.nanvar(bootstrap_result,axis=0) )   
                
                
    def score(self, X, y):
        x_df = pd.DataFrame(X)
        y_df = pd.DataFrame(y)        
        x_df.columns = ["col"+str(i) for i in range(x_df.shape[1])]  # type: ignore
        y_df.columns = ['label']  # type: ignore

        training_data = pd.concat([x_df, y_df], axis=1)
        self.__init__(label='label', 
                        get_oof=self.get_oof, 
                        presets=self.preset, 
                        num_stack_levels=self.num_stack_levels, 
                        hyperparameter=self.hyperparameter)
        
        super().fit(training_data, 
                    presets=self.preset, 
                    num_stack_levels=self.num_stack_levels, 
                    hyperparameters = self.hyperparameter)
        
        y_pred = self.predict(X)
        y_true = y

        RSS = ((y_true - y_pred)** 2).sum()
        TSS = ((y_true - y_true.mean()) ** 2).sum()
        R2 = 1 - (RSS/TSS)
        return R2
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        
        return self
    
    def get_params(self, deep = True):
        """to ensure compatibility with sklearn"""
        
        params = {}
        return params

    

def Get_data(n_train=500,outlier=0.1,signal=5):
    def f(x):
        return (x-1)**2*(x+1)
    def g(x):
        if (x>=0.5):
            return 2*np.sqrt(x-0.5)
        else:
            return 0
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
    X=pd.DataFrame(temp_x,columns =['feature_1','feature_2','feature_3','feature_4','feature_5'] )
    true_label=pd.DataFrame(temp_y,columns=['true_label'])
    given_label=pd.DataFrame(temp_y,columns=['given_label'])
    Data=pd.concat([X,given_label,true_label ], axis = 1)
    if (outlier!=0):
        out_list=random.sample(range(n_train),int(outlier*n_train))
        out_list.sort()
        out_id=np.array([0]*n_train) 
        out_id[out_list]=1
        for i in out_list:
            Data['given_label'][i]=Data['given_label'][i]+signal
    true_error=pd.DataFrame(out_id,columns=['true_error'])
    Data=pd.concat([Data,true_error ], axis = 1)
    return Data

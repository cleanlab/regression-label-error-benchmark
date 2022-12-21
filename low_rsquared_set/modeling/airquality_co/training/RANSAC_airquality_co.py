import os
import numpy as np 
import pandas as pd 
from sklearn.linear_model import RANSACRegressor
import sys
sys.path.insert(1, 'E:/research/conformal outlier dect/realdata/Nov30/regression-label-error-benchmark/evaluation')
from utils import  AutoGluonPredictor
from sklearn.metrics import confusion_matrix

path = "./../../../dataset/airquality_co.csv"
dataset_name = os.path.splitext(os.path.basename(path))[0]
data = pd.read_csv(path, index_col=0)
error_ratio_before_RANSAC=sum(data['true_error'])/len(data)

##RANSAC by residual
train_data = data.drop(['given_label','true_label','true_error','time_slot','day_of_week'], axis=1)
label=data['given_label']
AG_estimator = AutoGluonPredictor()

AG_RANSAC_estimator = RANSACRegressor(random_state=0, 
                                        base_estimator=AG_estimator, 
                                        min_samples=train_data.shape[0], 
                                        stop_score=0.9)

AG_RANSAC_estimator = AG_RANSAC_estimator.fit(train_data, label)

print("AG inliers: ", sum(AG_RANSAC_estimator.inlier_mask_))
print("AG outliers: ", sum(1 - AG_RANSAC_estimator.inlier_mask_))
cm = confusion_matrix((1 - AG_RANSAC_estimator.inlier_mask_), np.array(data.true_error))
error_ratio_after_RANSAC=1-(cm[0,0]+cm[1,1])/len(data)
print(error_ratio_before_RANSAC,error_ratio_after_RANSAC)

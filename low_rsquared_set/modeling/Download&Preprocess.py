import os
import pandas as pd 
import wget
import random 
import pickle
import glob

####Download data to dataset
output_directory="./../dataset/"
wget.download("https://cleanlab-public.s3.amazonaws.com/RegressionBenchmark/airquality_co.csv",out=output_directory)
wget.download("https://cleanlab-public.s3.amazonaws.com/RegressionBenchmark/airquality_nmhc.csv",out=output_directory)
wget.download("https://cleanlab-public.s3.amazonaws.com/RegressionBenchmark/airquality_no2.csv",out=output_directory)
wget.download("https://cleanlab-public.s3.amazonaws.com/RegressionBenchmark/label_aggregation_easy_concat.csv",out=output_directory)
wget.download("https://cleanlab-public.s3.amazonaws.com/RegressionBenchmark/label_aggregation_easy_diff.csv",out=output_directory)
wget.download("https://cleanlab-public.s3.amazonaws.com/RegressionBenchmark/metaphor_novelity_easy_concat.csv",out=output_directory)
wget.download("https://cleanlab-public.s3.amazonaws.com/RegressionBenchmark/metaphor_novelity_easy_diff.csv",out=output_directory)
wget.download("https://cleanlab-public.s3.amazonaws.com/RegressionBenchmark/standford_politeness_stack.csv",out=output_directory)
wget.download("https://cleanlab-public.s3.amazonaws.com/RegressionBenchmark/standford_politeness_wiki.csv",out=output_directory)
wget.download("https://cleanlab-public.s3.amazonaws.com/RegressionBenchmark/telomere_cq_scg.csv",out=output_directory)
wget.download("https://cleanlab-public.s3.amazonaws.com/RegressionBenchmark/telomere_cq_telo.csv",out=output_directory)
wget.download("https://cleanlab-public.s3.amazonaws.com/RegressionBenchmark/telomere_ts.csv",out=output_directory)

####Devide the data into training and testing set, where the outliers are all in the testing set
data_path = "./../dataset/"
all_data_path = glob.glob(data_path + "/*.csv")
for path in all_data_path:
    data = pd.read_csv(path, index_col=0)
    data_inliers=data[data['true_error']==0]
    data_outliers=data[data['true_error']==1]
    outliers_ratio=len(data_outliers)/len(data )
    test_idx=sorted(random.sample(sorted( data_inliers.index) , int(len(data_inliers )*0.4 ) ))
    train_idx=sorted( set(data_inliers.index )-set(test_idx) )
    training_data=data_inliers[  data_inliers.index.isin(train_idx)]
    testing_data=pd.concat([data_inliers[data_inliers.index.isin(test_idx) ],data_outliers ])
    pickle.dump(testing_data,open("Testing_data.pkl","wb"))
    pickle.dump(training_data,open("Training_data.pkl","wb"))
    dataset_name = os.path.splitext(os.path.basename(path))[0]
    savepath = "./../modeling/"+dataset_name+"/"
    if not os.path.isdir(savepath):
        os.mkdir(savepath) 
    pickle.dump(testing_data,open(savepath+"Testing_data.pkl","wb"))
    pickle.dump(training_data,open(savepath+"Training_data.pkl","wb"))
    

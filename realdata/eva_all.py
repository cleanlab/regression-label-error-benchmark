import os
import glob
import numpy as np
import pandas as pd
import pickle
from utils import compute_metrics

def score_residual(label: np.array, pred_values: np.array) -> np.array:
    """
    Returns label quality scores for each datapoint.
    Scores are calculated based on residual.

    Each score is continous value in range [0,1]
    1 - clean label (given label is likely correct).
    0 - dirty label (given label is likely incorrect).

    Parameters
    ----------
    label:
        original labels (or, given label) from dataset. Array of (N,)

    pred_values:
        predicted values from a custom regressor. Array of (N,)

    Returns
    -------
    scores:
        label quality scores for each datapoints

    """
    assert (
        label.shape == pred_values.shape
    ), f"shape of given_label {label.shape} and pred_values {pred_values.shape} are not same"

    residual = pred_values - label
    scores = np.exp(-abs(residual))
    return scores

def score_addition(label: np.array, pred_values: np.array,u_x: np.array,sig_x:np.array) -> np.array:
    """
    Returns label quality scores for each datapoint.
    Scores are calculated based on residual.

    Each score is continous value in range [0,1]
    1 - clean label (given label is likely correct).
    0 - dirty label (given label is likely incorrect).

    Parameters
    ----------
    label:
        original labels (or, given label) from dataset. Array of (N,)

    pred_values:
        predicted values from a custom regressor. Array of (N,)
        
    u_x:
        square root of bootstrap variance. Array of (N,)
        
    sig_x:
        square root of E(Y-\hat{f}(X))^2. Array of (N,)

    Returns
    -------
    scores:
        label quality scores for each datapoints

    """
    assert (
        label.shape == pred_values.shape
    ), f"shape of given_label {label.shape} and pred_values {pred_values.shape} are not same"

    residual = abs(pred_values - label)/(u_x+sig_x)
    scores = np.exp(-(residual))
    return scores

def score_geom(label: np.array, pred_values: np.array,u_x: np.array,sig_x:np.array) -> np.array:
    """
    Returns label quality scores for each datapoint.
    Scores are calculated based on residual.

    Each score is continous value in range [0,1]
    1 - clean label (given label is likely correct).
    0 - dirty label (given label is likely incorrect).

    Parameters
    ----------
    label:
        original labels (or, given label) from dataset. Array of (N,)

    pred_values:
        predicted values from a custom regressor. Array of (N,)
        
    u_x:
        square root of bootstrap variance. Array of (N,)
        
    sig_x:
        square root of E(Y-\hat{f}(X))^2. Array of (N,)

    Returns
    -------
    scores:
        label quality scores for each datapoints

    """
    assert (
        label.shape == pred_values.shape
    ), f"shape of given_label {label.shape} and pred_values {pred_values.shape} are not same"

    residual = abs(pred_values - label)/np.sqrt(u_x*sig_x)
    scores = np.exp(-(residual))
    return scores

def score_u(label: np.array, pred_values: np.array,u_x: np.array,sig_x:np.array) -> np.array:
    """
    Returns label quality scores for each datapoint.
    Scores are calculated based on residual.

    Each score is continous value in range [0,1]
    1 - clean label (given label is likely correct).
    0 - dirty label (given label is likely incorrect).

    Parameters
    ----------
    label:
        original labels (or, given label) from dataset. Array of (N,)

    pred_values:
        predicted values from a custom regressor. Array of (N,)
        
    u_x:
        square root of bootstrap variance. Array of (N,)
        
    sig_x:
        square root of E(Y-\hat{f}(X))^2. Array of (N,)

    Returns
    -------
    scores:
        label quality scores for each datapoints

    """
    assert (
        label.shape == pred_values.shape
    ), f"shape of given_label {label.shape} and pred_values {pred_values.shape} are not same"

    residual = abs(pred_values - label)/u_x
    scores = np.exp(-(residual))
    return scores

def score_sig(label: np.array, pred_values: np.array,u_x: np.array,sig_x:np.array) -> np.array:
    """
    Returns label quality scores for each datapoint.
    Scores are calculated based on residual.

    Each score is continous value in range [0,1]
    1 - clean label (given label is likely correct).
    0 - dirty label (given label is likely incorrect).

    Parameters
    ----------
    label:
        original labels (or, given label) from dataset. Array of (N,)

    pred_values:
        predicted values from a custom regressor. Array of (N,)
        
    u_x:
        square root of bootstrap variance. Array of (N,)
        
    sig_x:
        square root of E(Y-\hat{f}(X))^2. Array of (N,)

    Returns
    -------
    scores:
        label quality scores for each datapoints

    """
    assert (
        label.shape == pred_values.shape
    ), f"shape of given_label {label.shape} and pred_values {pred_values.shape} are not same"

    residual = abs(pred_values - label)/(sig_x+1e-5)
    scores = np.exp(-(residual))
    return scores

def generate_metrics(
    data: pd.DataFrame, pred_values: np.array,u_x: np.array,sig_x: np.array, score_type: str = "residual"
) -> dict:
    assert (
        data.shape[0] == pred_values.shape[0]
    ), f"shape of dataframe and length of pred_values are not same."

    given_label = data["given_label"]
    true_label = data["true_label"]
    true_error = data["true_error"]
    true_diff = given_label - true_label
    
    score_dict = {
        "residual": score_residual,
        "addition": score_addition,
        "geom": score_geom,
        # "marginal_density": score_marginal_density,
        # "discretised": score_discretised,
        "u": score_u,
        "sig": score_sig
    }
    # LOCATION-IF
    if score_type == "residual":
        scores = score_dict[score_type](given_label, pred_values)
    elif score_type == "marginal_density":
        scores = score_dict[score_type](given_label, bandwidth=0.05)
    elif score_type == "discretised":
        scores = score_dict[score_type](given_label, pred_values, num_bins=10)
    elif score_type == "addition":
        scores = score_dict[score_type](given_label, pred_values, u_x,sig_x)
    elif score_type == "geom":
        scores = score_dict[score_type](given_label, pred_values, u_x,sig_x)
    elif score_type == "u":
        scores = score_dict[score_type](given_label, pred_values, u_x,sig_x)
    elif score_type == "sig":
        scores = score_dict[score_type](given_label, pred_values, u_x,sig_x)
        
    metrics = compute_metrics(
        scores=np.array(scores),
        true_error=np.array(true_error),
        true_diff=np.array(true_diff),
    )
    return metrics

def get_all_metrics(data: pd.DataFrame, scoring_method: list, pred_dict: dict, u_x: dict,sig_x: dict) -> dict:
    metrics_data = {}
    for score_type in scoring_method:
        metrics_data[score_type] = {}
        for model_name, pred in pred_dict.items():
            metrics_data[score_type][model_name] = generate_metrics(
                data, pred,u_x=u_x[model_name],sig_x=sig_x[model_name] , score_type=score_type
            )
    return metrics_data

####evaluate for each dataset
models_to_be_considered = [
    "RandomForestMSE",
    "RandomForestMSE_BAG_L1",
    "LightGBM",
    "LightGBM_BAG_L1",
    "NeuralNetFastAI",
    "NeuralNetFastAI_BAG_L1",
    "WeightedEnsemble_L2",
    "WeightedEnsemble_L2_CV",
]

# Update scoring method list, if you are adding new scoring techniques. 
scoring_methods = ["residual", "addition"]

# Update with name of metric to be considered. These are from compute_metrics(utils.py), consider adding new metric there.  
metrics_considered = [
    "auroc",
    "auprc",
    "lift_at_num_errors",
    "lift_at_100",
]
data_path = "./../dataset/"
all_datasets=os.listdir(data_path)
dataset_names=[]
for path in all_datasets:
    dataset_names.append(os.path.splitext(os.path.basename(path))[0])

dataset_names.remove("standford_politeness_stack");dataset_names.remove("standford_politeness_wiki")
all_metrics={}

for data_name in dataset_names:
    testing_data=pd.read_csv(data_path+data_name+".csv", index_col=0)
    # reads all the predictions saved in mentioned base path for predictions. 
    predictions_path = "./../modeling/"+data_name+"/predictions/testing_set/"
    all_predictions_path = glob.glob(predictions_path + "/*.npy")

    # generate a dictionary of prediction with respect to model considered in models_to_be_considered. 
    predictions = {}
    for path in all_predictions_path:
        model_name = os.path.splitext(os.path.basename(path))[0]
        if model_name in models_to_be_considered:
            """predictions = {model_name: prediction}"""
            predictions[model_name] = np.load(path)
    

    # reads all the predictions of sigma saved in mentioned base path. 
    sig_pred_path = "./../modeling/"+data_name+"/predictions/sigma/"
    all_sig_pred_path = glob.glob(sig_pred_path + "/*.npy")

    # generate a dictionary of prediction with respect to model considered in models_to_be_considered. 
    sigmas = {}
    for path in all_sig_pred_path:
        model_name = os.path.splitext(os.path.basename(path))[0]
        if model_name in models_to_be_considered:
            """predictions = {model_name: prediction}"""
            sigmas[model_name] = np.sqrt( abs(np.load(path)))
            
    #generate a dictionary of u
    u_pred_path = "./../modeling/"+data_name+"/predictions/bootstrap/"
    all_u_pred_path = glob.glob(u_pred_path + "0/*.npy")

    # generate a dictionary of prediction with respect to model considered in models_to_be_considered. 
    u = {}
    n_bootstrap=20
    for path in all_u_pred_path:
        model_name = os.path.splitext(os.path.basename(path))[0]
        if model_name in models_to_be_considered:
            """predictions = {model_name: prediction}"""
            n_test=len(testing_data)
            bootstrap_result=np.zeros([n_bootstrap,int(n_test)])
            for i in range(n_bootstrap):
                bootstrap_result[i,:] = np.load(u_pred_path+str(i)+"/"+model_name+".npy" )
            u[model_name]=np.sqrt( np.nanvar(bootstrap_result,axis=0) )
    all_metrics[data_name]=get_all_metrics(
        testing_data, scoring_method=scoring_methods, pred_dict=predictions,u_x=u,sig_x=sigmas
    )
    
pickle.dump(all_metrics, open("All_metrics_ransac.pkl","wb"))
#Res_mat=pickle.load( open("All_metrics.pkl","rb"))


# Benchmarking algorithms to detect erroneous label values in regression datasets.

Codes to implement method proposed in Hang Zhou, Jonas Mueller, Mayank Kumar, Jane-Ling Wang and Jing Lei (2023). [Detecting Errors in Numerical Data via any Regression Model](https://arxiv.org/abs/2305.16583)

This repository is only for intended for scientific purposes. 
To find errors in your own regression data, you should instead use the official [cleanlab](https://github.com/cleanlab/cleanlab) library.


## Simulation

- `simulation`: Main folder for the simulation results in Section 5. 
    - `generat_data.py`: Run this code can get the data used in our simulation. The data is stored in `Data` folder.
    - `utils.py`: Consists of helper functions to generate data, etc. 
    - `conformal_atg.py`: Conformal methods for autogloun package. One can change the hyperparameter to use different regression regressors.
    - `conformal_sklearn.py`: Conformal methods for sklearn package. The default setting is the Random Forest regressor.
    - `Eva_before_removing.py`: To get AUPRC before in Table 3.
    - `Eva_after_removing.py`: To get AUPRC after in Table 3.

## Realdata
- `realdata`: Main folder for the simulation results in Section 6.
    - `data_preprocessing`: method to pre-preocess the data
    - `dataset`: folder to store the datasets
    - `eva_all`: evaluate the results to get Table 4.
    - `utils.py`: Consists of helper functions to generate data, etc. 
    - `modeling`: This folder contains all the information and code needed to train models on a specific dataset. Each dataset will have a folder named after itself i.e., airquality_co etc. Each dataset folder has three sub-directories: 
    - `predictions`: It stores predictions of different models considered during training. 
    - `trained_models`: Stores the trained models as required. 


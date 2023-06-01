# Benchmarking algorithms to detect erroneous label values in regression datasets.

This repository is only for intended for scientific purposes. 
To find errors in your own regression data, you should instead use the official [cleanlab](https://github.com/cleanlab/cleanlab) library.

# Change log 
We have created two set of benchmarks based on feature engineering. These benchmarks are named as low_rsquared_set and high_rsquared_set. 
low_rsquared_set is same as benchmarks that we were using. There has been minor changes in names, 2 datasets have been removed, 1 has been modified.

**low_rsquared_set:** high representation of datasets with low r-squared values observed during training.

**high_rsquared_set:** high representation of datasets with high r-squared values observed during training.

**common:** This is added in both the sets. This folder contains common methods applicable. It includes:
- `utils`: Contains methods to plot results, compute all the metrics and other supporting method. 
- `RANSAC`: Modified sklearn RANSAC Regressor to take custom loss function 
- `Regressor`: Custom regressor that removes x% of data every iteration and train with AutoGluon.

**1. low_rsquared_set:**
- airquality_co_reduced
- airquality_no2_reduced
- stanford_politeness_stack_furthest
- stanford_politeness_wiki_furthest
- telomere_cq_telo

    **Modified:**
    - airquality_co and airquality_no2 are renamed to airquality_co_reduced, airquality_no2_reduced respectively to create distinction between datasets in low_rsquared_set and high_rsquared_set. 
    - stanford_politeness_stack_furthest is similar to stanford_politeness_stack in terms of given_label generation. Number of features has been reduced in comparision to stanford_politeness_stack. In this new version, we have kept only those datasets that had annotator agreement. 
    - stanford_politeness_wiki has been renamed to stanford_politeness_wiki_furthest to create distinction between datasets in low_rsquared_set and high_rsquared_set.

    **Deleted:**
    - label_aggregation dataset has been deleted as it shared large chunk of features with metaphor_novelty. 
    - metaphor_novelty has been moved to high_rsquared_set as we were able to create only one set that had resonable r-squared value. 

    **No change:**
    - There is no change in telomere_cq_telo dataset. This is part of both the set of benchmarks. 


**2. high_rsquared_set**
- airquality_co_full
- airquality_no2_full
- metaphor_novelty_concat_average
- stanford_politeness_stack_HFSE_random 
- stanford_politeness_wiki_random 
- telomere_cq_telo


### Directory Structure
for each of low_rsquared_set and high_rsquared_set
```
.
├── README.md
├── dataset
├── evaluation
│   ├── common
│   │   ├── utils.py 
│   │   ├── RANSAC.py
│   │   ├── Regressor.py
│   ├── evaluation.ipynb
└── modeling
    └── airquality_co
        ├── predictions
        └── training
```

## Getting Started 
- Both high_rsquared_set and low_squared_set contains same folder structure. 
- `evaluation`: Main folder to start exploring. It consists of 2 files. 
    - `evaluation.ipynb`: This is where you should add new label quality scoring techniques. A baseline label quality scoring method is `score_residual()`. This notebook runs the label quality scoring techniques on all datasets, and evaluates their performance in detecting label errors via various metrics.
    - New label-quality-scoring techniques should be added to `evaluation.ipynb` rather than here.
    - Predictions generated through AutoGluon are available in `modeling/dataset_name/predictions`. The pipeline is developed under assumption that predictions (and other model outputs required by a label quality scoring function) are available in this folder.  

    - `common`: contain methods and modules that are common for use across all the datasets. 
        - `utils.py`: Consists of helper functions to generate plots, compute evaluation metrics, etc. 

- `modeling`: This folder contains all the information and code needed to train models on a specific dataset. Each dataset will have a folder named after itself i.e., airquality_co etc. Each dataset folder has three sub-directories: 
    - `predictions`: It stores predictions of different models considered during training. 
    - `trained_models`: Stores the trained models as required. 
    - `training`: It consist of notebook to perform trainig. It currently has defined pipeline to train models using AutoGluon. 

- `dataset`: folder to store the datasets

- It is recommended to save: 
    - the trained models in `modeling/dataset_name/trained_models`, 
    - predictions from new models in `modeling/dataset_name/predictions`. 
    - use same "model_name" throughout training, saving predictions, or wherever required.

## Using RANSAC with AutoGluon 

- Toy dataset can be created using `make_data`. Code is available in `utils.py`. 
- `utils.py` also has the code for `AutoGluonPredictor` class. This can be used to run sklearn or any other custom modules with AutoGluon. 
- By Default, `AutoGluonPredictor` considers cross-validation in regression problem. 
- Current defination of `AutoGluonPredictor` is as per requirements for sklearn modules. It is suggested to write custom modules in with methods used in the `AutoGluonPredictor` class.

- Example to use AutoGluonPredictor is available in `evaluation.ipynb`(section: AutoGluon+ RANSAC). 

Custom module must seek following methods from AutoGluonPredictor:

```
fit(X, y): Fit model to given training data and target values.

score(X, y): Returns the mean accuracy on the given test data, which is used for the stop criterion defined by stop_score. Additionally, the score is used to decide which of two equally large consensus sets is chosen as the better one.

predict(X): Returns predicted values using the linear model, which is used to compute residual error using loss function.
```


## Things to note

- Every label quality score in this repo should be such that LOWER values of the score correspond to datapoints that are MORE likely to have a label error.

- Each dataset has three special columns i.e., `given_label`, `true_label` and `true_error`. 
    - `given_label`: Observed response variable Y. 
    - `true_label`: Ground truth for variable Y. 
    - `true_error`: 0 or 1 entries which represent if given_labels are correct or not.
        ```
        0: label is correct for this datapoint
        1: label is incorrect  
        ```

- The `true_label` and `true_error` columns would be unavaible in real applications, and any label-quality-scores should NOT depend on these. In the version of the dataset used for training models, we remove these extra special columns (keeping `given_label`), so that you don't accidentally train your model on them.

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
    - `dataset`: folder to store the datasets
    - `eva_all`: evaluate the results to get Table 4.
    - `utils.py`: Consists of helper functions to generate data, etc. 
    - `modeling`: This folder contains all the information and code needed to train models on a specific dataset. Each dataset will have a folder named after itself i.e., airquality_co etc. Each dataset folder has three sub-directories: 
    - `predictions`: It stores predictions of different models considered during training. 
    - `trained_models`: Stores the trained models as required. 
    - `training`: It consist of notebook to perform trainig. It currently has defined pipeline to train models using AutoGluon. 

# Benchmarking algorithms to detect erroneous label values in regression datasets.

This repository is only for intended for scientific purposes. 
To find errors in your own regression data, you should instead use the official [cleanlab](https://github.com/cleanlab/cleanlab) library.


### Directory Structure
```
.
├── README.md
├── dataset
│   └── airquality_co.csv
├── evaluation
│   ├── __init__.py
│   ├── evaluation.ipynb
│   └── utils.py
└── modeling
    └── airquality_co
        ├── predictions
        ├── trained_models
        └── training
```

## Getting Started 

- `evaluation`: Main folder to start exploring. It consists of 2 files. 
    - `evaluation.ipynb`: This is where you should add new label quality scoring techniques. A baseline label quality scoring method is `score_residual()`. This notebook runs the label quality scoring techniques on all datasets, and evaluates their performance in detecting label errors via various metrics.
    - `utils.py`: Consists of helper functions to generate plots, compute evaluation metrics, etc. 
    - New label-quality-scoring techniques should be added to `evaluation.ipynb` rather than here.
    - Predictions generated through AutoGluon are available in `modeling/dataset_name/predictions`. The pipeline is developed under assumption that predictions (and other model outputs required by a label quality scoring function) are available in this folder.  

- `dataset`: folder to store the datasets

- `modeling`: This folder contains all the information and code needed to train models on a specific dataset. Each dataset will have a folder named after itself i.e., airquality_co etc. Each dataset folder has three sub-directories: 
    - `predictions`: It stores predictions of different models considered during training. 
    - `trained_models`: Stores the trained models as required. 
    - `training`: It consist of notebook to perform trainig. It currently has defined pipeline to train models using AutoGluon. 

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
## Notes by Hang
- You can find the results for all datasets in `evaluation/Plot.ipynb`.
- The predictions for models/sigma/u(x) by bootstrap are obtained by `modeling/training_models.py`.
- The results for differents scores are evaluated in `evaluation/eva.py`.

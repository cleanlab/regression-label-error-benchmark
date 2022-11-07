# regression-label-error-benchmark
Benchmark algorithms to detect erroneous label values in regression datasets

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

# Getting Started 
- `evaluation`: Main folder to start exploring. It consists of 2 files. 
    - `evaluation.ipynb`: use to check current base line and add new scoring techniques. Comments are added to assist through any updates that are required. Main baseline scoring method is `score_residual()`.
    - `utils.py`: it consists of helper functions to effectively generate plot, compute metrics at once and for other intermediate steps. 
    - New label-quality-scoring techniques must be added to `evaluation.ipynb`
    - Predictions generated through AutoGluon is available in `modeling/dataset_name/predictions`. The pipeline is developed under assumption that predictions are available in this folder.  

- `dataset`: folder to store the datasets

- `modeling`: This folder contains all the information and code needed to train models on a specific dataset. Each dataset will have a folder named after itself i.e., airquality_co etc. Each dataset folder has three sub-directories: 
    - `predictions`: It stores predictions of different models considered during training. 
    - `trained_models`: Stores the trained models as required. 
    - `training`: It consist of notebook to perform trainig. It currently has defined pipeline to train models using AutoGluon. 

- It is recommended to save: 
    - the trained models in `modeling/dataset_name/trained_models`, 
    - predictions from new models in `modeling/dataset_name/predictions`. 
    - use same "model_name" throughout training, saving predictions, or wherever required.

# Dataset Features: things to know
- Each dataset has three special columns i.e., `given_label`, `true_label` and `true_error`. 
- `given_label`: Observed response variable Y. 
- `true_label`: Ground truth for variable Y. 
- `true_error`: It represents if given_labels are correct or not.
    ```
    0: label correct
    1: label error  
    ```
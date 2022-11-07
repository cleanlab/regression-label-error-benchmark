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
    ├── predictions
    │   └── airquality_co
    └── training
        ├── airquality_co.ipynb
        └── trained_models
```
`dataset`: store dataset here. \
`evaluation`: this consists of all evaluation notebook and relevent helper functions(`utils.py`). \
`modeling`: Use this to train models (`training`) and save prediction (`predictions`). Each notebook consists of relevant code to train and save predictions for the specific dataset. \
if saving the trained models, try to use `trained_models` dir(cosidered as default directory for autogluon based training).
# Stanford CS 349D - Optimizing UDFs with Early Filters

## Overview
This project aims to reduce the computational cost of calling user-defined functions (UDFs) in data queries by using probabilistic predicates (PPs) as early filters. We construct PPs for each simple clause and incorporate them into a custom query optimizer designed for efficient data handling.

## Getting Started

Initialize your environment by following these steps:
```bash
conda create --name cs349d
conda activate cs349d
pip install -r requirements.txt
git submodule init
git submodule update
```

## Directory Structure
- **data/**: Contains datasets used for training and testing the PP models. This folder is crucial for ensuring that our models are trained on representative data.
- **dataloader/**: Includes scripts for loading and preprocessing data. These scripts are tailored to format the data correctly for model training.
- **ml_udf/**: Stores the user-defined functions (UDFs) for image datasets that are optimized by the probabilistic predicates, including [yolov5](https://github.com/ultralytics/yolov5/releases) and [FastRCNN](https://github.com/rbgirshick/fast-rcnn).
- **pp_models/**: Contains the machine learning models that serve as probabilistic predicates. These models are designed to predict the necessity of executing UDFs, thereby acting as early filters.
- **pp_params/**: Holds parameters files for loading and tuning the PP models. Adjusting these parameters can significantly affect the models' accuracy and efficiency.
- **query_optimizer/**: Contains the query optimizer module that integrates probabilistic predicates into data query processes. This module is key to achieving optimal performance.
- **qo_tests/**: Includes test scripts to run the query optimizer per simple clause.
- **query_test/**: Includes scripts for testing the ML UDF for simple queries. 

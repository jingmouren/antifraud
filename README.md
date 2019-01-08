# AntiFraud
A Fraud Detection Framework


## Requirements

-  Python>=3.5
-  numpy>=1.14.3
-  scikit-learn>=0.20.0

## Usage

#### Packaging
Git clone this repository, go to the home folder of *antifraud*, run: 

     python setup.py sdist

Pip installable package will be generated in dist/*.tar.gz
#### Install

Simply run below command:

    pip install ./dist/antifraud-0.1.0.tar.gz


#### General Options

You can check out the other options available to use with *Ternary* using:

     python -m antifraud --help

- --method, the processing method, includes: logistic, random-forest, xgboost, deep-forest, wide-deep, cnn,
                                 cnn-att, etc.
- --train, train data
- --test, test data


>Default database was configured in [config/antifraud.cfg](antifraud/config/antifraud.cfg)

#### Example
Generate annotator command:

     python -m antifraud --method randomForest --train data/train.csv --test data/test.csv



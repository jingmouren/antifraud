# AntiFraud
A Credit Card Fraud Detection Framework.

Source codes of papers:
- AAAI2020: Spatio-temporal attention-based neural network for credit card fraud detection
- TKDE2020: Graph Neural Network for Fraud Detection via Spatial-temporal Attention
- ICONIP2016: Credit card fraud detection using convolutional neural networks

## Requirements

-  Python>=3.5
-  numpy>=1.14.3
-  scikit-learn>=0.20.0
-  pytest>=3.6.3
-  pandas>=0.23.3
-  networkx>=2.0
-  scipy>=0.19.1
-  matplotlib>=2.0.2
-  tensorflow>=1.12.0
-  xgboo    st>=0.81

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

     python -m antifraud --method random-forest --train data/train.csv --test data/test.csv

## Citing

If you find *Antifraud* is useful for your research, please consider citing the following papers:

    @inproceedings{cheng2020spatio,
        title={Spatio-temporal attention-based neural network for credit card fraud detection},
        author={Cheng, Dawei and Xiang, Sheng and Shang, Chencheng and Zhang, Yiyi and Yang, Fangzhou and Zhang, Liqing},
        booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
        volume={34},
        number={01},
        pages={362--369},
        year={2020}
    }
    @article{cheng2020graph,
        title={Graph Neural Network for Fraud Detection via Spatial-temporal Attention},
        author={Cheng, Dawei and Wang, Xiaoyang and Zhang, Ying and Zhang, Liqing},
        journal={IEEE Transactions on Knowledge and Data Engineering},
        year={2020},
        publisher={IEEE}
    }
    @inproceedings{fu2016credit,
        title={Credit card fraud detection using convolutional neural networks},
        author={Fu, Kang and Cheng, Dawei and Tu, Yi and Zhang, Liqing},
        booktitle={International Conference on Neural Information Processing},
        pages={483--490},
        year={2016},
        organization={Springer}
    }
    

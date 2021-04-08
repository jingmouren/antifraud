from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from antifraud.config import Config
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument("-m", "--method",  required=True,
                        choices=['logistic', 'Adamboost', 'GBDT', 'LSTM', 'cnn',
                                 'cnn-att-2d', 'cnn-att-3d', 'other'],
                        help='The processing method: logistic, random-forest, xgboost, deep-forest, wide-deep, '
                             'cnn, cnn-att, other')
    parser.add_argument("-trf", "--trainfeature", default=None,
                        help="The train feature data")
    parser.add_argument("-trl", "--trainlabel", default=None,
                        help="The train label data")
    parser.add_argument("-tef", "--testfeature", default=None,
                        help="The test featrue data")
    parser.add_argument("-tel", "--testlabel", default=None,
                        help="The test label data")
    parser.add_argument("-rawdata", "--rawdata", default=None,
                        help="raw data file to be transformed to feature matrix.")
    parser.add_argument("-lamda", "--lamda", default="0",
                        help="(Only for cnn-att method) Time decay parameter.")
    args = parser.parse_args()
    config = Config().get_config()
    return args


def main(args):
    test_label = []
    pred_score = []
    if args.method == 'logistic':
        from antifraud.methods.LR import logistic
        pred_score,test_label = logistic(args.trainfeature, args.trainlabel, args.testfeature, args.testlabel)
    elif args.method == 'Adamboost':
        from antifraud.methods.AdaBM import adaBM
        pred_score,test_label = adaBM(args.trainfeature, args.trainlabel, args.testfeature, args.testlabel)
    elif args.method == 'GBDT':
        from antifraud.methods.GBDT import xgb_model
        pred_score,test_label = xgb_model(args.trainfeature, args.trainlabel, args.testfeature, args.testlabel)
    elif args.method == 'LSTM':
        from antifraud.methods.LSTM_seq import LSTM_model
        pred_score,test_label = LSTM_model(args.trainfeature, args.trainlabel, args.testfeature, args.testlabel)
    elif args.method == 'cnn':
        from antifraud.methods.CNN_max import cnn_model
        pred_score,test_label = cnn_model(args.trainfeature, args.trainlabel, args.testfeature, args.testlabel)
    elif args.method.startswith('cnn-att'):
        from antifraud.methods.cnn_att.att_cnn_main import att_main,load_att_data
        train_feature, train_label, test_feature, test_label = load_att_data()
        pred_score = att_main(train_feature, train_label, test_feature, test_label, args.method)
    try:
        from antifraud.metrics.normal_function import general_result
        print("model {} works on the data...".format(args.method))
        general_result(test_label, pred_score)
    except:
        import pickle
        import time
        cur_time = str(int(time.mktime(time.gmtime())))
        score_filename = cur_time + ".score"
        label_filename = cur_time + ".label"
        pickle.dump(pred_score, open(score_filename, "wb"))
        pickle.dump(test_label, open(label_filename, "wb"))
        print("Something went wrong.(Please check your predicted score!)")


if __name__ == "__main__":
    main(parse_args())

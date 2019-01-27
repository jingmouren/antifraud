from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from antifraud.config import Config
from antifraud.methods.random_forest import *
from antifraud.methods.logistic import *
from antifraud.methods.deep_forest.deep_forest import *
from antifraud.methods.wide_deep import *
from antifraud.methods.xgboost_model import *
from antifraud.feature_engineering.load_data import load_train_test

import logging

logger = logging.getLogger(__name__)


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument("-m", "--method",  required=True,
                        choices=['logistic', 'random-forest', 'xgboost', 'deep-forest', 'wide-deep', 'cnn',
                                 'cnn-att', 'other'],
                        help='The processing method: logistic, random-forest, xgboost, deep-forest, wide-deep, '
                             'cnn, cnn-att, other')
    parser.add_argument("-train", "--train", default=None,
                        help="The train data")
    parser.add_argument("-test", "--test", default=None,
                        help="The test data")
    args = parser.parse_args()
    config = Config().get_config()
    return args


def main(args):
    if args.method == 'random-forest':
        train_feature, train_label, test_feature, test_label = load_train_test(args.train, args.test)
        accuracy_rate = random_forest(train_feature, train_label, test_feature, test_label)
        print("model random-forest works on the data, the accuracy rate is: ", accuracy_rate)
    if args.method == 'logistic':
        train_feature, train_label, test_feature, test_label = load_train_test(args.train, args.test)
        accuracy_rate = logistic(train_feature, train_label, test_feature, test_label)
        print("model logistic works on the data, the accuracy rate is: ", accuracy_rate)
    if args.method == 'deep-forest':
        train_feature, train_label, test_feature, test_label = load_train_test(args.train, args.test)
        accuracy_rate = deep_forest(train_feature, train_label, test_feature, test_label)
        print("model deep-forest works on the data, the accuracy rate is: ", accuracy_rate)
    if args.method == 'wide-deep':
        print("model wide-deep works on the data, the result is: ")
        wide_deep()
    if args.method == 'xgboost':
        accuracy_rate = xgboost_model(args.train, args.test)
        print("model xgboost works on the data, the accuracy rate is: ", accuracy_rate)


if __name__ == "__main__":
    main(parse_args())

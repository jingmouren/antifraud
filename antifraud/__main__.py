from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from antifraud.config import Config
from antifraud.methods.random_forest import *
from antifraud.methods.logistic import *

import logging

logger = logging.getLogger(__name__)


def parse_args():

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')

    config = Config().get_config()

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
    return args


def main(args):

    if args.method == 'random-forest':
        random_forest()

    if args.method == 'logistic':
        logistic()


if __name__ == "__main__":
    main(parse_args())

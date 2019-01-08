import os
import configparser
import logging

CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
config_database_file = os.path.join(CONFIG_DIR, "antifraud.cfg")


class Config(object):

    def __init__(self, filepath=config_database_file):
        self.__config = configparser.ConfigParser()
        self.__config.read(filepath)

    def get_config(self):
        return self.__config

    @staticmethod
    def __get_log_level(levels):
        return {
            'logging.INFO': logging.INFO,
            'logging.DEBUG': logging.DEBUG,
            'logging.WARNING': logging.WARNING,
            'logging.ERROR': logging.ERROR,
        }[levels]

    def get_log_level(self):
        return self.__get_log_level(self.__config["log"]["log.level"])

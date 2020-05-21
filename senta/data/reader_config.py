# -*- coding: utf-8 -*
"""
:py:class:`ReaderConfig` is an abstract class representing
"""


class ReaderConfig(object):
    """ReaderConfig"""
    def __init__(self):
        self.data_path = None
        self.shuffle = False
        self.batch_size = 8
        self.sampling_rate = 1.0
        self.epoch = 1
        self.extra_params = {}

    def build(self, params_dict):
        """
        :param params_dict:
        :return:
        """
        self.data_path = params_dict["data_path"]
        self.shuffle = params_dict["shuffle"]
        self.batch_size = params_dict["batch_size"]
        self.sampling_rate = params_dict["sampling_rate"]
        self.epoch = params_dict["epoch"]

        self.extra_params = params_dict.get("extra_params", None)
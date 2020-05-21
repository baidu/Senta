# -*- coding: utf-8 -*
"""
:py:class:`Model` is an abstract class representing
"""


class Model(object):
    """
    计算图网络
    """

    def __init__(self, model_params):
        self.model_params = model_params 

    def forward(self, fields_dict, phase):
        """
        forward
        """
        raise NotImplementedError

    def fields_process(self, fields_dict, phase):
        """
        fields_process
        """
        raise NotImplementedError

    def make_embedding(self, fields_dict, phase):
        """
        make_embedding
        """
        raise NotImplementedError

    def optimizer(self, loss, is_fleet=False):
        """
        optimizer
        """
        raise NotImplementedError

    def parse_predict_result(self, predict_result):
        """
        parse_predict_result
        """
        raise NotImplementedError

    def get_metrics(self, fetch_output_dict, meta_info, phase):
        """
        get_metrics
        """
        raise NotImplementedError

    def metrics_show(self, result_evaluate):
        """
        metrics_show
        """
        raise NotImplementedError

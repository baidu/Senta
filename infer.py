# -*- coding: utf-8 -*
"""import"""
import os
import sys
from senta.inference.inference import Inference
from senta.common.register import RegisterSet
from senta.common import register
from senta.data.data_set import DataSet

import logging
from senta.utils import args
from senta.utils import params
from senta.utils import log


def dataset_reader_from_params(params_dict):
    """
    :param params_dict:
    :return:
    """
    dataset_reader = DataSet(params_dict)
    dataset_reader.build()

    return dataset_reader


def model_from_params(params_dict):
    """
    :param params_dict:
    :return:
    """
    model_name = params_dict.get("type")
    model_class = RegisterSet.models.__getitem__(model_name)
    model = model_class(params_dict)
    return model


def build_inference(params_dict, dataset_reader, model):
    """build trainer"""
    inference = Inference(param=params_dict, data_set_reader=dataset_reader, model_class=model)
    return inference


if __name__ == "__main__":
    args = args.build_common_arguments()
    log.init_log("./log/infer", level=logging.DEBUG)
    # 分类任务的预测
    param_dict = params.from_file(args.param_path)
    _params = params.replace_none(param_dict)

    register.import_modules()

    dataset_reader_params_dict = _params.get("dataset_reader")
    dataset_reader = dataset_reader_from_params(dataset_reader_params_dict)

    model_params_dict = _params.get("model")
    model = model_from_params(model_params_dict)

    inference_params_dict = _params.get("inference")
    inference = build_inference(inference_params_dict, dataset_reader, model)

    inference.do_inference()

    logging.info("os exit.")
    os._exit(0)



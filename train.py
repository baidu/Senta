# -*- coding: utf-8 -*
"""import"""
import os
import sys
from senta.common.register import RegisterSet
from senta.common import register
from senta.data.data_set import DataSet
import logging
from senta.utils import args
from senta.utils import params
from senta.utils import log

logging.getLogger().setLevel(logging.INFO)

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
    opt_params = params_dict.get("optimization", None)
    num_train_examples = 0
    # 按配置计算warmup_steps
    if opt_params and opt_params.__contains__("warmup_steps"):
        trainers_num = int(os.getenv("PADDLE_TRAINERS_NUM", "1"))
        num_train_examples = dataset_reader.train_reader.get_num_examples()
        batch_size_train = dataset_reader.train_reader.config.batch_size
        epoch_train = dataset_reader.train_reader.config.epoch
        max_train_steps = epoch_train * num_train_examples // batch_size_train // trainers_num

        warmup_steps = opt_params.get("warmup_steps", 0)

        if warmup_steps == 0:
            warmup_proportion = opt_params.get("warmup_proportion", 0.1)
            warmup_steps = int(max_train_steps * warmup_proportion)

        logging.info("Device count: %d" % trainers_num)
        logging.info("Num train examples: %d" % num_train_examples)
        logging.info("Max train steps: %d" % max_train_steps)
        logging.info("Num warmup steps: %d" % warmup_steps)

        opt_params = {}
        opt_params["warmup_steps"] = warmup_steps
        opt_params["max_train_steps"] = max_train_steps
        opt_params["num_train_examples"] = num_train_examples

        # combine params dict
        params_dict["optimization"].update(opt_params)

    model_name = params_dict.get("type")
    model_class = RegisterSet.models.__getitem__(model_name)
    model = model_class(params_dict)
    return model, num_train_examples


def build_trainer(params_dict, dataset_reader, model, num_train_examples=0):
    """build trainer"""
    trainer_name = params_dict.get("type", "CustomTrainer")
    trainer_class = RegisterSet.trainer.__getitem__(trainer_name)
    params_dict["num_train_examples"] = num_train_examples
    trainer = trainer_class(params=params_dict, data_set_reader=dataset_reader, model_class=model)
    return trainer


if __name__ == "__main__":
    args = args.build_common_arguments()
    log.init_log(os.path.join(args.log_dir, "train"), level=logging.DEBUG)
    param_dict = params.from_file(args.param_path)
    _params = params.replace_none(param_dict)
    # 记得import一下注册的模块
    register.import_modules()

    dataset_reader_params_dict = _params.get("dataset_reader")
    dataset_reader = dataset_reader_from_params(dataset_reader_params_dict)

    model_params_dict = _params.get("model")
    model, num_train_examples = model_from_params(model_params_dict)

    trainer_params_dict = _params.get("trainer")
    trainer = build_trainer(trainer_params_dict, dataset_reader, model, num_train_examples)

    trainer.train_and_eval()
    logging.info("end of run train and eval .....")

    logging.info("os exit.")
    os._exit(0)

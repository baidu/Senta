# *_*coding:utf-8 *_*
"""
CustomTrainer
"""
import collections
import time
import traceback
from collections import OrderedDict

import numpy as np
from paddle import fluid
from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler import fleet
import logging
import os

from senta.common.register import RegisterSet
from senta.common.rule import InstanceName
from senta.training.base_trainer import BaseTrainer
# from senta.training.trainer_pro import BaseTrainer


@RegisterSet.trainer.register
class CustomTrainer(BaseTrainer):
    """CustomTrainer:通用senta任务的trainer"""
    def __init__(self, params, data_set_reader, model_class):
        """
        :param params:
        :param data_set_reader:
        :param model_class:
        """
        BaseTrainer.__init__(self, params, data_set_reader, model_class)

    def train_and_eval(self):
        """
        :return:
        """
        if self.is_fleet and fleet.is_server():
            logging.debug("is fleet.server, over")
            return
        if self.is_fleet:
            logging.debug("worker_index%d start train...." % fleet.worker_index())

        num_train_examples = self.params.get("num_train_examples", 0)
        if num_train_examples == 0:
            num_train_examples = self.data_set_reader.train_reader.get_num_examples()

        self.data_set_reader.train_reader.run()
        steps = 1
        time_begin = time.time()
        if 'output_path' in self.params.keys() and self.params["output_path"]:
            save_checkpoints_path = os.path.join(self.params["output_path"], "save_checkpoints")
            save_inference_model_path = os.path.join(self.params["output_path"], "save_inference_model")
        else:
            save_checkpoints_path = "./output/save_checkpoints/"
            save_inference_model_path = "./output/save_inference_model/"
        try:
            while True:
                try:
                    if steps % self.params["train_log_step"] != 0:
                        self.run(InstanceName.TRAINING, need_fetch=False)
                    else:
                        metrics_tensor_value = self.run(InstanceName.TRAINING, need_fetch=True)
                        current_example, current_epoch = self.data_set_reader.train_reader.get_train_progress()
                        logging.info("epoch {0} progress {1}/{2} pyreader queue size {3}".
                                     format(current_epoch, current_example, num_train_examples,
                                            self.data_set_reader.train_reader.paddle_py_reader.queue.size()))

                        fetch_output_dict = collections.OrderedDict()
                        for key, value in zip(self.fetch_list_train_key, metrics_tensor_value):
                            fetch_output_dict[key] = value
                        time_end = time.time()
                        used_time = time_end - time_begin
                        meta_info = collections.OrderedDict()
                        meta_info[InstanceName.STEP] = steps
                        meta_info[InstanceName.GPU_ID] = self.gpu_id
                        meta_info[InstanceName.TIME_COST] = used_time

                        metrics_output = self.model_class.get_metrics(fetch_output_dict, meta_info,
                                                                      InstanceName.TRAINING)
                        if self.params.get("visualdl_log", False):
                            assert isinstance(metrics_output, OrderedDict), "metrics_output is must be OrderedDict"
                            self.visualdl_log(metrics_output, np.mean(fetch_output_dict[InstanceName.LOSS]), steps, 
                                              phase=InstanceName.TRAINING)
                        time_begin = time.time()

                    if steps % self.params["eval_step"] == 0:
                        if self.params["is_eval_dev"]:
                            self.evaluate(self.data_set_reader.dev_reader, InstanceName.EVALUATE, steps)
                        if self.params["is_eval_test"]:
                            self.evaluate(self.data_set_reader.test_reader, InstanceName.TEST, steps)
                    if self.trainer_id == 0:
                        if steps % self.params["save_model_step"] == 0:
                            self.save_models(save_checkpoints_path, save_inference_model_path, steps)
                    steps += 1
                    if "steps_for_test" in self.params and steps >= self.params["steps_for_test"]:
                        self.data_set_reader.train_reader.stop()
                        logging.debug("steps_for_test stop!")
                        break
                except fluid.core.EOFException:
                    self.data_set_reader.train_reader.stop()
                    break
            if self.params["is_eval_dev"]:
                logging.info("Final evaluate result: ")
                self.evaluate(self.data_set_reader.dev_reader, InstanceName.EVALUATE, steps)
            if self.params["is_eval_test"]:
                logging.info("Final test result: ")
                self.evaluate(self.data_set_reader.test_reader, InstanceName.TEST, steps)
        except Exception as e:
            logging.error('traceback.format_exc():%s' % traceback.format_exc())
            self.save_models(save_checkpoints_path, save_inference_model_path, steps)
            raise e

        self.save_models(save_checkpoints_path, save_inference_model_path, steps)

    def evaluate(self, reader, phase, step):
        """
        :param reader:
        :param phase:
        :param step:
        :return:
        """
        if not reader:
            raise ValueError("{0} reader is none".format(phase))
        reader.run()
        all_metrics_tensor_value = []
        i = 0
        time_begin = time.time()
        while True:
            try:
                metrics_tensor_value = self.run(phase=phase)
                if i == 0:
                    all_metrics_tensor_value = [[tensor] for tensor in metrics_tensor_value]
                else:
                    for j in range(len(metrics_tensor_value)):
                        one_tensor_value = all_metrics_tensor_value[j]
                        all_metrics_tensor_value[j] = one_tensor_value + [metrics_tensor_value[j]]
                i += 1
            except fluid.core.EOFException:
                reader.stop()
                break

        fetch_output_dict = collections.OrderedDict()
        for key, value in zip(self.fetch_list_evaluate_key, all_metrics_tensor_value):
            fetch_output_dict[key] = value
        time_end = time.time()
        used_time = time_end - time_begin

        meta_info = collections.OrderedDict()
        meta_info[InstanceName.STEP] = step
        meta_info[InstanceName.GPU_ID] = self.gpu_id
        meta_info[InstanceName.TIME_COST] = used_time
        metrics_output = self.model_class.get_metrics(fetch_output_dict, meta_info, phase)
        if self.params.get("visualdl_log", False):
            assert isinstance(metrics_output, OrderedDict), "the metrics_output must be OrderedDict"
            eval_loss = np.mean(fetch_output_dict[InstanceName.LOSS])
            self.visualdl_log(metrics_output, eval_loss, step, phase=phase)

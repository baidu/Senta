# *_*coding:utf-8 *_*
"""
GlueTaskTrainer
"""
import collections
import time
import traceback
from collections import OrderedDict

import numpy as np
from paddle import fluid
from paddle.fluid.incubate.fleet.collective import fleet
import logging
import os

from senta.common.register import RegisterSet
from senta.common.rule import InstanceName
from senta.training.base_trainer import BaseTrainer


@RegisterSet.trainer.register
class GlueTaskTrainer(BaseTrainer):
    """GlueTaskTrainer: Glue任务的trainer"""
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
        current_epoch = 0
        last_epoch = 0
        dev_score_history = []
        try:
            while True:
                try:
                    current_example, current_epoch = self.data_set_reader.train_reader.get_train_progress() 
                    if (steps % self.params["train_log_step"] != 0 or self.trainer_id != 0 \
                       and current_epoch == last_epoch):
                        self.run(InstanceName.TRAINING, need_fetch=False)
                    else:
                        metrics_tensor_value = self.run(InstanceName.TRAINING, need_fetch=True)
                        logging.info("epoch {0} progress {1}/{2} pyreader queue size {3}".
                                     format(current_epoch, current_example, num_train_examples,
                                            self.data_set_reader.train_reader.paddle_py_reader.queue.size()))
                        
                        current_example, current_epoch = self.data_set_reader.train_reader.get_train_progress()

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

                    if steps % self.params["eval_step"] == 0 or last_epoch != current_epoch:
                        if self.params["is_eval_dev"]:
                            rets = self.evaluate_iterface(self.data_set_reader.dev_reader, \
                                          InstanceName.EVALUATE, steps, current_epoch)
                            if self.trainer_id == 0:
                                dev_score_history.append(rets[0]['score'])
                        else:
                            rets = None
                            
                        if self.params["is_eval_test"]:
                            self.predict_iterface(self.data_set_reader.test_reader, \
                                          InstanceName.TEST, steps, current_epoch, rets, \
                                          dev_score_history)
                    if self.trainer_id == 0:
                        if steps % self.params["save_model_step"] == 0:
                            self.save_models(save_checkpoints_path, save_inference_model_path, steps)
                    steps += 1

                    if last_epoch != current_epoch:
                        last_epoch = current_epoch

                    if "steps_for_test" in self.params and steps >= self.params["steps_for_test"]:
                        self.data_set_reader.train_reader.stop()
                        logging.debug("steps_for_test stop!")
                        break
                except fluid.core.EOFException:
                    self.data_set_reader.train_reader.stop()
                    break
            if self.params["is_eval_dev"]:
                logging.info("Final evaluate result: ")
                rets = self.evaluate_iterface(self.data_set_reader.dev_reader, \
                                     InstanceName.EVALUATE, steps, current_epoch)
                if self.trainer_id == 0:
                    dev_score_history.append(rets[0]['score'])
            else:
                rets = None

            if self.params["is_eval_test"]:
                logging.info("Final test result: ")
                self.predict_iterface(self.data_set_reader.test_reader, InstanceName.TEST, \
                              steps, current_epoch, rets, dev_score_history)
            if self.params.get("diagnostic", False):
                logging.info("Final test on dianostic: ")
                # TODO
        except Exception as e:
            logging.error('traceback.format_exc():%s' % traceback.format_exc())
            self.save_models(save_checkpoints_path, save_inference_model_path, steps)
            raise e

        self.save_models(save_checkpoints_path, save_inference_model_path, steps)

    def evaluate_iterface(self, reader, phase, step, current_epoch):
        """evaluate_iterface"""
        #**************** multi dev files **************#
        #rets = []
        #for reader_ds in reader:
        #    logging.info('evaluating {}'.format(reader_ds.name))
        #    meta = self.evaluate(reader_ds, InstanceName.EVALUATE, step, current_epoch)
        #    rets.append(ret)
        
        #*************** single dev file ***************#
        rets = []
        meta = self.evaluate(reader, InstanceName.EVALUATE, step, current_epoch)
        rets.append(meta)

        return rets

    def predict_iterface(self, reader, phase, step, current_epoch, metas, dev_score_history):
        """predict iterface"""
        #***************** multi test file ****************#
        #for reader_f, save_f in zip(reader, save_dirs):
        #    save_path = save_f + '.' + str(current_epoch) + '.' + str(step)
        #    logging.info("testing {}, save to {}".format(reader_f.name, save_path))
        #    meta = self.evaluate(reader_f, InstanceName.TEST, step, current_eopch, metas)
        #    qids = meta["qids"]
        #    preds = meta["preds"]
        #    probs = meta["probs"]
        #    with open(save_path, 'w') as f:
        #        for id, s, p in  zip(qids, preds, probs):
        #            f.write('{}\t{}\t{}\n'.format(id, s, p))

        #    if is_best:
        #        with open(os.path.dirname(save_f) + 'best', 'w') as f:
        #            f.write("{}\n".format(dev_score_history[-1]))
        #            f.write('{}\n'.format(save_path))
        
        #***************** single test file ****************#
        meta = self.evaluate(reader, InstanceName.TEST, step, current_epoch, metas)
        if self.trainer_id == 0:
            qids = meta["qids"]
            preds = meta["preds"]
            probs = meta["probs"]
            test_save = self.params["test_save"]
            save_dirs = test_save.split(',')
            
            if not os.path.exists(os.path.dirname(save_dirs[0])):
                os.makedirs(os.path.dirname(save_dirs[0]))

            is_best = False
            if len(dev_score_history) > 1 and all(dev_score_history[-1] > i for i in \
                                                  dev_score_history[:-1]):
                is_best = True
            save_path = save_dirs[0] + '.' + str(current_epoch) + '.' + str(step)

            with open(save_path, 'w') as f:
                for id, s, p in  zip(qids, preds, probs):
                    f.write('{}\t{}\t{}\n'.format(id, s, p))

            if is_best:
                with open(os.path.dirname(save_dirs[0]) + 'best', 'w') as f:
                    f.write("{}\n".format(dev_score_history[-1]))
                    f.write('{}\n'.format(save_path))

    def evaluate(self, reader, phase, step, current_epoch, metas=None):
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
        meta_info["current_epoch"] = current_epoch
        meta_info["metric"] = self.params.get("metric", "simple_accuracy")
        meta, metrics_output = self.model_class.get_metrics(fetch_output_dict, meta_info, \
                                                            phase, metas=None)
        if self.params.get("visualdl_log", False):
            assert isinstance(metrics_output, OrderedDict), "the metrics_output must be OrderedDict"
            eval_loss = np.mean(fetch_output_dict[InstanceName.LOSS])
            self.visualdl_log(metrics_output, eval_loss, step, phase=phase)

        return meta

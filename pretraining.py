# -*- coding: utf-8 -*

""" main entrance to train ernie multitask language model """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle.fluid as fluid
from paddle.fluid.incubate.fleet.collective import fleet
import numpy as np
import collections
import argparse
import logging
import random
import copy
import gzip
import json
import sys
import os
import time

try:
    import paddlecloud.visual_util as visualdl
except ImportError:
    pass

logging.getLogger().setLevel(logging.INFO)
from senta.common.rule import InstanceName
from senta.models.ernie_multil_task_language_model import ErnieMTLM
from senta.models.ernie_skep_multil_task_language_model import ErnieSkepMTLM
from senta.models.roberta_language_model import RobertaLM
from senta.models.roberta_skep_language_model import RobertaSkepLM
from senta.modules.ernie import ErnieConfig
from senta.common.args import ArgumentGroup, print_arguments
from senta.utils.util_helper import save_infer_data_meta
from senta.data.tokenizer.tokenization_wp import FullTokenizer, GptBpeTokenizer
from senta.data.data_set_reader.ernie_pretrain_dataset_reader import ErniePretrainDataReader
from senta.data.data_set_reader.ernie_skep_pretrain_dataset_reader import ErnieSkepPretrainDataReader
from senta.data.data_set_reader.roberta_pretrain_dataset_reader_en import RobertaPretrainDataReaderEnglish
from senta.data.data_set_reader.roberta_skep_pretrain_dataset_reader_en import RobertaSkepPretrainDataReaderEnglish
import senta.utils.init as init
from senta.utils import log
from senta.training.base_trainer import BaseTrainer

parser = argparse.ArgumentParser(__doc__)
model_g = ArgumentGroup(parser, "model", "model configuration and paths.")
model_g.add_arg("ernie_config_path", str, "./config/ernie_config.json", \
        "Path to the json file for ernie model config.")

ckpt_vs_param = ("ERROR: config 'init_checkpoint' and 'init_parameters' "
                    "both are set! Only one of them should be set. "
                    "if you want warmstart checkpoint keep its learning_rate and moments, plese set 'init_checkpoint'. "
                    "if you want warmstart checkpoint with only its parameters, and you want reset a new learning_rate "
                    "by config, plese set 'init_parameters'")

model_g.add_arg("load_checkpoint", str, None, ckpt_vs_param)
model_g.add_arg("load_parameters", str, None, ckpt_vs_param)
model_g.add_arg("model_type", str, 'ernie_en', "The model architecture to be trained")
model_g.add_arg("pre_train_model", str, None, "not supported yet")
model_g.add_arg("checkpoints", str, "checkpoints", "Path to save checkpoints.")
model_g.add_arg("weight_sharing", bool, True, \
                    "If set, share weights between word embedding and masked lm.")
model_g.add_arg("generate_neg_sample", bool, False, \
        "If set, randomly generate negtive samples by positive samples.")

train_g = ArgumentGroup(parser, "training", "training options.")
train_g.add_arg("is_en", bool, True, "Whether to use en")
train_g.add_arg("epoch", int, 100, "Number of epoches for training.")
train_g.add_arg("learning_rate", float, 0.0001, "Learning rate used to train with warmup.")
train_g.add_arg("lr_scheduler", str, "linear_warmup_decay",
                "scheduler of learning rate.", choices=['linear_warmup_decay', 'noam_decay'])
train_g.add_arg("task_group_path", str, './data/en/task.json', "task_group path")
train_g.add_arg("weight_decay", float, 0.01, "Weight decay rate for L2 regularizer.")
train_g.add_arg("num_train_steps", int, 1000000, "Total steps to perform pretraining.")
train_g.add_arg("warmup_steps", int, 5000, "Total steps to perform warmup when pretraining.")
train_g.add_arg("save_model_step", int, 10000, "The steps interval to save checkpoints.")
train_g.add_arg("eval_step", int, 20, "The steps interval to evaluate model performance.")
train_g.add_arg("is_eval_dev", bool, True, "是否在训练的时候评估开发集，如果取值为1一定需要配置dev_reader及其数据路径。")
train_g.add_arg("is_eval_test", bool, False, "是否在训练的时候评估测试集，如果取值为1一定需要配置test_reader及其数据路径")
train_g.add_arg("use_fuse", bool, False, "Whether to use fuse_allreduce_ops.")
train_g.add_arg("do_recompute", bool, False, "Whether to use recompute.")
train_g.add_arg("nccl_comm_num", int, 1, "NCCL comm num.")
train_g.add_arg("hierarchical_allreduce_inter_nranks", int, 8, "Hierarchical allreduce inter ranks.")
train_g.add_arg("eval_batch_size", int, 1024, "eval_batch_size.")
train_g.add_arg("use_hierarchical_allreduce", bool, False, "Use hierarchical allreduce or not.")
train_g.add_arg("use_fp16", bool, False, "Whether to use fp16 mixed precision training.")
train_g.add_arg("use_dynamic_loss_scaling", bool, False, "Whether to use dynamic loss scaling.")
train_g.add_arg("init_loss_scaling", float, 1.0,
                "Loss scaling factor for mixed precision training, only valid when use_fp16 is enabled.")
train_g.add_arg("incr_every_n_steps", int, 1000, "Increases loss scaling every n consecutive.")
train_g.add_arg("decr_every_n_nan_or_inf", int, 2,
                "Decreases loss scaling every n accumulated steps with nan or inf gradients.")
train_g.add_arg("incr_ratio", float, 2.0,
                "The multiplier to use when increasing the loss scaling.")
train_g.add_arg("decr_ratio", float, 0.8,
                "The less-than-one-multiplier to use when decreasing.")
train_g.add_arg("using_spm", bool, True, ".")
train_g.add_arg("do_whole_word_mask", bool, False, ".")
train_g.add_arg("masking_strategy", str, "connective_masking", ".")
train_g.add_arg("ngram", int, 3, ".")
train_g.add_arg("num_iteration_per_drop_scope", int, 1, 
        ("num_iteration_per_drop_scope indicates how many iterations to clean"
        "up the temp variables which is generated during execution."))

log_g = ArgumentGroup(parser, "logging", "logging related.")
log_g.add_arg("train_log_step", int, 2, "The steps interval to print loss.")
log_g.add_arg("verbose", bool, False, "Whether to output verbose log.")

data_g = ArgumentGroup(parser, "data", "Data paths, vocab paths and data processing options")
data_g.add_arg("train_filelist", str, "", "Path to training filelist.")
data_g.add_arg("do_lower_case", bool, "", "Path to training filelist.")
data_g.add_arg("valid_filelist", str, "", "Path to valid filelist.")
data_g.add_arg("test_filelist", str, "", "Path to test filelist.")
data_g.add_arg("vocab_path", str, "", "Vocabulary path.")
data_g.add_arg("spm_model_file", str, "", "spm_model path.")
data_g.add_arg("bpe_vocab_file", str, "", "gpt bep vocab path.")
data_g.add_arg("bpe_json_file", str, "", "gpt bpe json path.")
data_g.add_arg("max_seq_len", int, 512, "Number of words of the longest seqence.")
data_g.add_arg("train_batch_size", int, 1024, "Total examples' number in batch for training. see also --in_tokens.")
data_g.add_arg("in_tokens", bool, False,
              "If set, the batch size will be the maximum number of tokens in one batch. "
              "Otherwise, it will be the maximum number of examples in one batch.")
data_g.add_arg('hack_old_data', bool, False, "Whether to support old train data format.",
        choices=[True, False])

run_type_g = ArgumentGroup(parser, "run_type", "running type options.")
run_type_g.add_arg("is_distributed", bool, False, "If set, then start distributed training.")
run_type_g.add_arg("PADDLE_USE_GPU", bool, True, "If set, use GPU for training.")
run_type_g.add_arg("use_fast_executor", bool, False, "If set, use fast parallel executor (in experiment).")
run_type_g.add_arg("do_test", bool, False, "Whether to perform evaluation on test data set.")
run_type_g.add_arg("shuffle_files", bool, True, "Whether to shuffle files.")
run_type_g.add_arg("visualdl", bool, False, "Whether to use visualdl")
run_type_g.add_arg("is_do_train", bool, True, "is_do_train")
run_type_g.add_arg("log_dir", str, "log", "output log dir")

class Readers(object):
    """ readers """
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

class PretrainingTrainer(BaseTrainer):
    """PretrainingTrainer：英文pretraining的trainer"""

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
        self.data_set_reader.train_reader.run()
        steps = 1
        save_checkpoints_path = os.path.join(self.params["checkpoints"], "save_checkpoints")
        time_begin = time.time()
        while True:
            try:
                if steps % self.params["train_log_step"] != 0:
                    self.run(InstanceName.TRAINING, need_fetch=False)
                else:
                    metrics_tensor_value = self.run(InstanceName.TRAINING, need_fetch=True)

                    fetch_list_dict = collections.OrderedDict()
                    for key, value in zip(self.fetch_list_train_key, metrics_tensor_value):
                        fetch_list_dict[key] = value
                    time_end = time.time()
                    used_time = time_end - time_begin
                    meta_info = collections.OrderedDict()

                    """ print train log """
                    log_info = ""
                    each_mask_lm_cost = fetch_list_dict['mask_lm_loss']
                    lm_w = fetch_list_dict['lm_weight']
                    learning_rate = fetch_list_dict["scheduled_lr"]
                    progress_out = self.data_set_reader.train_reader.get_progress()
                    epoch, current_file_index, total_file, current_file, mask_type = progress_out
                    metric = collections.OrderedDict()
                    metric["epoch"] = epoch
                    metric["progress"] = "{}/{}".format(current_file_index, total_file)
                    metric["step"] = steps
                    metric["loss"] = np.mean(fetch_list_dict[InstanceName.LOSS])
                    metric["ppl"] = np.exp(np.sum(each_mask_lm_cost * lm_w) / np.sum(lm_w))
                    for task in self.model_class.task_group:
                        name = task['task_name']
                        if name == 'mask_language_model':
                            continue
                        each_task_acc = fetch_list_dict["acc_" + name]
                        task_w = fetch_list_dict["task_weight_of_" + name]
                        acc = np.sum(each_task_acc * task_w) / np.sum(task_w)
                        metric["acc_" + name] = acc
                    metric["file"] = current_file
                    metric["mask_type"] = mask_type
                    metric["speed"] = "{} steps/s".format(self.params['train_log_step'] / used_time)
                    log_info += ", ".join([":".join([k, str(v)]) for k, v, in metric.items()])
                    if self.params['use_fp16']:
                        log_info += ", current_learning_rate:%f, loss_scaling:%f" \
                                    % (fetch_list_dict["scheduled_lr"], fetch_list_dict["loss_scaling"])
                    else:
                        log_info += ", current_learning_rate:{}".format(fetch_list_dict["scheduled_lr"])
                    time_begin = time.time()
                    logging.info(log_info)

                if steps % self.params["eval_step"] == 0:
                    if self.params["is_eval_dev"]:
                        self.evaluate(self.data_set_reader.dev_reader, InstanceName.EVALUATE, steps)
                    if self.params["is_eval_test"]:
                        self.evaluate(self.data_set_reader.test_reader, InstanceName.TEST, steps)

                if self.trainer_id == 0:
                    if steps % self.params["save_model_step"] == 0:
                        self.save_models(save_checkpoints_path, None, 
                                steps, save_inference=False)
                steps += 1
            except fluid.core.EOFException:
                self.save_models(save_checkpoints_path, None, 
                        steps, save_inference=False)
                self.data_set_reader.train_reader.stop()
                break
        if self.params["is_eval_dev"]:
            logging.info("Final evaluate result: ")
            self.evaluate(self.data_set_reader.dev_reader, InstanceName.EVALUATE, steps)
        if self.params["is_eval_test"]:
            logging.info("Final test result: ")
            self.evaluate(self.data_set_reader.test_reader, InstanceName.TEST, steps)

        self.save_models(save_checkpoints_path, None, 
                steps, save_inference=False)
        logging.info("Save checkpoint done!")
        logging.info("train and eval done!")

    def evaluate(self, reader, phase, steps):
        """
        :param reader:
        :param phase:
        :param program:
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
                metrics_tensor_value = self.run(phase, need_fetch=True)
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

        fetch_list_dict = collections.OrderedDict()
        for key, value in zip(self.fetch_list_evaluate_key, all_metrics_tensor_value):
            fetch_list_dict[key] = value
        time_end = time.time()
        used_time = time_end - time_begin
        log_info = ""
        lm_cost = 0
        lm_steps = 0
        cost = 0
        task_acc = collections.OrderedDict()
        task_steps = collections.OrderedDict()
        for step in range(i):
            lm_w = fetch_list_dict["lm_weight"][step]
            lm_cost += np.sum(fetch_list_dict['mask_lm_loss'][step] * lm_w)
            lm_steps += np.sum(lm_w)
            cost += np.mean(fetch_list_dict[InstanceName.LOSS][step])

            for task in self.model_class.task_group:
                name = task['task_name']
                if name == 'mask_language_model':
                    continue
                each_task_acc = fetch_list_dict["acc_" + name][step]
                task_w = fetch_list_dict["task_weight_of_" + name][step]
                task_acc[name] = task_acc.get(name, 0.0) + np.sum(each_task_acc * task_w)
                task_steps[name] = task_steps.get(name, 0.0) + np.sum(task_w)

        progress_out = self.data_set_reader.dev_reader.get_progress()
        epoch, current_file_index, total_file, current_file, mask_type = progress_out

        metric = collections.OrderedDict()
        metric["epoch"] = epoch
        metric["step"] = i
        metric["loss"] = "{}".format(cost / i)
        metric["ppl"] = "{}".format(np.exp(lm_cost / lm_steps))
        for task in self.model_class.task_group:
            name = task["task_name"]
            if name == 'mask_language_model':
                continue
            metric["acc_" + name] = task_acc[name] / task_steps[name]
        metric["speed"] = str(self.params['eval_step'] / i) + " steps/s"
        log_info += "[validation_set] " + ", ".join([":".join([k, str(v)]) for k, v, in metric.items()])
        log_info += "elapsed time: %f s" % (used_time)
        logging.info(log_info)


MODEL_CLASSES = {
    "ernie_1.0_ch": (ErnieConfig, ErnieMTLM, FullTokenizer, 
        PretrainingTrainer, ErniePretrainDataReader),
    "ernie_1.0_skep_ch": (ErnieConfig, ErnieSkepMTLM, FullTokenizer, 
        PretrainingTrainer, ErnieSkepPretrainDataReader),
    "ernie_2.0_en": (ErnieConfig, ErnieMTLM, FullTokenizer, 
        PretrainingTrainer, ErniePretrainDataReader),
    "ernie_2.0_skep_en": (ErnieConfig, ErnieSkepMTLM, FullTokenizer, 
        PretrainingTrainer, ErnieSkepPretrainDataReader),
    "roberta_en": (ErnieConfig, RobertaLM, GptBpeTokenizer, 
        PretrainingTrainer, RobertaPretrainDataReaderEnglish),
    "roberta_skep_en": (ErnieConfig, RobertaSkepLM, GptBpeTokenizer, 
        PretrainingTrainer, RobertaSkepPretrainDataReaderEnglish),
}


def main(args):
    """ main """
    log.init_log(os.path.join(args.log_dir, "train"), level=logging.DEBUG)
    task_group = json.load(open(args.task_group_path))
    config_class, model_class, tokenizer_class, trainer_class, reader_class = MODEL_CLASSES[args.model_type]
    
    config = config_class(args.ernie_config_path)
    model = model_class(config, args, task_group)
    if args.model_type in ["roberta_en", "roberta_skep_en"]:
        tokenizer = tokenizer_class(vocab_file=args.vocab_path, params=vars(args))
    else:
        tokenizer = tokenizer_class(vocab_file=args.vocab_path)
    
    args_eval = copy.deepcopy(args)
    args_eval.epoch=1
    readers = Readers(train_reader=reader_class(args, 'train_reader', tokenizer, task_group),
                      dev_reader=reader_class(args_eval, 'dev_reader', tokenizer, task_group, evaluate=True),
                      test_reader=None, predict_reader=None)

    params = vars(args)
    trainer = trainer_class(params, readers, model)

    trainer.train_and_eval()


if __name__ == "__main__":
    args = parser.parse_args()
    print_arguments(args)
    main(args)


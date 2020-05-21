# -*- coding: UTF-8 -*-
"""
本文件定义了Senta类，实现其情感分类，训练模型的接口。
"""

import os
import io
import shutil
import time
import requests
import tarfile
import logging
import numpy as np
from paddle.fluid.core import AnalysisConfig
from paddle.fluid.core import create_paddle_predictor
from senta.data.util_helper import convert_texts_to_ids, structure_fields_dict
from senta.utils.params import from_file, replace_none
from senta.utils.util_helper import array2tensor, check_cuda, text_type
from senta.common.register import RegisterSet
from senta.common import register
from senta.common.rule import InstanceName
from senta.data.data_set import DataSet
from senta.utils import args, params, log

logging.getLogger().setLevel(logging.INFO)
_get_abs_path = lambda path: os.path.normpath(os.path.join(os.getcwd(), os.path.dirname(__file__), path))

def get_http_url(url, file_name):
    """
    get_http_url
    """
    requests.packages.urllib3.disable_warnings()
    requests.adapters.DEFAULT_RETRIES = 5
    r = requests.get(url, verify=False, stream=True)
    with open(file_name, "wb") as f:
        for chunk in r.iter_content(chunk_size=512):
            f.write(chunk)

def untar(fname, dirs):
    """
    untar
    """
    try:
        t = tarfile.open(fname)
        t.extractall(path = dirs)
        return True
    except Exception as e:
        logging.error(e)
        return False

def download_data(data_url, md5_url):
    """
    download_data
    """
    model_files = _get_abs_path("model_files.tar.gz")
    md5_files = _get_abs_path("model_files.tar.gz.md5")
    md5_files_new = _get_abs_path("model_files.tar.gz.md5.new")
    model_files_prefix = _get_abs_path("model_files")

    get_http_url(md5_url, md5_files_new)
    if os.path.exists(model_files) and os.path.exists(md5_files):
        with open(md5_files, 'r') as fr:
            md5 = fr.readline().strip('\r\n').split('  ')[0]
        with open(md5_files_new, 'r') as fr:
            md5_new = fr.readline().strip('\r\n').split('  ')[0]
        if md5 == md5_new:
            return 0

    if os.path.exists(model_files):
        os.remove(model_files)
    if os.path.exists(model_files_prefix):
        shutil.move(model_files_prefix, model_files_prefix + '.' + str(int(time.time())))

    shutil.move(md5_files_new, md5_files)
    get_http_url(data_url, model_files)
    untar(model_files, _get_abs_path("./"))
    return 1

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
    dataset_reader = params_dict.get("dataset_reader")
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


class Senta(object):
    """docstring for Senta"""
    def __init__(self):
        super(Senta, self).__init__()
        self.__get_params()

    def __get_params(self):
        """
        __get_params
        """
        config_dir = _get_abs_path("config")
        param_path = os.path.join(config_dir, 'infer.json')
        param_dict = from_file(param_path)
        self._params = replace_none(param_dict)

    def __load_inference_model(self, model_path, use_gpu):
        """
        :param meta_path:
        :return:
        """
        check_cuda(use_gpu)
        config = AnalysisConfig(model_path + "/" + "model", model_path + "/" + "params")
        if use_gpu:
            config.enable_use_gpu(1024)
        else:
            config.disable_gpu()
            config.enable_mkldnn()
        inference = create_paddle_predictor(config.to_native_config())
        return inference

    def get_support_model(self):
        """
        get_support_model
        """
        pre_train_model = list(self._params.get("model_name").keys())
        return pre_train_model

    def get_support_task(self):
        """
        get_support_task
        """
        tasks = list(self._params.get("task_name").keys())
        return tasks

    def init_model(self, model_class="ernie_1.0_skep_large_ch", task="sentiment_classify", use_cuda=False):
        """
        init_model
        """
        ptm = self._params.get("model_name").get(model_class)
        ptm_id = ptm.get('type')
        task_id = self._params.get("task_name").get(task)
        model_dict = self._params.get("model_class").get(ptm_id + task_id)

        # step 1: get_init_model, if download
        data_url = model_dict.get("model_file_http_url")
        md5_url = model_dict.get("model_md5_http_url")
        is_download_data = download_data(data_url, md5_url)

        # step 2 get model_class
        register.import_modules()
        model_name = model_dict.get("type")
        self.model_class = RegisterSet.models.__getitem__(model_name)(model_dict)

        # step 3 init data params
        model_path = _get_abs_path(model_dict.get("inference_model_path"))
        data_params_path = model_path + "/infer_data_params.json"
        param_dict = from_file(data_params_path)
        param_dict = replace_none(param_dict)
        self.input_keys = param_dict.get("fields")

        # step 4 init env
        self.inference = self.__load_inference_model(model_path, use_cuda)

        # step 5: tokenizer
        tokenizer_info = model_dict.get("predict_reader").get('tokenizer')
        tokenizer_name = tokenizer_info.get('type')
        tokenizer_vocab_path = _get_abs_path(tokenizer_info.get('vocab_path'))
        tokenizer_params = None
        if tokenizer_info.__contains__("params"):
            tokenizer_params = tokenizer_info.get("params")
            bpe_v_file = tokenizer_params["bpe_vocab_file"]
            bpe_j_file = tokenizer_params["bpe_json_file"]
            tokenizer_params["bpe_vocab_file"] = _get_abs_path(bpe_v_file)
            tokenizer_params["bpe_json_file"] = _get_abs_path(bpe_j_file)

        tokenizer_class = RegisterSet.tokenizer.__getitem__(tokenizer_name)
        self.tokenizer = tokenizer_class(vocab_file=tokenizer_vocab_path,
                split_char=" ",
                unk_token="[UNK]",
                params=tokenizer_params)
        self.max_seq_len = 512
        self.truncation_type = 0
        self.padding_id = 1 if tokenizer_name == "GptBpeTokenizer" else 0

        self.inference_type = model_dict.get("inference_type", None)

        # step6: label_map
        label_map_file = model_dict.get("label_map_path", None)
        self.label_map = {}
        if isinstance(label_map_file, str):
            label_map_file = _get_abs_path(label_map_file)
            with open(label_map_file, 'r') as fr:
                for line in fr.readlines():
                    line = line.strip('\r\n')
                    items = line.split('\t')
                    idx, label = int(items[1]), items[0]
                    self.label_map[idx] = label

    
    def predict(self, texts_, aspects=None):
        """
        the sentiment classifier's function
        :param texts: a unicode string or a list of unicode strings.
        :return: sentiment prediction results.
        """
        if isinstance(texts_, text_type):
            texts_ = [texts_]

        if isinstance(aspects, text_type):
            aspects = [aspects]

        return_list = convert_texts_to_ids(texts_, self.tokenizer, self.max_seq_len, \
                self.truncation_type, self.padding_id)
        record_dict = structure_fields_dict(return_list, 0, need_emb=False)
        input_list = []
        for item in self.input_keys:
            kv = item.split("#")
            name = kv[0]
            key = kv[1]
            input_item = record_dict[InstanceName.RECORD_ID][key]
            input_list.append(input_item)
        inputs = [array2tensor(ndarray) for ndarray in input_list]
        result = self.inference.run(inputs)
        batch_result = self.model_class.parse_predict_result(result)
        results = []
        if self.inference_type == 'seq_lab':
            for text, probs in zip(texts_, batch_result):
                label = [self.label_map[l] for l in probs]
                results.append((text, label))
        else:
            for text, probs in zip(texts_, batch_result):
                label = self.label_map[np.argmax(probs)]
                results.append((text, label))
        return results

    def train(self, json_path):
        """
        the function use to retrain model
        :param model_save_dir: where to saving model after training
        """
        param_dict = params.from_file(json_path)
        _params = params.replace_none(param_dict)
        register.import_modules()

        dataset_reader_params_dict = _params.get("dataset_reader")
        dataset_reader = dataset_reader_from_params(dataset_reader_params_dict)

        model_params_dict = _params.get("model")
        model, num_train_examples = model_from_params(model_params_dict)

        trainer_params_dict = _params.get("trainer")
        trainer = build_trainer(trainer_params_dict, dataset_reader, model, num_train_examples)

        trainer.train_and_eval()
        logging.info("end of run train and eval .....")

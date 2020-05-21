# *_*coding:utf-8 *_*
"""
inference
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import time
import os

from senta.common.rule import InstanceName
from senta.utils import params
from senta.utils.util_helper import array2tensor
from paddle.fluid.core_avx import AnalysisConfig, create_paddle_predictor


class Inference(object):
    """Inferece:模型预测
    """
    def __init__(self, param, data_set_reader, model_class):
        """
        1.解析input_data的结构 2.解析参数，构造inference  3. 启动data_generator,开始预测 4.回掉预测结果到model中进行解析
        :param param: 运行的基本参数设置
        :param data_set_reader: 运行的基本参数设置
        :param model_class: 使用的是哪个model
        """
        self.data_set_reader = data_set_reader
        self.param = param
        self.model_class = model_class
        self.inference = None
        self.input_keys = []
        self.init_data_params()
        self.init_env()

    def load_inference_model(self, model_path, use_gpu):
        """
        :param meta_path:
        :return:
        """
        config = AnalysisConfig(model_path + "/" + "model", model_path + "/" + "params")
        if use_gpu:
            config.enable_use_gpu(1024)
        else:
            config.disable_gpu()
            config.enable_mkldnn()
        inference = create_paddle_predictor(config.to_native_config())
        return inference

    def init_env(self):
        """
        :return:
        """
        self.inference = self.load_inference_model(self.param["inference_model_path"], 
                self.param["PADDLE_USE_GPU"])
    
    def init_data_params(self):
        """
        :return:
        """
        model_path = self.param["inference_model_path"]
        data_params_path = model_path + "/infer_data_params.json"
        param_dict = params.from_file(data_params_path)
        param_dict = params.replace_none(param_dict)
        self.input_keys = param_dict.get("fields")

    def do_inference(self):
        """
        :return:
        """
        logging.info("start do inference....")
        test_save = self.param.get("test_save")
        #"./output/inference/test_out.tsv")
        if not os.path.exists(os.path.dirname(test_save)):
            os.makedirs(os.path.dirname(test_save))

        fw = open(test_save, 'w')
        total_time = 0
        reader = self.data_set_reader.predict_reader.data_generator()
        qid = 0

        inference_type = self.param.get("inference_type")
        label_map_file = self.param.get("vocab_path", None)
        if isinstance(label_map_file, str):
            label_map = {}
            with open(label_map_file, 'r') as fr:
                for line in fr.readlines():
                    line = line.strip('\r\n')
                    items = line.split('\t')
                    idx, label = int(items[1]), items[0]
                    label_map[idx] = label

        for sample in reader():
            sample_dict = self.data_set_reader.predict_reader.convert_fields_to_dict(sample, need_emb=False)
            input_list = []
            for item in self.input_keys:
                kv = item.split("#")
                name = kv[0]
                key = kv[1]
                item_instance = sample_dict[name]
                input_item = item_instance[InstanceName.RECORD_ID][key]
                input_list.append(input_item)

            inputs = [array2tensor(ndarray) for ndarray in input_list]
            begin_time = time.time()
            result = self.inference.run(inputs)
            end_time = time.time()
            total_time += end_time - begin_time
            batch_result = self.model_class.parse_predict_result(result)
            if inference_type == 'seq_lab':
                for label in batch_result:
                    label = [label_map[l] for l in label]
                    line = str(qid) + '\t' + ' '.join(label) + '\n'
                    fw.write(line)
                    qid += 1
            else:
                for item_prob in batch_result:
                    pred = item_prob.argmax()
                    score = item_prob.tolist()
                    line = str(qid) + '\t' + str(pred) + '\t' + str(score) + '\n'
                    fw.write(line)
                    qid += 1
        fw.close()
        logging.info("total_time:{}".format(total_time))

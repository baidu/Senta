# -*- coding: utf-8 -*
"""
ernie序列标注的model
"""
import collections
import logging
import math
from collections import OrderedDict

import numpy as np
from paddle import fluid
from senta.common.register import RegisterSet
from senta.common.rule import InstanceName
from senta.models.model import Model
from senta.modules.ernie import ErnieModel, ErnieConfig
import senta.metrics.metrics as metrics


@RegisterSet.models.register
class ErnieCrfSeqLabel(Model):
    """ErnieSeqLabel
    """

    def __init__(self, model_params):
        Model.__init__(self, model_params)

    def forward(self, fields_dict, phase):
        """前向计算组网部分包括loss值的计算,必须由子类实现
        :param: fields_dict: 序列化好的id
        :param: phase: 当前调用的阶段，如训练、预测，不同的阶段组网可以不一样
        :return: 一个dict数据，存放TARGET_FEED_NAMES, TARGET_PREDICTS, PREDICT_RESULT,LABEL,LOSS等所有你希望获取的数据
        """
        instance_text_a = fields_dict["text_a"]
        record_id_text_a = instance_text_a[InstanceName.RECORD_ID]
        text_a_src = record_id_text_a[InstanceName.SRC_IDS]
        text_a_pos = record_id_text_a[InstanceName.POS_IDS]
        text_a_sent = record_id_text_a[InstanceName.SENTENCE_IDS]
        text_a_mask = record_id_text_a[InstanceName.MASK_IDS]
        text_a_task = record_id_text_a[InstanceName.TASK_IDS]
        text_a_lens = record_id_text_a[InstanceName.SEQ_LENS]

        instance_label = fields_dict["label"]
        record_id_label = instance_label[InstanceName.RECORD_ID]
        label = record_id_label[InstanceName.SRC_IDS]
        label_lens = record_id_label[InstanceName.SEQ_LENS]
        unpad_labels = fluid.layers.sequence_unpad(label, length=label_lens)

        emb_dict = self.make_embedding(fields_dict, phase)
        emb_text_a = emb_dict["text_a"]
        unpad_emb = fluid.layers.sequence_unpad(emb_text_a, length=text_a_lens)

        num_labels = self.model_params.get("num_labels", 3)

        # demo config
        grnn_hidden_dim = 128  # 768
        crf_lr = 0.2
        bigru_num = 2
        init_bound = 0.1

        def _bigru_layer(input_feature):
            """define the bidirectional gru layer
            """
            pre_gru = fluid.layers.fc(
                input=input_feature,
                size=grnn_hidden_dim * 3,
                param_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.Uniform(
                        low=-init_bound, high=init_bound),
                    regularizer=fluid.regularizer.L2DecayRegularizer(
                        regularization_coeff=1e-4)))
            gru = fluid.layers.dynamic_gru(
                input=pre_gru,
                size=grnn_hidden_dim,
                param_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.Uniform(
                        low=-init_bound, high=init_bound),
                    regularizer=fluid.regularizer.L2DecayRegularizer(
                        regularization_coeff=1e-4)))

            pre_gru_r = fluid.layers.fc(
                input=input_feature,
                size=grnn_hidden_dim * 3,
                param_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.Uniform(
                        low=-init_bound, high=init_bound),
                    regularizer=fluid.regularizer.L2DecayRegularizer(
                        regularization_coeff=1e-4)))
            gru_r = fluid.layers.dynamic_gru(
                input=pre_gru_r,
                size=grnn_hidden_dim,
                is_reverse=True,
                param_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.Uniform(
                        low=-init_bound, high=init_bound),
                    regularizer=fluid.regularizer.L2DecayRegularizer(
                        regularization_coeff=1e-4)))

            bi_merge = fluid.layers.concat(input=[gru, gru_r], axis=1)
            return bi_merge

        input_feature = unpad_emb
        for i in range(bigru_num):
            bigru_output = _bigru_layer(input_feature)
            input_feature = bigru_output

        emission = fluid.layers.fc(
            size=num_labels,
            input=bigru_output,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Uniform(
                    low=-init_bound, high=init_bound),
                regularizer=fluid.regularizer.L2DecayRegularizer(
                    regularization_coeff=1e-4)))

        crf_cost = fluid.layers.linear_chain_crf(
            input=emission,
            label=unpad_labels,
            param_attr=fluid.ParamAttr(
                name='crfw',
                learning_rate=crf_lr))

        prediction = fluid.layers.crf_decoding(
            input=emission, param_attr=fluid.ParamAttr(name='crfw'))

        run_value = fluid.layers.chunk_eval(input=prediction, label=unpad_labels, chunk_scheme="IOB",
                                            num_chunk_types=int(math.ceil((num_labels - 1) / 2.0)))
        precision, recall, f1_score, num_infer_chunks, num_label_chunks, num_correct_chunks = run_value

        if phase == InstanceName.SAVE_INFERENCE:
            target_predict_list = [prediction]
            target_feed_name_list = [text_a_src.name, text_a_pos.name, text_a_sent.name,
                                     text_a_mask.name, text_a_lens.name]
            emb_params = self.model_params.get("embedding")
            ernie_config = ErnieConfig(emb_params.get("config_path"))
            if ernie_config.get('use_task_id', False):
                target_feed_name_list.append(text_a_task.name)

            forward_return_dict = {
                InstanceName.TARGET_FEED_NAMES: target_feed_name_list,
                InstanceName.TARGET_PREDICTS: target_predict_list
            }
            return forward_return_dict

        avg_cost = fluid.layers.mean(x=crf_cost)

        forward_return_dict = {
            # InstanceName.PREDICT_RESULT: prediction,
            # InstanceName.LABEL: label,
            "num_infer_chunks": num_infer_chunks,
            "num_label_chunks": num_label_chunks,
            "num_correct_chunks": num_correct_chunks,
            InstanceName.LOSS: avg_cost
        }
        return forward_return_dict

    def fields_process(self, fields_dict, phase):
        """对fields中序列化好的id按需做二次处理
        :return: 处理好的fields
        """

        return fields_dict

    def make_embedding(self, fields, phase):
        """构造embedding，按需调用
        :param fields:
        :param phase:
        :return: embedding_dict
        """
        emb_params = self.model_params.get("embedding")
        config_path = emb_params.get("config_path")
        use_fp16 = emb_params.get("use_fp16")
        ernie_config = ErnieConfig(config_path)

        instance_text_a = fields["text_a"]
        record_id_text_a = instance_text_a[InstanceName.RECORD_ID]
        text_a_src = record_id_text_a[InstanceName.SRC_IDS]
        text_a_pos = record_id_text_a[InstanceName.POS_IDS]
        text_a_sent = record_id_text_a[InstanceName.SENTENCE_IDS]
        text_a_mask = record_id_text_a[InstanceName.MASK_IDS]
        text_a_task = record_id_text_a[InstanceName.TASK_IDS]

        ernie = ErnieModel(
            src_ids=text_a_src,
            position_ids=text_a_pos,
            sentence_ids=text_a_sent,
            task_ids=text_a_task,
            input_mask=text_a_mask,
            config=ernie_config,
            use_fp16=use_fp16
        )
        emb_sequence = ernie.get_sequence_output()
        embedding_dict = {"text_a": emb_sequence}
        return embedding_dict

    def optimizer(self, loss, is_fleet=False):
        """
        :param loss:
        :param is_fleet:
        :return:
        """
        opt_params = self.model_params.get("optimization")

        optimizer_output_dict = OrderedDict()
        optimizer_output_dict['use_ernie_opt'] = True

        opt_args_dict = OrderedDict()
        opt_args_dict["loss"] = loss
        opt_args_dict["warmup_steps"] = opt_params["warmup_steps"]
        opt_args_dict["num_train_steps"] = opt_params["max_train_steps"]
        opt_args_dict["learning_rate"] = opt_params["learning_rate"]
        opt_args_dict["weight_decay"] = opt_params["weight_decay"]
        opt_args_dict["scheduler"] = opt_params["lr_scheduler"]
        opt_args_dict["use_fp16"] = self.model_params["embedding"].get("use_fp16", False)
        opt_args_dict["use_dynamic_loss_scaling"] = opt_params["use_dynamic_loss_scaling"]
        opt_args_dict["init_loss_scaling"] = opt_params["init_loss_scaling"]
        opt_args_dict["incr_every_n_steps"] = opt_params["incr_every_n_steps"]
        opt_args_dict["decr_every_n_nan_or_inf"] = opt_params["decr_every_n_nan_or_inf"]
        opt_args_dict["incr_ratio"] = opt_params["incr_ratio"]
        opt_args_dict["decr_ratio"] = opt_params["decr_ratio"]

        optimizer_output_dict["opt_args"] = opt_args_dict
        return optimizer_output_dict

    def parse_predict_result(self, predict_result):
        """按需解析模型预测出来的结果
        :param predict_result: 模型预测出来的结果
        :return:
        """
        output = predict_result[0]
        seq_lens = np.array(output.lod).reshape(-1)
        output_data = output.data.int64_data()
        start_index = 0
        batch_result = []
        for end_index in seq_lens[1:]:
            example_true_length = end_index - start_index
            one_example_infer_result = output_data[start_index + 1: start_index + example_true_length - 1]
            #logging.info(one_example_infer_result)
            start_index = end_index
            batch_result.append(one_example_infer_result)
        return batch_result

    def get_metrics(self, fetch_output_dict, meta_info, phase):
        """指标评估部分的动态计算和打印
        :param fetch_output_dict: executor.run过程中fetch出来的forward中定义的tensor
        :param meta_info：常用的meta信息，如step, used_time, gpu_id等
        :param phase: 当前调用的阶段，包含训练和评估
        :return:
        """
        # predictions = fetch_output_dict[InstanceName.PREDICT_RESULT]
        # label = fetch_output_dict[InstanceName.LABEL]

        num_infer_chunks = fetch_output_dict["num_infer_chunks"]
        num_label_chunks = fetch_output_dict["num_label_chunks"]
        num_correct_chunks = fetch_output_dict["num_correct_chunks"]

        metrics_chunk = metrics.Chunk()
        metrics_res = metrics_chunk.eval([num_infer_chunks, num_label_chunks, num_correct_chunks])

        precision = metrics_res["precision"]
        recall = metrics_res["recall"]
        f1_score = metrics_res["f1_score"]

        if phase == InstanceName.TRAINING:
            step = meta_info[InstanceName.STEP]
            time_cost = meta_info[InstanceName.TIME_COST]
            #logging.debug("phase = {0} f1 = {1} precision = {2} recall = {3} step = {4} time_cost = {5}".format(
            #    phase, f1_score, precision, recall, step, time_cost))
            logging.debug("phase = {0} step = {1} time_cost = {2}".format(
                phase, step, time_cost))
        if phase == InstanceName.EVALUATE or phase == InstanceName.TEST:
            time_cost = meta_info[InstanceName.TIME_COST]
            #logging.debug("phase = {0} f1 = {1} precision = {2} recall = {3} time_cost = {4}".format(
            #    phase, f1_score, precision, recall, time_cost))
            logging.debug("phase = {0} time_cost = {1}".format(
                phase, time_cost))

        metrics_return_dict = collections.OrderedDict()
        metrics_return_dict["f1"] = f1_score
        metrics_return_dict["precision"] = precision
        metrics_return_dict["recall"] = recall
        return metrics_return_dict

    def metrics_show(self, result_evaluate):
        """评估指标展示
        :return:
        """
        pass

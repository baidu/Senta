# -*- coding: utf-8 -*
"""
RobertaClassification
"""
import collections
import logging
from collections import OrderedDict

import numpy as np
from paddle import fluid
from senta.common.register import RegisterSet
from senta.common.rule import InstanceName
from senta.models.model import Model
from senta.modules.ernie import ErnieModel, ErnieConfig
import senta.metrics.metrics as metrics


@RegisterSet.models.register
class RobertaClassification(Model):
    """RobertaClassification
    """

    def __init__(self, model_params):
        Model.__init__(self, model_params)

    def forward(self, fields_dict, phase):
        """前向计算组网部分包括loss值的计算,必须由子类实现
        :param: fields_dict: 序列化好的id
        :param: phase: 当前调用的阶段，如训练、预测，不同的阶段组网可以不一样
        :return: 一个dict数据，存放TARGET_FEED_NAMES, TARGET_PREDICTS, PREDICT_RESULT,LABEL,LOSS等所有你希望获取的数据
        """
        fields_dict = self.fields_process(fields_dict, phase)
        instance_text_a = fields_dict["text_a"]
        record_id_text_a = instance_text_a[InstanceName.RECORD_ID]
        text_a_src = record_id_text_a[InstanceName.SRC_IDS]
        text_a_pos = record_id_text_a[InstanceName.POS_IDS]
        text_a_sent = record_id_text_a[InstanceName.SENTENCE_IDS]
        text_a_mask = record_id_text_a[InstanceName.MASK_IDS]
        text_a_task = record_id_text_a[InstanceName.TASK_IDS]

        instance_label = fields_dict["label"]
        record_id_label = instance_label[InstanceName.RECORD_ID]
        label = record_id_label[InstanceName.SRC_IDS]
        
        qid = None
        if "qid" in fields_dict.keys():
            instance_qid = fields_dict["qid"]
            record_id_qid = instance_qid[InstanceName.RECORD_ID]
            qid = record_id_qid[InstanceName.SRC_IDS]

        emb_dict = self.make_embedding(fields_dict, phase)
        emb_text_a = emb_dict["text_a"]

        num_labels = 2

        cls_feats = fluid.layers.dropout(
            x=emb_text_a,
            dropout_prob=0.1,
            dropout_implementation="upscale_in_train")
        
        logits = fluid.layers.fc(
            input=cls_feats,
            size=num_labels,
            param_attr=fluid.ParamAttr(
                name="_cls_out_w",
                initializer=fluid.initializer.TruncatedNormal(scale=0.02)),
            bias_attr=fluid.ParamAttr(
                name="_cls_out_b",
                initializer=fluid.initializer.Constant(0.)))

        if phase == InstanceName.SAVE_INFERENCE:
            """保存模型时需要的入参：表示预测时最终输出的结果"""
            probs = fluid.layers.softmax(logits)
            target_predict_list = [probs]
            """保存模型时需要的入参：表示模型预测时需要输入的变量名称和顺序"""
            target_feed_name_list = [text_a_src.name, text_a_pos.name, text_a_mask.name]
            emb_params = self.model_params.get("embedding")
            ernie_config = ErnieConfig(emb_params.get("config_path"))
            if ernie_config.get('use_task_id', False):
                target_feed_name_list.append(text_a_task.name)

            forward_return_dict = {
                InstanceName.TARGET_FEED_NAMES: target_feed_name_list,
                InstanceName.TARGET_PREDICTS: target_predict_list
            }
            return forward_return_dict
        
        ce_loss, probs = fluid.layers.softmax_with_cross_entropy(
            logits=logits, label=label, return_softmax=True)
        loss = fluid.layers.mean(x=ce_loss)

        num_seqs = fluid.layers.create_tensor(dtype="int64")
        accuracy = fluid.layers.accuracy(input=probs, label=label, total=num_seqs)

        """PREDICT_RESULT,LABEL,LOSS 是关键字，必须要赋值并返回"""

        forward_return_dict = {
            InstanceName.PREDICT_RESULT: probs,
            InstanceName.LABEL: label,
            InstanceName.LOSS: loss,
            "accuracy": accuracy,
            "num_seqs": num_seqs
        }
        if qid:
            forward_return_dict["qid"] = qid
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
        emb_sequence = ernie.get_pooled_output()
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
        output_data = output.data.float_data()
        batch_result = np.array(output_data).reshape((-1, 2))
        #for item_prob in batch_result:
        #    logging.info('\t'.join(map(str, item_prob.tolist())))
        return batch_result

    def get_metrics(self, fetch_output_dict, meta_info, phase):
        """指标评估部分的动态计算和打印
        :param fetch_output_dict: executor.run过程中fetch出来的forward中定义的tensor
        :param meta_info：常用的meta信息，如step, used_time, gpu_id等
        :param phase: 当前调用的阶段，包含训练和评估
        :return:
        """
        predictions = fetch_output_dict[InstanceName.PREDICT_RESULT]
        label = fetch_output_dict[InstanceName.LABEL]
        metrics_acc = metrics.Acc()
        acc = metrics_acc.eval([predictions, label])
        metrics_pres = metrics.Precision()
        precision = metrics_pres.eval([predictions, label])

        if phase == InstanceName.TRAINING:
            step = meta_info[InstanceName.STEP]
            time_cost = meta_info[InstanceName.TIME_COST]
            logging.debug("phase = {0} acc = {1} precision = {2} step = {3} time_cost = {4}".format(
                phase, acc, precision, step, time_cost))
        if phase == InstanceName.EVALUATE or phase == InstanceName.TEST:
            time_cost = meta_info[InstanceName.TIME_COST]
            logging.debug("phase = {0} acc = {1} precision = {2} time_cost = {3}".format(
                phase, acc, precision, time_cost))

        metrics_return_dict = collections.OrderedDict()
        metrics_return_dict["acc"] = acc
        metrics_return_dict["precision"] = precision
        return metrics_return_dict

    def metrics_show(self, result_evaluate):
        """评估指标展示
        :return:
        """
        pass

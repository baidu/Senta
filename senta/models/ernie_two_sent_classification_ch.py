# -*- coding: utf-8 -*
"""
ErnieClassificationPersonal
"""
import logging
import os
from collections import OrderedDict

import numpy as np
from paddle import fluid

from senta.common.register import RegisterSet
from senta.common.rule import InstanceName
from senta.models.ernie_classification import ErnieClassification
from senta.modules.ernie import ErnieConfig, ErnieModel
from senta.metrics.glue_eval import evaluate_mrr, evaluate_map
from senta.utils.multi_process_eval import MultiProcessEval


@RegisterSet.models.register
class ErnieTwoSentClassificationCh(ErnieClassification):
    """ErnieTwoSentClassificationCh"""

    def __init__(self, model_params):
        ErnieClassification.__init__(self, model_params)
        #logging.info("ErnieClassificationPersonal init....")

    def forward(self, fields_dict, phase):
        """
        create forwrad net
        """
        fields_dict = self.fields_process(fields_dict, phase)
        # text_a
        instance_text_a = fields_dict["text_a"]
        record_id_text_a = instance_text_a[InstanceName.RECORD_ID]
        text_a_src = record_id_text_a[InstanceName.SRC_IDS]
        text_a_pos = record_id_text_a[InstanceName.POS_IDS]
        text_a_sent = record_id_text_a[InstanceName.SENTENCE_IDS]
        text_a_mask = record_id_text_a[InstanceName.MASK_IDS]
        text_a_task = record_id_text_a[InstanceName.TASK_IDS]
        # label
        instance_label = fields_dict["label"]
        record_id_label = instance_label[InstanceName.RECORD_ID]
        label = record_id_label[InstanceName.SRC_IDS]
        # qid
        instance_qid = fields_dict["qid"]
        record_id_qid = instance_qid[InstanceName.RECORD_ID]
        qid = record_id_qid[InstanceName.SRC_IDS]

        # get output embedding of model
        emb_dict = self.make_embedding(fields_dict, phase)
        emb_text_a = emb_dict["text_a"]
        num_labels = self.model_params.get("num_labels")

        cls_feats = fluid.layers.dropout(
            x=emb_text_a,
            dropout_prob=0.1,
            dropout_implementation="upscale_in_train")

        logits = fluid.layers.fc(
            input=cls_feats,
            size=num_labels,
            param_attr=fluid.ParamAttr(
                name="cls_out_w",
                initializer=fluid.initializer.TruncatedNormal(scale=0.02)),
            bias_attr=fluid.ParamAttr(
                name="cls_out_b",
                initializer=fluid.initializer.Constant(0.)))

        if phase == InstanceName.SAVE_INFERENCE:
            probs = fluid.layers.softmax(logits)
            target_predict_list = [probs]
            target_feed_name_list = [text_a_src.name, text_a_pos.name, text_a_sent.name,
                                     text_a_mask.name]
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
        # num_seqs = fluid.layers.slice(fluid.layers.shape(label), axes=[0], starts=[0], ends=[1])

        forward_return_dict = {
            "accuracy": accuracy,
            InstanceName.LOSS: loss
        }
        if phase == InstanceName.EVALUATE or phase == InstanceName.TEST:
            forward_return_dict["qid"] = qid
            forward_return_dict["num_seqs"] = num_seqs
            forward_return_dict[InstanceName.PREDICT_RESULT] = probs
            forward_return_dict[InstanceName.LABEL] = label

        return forward_return_dict

    def fields_process(self, fields_dict, phase):
        """fields process"""
        return fields_dict

    def make_embedding(self, fields, phase):
        """make embedding"""
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

    def get_metrics(self, forward_output_dict, meta_info, phase):
        """get metrics"""
        loss = forward_output_dict[InstanceName.LOSS]
        acc = forward_output_dict["accuracy"]
        trainers_num = int(os.getenv("PADDLE_TRAINERS_NUM", "1"))

        if phase == InstanceName.TRAINING:
            # add normal log
            log_info = "step: %d, ave loss: %f, ave acc: %f, time_cost: %d "
            values = (meta_info[InstanceName.STEP], loss, acc, meta_info[InstanceName.TIME_COST])
            logging.info(log_info % values)

        elif phase == InstanceName.EVALUATE or phase == InstanceName.TEST:
            predictions = forward_output_dict[InstanceName.PREDICT_RESULT]
            label = forward_output_dict[InstanceName.LABEL]
            qid = forward_output_dict["qid"]
            num_seqs = forward_output_dict["num_seqs"]
            assert len(loss) == len(acc) == len(predictions) == len(label) == \
                   len(qid) == len(num_seqs), "the six results have different lenght"

            total_cost, total_acc, total_num_seqs, total_label_pos_num, total_pred_pos_num, \
            total_correct_num = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            qids, labels, scores = [], [], []
            for i in range(len(loss)):
                total_cost += np.sum(loss[i] * num_seqs[i])
                total_acc += np.sum(acc[i] * num_seqs[i])
                total_num_seqs += np.sum(num_seqs[i])
                labels.extend(label[i].reshape((-1)).tolist())
                if qid[i] is not None:
                    qids.extend(qid[i].reshape(-1).tolist())
                scores.extend(predictions[i][:, 1].reshape(-1).tolist())
                np_preds = np.argmax(predictions[i], axis=1).astype(np.float32)
                total_label_pos_num += np.sum(label[i])
                total_pred_pos_num += np.sum(np_preds)
                total_correct_num += np.sum(np.dot(np_preds, label[i]))

            output_path = "./output/tmpout"
            mul_pro_test = MultiProcessEval(output_path, phase, trainers_num,
                                                        meta_info[InstanceName.GPU_ID])
            if len(qids) == 0:
                log_info = "[%d_%s evaluation] ave loss: %f, ave acc: %f, data_num: %d, elapsed time: %f s"
                if mul_pro_test.dev_count == 1:
                    values = (meta_info[InstanceName.STEP], phase, total_cost / total_num_seqs, \
                              total_acc / total_num_seqs, \
                              total_num_seqs, meta_info[InstanceName.TIME_COST])
                    logging.info(log_info % values)
                else:
                    mul_pro_test.write_result([total_acc, total_num_seqs])
                    if mul_pro_test.gpu_id == 0:
                        acc_sum, data_num = mul_pro_test.concat_result(2)
                        values = (meta_info[InstanceName.STEP], phase, \
                                  total_cost / total_num_seqs, acc_sum / data_num, \
                                  int(data_num), meta_info[InstanceName.TIME_COST])
                        logging.info(log_info % values)
            else:
                is_print = True
                log_info = "[%d_%s evaluation] ave loss: %f, ave_acc: %f, mrr: %f, \
                        map: %f, p: %f, r: %f, f1: %f, data_num: %d, elapsed time: %f s"

                if mul_pro_test.dev_count > 1:
                    is_print = False
                    name_list = ["qids", "labels", "scores"]
                    mul_pro_test.write_result([total_correct_num, total_label_pos_num, \
                                               total_pred_pos_num], [qids, labels, scores], name_list)
                    if mul_pro_test.gpu_id == 0:
                        is_print = True
                        eval_index_all, eval_list_all = mul_pro_test.concat_result(3, 3, name_list)
                        total_correct_num, total_label_pos_num, total_pred_pos_num = eval_index_all
                        qids, labels, scores = [eval_list_all[name] for name in name_list]

                if is_print:
                    r = total_correct_num / total_label_pos_num
                    p = total_correct_num / total_pred_pos_num
                    f = 2 * p * r / (p + r)

                    assert len(qids) == len(labels) == len(scores)
                    preds = sorted(
                        zip(qids, scores, labels), key=lambda elem: (elem[0], -elem[1]))
                    mrr = evaluate_mrr(preds)
                    map = evaluate_map(preds)
                    values = (meta_info[InstanceName.STEP], phase, \
                              total_cost / total_num_seqs, total_acc / total_num_seqs, \
                              mrr, map, p, r, f, total_num_seqs, meta_info[InstanceName.TIME_COST])
                    logging.info(log_info, values)

                    logging.info(log_info % values)
        return None

#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Finetuning on classification tasks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import logging
from collections import OrderedDict

import paddle.fluid as fluid
import numpy as np
from senta.common.rule import InstanceName
from senta.common.register import RegisterSet
from senta.modules.ernie import ErnieModel, ErnieConfig
from senta.models.ernie_classification import ErnieClassification
from senta.utils.multi_process_eval import MultiProcessEval
from senta.metrics.glue_eval import acc_and_f1, matthews_corrcoef, pearson_and_spearman, \
                      simple_accuracy, evaluate_mrr, pgd_loss 


@RegisterSet.models.register
class ErnieTwoSentClassificationEn(ErnieClassification):
    """ErnieClassification_Personal"""

    def __init__(self, model_params):
        ErnieClassification.__init__(self, model_params)
        #logging.info("ErnieTwoSentClassificationEn init ....")
        self.is_classify = self.model_params["is_classify"]
        self.is_regression = self.model_params["is_regression"]

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
        
        cls_feats = fluid.layers.dropout(
            x=emb_text_a,
            dropout_prob=0.1,
            dropout_implementation="upscale_in_train")

        logits = fluid.layers.fc(
            input=cls_feats,
            size=self.model_params["num_labels"],
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
        
        assert self.is_classify != self.is_regression, \
                "is_classify or is_regression must be true and only oen of them can be true"
        num_seqs = fluid.layers.create_tensor(dtype="int64")

        if self.is_classify:
            if self.model_params["use_bce"]:
                # hard code for binary class
                one_hot_label = fluid.layers.one_hot(input=label, depth=2)
                loss = fluid.layers.sigmoid_cross_entropy_with_logits(logits, one_hot_label)
                loss = fluid.layers.reduce_sum(loss)
                probs = logits
            else:
                ce_loss, probs = fluid.layers.softmax_with_cross_entropy(
                    logits=logits, label=label, return_softmax=True)
                loss = fluid.layers.mean(x=ce_loss) 
            accuracy = fluid.layers.accuracy(input=probs, label=label, total=num_seqs)
        elif self.is_regression:
            if self.model_params["use_sigmoid"]:
                logits = fluid.layers.sigmoid(logits)
            cost = fluid.layers.square_error_cost(input=logits, label=label)
            loss = fluid.layers.mean(x=cost)
        else:
            raise ValueError('unsupported fine tune mode. only supported classify/regression')
        
        #if args_finetune.adv:
        #    task_fc_fn = partial(task_fc,
        #                        args=args,
        #                        is_prediction=False,
        #                        is_classify=args_finetune.is_classify,
        #                        is_regression=args_finetune.is_regression)

        #    adv_loss = pgd_loss(ernie, label, loss, task_fc_fn, args_finetune.adv_epsilon)
        #    loss = args.alpha * loss + (1 - args_finetune.alpha) * adv_loss

        if phase == InstanceName.TRAINING:
            forward_return_dict = {InstanceName.LOSS: loss}
            if self.is_classify:
                forward_return_dict["accuracy"] = accuracy
        elif phase == InstanceName.EVALUATE:
            forward_return_dict = {InstanceName.LOSS: loss}
            forward_return_dict["qid"] = qid
            forward_return_dict[InstanceName.LABEL] = label
            if self.is_classify:
                forward_return_dict["num_seqs"] = num_seqs
                forward_return_dict["accuracy"] = accuracy
                forward_return_dict[InstanceName.PREDICT_RESULT] = probs
            elif self.is_regression:
                forward_return_dict[InstanceName.PREDICT_RESULT] = logits
        elif phase == InstanceName.TEST:
            forward_return_dict = {"qid": qid}
            if self.is_classify:
                forward_return_dict[InstanceName.PREDICT_RESULT] = probs
            elif self.is_regression:
                forward_return_dict[InstanceName.PREDICT_RESULT] = logits

        return forward_return_dict

    def make_embedding(self, fields, phase):
        """make embedding"""
        emb_params = self.model_params.get("embedding")
        config_path = emb_params.get("config_path")
        ernie_config = ErnieConfig(config_path)
        use_fp16 = emb_params.get("use_fp16")

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
    
    def get_metrics(self, forward_output_dict, meta_info, phase, metas=None):
        """get metrics"""
        loss = forward_output_dict[InstanceName.LOSS]
        if self.is_classify:
            acc = forward_output_dict["accuracy"]
        trainers_num = int(os.getenv("PADDLE_TRAINERS_NUM"))

        if phase == InstanceName.TRAINING:
            # add normal log
            loss = forward_output_dict[InstanceName.LOSS]
            if self.is_classify:
                acc = forward_output_dict["accuracy"]
 
            if self.is_classify:
                log_info = "step: %d, ave loss: %f, ave acc: %f, elapsed time: %f s"
                values = (meta_info[InstanceName.STEP], loss, acc, \
                          meta_info[InstanceName.TIME_COST])
            elif self.is_regression:
                log_info = "step: %d, ave loss: %f, elapsed time: %f s"
                values = (meta_info[InstanceName.STEP], loss, meta_info[InstanceName.TIME_COST])

            logging.info(log_info % values)
            return None, None

        elif phase == InstanceName.EVALUATE:
            loss = forward_output_dict[InstanceName.LOSS]
            predictions = forward_output_dict[InstanceName.PREDICT_RESULT]
            label = forward_output_dict[InstanceName.LABEL]
            qid = forward_output_dict["qid"]
            if self.is_classify:
                num_seqs = forward_output_dict["num_seqs"]
                acc = forward_output_dict["accuracy"]
                assert len(loss) == len(acc) == len(predictions) == len(label) == \
                        len(qid) == len(num_seqs), "the six results have different length"
            elif self.is_regression:
                assert len(loss) == len(predictions) == len(label) == len(qid), \
                        "the four results have differrnt length"
            
            output_path = "./output/tmpout"
            mul_pro_test = MultiProcessEval(output_path, phase, trainers_num, 
                                                        meta_info[InstanceName.GPU_ID])
            meta = {}
            if self.is_classify:
                total_cost, total_acc, total_num_seqs, total_label_pos_num, total_pred_pos_num, \
                        total_correct_num = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                qids, labels, scores, preds = [], [], [], []
                for i in range(len(loss)):
                    total_cost += np.sum(loss[i] * num_seqs[i])
                    total_acc += np.sum(acc[i] * num_seqs[i])
                    total_num_seqs += np.sum(num_seqs[i])
                    labels.extend(label[i].reshape((-1)).tolist())
                    if qid[i] is not None:
                        qids.extend(qid[i].reshape(-1).tolist())
                    else:
                        qids.extend(list(range(len(label[i]))))
                    scores.extend(predictions[i][:, 1].reshape(-1).tolist())
                    np_preds = list(np.argmax(predictions[i], axis=1).astype(np.float32))
                    preds.extend([float(val) for val in np_preds])
                    total_label_pos_num += np.sum(label[i])
                    total_pred_pos_num += np.sum(np_preds)
                    total_correct_num += np.sum(np.dot(np_preds, label[i]))
                    
                #output_path = "./tmpout"
                #mul_pro_test = MultiProcessTestForClassifiy(output_path, phase, trainers_num, 
                #                                            meta_info[InstanceName.GPU_ID])
    
                if meta_info["metric"] == "acc_and_f1":
                    is_print = True
                    if mul_pro_test.dev_count == 1:
                        ret = acc_and_f1(preds, labels)
                    else:
                        is_print = False
                        name_list = ['preds', 'labels']
                        mul_pro_test.write_result([total_num_seqs], [preds, labels], name_list)
                        if mul_pro_test.gpu_id == 0:
                            is_print = True
                            eval_index_all, eval_list_all = mul_pro_test.concat_result(1, 2, \
                                                                                       name_list)
                            preds_concat, labels_concat = [eval_list_all[name] for name \
                                                           in name_list]
                            ret = acc_and_f1(preds_concat, labels_concat)
                            total_num_seqs = int(eval_index_all[0])
                    if is_print:
                        log_info = "[%d_%s evaluation] ave loss: %f, ave acc: %f, f1: %f, \
                                    data_num: %d, elapsed time: %f s" 
                        values = (meta_info[InstanceName.STEP], phase, \
                                  total_cost / total_num_seqs, \
                                  ret["acc"], ret["f1"], total_num_seqs, \
                                  meta_info[InstanceName.TIME_COST])
                        evaluate_info = log_info % values
                        meta['score'] = ret['f1']
                        logging.info(evaluate_info)
                
                elif meta_info["metric"] == "matthews_corrcoef":
                    is_print = True
                    if mul_pro_test.dev_count == 1:
                        ret = matthews_corrcoef(preds, labels)
                    else:
                        is_print = False
                        name_list = ['preds', 'labels']
                        mul_pro_test.write_result([total_num_seqs], [preds, labels], name_list)
                        if mul_pro_test.gpu_id == 0:
                            is_print = True
                            eval_index_all, eval_list_all = mul_pro_test.concat_result(1, 2, \
                                                                                      name_list)
                            preds_concat, labels_concat = [eval_list_all[name] for name in \
                                                          name_list]
                            ret = matthews_corrcoef(preds_concat, labels_concat)
                            total_num_seqs = int(eval_index_all[0])
                    if is_print:
                        log_info = "[%d_%s evaluation] ave loss: %f, matthews_corrcoef: %f, \
                                    data_num: %d, elapsed time: %f s"
                        values = (meta_info[InstanceName.STEP], phase, \
                                  total_cost / total_num_seqs, ret, total_num_seqs, \
                                  meta_info[InstanceName.TIME_COST])
                        evaluate_info = log_info % values
                        meta['score'] = ret
                        logging.info(evaluate_info)
                
                elif meta_info["metric"] == "matthews_corrcoef_and_accf1":
                    is_print = True
                    if mul_pro_test.dev_count == 1:
                        mat_ret = matthews_corrcoef(preds, labels)
                        sim_ret = acc_and_f1(preds, labels)
                    else:
                        is_print = False
                        name_list = ['preds', 'labels']
                        mul_pro_test.write_result([total_num_seqs], [preds, labels], name_list)
                        if mul_pro_test.gpu_id == 0:
                            is_print = True
                            eval_index_all, eval_list_all = mul_pro_test.concat_result(1, 2, \
                                                                                      name_list)
                            preds_concat, labels_concat = [eval_list_all[name] for name in \
                                                          name_list]
                            mat_ret = matthews_corrcoef(preds_concat, labels_concat)
                            sim_net = acc_and_f1(preds_concat, labels_concat)
                            total_num_seqs = int(eval_index_all[0])
                    if is_print:
                        log_info = "[%d_%s evaluation] ave loss: %f, matthews_corrcoef: %f, \
                                acc: %f, f1: %f, data_num: %d, elapsed time: %f s"
                        values = (meta_info[InstanceName.STEP], phase, \
                                  total_cost / total_num_seqs, mat_ret, sim_net["acc"], \
                                  sim_net["f1"], total_num_seqs, \
                                  meta_info[InstanceName.TIME_COST])
                        evaluate_info = log_info % values
                        meta['score'] = mat_ret
                        logging.info(evaluate_info)

                elif meta_info["metric"] == "pearson_and_spearman":
                    is_print = True
                    if mul_pro_test.dev_count == 1:
                        ret = pearson_and_spearman(scores, labels)
                    else:
                        is_print = False
                        name_list = ['scores', 'labels']
                        mul_pro_test.write_result([total_num_seqs], [scores, labels], name_list)
                        if mul_pro_test.gpu_id == 0:
                            is_print = True
                            eval_index_all, eval_list_all = mul_pro_test.concat_result(1, 2, \
                                                                                      name_list)
                            scores_concat, labels_concat = [eval_list_all[name] for name in \
                                                           name_list]
                            ret = pearson_and_spearman(scores_concat, labels_concat)
                            total_num_seqs = int(eval_index_all[0])
                    
                    if is_print:
                        log_info = "[%d_%s evaluation] ave loss: %f, pearson: %f, spearman: %f, \
                                corr: %f, data_num: %d, elapsed time: %f s"
                        values = (meta_info[InstanceName.STEP], phase, \
                                 total_cost / total_num_seqs, ret['pearson'], ret['spearman'], \
                                 ret['corr'], total_num_seqs, \
                                 meta_info[InstanceName.TIME_COST])
                        evaluate_info = log_info % values
                        meta['score'] = (ret['pearson'] + ret['spearmanr']) / 2.0
                        logging.info(evaluate_info)

                elif meta_info["metric"] == "simple_accuracy":
                    is_print = True
                    if mul_pro_test.dev_count == 1:
                        ret = simple_accuracy(preds, labels)
                    else:
                        is_print = False
                        name_list = ['preds', 'labels']
                        mul_pro_test.write_result([total_num_seqs], [preds, labels], name_list)
                        if mul_pro_test.gpu_id == 0:
                            is_print = True
                            eval_index_all, eval_list_all = mul_pro_test.concat_result(1, 2, \
                                                                                      name_list)
                            preds_concat, labels_concat = [eval_list_all[name] for name in \
                                                          name_list]
                            ret = simple_accuracy(preds_concat, labels_concat)
                            total_num_seqs = int(eval_index_all[0])
                    if is_print:
                        log_info = "[%d_%s evaluation] ave loss: %f, acc: %f, \
                                data_num: %d, elapsed time: %f s"
                        values = (meta_info[InstanceName.STEP], phase, \
                                  total_cost / total_num_seqs, ret, total_num_seqs, \
                                  meta_info[InstanceName.TIME_COST])
                        evaluate_info = log_info % values
                        meta['score'] = ret
                        logging.info(evaluate_info)
        
                elif meta_info["metric"] == "acc_and_f1_and_mrr":
                    is_print = True
                    log_info = "[%d_%s evaluation] ave loss: %f, ave_acc: %f, f1: %f, \
                            mrr: %f, data_num: %d, elapsed time: %f s"
                    name_list = ['qids', 'labels', 'scores', 'preds']
                    if mul_pro_test.dev_count > 1:
                        is_print = False
                        mul_pro_test.write_result([total_num_seqs], [qids, labels, \
                                                  scores, preds], name_list)
                        if mul_pro_test.gpu_id == 0:
                            is_print = True
                            eval_index_all, eval_list_all = mul_pro_test.concat_result(1, 4, \
                                                                                       name_list)
                            qids, labels, scores, preds = [eval_list_all[name] for name in \
                                                          name_list]
                            total_num_seqs = int(eval_index_all[0])

                    if is_print:
                        assert len(qids) == len(labels) == len(scores) == len(preds)
                        ret_a = acc_and_f1(preds, labels)
                        preds = sorted(
                            zip(qids, scores, labels), key=lambda elem: (elem[0], -elem[1]))
                        ret_b = mul_pro_test.evaluate_mrr(preds)
                        values = (meta_info[InstanceName.STEP], phase, \
                                total_cost / total_num_seqs, ret_a['acc'], ret_a['f1'], \
                                total_num_seqs, meta_info[InstanceName.TIME_COST])
                        evaluate_info = log_info % values
                        meta['score'] = ret_a['f1']
                        logging.info(evaluate_info)

                else:
                    raise ValueError('unsupported metric {}'.format(meta_info["metric"]))

            elif self.is_regression:
                total_num_seqs = 0.0
                qids, labels, scores = [], [], []
                for i in range(len(loss)):
                    labels.extend(label[i].reshape((-1)).tolist())
                    if qid[i] is not None:
                        qids.extend(qid[i].reshape(-1).tolist())
                    else:
                        qids.extend(list(range(len(label[i]))))
                    scores.extend(predictions[i].reshape(-1).tolist())
                
                if meta_info["metric"] == "pearson_and_spearman":
                    is_print = True
                    if mul_pro_test.dev_count == 1:
                        ret = pearson_and_spearman(scores, labels)
                    else:
                        is_print = False
                        name_list = ['scores', 'labels']
                        mul_pro_test.write_result([], [scores, labels], name_list)
                        if mul_pro_test.gpu_id == 0:
                            is_print = True
                            eval_index_all, eval_list_all = mul_pro_test.concat_result(0, 2, \
                                                                                      name_list)
                            scores_concat, labels_concat = [eval_list_all[name] for name in \
                                                           name_list]
                            ret = pearson_and_spearman(scores_concat, labels_concat)
                    if is_print:
                        log_info = "[%d_%s evaluation] ave loss: %f, pearson: %f, spearman: %f, \
                                corr: %f, elapsed time: %f s"
                        values = (meta_info[InstanceName.STEP], phase, 0.0, \
                                  ret['pearson'], ret['spearmanr'], ret['corr'], \
                                  meta_info[InstanceName.TIME_COST])
                        evaluate_info = log_info % values
                        meta['score'] = ret['pearson']
                        logging.info(evaluate_info)

                elif meta_info["metric"] == "matthews_corrcoef":
                    best_score = -1000000
                    best_thresh = None
                    is_print = True
                    log_info = "[%d_%s evaluation] ave loss: %f, matthews_corrcoef: %f, \
                                data_num: %d, best_thresh: %f, elapsed time: %f s"
                    name_list = ['scores', 'labels']
                    if mul_pro_test.dev_count > 1:
                        is_print = False
                        mul_pro_test.write_result([], [scores, labels], name_list)
                        if mul_pro_test.gpu_id == 0:
                            is_print = True
                            eval_index_all, eval_list_all = mul_pro_test.concat_result(0, 2, \
                                                                                      name_list)
                            scores, labels = [eval_list_all[name] for name in \
                                              name_list]

                    if is_print:
                        scores = np.array(scores)
                        scores = 1 / (1 + np.exp(-scores))
                        for s in range(0, 10000):
                            T = s / 1000.0
                            pred = (scores > T).astype('int')
                            matt_score = matthews_corrcoef(pred, labels)
                            if matt_score > best_score:
                                best_score = matt_score
                                best_thresh = T

                        values = (meta_info[InstanceName.STEP], phase, 0.0, best_score, \
                                total_num_seqs, best_thresh, meta_info[InstanceName.TIME_COST])
                        evaluate_info = log_info % values
                        logging.info(evaluate_info)

                else:
                    raise ValueError('unsupported metric {}'.format(meta_info["metric"]))

            return meta, None

        elif phase == InstanceName.TEST:
            meta = {}
            predictions = forward_output_dict[InstanceName.PREDICT_RESULT]
            qid = forward_output_dict["qid"]
            assert len(predictions) == len(qid), "the two fields must have same length"
            
            output_path = "./output/tmpout"
            mul_pro_test = MultiProcessEval(output_path, phase, trainers_num, 
                                                        meta_info[InstanceName.GPU_ID])

            qids, scores, probs = [], [], []
            preds = []
            for i in range(len(qid)):
                qids.extend(qid[i].reshape(-1).tolist())
                if self.is_classify:
                    np_preds = list(np.argmax(predictions[i], axis=1).astype(np.float32))
                    preds.extend([float(val) for val in np_preds])
                elif self.is_regression:
                    preds.extend(predictions[i].reshape(-1).tolist())
                probs.append(predictions[i])

            if isinstance(metas, dict) and "best_thresh" in metas:
                preds = (np.array(preds) > metas['best_thresh'].astype('int'))
                preds = list(preds)
            probs = np.concatenate(probs, axis=0).reshape([len(qids), -1]).tolist()

            is_print = True
            name_list = ['qids', 'preds', 'probs']
            if mul_pro_test.dev_count > 1:
                is_print = False
                mul_pro_test.write_result([], [qids, preds, probs], name_list)
                if mul_pro_test.gpu_id == 0:
                    is_print = True
                    eval_index_all, eval_list_all = mul_pro_test.concat_result(0, 3, name_list)
                    qids, preds, probs = [eval_list_all[name] for name in name_list]
                
            if is_print:
                meta = {
                    "qids": qids,
                    "preds": preds,
                    "probs": probs
                }

            return meta, None


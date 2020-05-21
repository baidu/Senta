# -*- coding: utf-8 -*
"""
ErnieOneSentClassification
"""
import logging
import os
import sys

from collections import OrderedDict
import numpy as np
from senta.metrics.glue_eval import acc_and_f1, matthews_corrcoef, pearson_and_spearman, \
                      simple_accuracy, evaluate_mrr, pgd_loss
from senta.common.register import RegisterSet
from senta.common.rule import InstanceName
from senta.models.ernie_classification import ErnieClassification
from senta.utils.multi_process_eval import MultiProcessEval


@RegisterSet.models.register
class ErnieOneSentClassificationCh(ErnieClassification):
    """ErnieOneSentClassificationCh"""

    def get_metrics(self, fetch_output_dict, meta_info, phase, metas=None):
        """get metrics"""
        forward_output_dict = fetch_output_dict
        trainers_num = int(os.getenv("PADDLE_TRAINERS_NUM"))
        meta_info["metric"] = meta_info.get("metric", "acc_and_f1")
        if phase == InstanceName.TRAINING:
            # add normal log
            loss = forward_output_dict[InstanceName.LOSS]
            acc = forward_output_dict["accuracy"]

            log_info = "step: %d, ave loss: %f, elapsed time: %f s"
            values = (meta_info[InstanceName.STEP], loss, meta_info[InstanceName.TIME_COST])
            logging.info(log_info % values)
            return None, None

        elif phase == InstanceName.EVALUATE:
            loss = forward_output_dict[InstanceName.LOSS]
            predictions = forward_output_dict[InstanceName.PREDICT_RESULT]
            label = forward_output_dict[InstanceName.LABEL]
            qid = forward_output_dict["qid"]
            num_seqs = forward_output_dict["num_seqs"]
            acc = forward_output_dict["accuracy"]
            assert len(loss) == len(predictions) == len(label) == \
                    len(qid) == len(num_seqs), "the five results have different length"

            output_path = "./output/tmpout"
            mul_pro_test = MultiProcessEval(output_path, phase, trainers_num,
                                                        meta_info[InstanceName.GPU_ID])
            meta = {}
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
                np_preds = list(np.argmax(predictions[i], axis=1).astype(np.float32))
                preds.extend([float(val) for val in np_preds])
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

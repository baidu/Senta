# -*- coding: utf-8 -*
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

"""
metrics
"""

import numpy as np
import logging

from sklearn import metrics
from paddle import fluid
from six.moves import xrange

log = logging.getLogger(__name__)

__all__ = [
    'Metrics', 'F1', 'Recall', 'Precision', 'Acc'
]


class Metrics(object):
    """Metrics"""
    def eval(self, run_value):
        """need overwrite， run_value是动态fetch回来的值，按需要进行计算和打印"""
        raise NotImplementedError


class Chunk(Metrics):
    """Chunk"""
    def eval(self, run_value):
        chunk_metrics = fluid.metrics.ChunkEvaluator()
        num_infer_chunks, num_label_chunks, num_correct_chunks = run_value
        if isinstance(num_infer_chunks[0], np.ndarray):
            for i in range(len(num_infer_chunks)):
                chunk_metrics.update(np.array(num_infer_chunks[i][0]), np.array(num_label_chunks[i][0]),
                                     np.array(num_correct_chunks[i][0]))
        else:
            for i in range(len(num_infer_chunks)):
                chunk_metrics.update(np.array(num_infer_chunks[i]), np.array(num_label_chunks[i]),
                                     np.array(num_correct_chunks[i]))
        precision, recall, f1_score = chunk_metrics.eval()
        result = {"precision": precision, "recall": recall, "f1_score": f1_score}
        return result


class Acc(Metrics):
    """Acc"""
    def eval(self, run_value):
        predict, label = run_value
        if isinstance(predict, list):
            tmp_arr = []
            for one_batch in predict:
                for pre in one_batch:
                    tmp_arr.append(np.argmax(pre))
        else:
            tmp_arr = []
            for pre in predict:
                tmp_arr.append(np.argmax(pre))

        predict_arr = np.array(tmp_arr)

        if isinstance(label, list):
            tmp_arr = []
            for one_batch in label:
                batch_arr = [one_label for one_label in one_batch]
                tmp_arr.extend(batch_arr)
            label_arr = np.array(tmp_arr)
        else:
            label_arr = np.array(label.flatten())

        score = metrics.accuracy_score(label_arr, predict_arr)

        return score


class Precision(Metrics):
    """Precision"""
    def eval(self, run_value):
        predict, label = run_value
        if isinstance(predict, list):
            tmp_arr = []
            for one_batch in predict:
                for pre in one_batch:
                    tmp_arr.append(np.argmax(pre))
        else:
            tmp_arr = []
            for pre in predict:
                tmp_arr.append(np.argmax(pre))

        predict_arr = np.array(tmp_arr)

        if isinstance(label, list):
            tmp_arr = []
            for one_batch in label:
                batch_arr = [one_label for one_label in one_batch]
                tmp_arr.extend(batch_arr)
            label_arr = np.array(tmp_arr)
        else:
            label_arr = np.array(label.flatten())

        score = metrics.precision_score(label_arr, predict_arr, average="macro")
        # logging.info("sklearn precision macro score = ", score)
        return score


class Recall(Metrics):
    """Recall"""
    def eval(self, run_value):
        predict, label = run_value
        predict_arr = None
        label_arr = None
        if isinstance(predict, list):
            tmp_arr = []
            for one_batch in predict:
                for pre in one_batch:
                    tmp_arr.append(np.argmax(pre))
        else:
            tmp_arr = []
            for pre in predict:
                tmp_arr.append(np.argmax(pre))

        predict_arr = np.array(tmp_arr)

        if isinstance(label, list):
            tmp_arr = []
            for one_batch in label:
                batch_arr = [one_label for one_label in one_batch]
                tmp_arr.extend(batch_arr)
            label_arr = np.array(tmp_arr)
        else:
            label_arr = np.array(label.flatten())

        score = metrics.recall_score(label_arr, predict_arr, average="macro")
        # logging.info("sklearn recall macro score = ", score)
        return score


class F1(Metrics):
    """F1"""

    def eval(self, run_value):
        predict, label = run_value
        predict_arr = None
        label_arr = None
        if isinstance(predict, list):
            tmp_arr = []
            for one_batch in predict:
                for pre in one_batch:
                    tmp_arr.append(np.argmax(pre))
        else:
            tmp_arr = []
            for pre in predict:
                tmp_arr.append(np.argmax(pre))

        predict_arr = np.array(tmp_arr)

        if isinstance(label, list):
            tmp_arr = []
            for one_batch in label:
                batch_arr = [one_label for one_label in one_batch]
                tmp_arr.extend(batch_arr)
            label_arr = np.array(tmp_arr)
        else:
            label_arr = np.array(label.flatten())

        score = metrics.f1_score(label_arr, predict_arr, average="macro")
        # logging.info("sklearn f1 macro score = ", score)
        return score


class Auc(Metrics):
    "Auc"
    def eval(self, run_value):
        predict, label = run_value
        predict_arr = None
        label_arr = None

        if isinstance(predict, list):
            tmp_arr = []
            for one_batch in predict:
                for pre in one_batch:
                    assert len(pre) == 2, "auc metrics only support binary classification, " \
                                          "and the positive label must be 1, negtive label must be 0"
                    tmp_arr.append(pre[1])
        else:
            tmp_arr = []
            for pre in predict:
                assert len(pre) == 2, "auc metrics only support binary classification, " \
                                      " and the positive label must be 1, negtive label must be 0"
                tmp_arr.append(pre[1])

        predict_arr = np.array(tmp_arr)

        if isinstance(label, list):
            tmp_arr = []
            for one_batch in label:
                batch_arr = [one_label for one_label in one_batch]
                tmp_arr.extend(batch_arr)
            label_arr = np.array(tmp_arr)
        else:
            label_arr = np.array(label.flatten())

        fpr, tpr, thresholds = metrics.roc_curve(label_arr, predict_arr)
        score = metrics.auc(fpr, tpr)
        return float("%.4f" % score)


class Pn(Metrics):
    """Pn"""
    def eval(self, run_value):
        pos_score, neg_score = run_value
        wrong_cnt = np.sum(pos_score <= neg_score)
        right_cnt = np.sum(pos_score > neg_score)
        if wrong_cnt == 0:
            pn = float("inf")
        else:
            pn = float(right_cnt) / wrong_cnt

        return pn


class Ppl(Metrics):
    """ppl"""

    def eval(self, run_value):
        label_len, loss = run_value
        cost_train = np.mean(loss)

        if isinstance(label_len, list):
            tmp_arr = []
            for one_batch in label_len:
                batch_arr = [one_len for one_len in one_batch]
                tmp_arr.extend(batch_arr)
            label_len = np.array(tmp_arr)

        word_count = np.sum(label_len)
        total_loss = cost_train * label_len.shape[0]
        try_loss = total_loss / word_count
        ppl = np.exp(total_loss / word_count)

        result = {"ave_loss": try_loss, "ppl": ppl}
        return result


def chunk_eval(np_labels, np_infers, np_lens, tag_num, dev_count=1):
    """ chunk_eval """
    def extract_bio_chunk(seq):
        """ extract_bio_chunk """
        chunks = []
        cur_chunk = None
        null_index = tag_num - 1
        for index in xrange(len(seq)):
            tag = seq[index]
            tag_type = tag // 2
            tag_pos = tag % 2

            if tag == null_index:
                if cur_chunk is not None:
                    chunks.append(cur_chunk)
                    cur_chunk = None
                continue

            if tag_pos == 0:
                if cur_chunk is not None:
                    chunks.append(cur_chunk)
                    cur_chunk = {}
                cur_chunk = {"st": index, "en": index + 1, "type": tag_type}

            else:
                if cur_chunk is None:
                    cur_chunk = {"st": index, "en": index + 1, "type": tag_type}
                    continue

                if cur_chunk["type"] == tag_type:
                    cur_chunk["en"] = index + 1
                else:
                    chunks.append(cur_chunk)
                    cur_chunk = {"st": index, "en": index + 1, "type": tag_type}

        if cur_chunk is not None:
            chunks.append(cur_chunk)
        return chunks

    null_index = tag_num - 1
    num_label = 0
    num_infer = 0
    num_correct = 0
    labels = np_labels.reshape([-1]).astype(np.int32).tolist()
    infers = np_infers.reshape([-1]).astype(np.int32).tolist()
    all_lens = np_lens.reshape([dev_count, -1]).astype(np.int32).tolist()

    base_index = 0
    for dev_index in xrange(dev_count):
        lens = all_lens[dev_index]
        max_len = 0
        for l in lens:
            max_len = max(max_len, l)

        for i in xrange(len(lens)):
            seq_st = base_index + i * max_len + 1
            seq_en = seq_st + (lens[i] - 2)
            infer_chunks = extract_bio_chunk(infers[seq_st:seq_en])
            label_chunks = extract_bio_chunk(labels[seq_st:seq_en])
            num_infer += len(infer_chunks)
            num_label += len(label_chunks)

            infer_index = 0
            label_index = 0
            while label_index < len(label_chunks) \
                   and infer_index < len(infer_chunks):
                if infer_chunks[infer_index]["st"] \
                    < label_chunks[label_index]["st"]:
                    infer_index += 1
                elif infer_chunks[infer_index]["st"] \
                    > label_chunks[label_index]["st"]:
                    label_index += 1
                else:
                    if infer_chunks[infer_index]["en"] \
                        == label_chunks[label_index]["en"] \
                        and infer_chunks[infer_index]["type"] \
                        == label_chunks[label_index]["type"]:
                        num_correct += 1

                    infer_index += 1
                    label_index += 1

        base_index += max_len * len(lens)

    return num_label, num_infer, num_correct


def calculate_f1(num_label, num_infer, num_correct):
    """ calculate_f1 """
    if num_infer == 0:
        precision = 0.0
    else:
        precision = num_correct * 1.0 / num_infer

    if num_label == 0:
        recall = 0.0
    else:
        recall = num_correct * 1.0 / num_label

    if num_correct == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1

class LmPpl(Metrics):
    """ppl for language model"""
    def eval(self, run_value):
        label_len, loss = run_value
        total_ppl = 0.0
        for seq_loss in loss:
            avg_ppl = np.exp(seq_loss)
            seq_ppl = np.mean(avg_ppl)
            total_ppl += seq_ppl
        ave_ppl = total_ppl / len(loss)
        return ave_ppl

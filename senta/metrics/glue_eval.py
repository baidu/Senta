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
"""ultis help and eval functions for glue ."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np

from scipy.stats import pearsonr, spearmanr
from six.moves import xrange
import paddle.fluid as fluid
from functools import partial

def scale_l2(x, norm_length):
    """
    # copy lines from here https://github.com/tensorflow/models/blob/master/research/adversarial_text/adversarial_losses.py#L190
    # shape(x) = (batch, num_timesteps, d)
    # Divide x by max(abs(x)) for a numerically stable L2 norm.
    # 2norm(x) = a * 2norm(x/a)
    # Scale over the full sequence, dims (1, 2)
    """

    alpha = fluid.layers.reduce_max(fluid.layers.abs(x), dim=1, keep_dim=True) + 1e-12
    l2_norm = alpha * fluid.layers.sqrt(
            fluid.layers.reduce_sum(fluid.layers.pow(x / alpha), dim=1, keep_dim=True) + 1e-6)
    x_unit = x / l2_norm
    return norm_length * x_unit


def pgd_loss(ernie, labels, loss, task_fc_fn, epsilon=0.25):
    """ refer code from
    https://github.com/tensorflow/models/blob/master/research/adversarial_text/adversarial_losses.py#L145
    but we didn't use the vat loss for now
    """

    #TODO any difference with fleet_main_program or ParallelProgram or TrainProgram?
    program = fluid.default_main_program()

    param_grads = fluid.backward.append_backward(loss, parameter_list=[ernie._word_emb_name])

    # in the VAT paper code, the d is draw from a norm distribution, what's the advantage? why not use the
    # gradient of the emb directly?
    # d = fluid.layers.random_normal(shape=emb.shape)
    d = filter(lambda p: p[0].name == ernie._word_emb_name, param_grads)[0][1]
    emb = program.block(0).var(ernie._word_emb_name)

    #for _ in range(args.K_iteration):
    K_iteration = 8
    small_constant_for_finite_diff = 1e-5
    emb_hat = emb

    d = fluid.layers.gaussian_random(emb.shape)

    # it seems it can be implemented by the while loop
    for _ in range(K_iteration):
        #d = xi * utils_tf.l2_batch_normalize(d)
        d = scale_l2(d, small_constant_for_finite_diff)
        #logits_d = model.get_logits(x + d)
        #kl = utils_tf.kl_with_logits(logits, logits_d)

        emb_hat = emb_hat + d
        ernie._build_model(emb=emb_hat)
        graph_vars = task_fc_fn(ernie, labels)

        gradient = filter(lambda p: p[0].name == ernie._word_emb_name, param_grads)[0][1]
        gradient.stop_gradient = True
        d = gradient
        #Hd = tf.gradients(kl, d)[0]
        #d = tf.stop_gradient(Hd)

    d = scale_l2(d, small_constant_for_finite_diff)
    emb_hat = emb_hat + d
    ernie._build_model(emb=emb_hat)
    graph_vars = task_fc_fn(ernie, labels)

    return graph_vars['loss']

def matthews_corrcoef(preds, labels):
    """matthews_corrcoef"""
    preds = np.array(preds)
    labels = np.array(labels)
    tp = np.sum((labels == 1) & (preds == 1))
    tn = np.sum((labels == 0) & (preds == 0))
    fp = np.sum((labels == 0) & (preds == 1))
    fn = np.sum((labels == 1) & (preds == 0))

    mcc = ( (tp * tn) - (fp * fn)) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) )
    return mcc


def f1_score(preds, labels):
    """f1_score"""
    preds = np.array(preds)
    labels = np.array(labels)

    tp = np.sum((labels == 1) & (preds == 1))
    tn = np.sum((labels == 0) & (preds == 0))
    fp = np.sum((labels == 0) & (preds == 1))
    fn = np.sum((labels == 1) & (preds == 0))
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    f1 = (2 * p * r) / (p + r + 1e-8)
    return f1


def pearson_and_spearman(preds, labels):
    """pearson_and_spearman"""
    preds = np.array(preds)
    labels = np.array(labels)

    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def acc_and_f1(preds, labels):
    """acc_and_f1"""
    preds = np.array(preds)
    labels = np.array(labels)

    acc = simple_accuracy(preds, labels)
    f1 = f1_score(preds, labels)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def simple_accuracy(preds, labels):
    """simple_accuracy"""
    preds = np.array(preds)
    labels = np.array(labels)
    return (preds == labels).mean()

def evaluate_mrr(preds):
    """evaluate_mrr"""
    last_qid = None
    total_mrr = 0.0
    qnum = 0.0
    rank = 0.0
    correct = False
    for qid, score, label in preds:
        if qid != last_qid:
            rank = 0.0
            qnum += 1
            correct = False
            last_qid = qid

        rank += 1
        if not correct and label != 0:
            total_mrr += 1.0 / rank
            correct = True

    return total_mrr / qnum


def evaluate_map(preds):
    """evaluate_map"""
    def singe_map(st, en):
        """singe_map"""
        total_p = 0.0
        correct_num = 0.0
        for index in xrange(st, en):
            if int(preds[index][2]) != 0:
                correct_num += 1
                total_p += correct_num / (index - st + 1)
        if int(correct_num) == 0:
            return 0.0
        return total_p / correct_num

    last_qid = None
    total_map = 0.0
    qnum = 0.0
    st = 0
    for i in xrange(len(preds)):
        qid = preds[i][0]
        if qid != last_qid:
            qnum += 1
            if last_qid is not None:
                total_map += singe_map(st, i)
            st = i
            last_qid = qid

    total_map += singe_map(st, len(preds))
    return total_map / qnum

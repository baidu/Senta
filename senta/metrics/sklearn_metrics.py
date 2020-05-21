# -*- coding: utf-8 -*
"""import"""
import numpy as np
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class SKLearnClassify(object):
    """SKLearnClassify
    """
    @staticmethod
    def evaluate(output, evaluate_types, average="macro"):
        """
        :param output:
        :param evaluate_types:
        :param average:
        :return:
        """
        if not evaluate_types:
            return
        evas = str.split(evaluate_types, ",")
        if len(evas) == 0:
            return
        ret = {}
        for eva in evas:
            if eva == "acc":
                ret[eva] = SKLearnClassify.sk_learn_acc(output)
            elif eva == "auc":
                ret[eva] = SKLearnClassify.sk_learn_auc(output)
            elif eva == "f1":
                ret[eva] = SKLearnClassify.sk_learn_f1(output, average=average)
            elif eva == "precision":
                ret[eva] = SKLearnClassify.sk_learn_precision_score(output, average=average)
            elif eva == "recall":
                ret[eva] = SKLearnClassify.sk_learn_recall_score(output, average=average)

        return ret

    @staticmethod
    def sk_learn_acc(output):
        """
        :param output:
        :return:
        """
        predict = output["classify_infer"]
        label = output["label"]
        predict_arr = None
        label_arr = None

        if isinstance(predict, list):
            predict_arr = np.array(predict).astype('int64')
        else:
            tmp_arr = []
            for pre in predict:
                tmp_arr.append(np.argmax(pre))
            predict_arr = np.array(tmp_arr)

        if isinstance(label, list):
            label_arr = np.array(label).astype('int64')
        else:
            label_arr = np.array(label.flatten())

        score = accuracy_score(label_arr, predict_arr)
        # logging.info("sklearn acc score = ", score)
        return score

    @staticmethod
    def sk_learn_auc(output):
        """
        :param output:
        :return:
        """
        predict = output["classify_infer"]
        label = output["label"]
        assert len(predict[0]) == 2, "auc metrics only support binary classification, \
                                      and the positive label must be 1, negtive label must be 0"
        predict_arr = []
        for pre in predict:
            pos_prob = pre[1]
            predict_arr.append(pos_prob)

        # y = np.array([1, 1, 2, 2])
        # pred = np.array([0.1, 0.4, 0.35, 0.8])

        fpr, tpr, thresholds = metrics.roc_curve(np.array(label.flatten()), np.array(predict_arr))
        score = metrics.auc(fpr, tpr)
        # logging.info("sklearn auc score = ", score)
        return score

    @staticmethod
    def sk_learn_f1(output, average="macro"):
        """
        :param output:
        :param average:
        :return:
        """
        predict = output["classify_infer"]
        label = output["label"]

        predict_arr = None
        label_arr = None

        if isinstance(predict, list):
            predict_arr = np.array(predict).astype('int64')
        else:
            tmp_arr = []
            for pre in predict:
                tmp_arr.append(np.argmax(pre))
            predict_arr = np.array(tmp_arr)

        if isinstance(label, list):
            label_arr = np.array(label).astype('int64')
        else:
            label_arr = np.array(label.flatten())

        score = f1_score(label_arr, predict_arr, average=average)
        # logging.info("sklearn f1 macro score = ", score)
        return score

    @staticmethod
    def sk_learn_precision_score(output, average="macro"):
        """
        :param output:
        :param average:
        :return:
        """
        predict = output["classify_infer"]
        label = output["label"]
        predict_arr = None
        label_arr = None

        if isinstance(predict, list):
            predict_arr = np.array(predict).astype('int64')
        else:
            tmp_arr = []
            for pre in predict:
                tmp_arr.append(np.argmax(pre))
            predict_arr = np.array(tmp_arr)

        if isinstance(label, list):
            label_arr = np.array(label).astype('int64')
        else:
            label_arr = np.array(label.flatten())

        score = precision_score(label_arr, predict_arr, average=average)
        # logging.info("sklearn precision macro score = ", score)
        return score

    @staticmethod
    def sk_learn_recall_score(output, average="macro"):
        """
        :param output:
        :param average:
        :return:
        """
        predict = output["classify_infer"]
        label = output["label"]
        predict_arr = None
        label_arr = None

        if isinstance(predict, list):
            predict_arr = np.array(predict).astype('int64')
        else:
            tmp_arr = []
            for pre in predict:
                tmp_arr.append(np.argmax(pre))
            predict_arr = np.array(tmp_arr)

        if isinstance(label, list):
            label_arr = np.array(label).astype('int64')
        else:
            label_arr = np.array(label.flatten())

        score = recall_score(label_arr, predict_arr, average=average)
        # logging.info("sklearn recall macro score = ", score)
        return score


class SimNetEvaluate(object):
    """SimNetEvaluate
    """
    @staticmethod
    def evaluate(output, evaluate_types):
        """
        :param output:
        :param evaluate_types:
        :return:
        """
        if not evaluate_types:
            return
        evas = str.split(evaluate_types, ",")
        if len(evas) == 0:
            return
        ret = {}
        for eva in evas:
            if eva == "acc":
                ret[eva] = SimNetEvaluate.simnet_acc(output)
            elif eva == "auc":
                continue
            elif eva == "f1":
                continue
            elif eva == "precision":
                continue
            elif eva == "recall":
                continue

        return ret

    @staticmethod
    def simnet_acc(output):
        """
        :param output:
        :return:
        """
        pos_score = output["match_pos_score"]
        neg_score = output["match_neg_score"]
        sub = pos_score - neg_score
        right_count = 0
        sum_count = 0
        for score in sub:
            sum_count += 1
            if score > 0:
                right_count += 1
        right_count = right_count * 1.0
        acc = right_count / sum_count

        return float(acc)


class SequenceLabelEvaluate(object):
    """SequenceLabelEvaluate
    """
    @staticmethod
    def evaluate(output, evaluate_types):
        """
        :param output:
        :param evaluate_types:
        :return:
        """
        if not evaluate_types:
            return
        evas = str.split(evaluate_types, ",")
        if len(evas) == 0:
            return
        ret = {}
        for eva in evas:
            if eva == "acc":
                continue
            elif eva == "auc":
                continue
            elif eva == "f1":
                ret[eva] = np.mean(output["f1_score"])
            elif eva == "precision":
                ret[eva] = np.mean(output["precision"])
            elif eva == "recall":
                ret[eva] = np.mean(output["recall"])

        return ret

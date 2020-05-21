# -*- coding: utf-8 -*

""" pretraining reader for chinese"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import six
import logging
import gzip
import copy
import paddle.fluid as fluid
import numpy as np
from six.moves import xrange



def multi_sent_sorted(token_ids, sent_ids, pos_ids, label, seg_labels, cls_id, sep_id):
    """multi sent sorted"""
    start = 0
    token_ids_list = []
    seg_labels_list = []
    while True:
        try:
            sep_index = token_ids[start + 1:].index(sep_id)
            end = start + 1 + sep_index
            token_ids_list.append(token_ids[start + 1: end])
            seg_labels_list.append(seg_labels[start + 1: end])
            start = end
        except Exception as e:
            break
    premutation_1_sent = [[0]]
    premutation_2_sent = [[0, 1], [1, 0]]
    premutation_3_sent = [[0, 1, 2], [0, 2, 1],
                          [1, 0, 2], [1, 2, 0],
                          [2, 0, 1], [2, 1, 0]]
    premutation_4_sent = [[0, 1, 2, 3], [0, 1, 3, 2],
                          [0, 2, 1, 3], [0, 2, 3, 1],
                          [0, 3, 1, 2], [0, 3, 2, 1],
                          [1, 0, 2, 3], [1, 0, 3, 2],
                          [1, 2, 0, 3], [1, 2, 3, 0],
                          [1, 3, 0, 2], [1, 3, 2, 0],
                          [2, 0, 1, 3], [2, 0, 3, 1],
                          [2, 1, 0, 3], [2, 1, 3, 0],
                          [2, 3, 0, 1], [2, 3, 1, 0],
                          [3, 0, 1, 2], [3, 0, 2, 1],
                          [3, 1, 0, 2], [3, 1, 2, 0],
                          [3, 2, 0, 1], [3, 2, 1, 0]]

    shuffle_token_ids = [cls_id]
    shuffle_sent_ids = [0]
    shuffle_seg_labels = [-1]
    shuffle_label = 0
    if label == 0 and len(token_ids_list) == 2:
        choice_index = np.random.choice(2)
        for index, order in enumerate(premutation_2_sent[choice_index]):
            shuffle_token_ids += token_ids_list[order] + [sep_id]
            shuffle_sent_ids += [index] * len(token_ids_list[order]) + [index]
            shuffle_seg_labels += seg_labels_list[order] + [-1]
        shuffle_label = label + choice_index
    elif label == 2 and len(token_ids_list) == 3:
        choice_index = np.random.choice(6)
        for index, order in enumerate(premutation_3_sent[choice_index]):
            shuffle_token_ids += token_ids_list[order] + [sep_id]
            shuffle_sent_ids += [index] * len(token_ids_list[order]) + [index]
            shuffle_seg_labels += seg_labels_list[order] + [-1]
        shuffle_label = label + choice_index
    elif label == 8 and len(token_ids_list) == 4:
        choice_index = np.random.choice(24)
        for index, order in enumerate(premutation_4_sent[choice_index]):
            shuffle_token_ids += token_ids_list[order] + [sep_id]
            shuffle_sent_ids += [index] * len(token_ids_list[order]) + [index]
            shuffle_seg_labels += seg_labels_list[order] + [-1]
        shuffle_label = label + choice_index
    else:
        logging.error("format error")

    """
    if label == 0 and len(token_ids_list) == 1:
        choice_index = rng.choice(1)
        for index, order in enumerate(premutation_1_sent[choice_index]):
            shuffle_token_ids += token_ids_list[order] + [sep_id]
            shuffle_sent_ids += [index] * len(token_ids_list[order]) + [index]
            shuffle_seg_labels += seg_labels_list[order] + [-1]
            shuffle_label = label + choice_index
    elif label == 1 and len(token_ids_list) == 2:
        choice_index = rng.choice(2)
        for index, order in enumerate(premutation_2_sent[choice_index]):
            shuffle_token_ids += token_ids_list[order] + [sep_id]
            shuffle_sent_ids += [index] * len(token_ids_list[order]) + [index]
            shuffle_seg_labels += seg_labels_list[order] + [-1]
            shuffle_label = label + choice_index
    elif label == 3 and len(token_ids_list) == 3:
        choice_index = rng.choice(6)
        for index, order in enumerate(premutation_3_sent[choice_index]):
            shuffle_token_ids += token_ids_list[order] + [sep_id]
            shuffle_sent_ids += [index] * len(token_ids_list[order]) + [index]
            shuffle_seg_labels += seg_labels_list[order] + [-1]
            shuffle_label = label + choice_index
    elif label == 9 and len(token_ids_list) == 4:
        choice_index = rng.choice(24)
        for index, order in enumerate(premutation_4_sent[choice_index]):
            shuffle_token_ids += token_ids_list[order] + [sep_id]
            shuffle_sent_ids += [index] * len(token_ids_list[order]) + [index]
            shuffle_seg_labels += seg_labels_list[order] + [-1]
            shuffle_label = label + choice_index
    else:
        logging.error('row_token %s' % token_ids)
        logging.error('res %s %s %s' % (label, len(token_ids_list), token_ids_list))
        logging.error("format error")
    """


    return (shuffle_token_ids, shuffle_sent_ids, pos_ids, shuffle_label, shuffle_seg_labels)


def shuffle_entity(batch_tokens, seg_labels, total_token_num):
    """shuffle entity"""
    max_len = max([len(sent) for sent in batch_tokens])
    prob_mask = np.random.rand(total_token_num)
    # Note: the first token is [CLS], so [low=1]
    pre_sent_len = 0
    prob_index = 0
    for sent_index, sent in enumerate(batch_tokens):
        prob_index += pre_sent_len
        beg = 0
        for token_index, token in enumerate(sent):
            seg_label = seg_labels[sent_index][token_index]
            if seg_label == 1:
                continue
            if beg == 0:
                if seg_label != -1:
                    beg = token_index
                continue

            prob = prob_mask[prob_index + beg]
            if prob > 0.10:
                pass
            else:
                tmp = sent[beg: token_index]
                np.random.shuffle(tmp)
                sent[beg: token_index] = tmp

            if seg_label == -1:
                beg = 0
            else:
                beg = token_index
        pre_sent_len = len(sent)

    return batch_tokens


def mask(batch_tokens,
         seg_labels,
         mask_word_tags,
         total_token_num,
         vocab_size,
         CLS=1,
         SEP=2,
         MASK=3):
    """
    Add mask for batch_tokens, return out, mask_label, mask_pos;
    Note: mask_pos responding the batch_tokens after padded;
    """
    max_len = max([len(sent) for sent in batch_tokens])
    mask_label = []
    mask_pos = []
    mask_pos_ids = []
    sbo_pos_left = []
    sbo_pos_right = []
    prob_mask = np.random.rand(total_token_num)
    # Note: the first token is [CLS], so [low=1]
    replace_ids = np.random.randint(1, high=vocab_size, size=total_token_num)
    pre_sent_len = 0
    prob_index = 0
    for sent_index, sent in enumerate(batch_tokens):
        mask_flag = False
        mask_word = mask_word_tags[sent_index]
        prob_index += pre_sent_len
        if mask_word:
            beg = 0
            for token_index, token in enumerate(sent):
                seg_label = seg_labels[sent_index][token_index]
                if seg_label == 1:
                    continue
                if beg == 0:
                    if seg_label != -1:
                        beg = token_index
                    continue

                prob = prob_mask[prob_index + beg]
                if prob > 0.15:
                    pass
                else:
                    for index in xrange(beg, token_index):
                        prob = prob_mask[prob_index + index]
                        base_prob = 1.0
                        if index == beg:
                            base_prob = 0.15
                        if base_prob * 0.2 < prob <= base_prob:
                            mask_label.append(sent[index])
                            sent[index] = MASK
                            mask_flag = True
                            mask_pos.append(sent_index * max_len + index)
                            mask_pos_ids.append(index)
                            sbo_pos_left.append(sent_index * max_len + beg)
                            sbo_pos_right.append(sent_index * max_len + token_index - 1)
                        elif base_prob * 0.1 < prob <= base_prob * 0.2:
                            mask_label.append(sent[index])
                            sent[index] = replace_ids[prob_index + index]
                            mask_flag = True
                            mask_pos.append(sent_index * max_len + index)
                            mask_pos_ids.append(index)
                            sbo_pos_left.append(sent_index * max_len + beg)
                            sbo_pos_right.append(sent_index * max_len + token_index - 1)
                        else:
                            mask_label.append(sent[index])
                            mask_pos.append(sent_index * max_len + index)
                            mask_pos_ids.append(index)
                            sbo_pos_left.append(sent_index * max_len + beg)
                            sbo_pos_right.append(sent_index * max_len + token_index - 1)

                if seg_label == -1:
                    beg = 0
                else:
                    beg = token_index
        else:
            for token_index, token in enumerate(sent):
                prob = prob_mask[prob_index + token_index]
                if prob > 0.15:
                    continue
                elif 0.03 < prob <= 0.15:
                    # mask
                    if token != SEP and token != CLS:
                        mask_label.append(sent[token_index])
                        sent[token_index] = MASK
                        mask_flag = True
                        mask_pos.append(sent_index * max_len + token_index)
                        mask_pos_ids.append(token_index)
                        sbo_pos_left.append(sent_index * max_len + token_index - 1)
                        sbo_pos_right.append(sent_index * max_len + token_index + 1)

                elif 0.015 < prob <= 0.03:
                    # random replace
                    if token != SEP and token != CLS:
                        mask_label.append(sent[token_index])
                        sent[token_index] = replace_ids[prob_index +
                                                        token_index]
                        mask_flag = True
                        mask_pos.append(sent_index * max_len + token_index)
                        mask_pos_ids.append(token_index)
                        sbo_pos_left.append(sent_index * max_len + token_index - 1)
                        sbo_pos_right.append(sent_index * max_len + token_index + 1)

                else:
                    # keep the original token
                    if token != SEP and token != CLS:
                        mask_label.append(sent[token_index])
                        mask_pos.append(sent_index * max_len + token_index)
                        mask_pos_ids.append(token_index)
                        sbo_pos_left.append(sent_index * max_len + token_index - 1)
                        sbo_pos_right.append(sent_index * max_len + token_index + 1)

        pre_sent_len = len(sent)

    mask_label = np.array(mask_label).astype("int64").reshape([-1, 1])
    mask_pos = np.array(mask_pos).astype("int64").reshape([-1, 1])
    mask_pos_ids = np.array(mask_pos_ids).astype("int64").reshape([-1, 1])
    sbo_pos_left = np.array(sbo_pos_left).astype("int64").reshape([-1, 1])
    sbo_pos_right = np.array(sbo_pos_right).astype("int64").reshape([-1, 1])

    mask_out = (batch_tokens, mask_label, mask_pos, mask_pos_ids, sbo_pos_left, sbo_pos_right)
    return mask_out


def prepare_batch_data(insts,
                       total_token_num,
                       task_index,
                       lm_weight,
                       task_num,
                       voc_size=0,
                       pad_id=None,
                       cls_id=None,
                       sep_id=None,
                       mask_id=None,
                       return_input_mask=True,
                       return_max_len=True,
                       return_num_token=False):
    """get prepare batch data"""
    batch_src_ids = [inst[0] for inst in insts]
    batch_sent_ids = [inst[1] for inst in insts]
    batch_pos_ids = [inst[2] for inst in insts]
    batch_task_ids = [inst[3] for inst in insts]
    labels = [inst[4] for inst in insts]
    labels = np.array(labels).astype("int64").reshape([-1, 1])
    seg_labels = [inst[5] for inst in insts]
    mask_word_tags = [inst[6] for inst in insts]

    # First step: do mask without padding
    assert mask_id >= 0, "[FATAL] mask_id must >= 0"

    not_mask = False
    if lm_weight < 0.01:
        not_mask = True

    if not_mask:
        out = copy.deepcopy(batch_src_ids)
        out = shuffle_entity(batch_src_ids, seg_labels, total_token_num)
        mask_out = mask(
            batch_src_ids,
            seg_labels,
            mask_word_tags,
            total_token_num,
            vocab_size=voc_size,
            CLS=cls_id,
            SEP=sep_id,
            MASK=mask_id)
        _, mask_label, mask_pos, mask_pos_ids, sbo_pos_left, sbo_pos_right = mask_out
    else:
        mask_out = mask(
            batch_src_ids,
            seg_labels,
            mask_word_tags,
            total_token_num,
            vocab_size=voc_size,
            CLS=cls_id,
            SEP=sep_id,
            MASK=mask_id)
        out, mask_label, mask_pos, mask_pos_ids, sbo_pos_left, sbo_pos_right = mask_out

    # Second step: padding
    src_id, self_input_mask = pad_batch_data(
        out, pad_idx=pad_id, return_input_mask=True)
    pos_id = pad_batch_data(batch_pos_ids, pad_idx=pad_id)
    sent_id = pad_batch_data(batch_sent_ids, pad_idx=pad_id)
    task_id = pad_batch_data(batch_task_ids, pad_idx=pad_id)
    lm_w = np.array([lm_weight]).astype("float32")

    return_list = [
        src_id, pos_id, sent_id, task_id, self_input_mask, mask_label, mask_pos, lm_w, sbo_pos_left, sbo_pos_right,
        mask_pos_ids
    ]

    for i in xrange(task_num):
        if i == task_index:
            return_list.append(labels)
            return_list.append(np.array([1.0]).astype("float32"))
        else:
            return_list.append(np.zeros_like(labels))
            return_list.append(np.array([0.0]).astype("float32"))

    return return_list


def pad_batch_data(insts,
                   pad_idx=0,
                   return_pos=False,
                   return_input_mask=False,
                   return_max_len=False,
                   return_num_token=False,
                   return_seq_lens=False):
    """
    Pad the instances to the max sequence length in batch, and generate the
    corresponding position data and attention bias.
    """
    return_list = []
    max_len = max(len(inst) for inst in insts)

    inst_data = np.array(
        [inst + list([pad_idx] * (max_len - len(inst))) for inst in insts])
    return_list += [inst_data.astype("int64").reshape([-1, max_len, 1])]

    # position data
    if return_pos:
        inst_pos = np.array([
            list(range(0, len(inst))) + [pad_idx] * (max_len - len(inst))
            for inst in insts
        ])

        return_list += [inst_pos.astype("int64").reshape([-1, max_len, 1])]

    if return_input_mask:
        # This is used to avoid attention on paddings.
        input_mask_data = np.array([[1] * len(inst) + [0] *
                                    (max_len - len(inst)) for inst in insts])
        input_mask_data = np.expand_dims(input_mask_data, axis=-1)
        return_list += [input_mask_data.astype("float32")]

    if return_max_len:
        return_list += [max_len]

    if return_num_token:
        num_token = 0
        for inst in insts:
            num_token += len(inst)
        return_list += [num_token]

    if return_seq_lens:
        seq_lens = np.array([len(inst) for inst in insts])
        return_list += [seq_lens.astype("int64").reshape([-1, 1])]

    return return_list if len(return_list) > 1 else return_list[0]


class FluidDataType(object):                                                                                                                                 
    """ FluidDataType data struct wrapper """                                                                                                                
    def __init__(self, shape, dtype, lod_level):                                                                                                             
        self.shape = shape                                                                                                                                   
        self.dtype = dtype                                                                                                                                   
        self.lod_level = lod_level


class ErniePretrainDataReader(object):
    """ErniePretrainDataReader"""
    def __init__(self, args, pyreader_name, tokenizer, task_group, evaluate=False):

        self.args = args
        self.vocab = self.load_vocab(args.vocab_path)
        self.task_group = task_group
        self.batch_size = args.train_batch_size
        self.shuffle_files = args.shuffle_files
        self.epoch = args.epoch
        self.current_epoch = 0
        self.current_file_index = 0
        self.total_file = 0
        self.current_file = None
        self.voc_size = len(self.vocab)
        self.max_seq_len = args.max_seq_len
        self.pad_id = self.vocab["[PAD]"]
        self.cls_id = self.vocab["[CLS]"]
        self.sep_id = self.vocab["[SEP]"]
        self.mask_id = self.vocab["[MASK]"]
        if self.args.model_type == "ernie_2.0_en":
            self.input_slots = 7
        else:
            self.input_slots = 5
        self.generate_neg_sample = args.generate_neg_sample

        assert self.batch_size > 100, "Current batch size means total token's number, \
                                       it should not be set to too small number."

        if args.hack_old_data:
            self.input_slots = 4

        if 'test_reader' == pyreader_name:
            self.epoch = 1
            self.shuffle_files = False

        self.pyreader_name = pyreader_name
        L = self.args.max_seq_len

        self.data_types = collections.OrderedDict()
        self.data_types["src_ids"] = FluidDataType([-1, L, 1], 'int64', 0)
        self.data_types["pos_ids"] = FluidDataType([-1, L, 1], 'int64', 0)
        self.data_types["sent_ids"] = FluidDataType([-1, L, 1], 'int64', 0)
        self.data_types["task_ids"] = FluidDataType([-1, L, 1], 'int64', 0)
        self.data_types["input_mask"] = FluidDataType([-1, L, L], 'float', 0)
        self.data_types["mask_label"] = FluidDataType([-1, 1], 'int64', 0)
        self.data_types["mask_pos"] = FluidDataType([-1, 1], 'int64', 0)
        self.data_types["lm_weight"] = FluidDataType([], 'float', 0)
        self.data_types["sbo_pos_left"] = FluidDataType([-1, 1], 'int64', 0)
        self.data_types["sbo_pos_right"] = FluidDataType([-1, 1], 'int64', 0)
        self.data_types["mask_pos_ids"] = FluidDataType([-1, 1], 'int64', 0)

        for task in self.task_group:
            self.data_types[task['task_name']] = FluidDataType([-1, 1], 'int64', 0)
            self.data_types[task['task_name'] + '_weight'] = FluidDataType([], 'float', 0)

    def get_progress(self):
        """return current progress of traning data
        """
        progress_out = (self.current_epoch, self.current_file_index, \
                self.total_file, self.current_file, self.mask_type)
        return progress_out

    def parse_line(self, line, max_seq_len=512, task_index=None):
        """ parse one line to token_ids, sentence_ids, pos_ids, label
        """
        line = line.strip().split(";")
        assert len(line) == self.input_slots, \
            "One sample must have %d fields!" % self.input_slots

        if self.input_slots == 4:
            (token_ids, sent_ids, pos_ids, label) = line
            token_ids = [int(token) for token in token_ids.split(" ")]
            sent_ids = [int(token) for token in sent_ids.split(" ")]
            pos_ids = [int(token) for token in pos_ids.split(" ")]
            # fake seg_labels
            seg_labels = [0, ] * len(token_ids)
            seg_labels[0] = -1
            start = 0
            while True:
                try:
                    sep_index = token_ids[start + 1:].index(self.sep_id)
                    end = start + 1 + sep_index
                    seg_labels[end] = -1
                    start = end
                except Exception as e:
                    break

        if self.input_slots == 5:
            (token_ids, sent_ids, pos_ids, seg_labels, label) = line
            token_ids = [int(token) for token in token_ids.split(" ")]
            sent_ids = [int(token) for token in sent_ids.split(" ")]
            pos_ids = [int(token) for token in pos_ids.split(" ")]
            seg_labels = [int(seg_label) for seg_label in seg_labels.split(" ")]

        if self.input_slots == 7:
            (token_ids, sent_ids, pos_ids, seg_labels, _, _, label) = line
            token_ids = [int(token) for token in token_ids.split(" ")]
            sent_ids = [int(token) for token in sent_ids.split(" ")]
            pos_ids = [int(token) for token in pos_ids.split(" ")]
            seg_labels = [int(seg_label) for seg_label in seg_labels.split(" ")]

        label = int(label)

        task = self.task_group[task_index]
        data_func = task.get("data_func", None)
        if data_func and data_func != "":
            token_ids, sent_ids, pos_ids, label, seg_labels = \
                eval(data_func)(token_ids, sent_ids,
                                                pos_ids, label, seg_labels, self.cls_id, self.sep_id)

        if isinstance(task_index, int):
            task_ids = [task_index] * len(token_ids)
        else:
            task_ids = [0] * len(token_ids)

        #logging.info("token:%d\tsent:%d\tpos:%d\tseg:%d\ttask:%d" % (len(token_ids), len(sent_ids), len(pos_ids), len(seg_labels), len(task_ids)))
        assert len(token_ids) == len(sent_ids) == len(pos_ids) == len(
            seg_labels) == len(task_ids
        ), "[Must be true]len(token_ids) == len(sent_ids) == len(pos_ids) == len(seg_labels) == len(task_ids)"

        if len(token_ids) > max_seq_len:
            return None
        return [token_ids, sent_ids, pos_ids, task_ids, label, seg_labels]

    def read_file(self, file, task_index):
        """ read file"""
        # assert file.endswith('.gz'), "[ERROR] %s is not a gzip file" % file
        with gzip.open(file, "rt") as f:
            lines = f.readlines()
            np.random.shuffle(lines)
            for line in lines:
                parsed_line = self.parse_line(
                    line, max_seq_len=self.max_seq_len, task_index=task_index)
                if parsed_line is None:
                    continue
                yield parsed_line

    def convert_to_unicode(self, text):
        """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
        if six.PY3:
            if isinstance(text, str):
                return text
            elif isinstance(text, bytes):
                return text.decode("utf-8", "ignore")
            else:
                raise ValueError("Unsupported string type: %s" % (type(text)))
        elif six.PY2:
            if isinstance(text, str):
                return text.decode("utf-8", "ignore")
            elif isinstance(text, unicode):
                return text
            else:
                raise ValueError("Unsupported string type: %s" % (type(text)))
        else:
            raise ValueError("Not running on Python2 or Python 3?")

    def load_vocab(self, vocab_file):
        """Loads a vocabulary file into a dictionary."""
        vocab = collections.OrderedDict()
        fin = open(vocab_file)
        for num, line in enumerate(fin):
            items = self.convert_to_unicode(line.strip()).split("\t")
            if len(items) > 2:
                break
            token = items[0]
            index = items[1] if len(items) == 2 else num
            token = token.strip()
            vocab[token] = int(index)
        return vocab

    def random_pair_neg_samples(self, pos_samples):
        """ randomly generate negtive samples using pos_samples

            Args:
                pos_samples: list of positive samples

            Returns:
                neg_samples: list of negtive samples
        """
        np.random.shuffle(pos_samples)
        num_sample = len(pos_samples)
        neg_samples = []
        miss_num = 0

        def split_sent(sample, max_len, sep_id):
            """split sent"""
            token_seq, type_seq, pos_seq, task_seq, label, seg_labels = sample
            sep_index = token_seq.index(sep_id)
            left_len = sep_index - 1
            if left_len <= max_len:
                return (token_seq[1:sep_index], seg_labels[1:sep_index])
            else:
                return [
                    token_seq[sep_index + 1:-1], seg_labels[sep_index + 1:-1]
                ]

        for i in range(num_sample):
            pair_index = (i + 1) % num_sample
            left_tokens, left_seg_labels = split_sent(
                pos_samples[i], (self.max_seq_len - 3) // 2, self.sep_id)
            right_tokens, right_seg_labels = split_sent(
                pos_samples[pair_index],
                self.max_seq_len - 3 - len(left_tokens), self.sep_id)

            token_seq = [self.cls_id] + left_tokens + [self.sep_id] + \
                        right_tokens + [self.sep_id]
            if len(token_seq) > self.max_seq_len:
                miss_num += 1
                continue
            type_seq = [0] * (len(left_tokens) + 2) + [1] * (len(right_tokens) + 1)
            pos_seq = range(len(token_seq))
            task_seq = [task_seq[0]] * len(token_seq)
            seg_label_seq = [-1] + left_seg_labels + [-1] + right_seg_labels + [-1]

            assert len(token_seq) == len(type_seq) == len(pos_seq) == len(seg_label_seq) == len(task_seq), \
                "[ERROR]len(src_id) == lne(sent_id) == len(pos_id) must be True"
            neg_samples.append([token_seq, type_seq, pos_seq, task_seq, 0, seg_label_seq])

        return neg_samples, miss_num

    def mixin_negtive_samples(self, pos_sample_generator, buffer=1000):
        """ 1. generate negtive samples by randomly group sentence_1 and sentence_2 of positive samples
            2. combine negtive samples and positive samples

            Args:
                pos_sample_generator: a generator producing a parsed positive sample, which is a list: [token_ids, sent_ids, pos_ids, 1]

            Returns:
                sample: one sample from shuffled positive samples and negtive samples
        """
        pos_samples = []
        num_total_miss = 0
        pos_sample_num = 0
        try:
            while True:
                while len(pos_samples) < buffer:
                    pos_sample = next(pos_sample_generator)
                    label = pos_sample[3]
                    assert label == 1, "positive sample's label must be 1"
                    pos_samples.append(pos_sample)
                    pos_sample_num += 1

                neg_samples, miss_num = self.random_pair_neg_samples(
                    pos_samples)
                num_total_miss += miss_num
                samples = pos_samples + neg_samples
                pos_samples = []
                np.random.shuffle(samples)
                for sample in samples:
                    yield sample
        except StopIteration:
            logging.error("stopiteration: reach end of file")
            if len(pos_samples) == 1:
                yield pos_samples[0]
            elif len(pos_samples) == 0:
                yield None
            else:
                neg_samples, miss_num = self.random_pair_neg_samples(
                    pos_samples)
                num_total_miss += miss_num
                samples = pos_samples + neg_samples
                pos_samples = []
                np.random.shuffle(samples)
                for sample in samples:
                    yield sample
            logging.info("miss_num:%d\tideal_total_sample_num:%d\tmiss_rate:%f" %
                  (num_total_miss, pos_sample_num * 2,
                   num_total_miss / (pos_sample_num * 2)))

    def create_reader(self):
        """ 初始化paddle_py_reader，必须要初始化，否则会抛出异常
            create reader 定义的这些 dtypes 要和 prepare_batch_data 一一对应
            :return:
        """

        py_reader = fluid.layers.py_reader(
            capacity=70,
            shapes=[f.shape for f in self.data_types.values()],
            dtypes=[f.dtype for f in self.data_types.values()],
            lod_levels=[f.lod_level for f in self.data_types.values()],
            name=self.pyreader_name,
            use_double_buffer=True)
        self.paddle_py_reader = py_reader

        self.fields_dict_ = collections.OrderedDict()
        input_placeholders = fluid.layers.read_file(py_reader)

        assert len(input_placeholders) == len(self.data_types)
        for op, (name, _) in zip(input_placeholders, self.data_types.items()):
            self.fields_dict_[name] = op

        return py_reader

    def data_generator(self):
        """
        data_generator
        """
        filelist_key = "train_filelist"
        if 'dev_reader' == self.pyreader_name:
            filelist_key = "valid_filelist"

        all_files = []
        task_probs = []
        sum = 0.0
        for task in self.task_group:
            all_files.append(open(task[filelist_key]).readlines())
            task_probs.append(task["prob"])
            sum += task["prob"]
        for i in xrange(len(task_probs)):
            task_probs[i] = task_probs[i] / sum
        task_probs = np.array(task_probs).ravel()

        def wrapper():
            """wrapper"""
            def reader(task_index):
                """reader"""
                files = all_files[task_index]
                for epoch in range(self.epoch):
                    if self.shuffle_files:
                        np.random.shuffle(files)
                    for index, file in enumerate(files):
                        file, mask_word_prob = file.strip().split("\t")
                        mask_word = (np.random.random() < float(mask_word_prob))

                        if mask_word:
                            self.mask_type = "mask_word"
                        else:
                            self.mask_type = "mask_char"

                        sample_generator = self.read_file(file, task_index)
                        if not 'test_reader' == self.pyreader_name and self.generate_neg_sample:
                            sample_generator = self.mixin_negtive_samples(
                                sample_generator)

                        for sample in sample_generator:
                            self.current_epoch = epoch + 1
                            self.current_file_index = index + 1
                            self.current_file = file
                            self.total_file = len(files)

                            if sample is None:
                                continue
                            sample.append(mask_word)
                            yield sample

            def batch_reader(reader, batch_size):
                """batch reader"""
                batch, total_token_num, max_len = [], 0, 0
                dev_count = 1
                buff = []
                readers = []
                for i in xrange(len(task_probs)):
                    buff.append(None)
                    readers.append(reader(i))
                task_indices = range(len(task_probs))

                end_times = 0
                while end_times < 50:
                    task_index = np.random.choice(task_indices, p=task_probs)
                    dev_num = 0
                    cur_reader = readers[task_index]

                    while dev_num < dev_count:
                        if buff[task_index] is not None:
                            cur_len = len(buff[task_index][0])
                            max_len = max(max_len, cur_len)
                            batch.append(buff[task_index])
                            total_token_num += cur_len
                            buff[task_index] = None

                        parsed_line = next(cur_reader, None)
                        if parsed_line is None:
                            end_times += 1
                            dev_num += 1
                            if len(batch) > 0:
                                yield batch, total_token_num, task_index, self.task_group[task_index]["lm_weight"]
                                batch, total_token_num, max_len = [], 0, 0
                            continue

                        end_times = 0
                        cur_len = len(parsed_line[0])
                        max_len = max(max_len, cur_len)
                        if (len(batch) + 1) * max_len > batch_size:
                            yield batch, total_token_num, task_index, self.task_group[task_index]["lm_weight"]
                            batch, total_token_num, max_len = [], 0, 0
                            dev_num += 1
                            buff[task_index] = parsed_line
                        else:
                            batch.append(parsed_line)
                            total_token_num += cur_len

            for batch_data, total_token_num, task_index, lm_weight in batch_reader(reader,
                                                                                   self.batch_size):
                yield prepare_batch_data(
                    batch_data,
                    total_token_num,
                    task_index,
                    lm_weight,
                    len(self.task_group),
                    voc_size=self.voc_size,
                    pad_id=self.pad_id,
                    cls_id=self.cls_id,
                    sep_id=self.sep_id,
                    mask_id=self.mask_id,
                    return_input_mask=True,
                    return_max_len=False,
                    return_num_token=False)

        return wrapper

    def instance_fields_dict(self):
        """ instance_fields_dict implement the interface """
        return self.fields_dict_

    def run(self):
        """
        配置py_reader对应的数据生成器，并启动
        :return:
        """
        if self.paddle_py_reader:
            self.paddle_py_reader.decorate_tensor_provider(self.data_generator())
            self.paddle_py_reader.start()
            logging.info("set data_generator and start.......")
        else:
            raise ValueError("paddle_py_reader is None")

    def stop(self):
        """
        py_reader 停止
        """
        if self.paddle_py_reader:
            self.paddle_py_reader.reset()
        else:
            raise ValueError("paddle_py_reader is None")

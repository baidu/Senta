import os
import numpy as np
import random

import paddle.fluid as fluid
import paddle
import io


def get_predict_label(pos_prob):
    neg_prob = 1 - pos_prob
    # threshold should be (1, 0.5)
    neu_threshold = 0.55
    if neg_prob > neu_threshold:
        class3_label = 0
    elif pos_prob > neu_threshold:
        class3_label = 2
    else:
        class3_label = 1
    if pos_prob >= neg_prob:
        class2_label = 2
    else:
        class2_label = 0
    return class3_label, class2_label


def to_lodtensor(data, place):
    """
    convert ot LODtensor
    """
    seq_lens = [len(seq) for seq in data]
    cur_len = 0
    lod = [cur_len]
    for l in seq_lens:
        cur_len += l
        lod.append(cur_len)
    flattened_data = np.concatenate(data, axis=0).astype("int64")
    flattened_data = flattened_data.reshape([len(flattened_data), 1])
    res = fluid.LoDTensor()
    res.set(flattened_data, place)
    res.set_lod([lod])
    return res


def data2tensor(data, place):
    """
    data2tensor
    """
    input_seq = to_lodtensor(list(map(lambda x: x[0], data)), place)
    return {"words": input_seq}


def data_reader(file_path, word_dict, is_shuffle=True):
    """
    Convert word sequence into slot
    """
    unk_id = len(word_dict)
    all_data = []
    with io.open(file_path, "r", encoding='utf8') as fin:
        for line in fin:
            cols = line.strip().split("\t")
            label = int(cols[0])
            wids = [word_dict[x] if x in word_dict else unk_id
                    for x in cols[1].split(" ")]
            all_data.append((wids, label))
    if is_shuffle:
        random.shuffle(all_data)

    def reader():
        for doc, label in all_data:
            yield doc, label
    return reader


def load_vocab(file_path):
    """
    load the given vocabulary
    """
    vocab = {}
    with io.open(file_path, 'r', encoding='utf8') as f:
        wid = 0
        for line in f:
            if line.strip() not in vocab:
                vocab[line.strip()] = wid
                wid += 1
    vocab["<unk>"] = len(vocab)
    return vocab


def prepare_data(data_path, word_dict_path,
                 batch_size, mode):
    """
    prepare data
    """
    assert os.path.exists(
        word_dict_path), "The given word dictionary dose not exist."
    if mode == "train":
        assert os.path.exists(
            data_path), "The given training data does not exist."
    if mode == "eval" or mode == "infer":
        assert os.path.exists(
            data_path), "The given test data does not exist."

    word_dict = load_vocab(word_dict_path)
    if mode == "train":
        train_reader = paddle.batch(data_reader(data_path, word_dict, True),
                                    batch_size)
        return word_dict, train_reader
    else:
        test_reader = paddle.batch(data_reader(data_path, word_dict, False),
                                   batch_size)
        return word_dict, test_reader

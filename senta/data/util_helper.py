# -*- coding: utf-8 -*
"""
:py:`util_helper`
"""
import paddle
import numpy as np

from senta.data.field import Field
from senta.common.rule import InstanceName, FieldLength
from senta.utils.util_helper import truncation_words

def convert_text_to_id(text, field_config):
    """将一个明文样本转换成id
    :param text: 明文文本
    :param field_config : Field类型
    :return:
    """
    if not text:
        raise ValueError("text input is None")
    if not isinstance(field_config, Field):
        raise TypeError("field_config input is must be Field class")

    if field_config.need_convert:
        tokenizer = field_config.tokenizer
        tokens = tokenizer.tokenize(text)
        ids = tokenizer.convert_tokens_to_ids(tokens)
    else:
        ids = text.split(" ")

    # 加上截断策略
    if len(ids) > field_config.max_seq_len:
        ids = truncation_words(ids, field_config.max_seq_len, field_config.truncation_type)

    return ids


def padding_batch_data(insts,
                   pad_idx=0,
                   return_seq_lens=False,
                   paddle_version_code=1.6):
    """
    :param insts:
    :param pad_idx:
    :param return_seq_lens:
    :param paddle_version_code:
    :return:
    """
    return_list = []
    max_len = max(len(inst) for inst in insts)
    # Any token included in dict can be used to pad, since the paddings' loss
    # will be masked out by weights and make no effect on parameter gradients.

    inst_data = np.array(
        [inst + list([pad_idx] * (max_len - len(inst))) for inst in insts])
    return_list += [inst_data.astype("int64").reshape([-1, max_len, 1])]

    if return_seq_lens:
        if paddle_version_code <= 1.5:
            seq_lens_type = [-1, 1]
        else:
            seq_lens_type = [-1]
        seq_lens = np.array([len(inst) for inst in insts])
        return_list += [seq_lens.astype("int64").reshape(seq_lens_type)]

    return return_list if len(return_list) > 1 else return_list[0]


def mask_batch_data(insts, return_seq_lens=False, paddle_version_code=1.6):
    """
    :param insts:
    :param return_seq_lens:
    :param paddle_version_code:
    :return:
    """
    return_list = []
    max_len = max(len(inst) for inst in insts)

    input_mask_data = np.array([[1] * len(inst) + [0] *
                                (max_len - len(inst)) for inst in insts])
    input_mask_data = np.expand_dims(input_mask_data, axis=-1)
    return_list += [input_mask_data.astype("float32")]

    if return_seq_lens:
        if paddle_version_code <= 1.5:
            seq_lens_type = [-1, 1]
        else:
            seq_lens_type = [-1]
        seq_lens = np.array([len(inst) for inst in insts])
        return_list += [seq_lens.astype("int64").reshape(seq_lens_type)]

    return return_list if len(return_list) > 1 else return_list[0]


def pad_batch_data(insts,
                   insts_data_type="int64",
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
    # Any token included in dict can be used to pad, since the paddings' loss
    # will be masked out by weights and make no effect on parameter gradients.

    # id
    inst_data = np.array(
        [inst + list([pad_idx] * (max_len - len(inst))) for inst in insts])
    return_list += [inst_data.astype(insts_data_type).reshape([-1, max_len, 1])]

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
        if paddle.__version__[:3] <= '1.5':
            seq_lens_type = [-1, 1]
        else:
            seq_lens_type = [-1]
        seq_lens = np.array([len(inst) for inst in insts])
        return_list += [seq_lens.astype("int64").reshape(seq_lens_type)]

    return return_list if len(return_list) > 1 else return_list[0]


def generate_pad_batch_data(insts,
                   insts_data_type="int64",
                   pad_idx=0,
                   return_pos=False,
                   return_input_mask=False,
                   return_max_len=False,
                   return_num_token=False,
                   return_seq_lens=False,
                   paddle_version_code=1.6):
    """
    Pad the instances to the max sequence length in batch, and generate the
    corresponding position data and attention bias.
    """
    return_list = []
    max_len = max(len(inst) for inst in insts)
    # Any token included in dict can be used to pad, since the paddings' loss
    # will be masked out by weights and make no effect on parameter gradients.

    # id
    inst_data = np.array(
        [inst + list([pad_idx] * (max_len - len(inst))) for inst in insts])
    return_list += [inst_data.astype(insts_data_type).reshape([-1, max_len])]

    # position data
    if return_pos:
        inst_pos = np.array([
            list(range(0, len(inst))) + [pad_idx] * (max_len - len(inst))
            for inst in insts
        ])

        return_list += [inst_pos.astype("int64").reshape([-1, max_len])]

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
        if paddle_version_code <= 1.5:
            seq_lens_type = [-1, 1]
        else:
            seq_lens_type = [-1]
        seq_lens = np.array([len(inst) for inst in insts])
        return_list += [seq_lens.astype("int64").reshape(seq_lens_type)]

    return return_list if len(return_list) > 1 else return_list[0]


def convert_texts_to_ids(batch_text_a, tokenizer=None, max_seq_len=512, truncation_type=0, padding_id=0):
    src_ids = []
    position_ids = []
    task_ids = []
    sentence_ids = []

    for text in batch_text_a:
        tokens_text = tokenizer.tokenize(text)
        # 加上截断策略
        if len(tokens_text) > max_seq_len - 2:
            tokens_text = truncation_words(tokens_text, max_seq_len - 2, truncation_type)
        tokens = []
        tokens.append("[CLS]")
        for token in tokens_text:
            tokens.append(token)
        tokens.append("[SEP]")
        src_id = tokenizer.convert_tokens_to_ids(tokens)

        src_ids.append(src_id)
        pos_id = list(range(len(src_id)))
        task_id = [0] * len(src_id)
        sentence_id = [0] * len(src_id)
        position_ids.append(pos_id)
        task_ids.append(task_id)
        sentence_ids.append(sentence_id)

    return_list = []
    padded_ids, input_mask, batch_seq_lens = pad_batch_data(src_ids,
                                                            pad_idx=padding_id,
                                                            return_input_mask=True,
                                                            return_seq_lens=True)
    sent_ids_batch = pad_batch_data(sentence_ids, pad_idx=padding_id)
    pos_ids_batch = pad_batch_data(position_ids, pad_idx=padding_id)
    task_ids_batch = pad_batch_data(task_ids, pad_idx=padding_id)

    return_list.append(padded_ids)  # append src_ids
    return_list.append(sent_ids_batch)  # append sent_ids
    return_list.append(pos_ids_batch)  # append pos_ids
    return_list.append(input_mask)  # append mask
    return_list.append(task_ids_batch)  # append task_ids
    return_list.append(batch_seq_lens)  # append seq_lens

    return return_list


def structure_fields_dict(fields_id, start_index, need_emb=True):
    """静态图调用的方法，生成一个dict， dict有两个key:id , emb. id对应的是pyreader读出来的各个field产出的id，emb对应的是各个
    field对应的embedding
    :param fields_id: pyreader输出的完整的id序列
    :param start_index:当前需要处理的field在field_id_list中的起始位置
    :param need_emb:是否需要embedding（预测过程中是不需要embedding的）
    :return:
    """
    record_id_dict = {}
    record_id_dict[InstanceName.SRC_IDS] = fields_id[start_index]
    record_id_dict[InstanceName.SENTENCE_IDS] = fields_id[start_index + 1]
    record_id_dict[InstanceName.POS_IDS] = fields_id[start_index + 2]
    record_id_dict[InstanceName.MASK_IDS] = fields_id[start_index + 3]
    record_id_dict[InstanceName.TASK_IDS] = fields_id[start_index + 4]
    record_id_dict[InstanceName.SEQ_LENS] = fields_id[start_index + 5]

    record_emb_dict = None

    record_dict = {}
    record_dict[InstanceName.RECORD_ID] = record_id_dict
    record_dict[InstanceName.RECORD_EMB] = record_emb_dict

    return record_dict

def get_field_length():
    """获取当前这个field在进行了序列化之后，在field_id_list中占多少长度
    :return:
    """
    return FieldLength.ERNIE_TEXT_FIELD

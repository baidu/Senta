"""task reader"""
import csv
import logging
import numpy as np
import traceback
from paddle import fluid
from collections import namedtuple

from senta.common.register import RegisterSet
from senta.common.rule import InstanceName
from senta.data.data_set_reader.basic_dataset_reader_without_fields import TaskBaseReader

from senta.data.util_helper import pad_batch_data


@RegisterSet.data_set_reader.register
class TwoSentClassifyReaderCh(TaskBaseReader):
    """classify reader"""

    def __init__(self, name, fields, config):
        TaskBaseReader.__init__(self, name, fields, config,
                                vocab_path=config.extra_params.get("vocab_path"),
                                label_map_config=config.extra_params.get("label_map_config", None),
                                max_seq_len=config.extra_params.get("max_seq_len", 512),
                                do_lower_case=config.extra_params.get("do_lower_case", False),
                                in_tokens=config.extra_params.get("in_tokens", False),
                                use_multi_gpu_test=config.extra_params.get("use_multi_gpu_test", False))

    def create_reader(self):
        """create reader"""
        shapes=[[-1, self.max_seq_len, 1], [-1, self.max_seq_len, 1], [-1, self.max_seq_len, 1],
                [-1, self.max_seq_len, 1], [-1, self.max_seq_len, 1], [-1, 1], [-1, 1]]
        dtypes=['int64', 'int64', 'int64', 'int64', 'float32', 'int64', 'int64']
        lod_levels=[0, 0, 0, 0, 0, 0, 0]
        
        self.paddle_py_reader = fluid.layers.py_reader(
        capacity=50,
        shapes=shapes,
        dtypes=dtypes,
        lod_levels=lod_levels,
        name=self.name,
        use_double_buffer=True)

        logging.debug("{0} create py_reader shape = {1}, types = {2}, \
                      level = {3}: ".format(self.name, shapes, dtypes, lod_levels))

    def convert_fields_to_dict(self, field_list, need_emb=False):
        """convert fileds to dict"""
        fields_instance = {}

        record_id_dict_text_a = {
            InstanceName.SRC_IDS: field_list[0],
            InstanceName.SENTENCE_IDS: field_list[1],
            InstanceName.POS_IDS: field_list[2],
            InstanceName.TASK_IDS: field_list[3],
            InstanceName.MASK_IDS: field_list[4]
        }
        record_dict_text_a = {
            InstanceName.RECORD_ID: record_id_dict_text_a,
            InstanceName.RECORD_EMB: None
        }
        fields_instance["text_a"] = record_dict_text_a

        record_id_dict_label = {
            InstanceName.SRC_IDS: field_list[5]
        }
        record_dict_label = {
            InstanceName.RECORD_ID: record_id_dict_label,
            InstanceName.RECORD_EMB: None
        }
        fields_instance["label"] = record_dict_label

        record_id_dict_qid = {
            InstanceName.SRC_IDS: field_list[6]
        }
        record_dict_qid = {
            InstanceName.RECORD_ID: record_id_dict_qid,
            InstanceName.RECORD_EMB: None
        }
        fields_instance["qid"] = record_dict_qid

        return fields_instance


    def read_files(self, file_path, quotechar=None):
        """Reads a tab separated value file."""
        with open(file_path, "r") as f:
            try:
                examples = []
                reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
                headers = next(reader)
                text_indices = [
                    index for index, h in enumerate(headers) if h != "label"
                ]
                Example = namedtuple('Example', headers)
                for line in reader:
                    for index, text in enumerate(line):
                        if index in text_indices:
                            line[index] = text.replace(' ', '')
                    example = Example(*line)
                    examples.append(example)
                return examples
            except Exception:
                logging.error("error in read tsv")
                logging.error("traceback.format_exc():\n%s" % traceback.format_exc())

    def serialize_batch_records(self, batch_records):
        """pad batch records"""
        batch_token_ids = [record.token_ids for record in batch_records]
        batch_text_type_ids = [record.text_type_ids for record in batch_records]
        batch_position_ids = [record.position_ids for record in batch_records]
        batch_task_ids = [record.task_ids for record in batch_records]
        if "predict" not in self.name:
            batch_labels = [record.label_id for record in batch_records]
            batch_labels = np.array(batch_labels).astype("int64").reshape([-1, 1])
        else:
            batch_labels = np.array([]).astype("int64").reshape([-1, 1])

        if batch_records[0].qid:
            batch_qids = [record.qid for record in batch_records]
            batch_qids = np.array(batch_qids).astype("int64").reshape([-1, 1])
        else:
            batch_qids = np.array([]).astype("int64").reshape([-1, 1])

        # padding
        padded_token_ids, input_mask = pad_batch_data(
            batch_token_ids, pad_idx=self.pad_id, return_input_mask=True)
        padded_text_type_ids = pad_batch_data(
            batch_text_type_ids, pad_idx=self.pad_id)
        padded_position_ids = pad_batch_data(
            batch_position_ids, pad_idx=self.pad_id)
        padded_task_ids = pad_batch_data(
            batch_task_ids, pad_idx=0)

        return_list = [
            padded_token_ids, padded_text_type_ids, padded_position_ids,
            padded_task_ids, input_mask, batch_labels, batch_qids
        ]

        return return_list

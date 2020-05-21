# -*- coding: utf-8 -*
"""
:py:class:`BasicDataSetReader`
"""
import csv
import os
import traceback
import logging
from collections import namedtuple
import numpy as np
from paddle import fluid

from senta.common.register import RegisterSet
from senta.data.data_set_reader.base_dataset_reader import BaseDataSetReader


@RegisterSet.data_set_reader.register
class BasicDataSetReader(BaseDataSetReader):
    """BasicDataSetReader:一个基础的data_set_reader，实现了文件读取，id序列化，token embedding化等基本操作
    """

    def create_reader(self):
        """ 初始化paddle_py_reader，必须要初始化，否则会抛出异常
        :return:
        """
        if not self.fields:
            raise ValueError("fields can't be None")

        shapes = []
        types = []
        levels = []
        for item in self.fields:
            if not item.field_reader:
                raise ValueError("{0}'s field_reader is None".format(item.name))
            i_shape, i_type, i_level = item.field_reader.init_reader()
            shapes.extend(i_shape)
            types.extend(i_type)
            levels.extend(i_level)

        self.paddle_py_reader = fluid.layers.py_reader(
            capacity=50,
            shapes=shapes,
            name=self.name,
            dtypes=types,
            lod_levels=levels,
            use_double_buffer=True)

        logging.debug("{0} create py_reader shape = {1} , types = {2} , level = {3} : ".format(self.name,
                                                                                               shapes, types, levels))

    def instance_fields_dict(self):
        """调用pyreader，得到fields_id, 视情况构造embedding，然后结构化成dict类型返回给组网部分
        :return: 实例化的dict，保存了各个field的id和embedding(可以没有，是情况而定), 给trainer用
        """
        fields_id = fluid.layers.read_file(self.paddle_py_reader)
        logging.info("fluid.layers.read_file......")
        fields_instance = self.convert_fields_to_dict(fields_id)
        return fields_instance

    def convert_fields_to_dict(self, field_list, need_emb=True):
        """实例化fields_dict，保存了各个field的id和embedding(可以没有，是情况而定),
        当need_emb=False的时候，可以直接给predictor调用
        :param field_list:
        :param need_emb:
        :return: dict
        """
        start_index = 0
        fields_instance = {}
        for index, filed in enumerate(self.fields):
            logging.info("index %d, name %s" % (start_index, filed.name))
            item_dict = filed.field_reader.structure_fields_dict(field_list, start_index, need_emb=need_emb)
            fields_instance[filed.name] = item_dict
            start_index += filed.field_reader.get_field_length()

        return fields_instance

    def data_generator(self):
        """
        :return:
        """
        assert os.path.isdir(self.config.data_path), "%s must be a directory that stores data files" \
                                                     % self.config.data_path
        data_files = os.listdir(self.config.data_path)

        assert len(data_files) > 0, "%s is an empty directory" % self.config.data_path

        def wrapper():
            """
            :return:
            """
            for epoch_index in range(self.config.epoch):
                self.current_example = 0
                self.current_epoch = epoch_index

                for input_file in data_files:
                    examples = self.read_files(os.path.join(self.config.data_path, input_file))
                    if self.config.shuffle:
                        np.random.shuffle(examples)

                    for batch_data in self.prepare_batch_data(examples, self.config.batch_size):
                        yield batch_data

        return wrapper

    def read_files(self, file_path, quotechar=None):
        """读取明文文件
        :param file_path
        :return: 以namedtuple数组形式输出明文样本对应的实例
        """
        with open(file_path, "r") as f:
            try:
                examples = []
                reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
                len_fields = len(self.fields)
                field_names = []
                for filed in self.fields:
                    field_names.append(filed.name)

                self.Example = namedtuple('Example', field_names)

                for line in reader:
                    if len(line) == len_fields:
                        example = self.Example(*line)
                        examples.append(example)
                    else:
                        logging.warn('fileds in file %s not match: got %d, expect %d'\
                                % (file_path, len(line), len_fields))
                return examples

            except Exception:
                logging.error("error in read tsv")
                logging.error("traceback.format_exc():\n%s" % traceback.format_exc())

    def prepare_batch_data(self, examples, batch_size):
        """将明文样本按照py_reader需要的格式序列化成一个个batch输出
        :param examples:
        :param batch_size:
        :return:
        """
        batch_records = []
        for index, example in enumerate(examples):
            self.current_example += 1
            if len(batch_records) < batch_size:
                batch_records.append(example)
            else:
                yield self.serialize_batch_records(batch_records)
                batch_records = [example]

        if batch_records:
            yield self.serialize_batch_records(batch_records)

    def serialize_batch_records(self, batch_records):
        """
        :param batch_records:
        :return:
        """
        return_list = []
        example = batch_records[0]
        for index, key in enumerate(example._fields):
            text_batch = []
            for record in batch_records:
                text_batch.append(record[index])

            id_list = self.fields[index].field_reader.convert_texts_to_ids(text_batch)
            return_list.extend(id_list)

        return return_list

    def get_train_progress(self):
        """Gets progress for training phase."""
        return self.current_example, self.current_epoch

    def get_num_examples(self):
        """get number of example"""
        data_files = os.listdir(self.config.data_path)
        assert len(data_files) > 0, "%s is an empty directory" % self.config.data_path
        sum_examples = 0
        for input_file in data_files:
            examples = self.read_files(os.path.join(self.config.data_path, input_file))
            sum_examples += len(examples)
        self.num_examples = sum_examples
        return self.num_examples


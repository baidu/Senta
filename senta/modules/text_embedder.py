"""
embedding layers
"""

import paddle.fluid as fluid
import paddle.fluid.param_attr as attr
import paddle.fluid.layers as layers
import numpy as np
import sys
import os
import logging
import time, datetime
import time


class EmbeddingLayer(object):
    """
    Embedding Layer class
    """

    def __init__(self, dict_dim, emb_dim, vec_path, dict_path, name="emb", init_w2v=False):
        """
        :param dict_dim:
        :param emb_dim:
        :param vec_path:
        :param dict_patch:
        :param init_w2v:
        """
        self.dict_dim = dict_dim
        self.emb_dim = emb_dim
        self.vec_path = vec_path
        self.dict_path = dict_path
        self.name = name
        self.init_w2v = init_w2v
        
        if self.init_w2v:
            self.weight_data = EmbeddingLayer.v2np(self)
    
    def v2np(self):
        """
        vector to npy
        """
        word2id = {}
        id2word = []
        for line in open(self.dict_path, "r"):
            word, idx = line.rstrip().split("\t")
            word2id[word] = int(idx)
            id2word.append(word)
        voc_size = len(word2id)
        logging.info("the size of the vocab is %d" % voc_size)
        logging.info("loading word2vec from %s" % self.vec_path)
        logging.info("please wait for a minute.")
        start = time.time()
        vecs = []
        word2vec= {}
        with open(self.vec_path, "r") as f:
            f.readline()
            for line in f:
                info = line.strip("\n").split(" ")
                word = info[0]
                if word not in word2id:
                    continue
                vector = info[1:]
                if len(vector) != self.emb_dim:
                    logging.info(len(vector))
                assert(len(vector) == self.emb_dim)
                word2vec[word] = np.asarray(vector, dtype='float32')

        for word in id2word:
            if word in word2vec:
                vecs.append(word2vec[word])
            else:
                vecs.append(np.random.uniform(-0.05, 0.05, size=[self.emb_dim]).astype(np.float32))
        vecs = np.stack(vecs)
        
        end = time.time()
        logging.info("Spent %s on loading word2vec." % str(datetime.timedelta(seconds = end - start)))

        return vecs

    def ops(self, input):
        """
        :param input:
        """
        if self.init_w2v:
            #weight_data = EmbeddingLayer.v2np(self)
            w_param_attrs = fluid.ParamAttr(
                name=self.name, 
                learning_rate=1,
                initializer=fluid.initializer.NumpyArrayInitializer(self.weight_data),
                trainable=True)
            emb = fluid.layers.embedding(
                input=input,
                size=[self.dict_dim, self.emb_dim],
                is_sparse=False,
                param_attr=w_param_attrs,
                dtype='float32')
        else:    
            emb = fluid.layers.embedding(
                input=input,
                size=[self.dict_dim, self.emb_dim],
                is_sparse=False,
                param_attr=attr.ParamAttr(name=self.name),
                dtype='float32')

        return emb

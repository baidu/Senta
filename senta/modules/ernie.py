# *_*coding:utf-8 *_*
"""
import
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import json
import logging

import paddle.fluid as fluid
import six

from senta.modules.transformer_encoder import encoder, pre_process_layer
from senta.modules.transformer_encoder import gelu

class ErnieModel(object):
    """
    ErnieModel
    """

    def __init__(self,
                 src_ids,
                 position_ids,
                 sentence_ids,
                 task_ids,
                 input_mask,
                 config,
                 weight_sharing=True,
                 use_fp16=False):
        """
        :param src_ids:
        :param position_ids:
        :param sentence_ids:
        :param task_ids:
        :param input_mask:
        :param config:
        :param weight_sharing:
        :param use_fp16:
        """
        self._hidden_size = config.get('hidden_size', 768)
        self._emb_size = config.get('emb_size', self._hidden_size)
        self._n_layer = config.get('num_hidden_layers', 12)
        self._n_head = config.get('num_attention_heads', 12)
        self._voc_size = config.get('vocab_size', 30522)
        self._max_position_seq_len = config.get('max_position_embeddings', 512)
        self._param_share = config.get('param_share', "normal")
        self._pre_encoder_cmd = config.get('pre_encoder_cmd', "nd")
        self._preprocess_cmd = config.get('preprocess_cmd', "")
        self._postprocess_cmd = config.get('postprocess_cmd', "dan")
        self._epsilon = config.get('epsilon', 1e-05)
        self._emb_mapping_in = config.get('emb_mapping_in', False)
        self._n_layer_per_block = config.get('n_layer_per_block', 1)

        if config.has('sent_type_vocab_size'):
            self._sent_types = config['sent_type_vocab_size']
        else:
            self._sent_types = config.get('type_vocab_size', 2)

        self._use_sentence_id = config.get('use_sentence_id', True)
        self._use_task_id = config.get('use_task_id', False)
        if self._use_task_id:
            self._task_types = config.get('task_type_vocab_size', 3)
        self._hidden_act = config.get('hidden_act', 'gelu')
        self._prepostprocess_dropout = config.get('hidden_dropout_prob', 0.1)
        self._attention_dropout = config.get('attention_probs_dropout_prob', 0.1)
        self._weight_sharing = weight_sharing

        self._word_emb_name = "word_embedding"
        self._pos_emb_name = "pos_embedding"
        self._sent_emb_name = "sent_embedding"
        self._task_emb_name = "task_embedding"
        self._dtype = "float16" if use_fp16 else "float32"
        self._emb_dtype = "float32"
        # Initialize all weigths by truncated normal initializer, and all biases
        # will be initialized by constant zero by default.
        self._param_initializer = fluid.initializer.TruncatedNormal(
            scale=config.get('initializer_range', 0.02))

        self._build_model(src_ids, position_ids, sentence_ids, task_ids,
                          input_mask)

    def _build_model(self, src_ids, position_ids, sentence_ids, task_ids,
                     input_mask):
        """
        :param src_ids:
        :param position_ids:
        :param sentence_ids:
        :param task_ids:
        :param input_mask:
        :return:
        """
        # padding id in vocabulary must be set to 0
        emb_out = fluid.layers.embedding(
            input=src_ids,
            dtype=self._emb_dtype,
            size=[self._voc_size, self._emb_size],
            param_attr=fluid.ParamAttr(
                name=self._word_emb_name, initializer=self._param_initializer),
            is_sparse=False)

        position_emb_out = fluid.layers.embedding(
            input=position_ids,
            dtype=self._emb_dtype,
            size=[self._max_position_seq_len, self._emb_size],
            param_attr=fluid.ParamAttr(
                name=self._pos_emb_name, initializer=self._param_initializer))

        emb_out = emb_out + position_emb_out

        if self._use_sentence_id:
            sent_emb_out = fluid.layers.embedding(
                sentence_ids,
                dtype=self._emb_dtype,
                size=[self._sent_types, self._emb_size],
                param_attr=fluid.ParamAttr(
                    name=self._sent_emb_name, initializer=self._param_initializer))

            emb_out = emb_out + sent_emb_out

        if self._use_task_id:
            task_emb_out = fluid.layers.embedding(
                task_ids,
                size=[self._task_types, self._emb_size],
                dtype=self._emb_dtype,
                param_attr=fluid.ParamAttr(
                    name=self._task_emb_name,
                    initializer=self._param_initializer))

            emb_out = emb_out + task_emb_out

        emb_out = pre_process_layer(
            emb_out, self._pre_encoder_cmd, self._prepostprocess_dropout, name='pre_encoder', epsilon=self._epsilon)

        if self._emb_mapping_in:
            emb_out = fluid.layers.fc(input=emb_out,
                          num_flatten_dims=2,
                          size=self._hidden_size,
                          param_attr=fluid.ParamAttr(
                              name='emb_hidden_mapping',
                              initializer=self._param_initializer),
                          bias_attr='emb_hidden_mapping_bias')

        if self._dtype == "float16":
            emb_out = fluid.layers.cast(x=emb_out, dtype=self._dtype)
            input_mask = fluid.layers.cast(x=input_mask, dtype=self._dtype)
        self_attn_mask = fluid.layers.matmul(
            x=input_mask, y=input_mask, transpose_y=True)

        self_attn_mask = fluid.layers.scale(
            x=self_attn_mask, scale=10000.0, bias=-1.0, bias_after_scale=False)
        n_head_self_attn_mask = fluid.layers.stack(
            x=[self_attn_mask] * self._n_head, axis=1)
        n_head_self_attn_mask.stop_gradient = True

        self._enc_out, self._checkpoints= encoder(
            enc_input=emb_out,
            attn_bias=n_head_self_attn_mask,
            n_layer=self._n_layer,
            n_head=self._n_head,
            d_key=self._hidden_size // self._n_head,
            d_value=self._hidden_size // self._n_head,
            d_model=self._hidden_size,
            d_inner_hid=self._hidden_size * 4,
            prepostprocess_dropout=self._prepostprocess_dropout,
            attention_dropout=self._attention_dropout,
            relu_dropout=0,
            hidden_act=self._hidden_act,
            preprocess_cmd=self._preprocess_cmd,
            postprocess_cmd=self._postprocess_cmd,
            param_initializer=self._param_initializer,
            name='encoder',
            param_share=self._param_share,
            epsilon=self._epsilon,
            n_layer_per_block=self._n_layer_per_block)
        if self._dtype == "float16":
            self._enc_out = fluid.layers.cast(
                x=self._enc_out, dtype=self._emb_dtype)

    def get_sequence_output(self):
        """
        :return:
        """
        return self._enc_out
    
    def get_pooled_output(self):
        """Get the first feature of each sequence for classification"""
        next_sent_feat = fluid.layers.slice(
            input=self._enc_out, axes=[1], starts=[0], ends=[1])

        """
        if self._dtype == "float16":        
            next_sent_feat = fluid.layers.cast(
                x=next_sent_feat, dtype=self._emb_dtype)

        next_sent_feat = fluid.layers.fc(
            input=next_sent_feat,
            size=self._emb_size,
            param_attr=fluid.ParamAttr(
                name="mask_lm_trans_fc.w_0", initializer=self._param_initializer),
            bias_attr="mask_lm_trans_fc.b_0")
        """

        """
        next_sent_feat = fluid.layers.fc(
            input=next_sent_feat,
            size=self._emb_size,
            param_attr=fluid.ParamAttr(
                name="mask_lm_trans_fc.w_0", initializer=self._param_initializer),
            bias_attr="mask_lm_trans_fc.b_0")

        """
        next_sent_feat = fluid.layers.fc(
            input=next_sent_feat,
            size=self._hidden_size,
            act="tanh",
            param_attr=fluid.ParamAttr(
                name="pooled_fc.w_0", initializer=self._param_initializer),
            bias_attr="pooled_fc.b_0")
        return next_sent_feat

    def get_lm_output(self, mask_label, mask_pos):
        """Get the loss & accuracy for pretraining"""
        mask_pos = fluid.layers.cast(x=mask_pos, dtype='int32')
        # extract the first token feature in each sentence
        self.next_sent_feat = self.get_pooled_output()
        reshaped_emb_out = fluid.layers.reshape(
            x=self._enc_out, shape=[-1, self._hidden_size])
        # extract masked tokens' feature
        mask_feat = fluid.layers.gather(input=reshaped_emb_out, index=mask_pos)

        if self._dtype == "float16":
            mask_feat = fluid.layers.cast(x=mask_feat, dtype=self._emb_dtype)

        # transform: fc
        if self._hidden_act == 'gelu' or self._hidden_act == 'gelu.precise':
            _hidden_act = 'gelu'
        elif self._hidden_act == 'gelu.approximate':
            _hidden_act = None
        else:
            _hidden_act = self._hidden_act
        mask_trans_feat = fluid.layers.fc(
            input=mask_feat,
            size=self._emb_size,
            act=_hidden_act,
            param_attr=fluid.ParamAttr(
                name='mask_lm_trans_fc.w_0',
                initializer=self._param_initializer),
            bias_attr=fluid.ParamAttr(name='mask_lm_trans_fc.b_0'))
        if self._hidden_act == 'gelu.approximate':
            mask_trans_feat = gelu(mask_trans_feat)
        else:
            pass
        # transform: layer norm
        mask_trans_feat = fluid.layers.layer_norm(
            mask_trans_feat,
            begin_norm_axis=len(mask_trans_feat.shape) - 1,
            param_attr=fluid.ParamAttr(
                name='mask_lm_trans_layer_norm_scale',
                initializer=fluid.initializer.Constant(1.)),
            bias_attr=fluid.ParamAttr(
                name='mask_lm_trans_layer_norm_bias',
                initializer=fluid.initializer.Constant(1.)))
        # transform: layer norm
        # mask_trans_feat = pre_process_layer(
        #    mask_trans_feat, 'n', name='mask_lm_trans')

        mask_lm_out_bias_attr = fluid.ParamAttr(
            name="mask_lm_out_fc.b_0",
            initializer=fluid.initializer.Constant(value=0.0))
        if self._weight_sharing:
            fc_out = fluid.layers.matmul(
                x=mask_trans_feat,
                y=fluid.default_main_program().global_block().var(
                    self._word_emb_name),
                transpose_y=True)
            fc_out += fluid.layers.create_parameter(
                shape=[self._voc_size],
                dtype=self._emb_dtype,
                attr=mask_lm_out_bias_attr,
                is_bias=True)

        else:
            fc_out = fluid.layers.fc(input=mask_trans_feat,
                                     size=self._voc_size,
                                     param_attr=fluid.ParamAttr(
                                         name="mask_lm_out_fc.w_0",
                                         initializer=self._param_initializer),
                                     bias_attr=mask_lm_out_bias_attr)

        mask_lm_loss = fluid.layers.softmax_with_cross_entropy(
            logits=fc_out, label=mask_label)
        mean_mask_lm_loss = fluid.layers.mean(mask_lm_loss)

        return mean_mask_lm_loss

    def get_task_output(self, task, task_labels):
        """
        :param task:
        :param task_labels:
        :return:
        """
        task_fc_out = fluid.layers.fc(input=self.next_sent_feat,
                                      size=task["num_labels"],
                                      param_attr=fluid.ParamAttr(
                                          name=task["task_name"] + "_fc.w_0",
                                          initializer=self._param_initializer),
                                      bias_attr=task["task_name"] + "_fc.b_0")
        task_loss, task_softmax = fluid.layers.softmax_with_cross_entropy(
            logits=task_fc_out, label=task_labels, return_softmax=True)
        task_acc = fluid.layers.accuracy(input=task_softmax, label=task_labels)
        mean_task_loss = fluid.layers.mean(task_loss)
        return mean_task_loss, task_acc


class ErnieConfig(object):
    """parse ernie config"""

    def __init__(self, config_path):
        """
        :param config_path:
        """
        self._config_dict = self._parse(config_path)

    def _parse(self, config_path):
        """
        :param config_path:
        :return:
        """
        try:
            with open(config_path, 'r') as json_file:
                config_dict = json.load(json_file)
        except Exception:
            raise IOError("Error in parsing Ernie model config file '%s'" %
                          config_path)
        else:
            return config_dict

    def __getitem__(self, key):
        """
        :param key:
        :return:
        """
        return self._config_dict.get(key, None)

    def has(self, key):
        """
        :param key:
        :return:
        """
        if key in self._config_dict:
            return True
        return False

    def get(self, key, default_value):
        """
        :param key,default_value:
        :retrun:
        """
        if key in self._config_dict:
            return self._config_dict[key]
        else:
            return default_value

    def print_config(self):
        """
        :return:
        """
        for arg, value in sorted(six.iteritems(self._config_dict)):
            logging.info('%s: %s' % (arg, value))
        logging.info('------------------------------------------------')

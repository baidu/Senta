# -*- coding: utf-8 -*
"""
:py:class:`ErnieTokenEmbedding`
"""

from senta.common.rule import InstanceName
from senta.modules.ernie import ErnieModel, ErnieConfig
from senta.modules.token_embedding.base_token_embedding import BaseTokenEmbedding


class ErnieTokenEmbedding(BaseTokenEmbedding):
    """ErnieTokenEmbedding: 使用ernie的embedding，训练过程中不断finetune
    """
    def __init__(self, emb_dim, vocab_size, params_path):
        BaseTokenEmbedding.__init__(self, emb_dim, vocab_size)
        self.params_path = params_path
        self.ernie_config = ErnieConfig(params_path)
        self.use_fp16 = False

    def build(self):
        """
        添加一些自顶一个初始化信息，如参数名称
        :return:
        """
        pass

    def get_token_embedding(self, tokens_dict):
        """
        :param tokens_dict:
        :return:
        """
        ernie = ErnieModel(
            src_ids=tokens_dict[InstanceName.SRC_IDS],
            position_ids=tokens_dict[InstanceName.POS_IDS],
            sentence_ids=tokens_dict[InstanceName.SENTENCE_IDS],
            task_ids=tokens_dict[InstanceName.TASK_IDS],
            input_mask=tokens_dict[InstanceName.MASK_IDS],
            config=self.ernie_config,
            use_fp16=self.use_fp16
        )

        emb_dict = {
            InstanceName.SEQUENCE_EMB: ernie.get_sequence_output(),
            InstanceName.POOLED_EMB: ernie.get_pooled_output()
        }
        return emb_dict


    def get_output_dim(self):
        """
        :return:
        """
        pass

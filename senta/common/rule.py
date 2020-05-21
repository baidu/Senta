# -*- coding: utf-8 -*
"""
some rule
"""


class MaxTruncation(object):
    """MaxTruncation：超长截断规则
    """
    KEEP_HEAD = 0  # 从头开始到最大长度截断
    KEEP_TAIL = 1  # 从头开始到max_len-1的位置截断，末尾补上最后一个id（词或字）
    KEEP_BOTH_HEAD_TAIL = 2  # 保留头和尾两个位置，然后按keep_head方式截断


class EmbeddingType(object):
    """EmbeddingType:文本数据需要转换的embedding类型：no_emb , ernie_emb
    """
    NONE_EMBEDDING = 0  # 不需要emb
    ERNIE_EMBEDDING = 1  # 用ernie生成emb
    FLUID_EMBEDDING = 2  # 使用fluid的op生成emb


class DataShape(object):
    """DataShape:输入的数据类型
    """
    STRING = "string"  # string
    INT = "int"  # int64
    FLOAT = "float"  # float32


class InstanceName(object):
    """InstanceName:一些常用的命名
    """
    RECORD_ID = "id"
    RECORD_EMB = "emb"
    SRC_IDS = "src_ids"
    MASK_IDS = "mask_ids"
    SEQ_LENS = "seq_lens"
    SENTENCE_IDS = "sent_ids"
    POS_IDS = "pos_ids"
    TASK_IDS = "task_ids"

    # seq2seq的label域相关key
    TRAIN_LABEL_SRC_IDS = "train_label_src_ids"
    TRAIN_LABEL_MASK_IDS = "train_label_mask_ids"
    TRAIN_LABEL_SEQ_LENS = "train_label_seq_lens"
    INFER_LABEL_SRC_IDS = "infer_label_src_ids"
    INFER_LABEL_MASK_IDS = "infer_label_mask_ids"
    INFER_LABEL_SEQ_LENS = "infer_label_seq_lens"

    SEQUENCE_EMB = "sequence_output"  # 词级别的embedding
    POOLED_EMB = "pooled_output"  # 句子级别的embedding

    TARGET_FEED_NAMES = "target_feed_name"  # 保存模型时需要的入参：表示模型预测时需要输入的变量名称和顺序
    TARGET_PREDICTS = "target_predicts"  # 保存模型时需要的入参：表示预测时最终输出的结果
    PREDICT_RESULT = "predict_result"  # 训练过程中需要传递的预测结果
    LABEL = "label"  # label
    LOSS = "loss"  # loss
    # CRF_EMISSION = "crf_emission"  # crf_emission

    TRAINING = "training"  # 训练过程
    EVALUATE = "evaluate"  # 评估过程
    TEST = "test" # 测试过程
    SAVE_INFERENCE = "save_inference"  # 保存inference model的过程
    
    STEP = "steps"
    SPEED = "speed"
    TIME_COST = "time_cost"
    GPU_ID = "gpu_id"

class FieldLength(object):
    """一个field在序列化成field_id_list的时候，占的长度是多少
    """
    CUSTOM_TEXT_FIELD = 3
    ERNIE_TEXT_FIELD = 6
    SINGLE_SCALAR_FIELD = 1
    ARRAY_SCALAR_FIELD = 2
    BASIC_TEXT_FIELD = 2
    GENERATE_LABEL_FIELD = 6

class FleetMode(object):
    """Fleet模式
    """
    NO_FLEET = "NO_FLEET"
    CPU_MODE = "CPU"
    GPU_MODE = "GPU"

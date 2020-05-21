# -*- coding: utf-8 -*
"""import"""
import json
import unicodedata
from collections import OrderedDict
import six
from paddle.fluid.core_avx import PaddleTensor
from senta.common.rule import MaxTruncation
import sys
import os
import paddle
import paddle.fluid as fluid

try:
    import pkg_resources
    get_module_res = lambda *res: pkg_resources.resource_stream(__name__,
                                                                os.path.join(*res))
except ImportError:
    get_module_res = lambda *res: open(os.path.normpath(os.path.join(
                            os.getcwd(), os.path.dirname(__file__), *res)), 'rb')

PY2 = sys.version_info[0] == 2

default_encoding = sys.getfilesystemencoding()

if PY2:
    text_type = unicode
    string_types = (str, unicode)

    iterkeys = lambda d: d.iterkeys()
    itervalues = lambda d: d.itervalues()
    iteritems = lambda d: d.iteritems()

else:
    text_type = str
    string_types = (str,)
    xrange = range

    iterkeys = lambda d: iter(d.keys())
    itervalues = lambda d: iter(d.values())
    iteritems = lambda d: iter(d.items())

def strdecode(sentence):
    """
    string to unicode
    :param sentence: a string of  utf-8 or gbk
    :return: input's unicode result
    """
    if not isinstance(sentence, text_type):
        try:
            sentence = sentence.decode('utf-8')
        except UnicodeDecodeError:
            sentence = sentence.decode('gbk', 'ignore')
    return sentence


def check_cuda(use_cuda):
    """
    check_cuda
    """
    err = \
    "\nYou can not set use_cuda = True in the model because you are using paddlepaddle-cpu.\n \
    Please: 1. Install paddlepaddle-gpu to run your models on GPU or 2. Set use_cuda = False to run models on CPU.\n"
    try:
        if use_cuda == True and fluid.is_compiled_with_cuda() == False:
            print(err)
            sys.exit(1)
    except Exception as e:
        pass

def parse_data_config(config_path):
    """
    :param config_path:
    :return:
    """
    try:
        with open(config_path) as json_file:
            config_dict = json.load(json_file, object_pairs_hook=OrderedDict)
    except Exception:
        raise IOError("Error in parsing Ernie model config file '%s'" % config_path)
    else:
        return config_dict


def parse_version_code(version_str, default_version_code=1.5):
    """
    parser paddle fluid version code to float type
    :param version_str:
    :param default_version_code:
    :return:
    """
    if version_str:
        v1 = version_str.split(".")[0:2]
        v_code_str = ".".join(v1)
        v_code = float(v_code_str)
        return v_code
    else:
        return default_version_code


def truncation_words(words, max_seq_length, truncation_type):
    """
    :param words:
    :param max_seq_length:
    :param truncation_type:
    :return:
    """
    if len(words) > max_seq_length:
        if truncation_type == MaxTruncation.KEEP_HEAD:
            words = words[0: max_seq_length]
        elif truncation_type == MaxTruncation.KEEP_TAIL:
            tmp = words[0: max_seq_length - 1]
            tmp.append(words[-1])
            words = tmp
        elif truncation_type == MaxTruncation.KEEP_BOTH_HEAD_TAIL:
            tmp = words[1: max_seq_length - 2]
            tmp.insert(0, words[0])
            tmp.insert(max_seq_length - 1, words[-1])
            words = tmp
        else:
            words = words[0: max_seq_length]

    return words


def truncate_seq_pair(tokens_a, tokens_b, max_length):
    """
    :param tokens_a:
    :param tokens_a:
    :param max_length:
    :return:
    """
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def convert_to_unicode(text):
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


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a peice of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


def is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
        (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


def printable_text(text):
    """Returns text encoded in a way suitable for print or `tf.logging`."""

    # These functions want `str` for both Python2 and Python3, but in one case
    # it's a Unicode string and in the other it's a byte string.
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text
        elif isinstance(text, unicode):
            return text.encode("utf-8")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def array2tensor(arr_data):
    """ convert numpy array to PaddleTensor"""
    tensor_data = PaddleTensor(arr_data)
    return tensor_data


def save_infer_data_meta(data_dict, save_file):
    """
    :param data_dict:
    :param save_file:
    :return:
    """

    json_str = json.dumps(data_dict)
    with open(save_file, 'w') as json_file:
        json_file.write(json_str)


import logging
import argparse
import ast

import paddle.fluid as fluid
import numpy as np

import utils
from nets import bow_net
from nets import cnn_net
from nets import lstm_net
from nets import bilstm_net
from nets import gru_net

logger = logging.getLogger("paddle-fluid")
logger.setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser("Sentiment Classification.")
    # training data path
    parser.add_argument(
        "--train_data_path",
        type=str,
        required=False,
        help="The path of trainning data. Should be given in train mode!")
    # test data path
    parser.add_argument(
        "--test_data_path",
        type=str,
        required=False,
        help="The path of test data. Should be given in eval or infer mode!")
    # word_dict path
    parser.add_argument(
        "--word_dict_path",
        type=str,
        required=True,
        help="The path of word dictionary.")
    # current mode
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=['train', 'eval', 'infer'],
        help="train/eval/infer mode")
    # model type
    parser.add_argument(
        "--model_type",
        type=str,
        default="bilstm_net",
        help="type of model")
    # model save path
    parser.add_argument(
        "--model_path",
        type=str,
        default="models",
        required=True,
        help="The path to saved the trained models.")
    # Number of passes for the training task.
    parser.add_argument(
        "--num_passes",
        type=int,
        default=10,
        help="Number of passes for the training task.")
    # Batch size
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="The number of training examples in one forward/backward pass.")
    # lr value for training
    parser.add_argument(
        "--lr",
        type=float,
        default=0.002,
        help="The lr value for training.")
    # Whether to use gpu
    parser.add_argument(
        "--use_gpu",
        type=ast.literal_eval,
        default=False,
        help="Whether to use gpu to train the model.")
    # parallel train
    parser.add_argument(
        "--is_parallel",
        type=ast.literal_eval,
        default=False,
        help="Whether to train the model in parallel.")
    args = parser.parse_args()
    return args


def train_net(train_reader,
              word_dict,
              network,
              use_gpu,
              parallel,
              save_dirname,
              lr=0.002,
              batch_size=128,
              pass_num=30):
    """
    train network
    """
    if network == "bilstm_net":
        network = bilstm_net
    elif network == "bow_net":
        network = bow_net
    elif network == "cnn_net":
        network = cnn_net
    elif network == "lstm_net":
        network = lstm_net
    elif network == "gru_net":
        network = gru_net
    else:
        print ("unknown network type")
        return
    # word seq data
    data = fluid.layers.data(
        name="words", shape=[1], dtype="int64", lod_level=1)
    # label data
    label = fluid.layers.data(name="label", shape=[1], dtype="int64")

    cost, acc, pred = network(data, label, len(word_dict) + 1)
    cost = fluid.layers.mean(cost)
    acc = fluid.layers.mean(acc)

    # set optimizer
    sgd_optimizer = fluid.optimizer.Adagrad(learning_rate=lr)
    sgd_optimizer.minimize(cost)

    # set place, executor, datafeeder
    place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    feeder = fluid.DataFeeder(feed_list=[data, label], place=place)

    # initilize parameters
    exe.run(fluid.default_startup_program())

    # parallelize it
    train_exe = fluid.ParallelExecutor(use_cuda=use_gpu, loss_name=cost.name) \
        if args.is_parallel else exe

    # start training...
    for pass_id in range(pass_num):
        data_size, data_count, total_acc, total_cost = 0, 0, 0.0, 0.0
        for data in train_reader():
            # train a batch
            avg_cost_np, avg_acc_np = train_exe.run(
                                              feed=feeder.feed(data),
                                              fetch_list=[cost.name, acc.name])
            data_size = len(data)
            total_acc += data_size * np.sum(avg_acc_np)
            total_cost += data_size * np.sum(avg_cost_np)
            data_count += data_size * len(avg_acc_np)
        avg_cost = total_cost / data_count
        avg_acc = total_acc / data_count
        print("[train info]: pass_id: %d, avg_acc: %f, avg_cost: %f" %
              (pass_id, avg_acc, avg_cost))
        epoch_model = save_dirname + "/" + "epoch" + str(pass_id)
        # save the model
        fluid.io.save_inference_model(epoch_model, ["words"], pred, exe)


def eval_net(test_reader, use_gpu, model_path=None):
    """
    Evaluation function
    """
    if model_path is None:
        print (str(model_path) + "can not be found")
        return
    # set place, executor
    place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)

    inference_scope = fluid.core.Scope()
    with fluid.scope_guard(inference_scope):
        # load the saved model
        [inference_program, feed_target_names,
            fetch_targets] = fluid.io.load_inference_model(model_path, exe)
        
        # compute 2class and 3class accuracy
        class2_acc, class3_acc = 0.0, 0.0
        total_count, neu_count = 0, 0

        for data in test_reader():
            # infer a batch
            pred = exe.run(inference_program,
                           feed=utils.data2tensor(data, place),
                           fetch_list=fetch_targets,
                           return_numpy=True)
            for i, val in enumerate(data):
                class3_label, class2_label = utils.get_predict_label(pred[0][i, 1])
                true_label = val[1]
                if class2_label == true_label:
                    class2_acc += 1
                if class3_label == true_label:
                    class3_acc += 1
                if true_label == 1.0:
                    neu_count += 1

            total_count += len(data)

        class2_acc = class2_acc / (total_count - neu_count)
        class3_acc = class3_acc / total_count
        print("[test info] model_path: %s, class2_acc: %f, class3_acc: %f" %
              (model_path, class2_acc, class3_acc))


def infer_net(test_reader, use_gpu, model_path=None):
    """
    Inference function
    """
    if model_path is None:
        print(str(model_path) + "can not be found")
        return
    # set place, executor
    place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)

    inference_scope = fluid.core.Scope()
    with fluid.scope_guard(inference_scope):
        # load the saved model
        [inference_program, feed_target_names,
            fetch_targets] = fluid.io.load_inference_model(model_path, exe)

        for data in test_reader():
            # infer a batch
            pred = exe.run(inference_program,
                           feed=utils.data2tensor(data, place),
                           fetch_list=fetch_targets,
                           return_numpy=True)
            for i, val in enumerate(data):
                class3_label, class2_label = utils.get_predict_label(pred[0][i, 1])
                pos_prob = pred[0][i, 1]
                neg_prob = 1 - pos_prob
                print("predict label: %d, pos_prob: %f, neg_prob: %f" %
                      (class3_label, pos_prob, neg_prob))


def main(args):

    # train mode
    if args.mode == "train":
        # prepare_data to get word_dict, train_reader
        word_dict, train_reader = utils.prepare_data(
            args.train_data_path, args.word_dict_path, args.batch_size,
            args.mode)
        
        train_net(
            train_reader,
            word_dict,
            args.model_type,
            args.use_gpu,
            args.is_parallel,
            args.model_path,
            args.lr,
            args.batch_size,
            args.num_passes)
    
    # eval mode
    elif args.mode == "eval":
        # prepare_data to get word_dict, test_reader    
        word_dict, test_reader = utils.prepare_data(
            args.test_data_path, args.word_dict_path, args.batch_size,
            args.mode)
        eval_net(
            test_reader,
            args.use_gpu,
            args.model_path)
    
    # infer mode
    elif args.mode == "infer":
        # prepare_data to get word_dict, test_reader
        word_dict, test_reader = utils.prepare_data(
            args.test_data_path, args.word_dict_path, args.batch_size,
            args.mode)
        infer_net(
            test_reader,
            args.use_gpu,
            args.model_path)


if __name__ == "__main__":
    args = parse_args()
    main(args) 

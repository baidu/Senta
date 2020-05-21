# *_*coding:utf-8 *_*
"""
BaseTrainer
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from collections import OrderedDict

from paddle.fluid.incubate.fleet.collective import fleet

import multiprocessing
import paddle.fluid as fluid
import os
import logging
import senta.utils.init as init
from senta.common.rule import InstanceName
from senta.utils.util_helper import save_infer_data_meta
from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler import fleet
from paddle.fluid.incubate.fleet.base import role_maker
import threading


class BaseTrainer(object):
    """BaseTrainer"""

    def __init__(self, params, data_set_reader, model_class):
        """
        1.运行环境初始化 2.program初始化 3.计算图网络导入 4.模型参数导入 5.运行(reader) 6.模型导出
        :param params: 运行的基本参数设置
        :param data_set_reader: 运行的基本参数设置
        :param model_class: 使用的是哪个model
        """
        self.data_set_reader = data_set_reader
        self.params = params
        self.model_class = model_class
        self.random_seed = self.params.get("random_seed", 0)
        self.forward_train_output = None
        self.optimizer_output_dict = None
        self.fetch_list_train = []
        self.fetch_list_evaluate = []
        self.is_fleet = False
        self.init_program()
        self.init_env()
        self.init_net()
        self.executor.run(self.startup_program)
        self.prepare_fleet_paddle_cloud(self.is_fleet)
        if self.params["load_checkpoint"] or self.params["load_parameters"]:
            self.load_model_params("net_model")
        elif self.params["pre_train_model"]:
            self.load_model_params("pre_train_model")
        self.build_executor()

    def init_env(self):
        """
        :return:
        """
        # multi nodes
        self.num_trainers = 1
        self.trainer_id = 0
        self.is_local = self.params.get("PADDLE_IS_LOCAL", False)
        # cpu multi
        if self.params["PADDLE_USE_GPU"]:
            gpus = os.getenv('FLAGS_selected_gpus', '0').split(",")
            self.gpu_id = int(gpus[0])
            run_place = fluid.CUDAPlace(int(gpus[0]))
            if "is_distributed" in self.params and self.params["is_distributed"]:
                self.dev_count = len(gpus)
            else:
                self.dev_count = fluid.core.get_cuda_device_count()
            #logging.debug("gpu count %d" % self.dev_count)
            self.prepare_nccl2_env(self.is_local)
            logging.debug("finish prepare nccl2 env")
        else:
            run_place = fluid.CPUPlace()
            self.dev_count = int(os.environ.get('CPU_NUM', multiprocessing.cpu_count()))
            self.prepare_cpumulti_env(self.is_local)
            self.gpu_id = None
            logging.debug("finish prepare cpu multi")
        self.executor = fluid.Executor(run_place)

        # parallel executor relevant config
        self.num_iteration_per_drop_scope = self.params.get("num_iteration_per_drop_scope", 1)
        self.use_fast_executor = self.params.get("use_fast_executor", False)

    def init_program(self):
        """
        :return:
        """
        param_dict = OrderedDict()

        self.startup_program = fluid.Program()
        if self.random_seed is not None:
            self.startup_program.random_seed = self.random_seed
        self.train_program = fluid.Program()
        self.test_program = fluid.Program()
        self.evaluate_program = fluid.Program()
        self.save_inference_program = fluid.Program()

    def load_model_params(self, load_model):
        """
        :return:
        """
        if load_model == "net_model":
            if self.params["load_checkpoint"] and self.params["load_parameters"]:
                raise ValueError(
                    "ERROR: config 'load_checkpoint' and 'load_parameters' "
                    "both are set! Only one of them should be set. "
                    "if you want warmstart checkpoint keep its learning_rate and moments, plese set 'load_checkpoint'. "
                    "if you want warmstart checkpoint with only its parameters, and you want reset a new learning_rate "
                    "by config, plese set 'load_parameters'")
            if self.params["load_checkpoint"]:
                init.init_checkpoint(exe=self.executor, init_checkpoint_path=self.params["load_checkpoint"],
                                     main_program=self.startup_program, use_fp16=self.params.get("use_fp16", False))
            elif self.params["load_parameters"]:
                init.init_pretraining_params(exe=self.executor, pretraining_params_path=self.params["load_parameters"],
                                             main_program=self.startup_program,
                                             use_fp16=self.params.get("use_fp16", False))
        elif load_model == "pre_train_model":
            # pretrain_embedding_path = self.get_pretrain_embedding_path()
            for pre_train_model in self.params["pre_train_model"]:
                logging.info("pre_train_model's name = %s" % pre_train_model["name"])
                params_path = pre_train_model["params_path"]
                init.init_pretraining_params(exe=self.executor,
                                             pretraining_params_path=params_path,
                                             main_program=self.startup_program,
                                             use_fp16=self.params.get("use_fp16", False))

    def init_net(self):
        """
        初始化网络
        :return:
        """
        self.init_train_net()
        self.test_program = self.init_evaluate_net(self.data_set_reader.test_reader, self.test_program)
        self.evaluate_program = self.init_evaluate_net(self.data_set_reader.dev_reader,
                                                       self.evaluate_program)
        self.init_save_inference_net()

    def init_train_net(self):
        """
        训练网络初始化，前向+后向
        :return:
        """
        with fluid.program_guard(self.train_program, self.startup_program):
            with fluid.unique_name.guard():
                self.data_set_reader.train_reader.create_reader()
                fields_dict = self.data_set_reader.train_reader.instance_fields_dict()
                self.forward_train_output = self.model_class.forward(fields_dict, phase=InstanceName.TRAINING)

                self.optimizer_output_dict = self.model_class.optimizer(self.forward_train_output[InstanceName.LOSS],
                                                                        self.is_fleet)
                if isinstance(self.optimizer_output_dict, dict):
                    if "use_ernie_opt" in self.optimizer_output_dict:
                        opt_args = self.optimizer_output_dict["opt_args"]
                        self.optimizer_output_dict = optimization(train_program=self.train_program,
                                                                  startup_prog=self.startup_program,
                                                                  **opt_args)
                else:
                    self.optimizer_output_dict = {}
                self.forward_train_output.update(self.optimizer_output_dict)
                self.fetch_list_train = list(self.forward_train_output.values())
                self.fetch_list_train_key = list(self.forward_train_output.keys())

    def init_evaluate_net(self, reader, program):
        """初始化评估过程的网络，网络只有前向
        :return:
        """
        if reader:
            with fluid.program_guard(program, self.startup_program):
                with fluid.unique_name.guard():
                    reader.create_reader()
                    fields_dict = reader.instance_fields_dict()
                    self.forward_evaluate_output = self.model_class.forward(fields_dict, phase=InstanceName.EVALUATE)
                    self.fetch_list_evaluate = list(self.forward_evaluate_output.values())
                    self.fetch_list_evaluate_key = list(self.forward_evaluate_output.keys())
            program = program.clone(for_test=True)
        return program

    def init_save_inference_net(self):
        """初始化用来保存inference model的网络，只有前向，且是裁切过后的网络。
        :return:
        """
        if self.data_set_reader.predict_reader:
            with fluid.program_guard(self.save_inference_program, self.startup_program):
                with fluid.unique_name.guard():
                    self.data_set_reader.predict_reader.create_reader()
                    fields_dict = self.data_set_reader.predict_reader.instance_fields_dict()
                    forward_output_dict = self.model_class.forward(fields_dict, phase=InstanceName.SAVE_INFERENCE)
                    target_feed_list = forward_output_dict[InstanceName.TARGET_FEED_NAMES]
                    self.infer_dict = self.get_infer_data_meta(target_feed_list, fields_dict)
                    self.feed_target_names = target_feed_list

                    logging.info('...infer dict...')
                    logging.info(self.infer_dict)
                    logging.info('...feed target names...')
                    logging.info(self.feed_target_names)
                    self.inference_output = forward_output_dict[InstanceName.TARGET_PREDICTS]
            self.save_inference_program = self.save_inference_program.clone(for_test=True)

    def run(self, phase, need_fetch=True):
        """run and fetch
        :param phase:
        :param need_fetch
        :return:
        """
        fetch_output = []

        if phase == InstanceName.TRAINING:
            if need_fetch:
                fetch_output = self.train_exe.run(fetch_list=self.fetch_list_train)
            else:
                self.train_exe.run(fetch_list=[])
        elif phase == InstanceName.TEST:
            fetch_output = self.executor.run(program=self.test_program,
                                             fetch_list=self.fetch_list_evaluate)
        elif phase == InstanceName.EVALUATE:
            fetch_output = self.executor.run(program=self.evaluate_program,
                                             fetch_list=self.fetch_list_evaluate)
        return fetch_output

    def train_and_eval(self):
        """
        :param fetch_list_value:
        :param fetch_list_key:
        :param steps:
        :param phase:
        :return:
        """
        raise NotImplementedError

    def evaluate(self, reader, phase, step):
        """
        :param reader:
        :param phase:
        :param program:
        :param step:
        :return:
        """
        raise NotImplementedError

    def visualdl_log(self, metrics_output, train_loss, steps, phase):
        """log可视化，仅限于paddlecloud 平台任务
        :param metrics_output:
        :param train_loss:
        :param steps:
        :param phase:
        :return:
        """
        logging.info("{phase} log: steps {steps}, loss {loss}, metrics: {metrics}".format(phase=phase,
                                                                                          steps=steps, loss=train_loss,
                                                                                          metrics=metrics_output))
        try:
            if metrics_output and len(metrics_output) != 0:
                import paddlecloud.visual_util as visualdl
                x_dic = {"x_name": "step", "x_value": steps}
                y_ls = []
                for key, value in metrics_output.items():
                    y = {}
                    y["y_name"] = key
                    y["y_value"] = value
                    y_ls.append(y)
                visualdl.show_fluid_trend(x_dic, y_ls, tag="train")
        except Exception:
            logging.error("import paddlecloud.visual_util failed")

    def build_executor(self):
        """
        :return:
        """
        if self.is_fleet:
            exec_strategy = fluid.ExecutionStrategy()
            exec_strategy.num_threads = self.dev_count
            build_strategy = fluid.BuildStrategy()
            build_strategy.async_mode = False
            logging.info("CPU_NUM = %d" % self.dev_count)
            if self.dev_count > 1:
                build_strategy.reduce_strategy = fluid.BuildStrategy.ReduceStrategy.Reduce
            self.train_exe = fluid.ParallelExecutor(
                use_cuda=self.params["PADDLE_USE_GPU"],
                loss_name=self.forward_train_output[InstanceName.LOSS].name,
                main_program=self.train_program,
                build_strategy=build_strategy,
                exec_strategy=exec_strategy)
        else:
            exec_strategy = fluid.ExecutionStrategy()
            exec_strategy.num_threads = self.dev_count
            exec_strategy.num_iteration_per_drop_scope = self.num_iteration_per_drop_scope
            if self.use_fast_executor:
                exec_strategy.use_experimental_executor = True
            self.train_exe = fluid.ParallelExecutor(
                use_cuda=self.params["PADDLE_USE_GPU"],
                loss_name=self.forward_train_output[InstanceName.LOSS].name,
                exec_strategy=exec_strategy,
                main_program=self.train_program,
                num_trainers=self.num_trainers,
                trainer_id=self.trainer_id)

    def prepare_cpumulti_env(self, is_local):
        """
        :param is_local:
        :return:
        """
        if is_local:
            self.is_fleet = False
        else:
            role = role_maker.PaddleCloudRoleMaker()
            fleet.init(role)
            self.is_fleet = True
            logging.debug("init fleet cpu multi")

    def prepare_fleet_paddle_cloud(self, is_fleet):
        """
        :param is_local:
        :return:
        """
        if is_fleet == False:
            self.executor.run(self.startup_program)
        else:
            if fleet.is_worker():
                self.trainer_id = fleet.worker_index()
            if fleet.is_server():
                logging.info("init and run fleet server")
                fleet.init_server()
                fleet.run_server()
            elif fleet.is_worker():
                logging.info("init and run fleet worker")
                fleet.init_worker()
                self.executor.run(self.startup_program)

    def prepare_nccl2_env(self, is_local):
        """
        :param is_local:
        :return:
        """
        if not is_local:
            logging.debug("is_distributed: %s" % self.params["is_distributed"])
            if self.params["is_distributed"]:
                trainer_id = int(os.getenv("PADDLE_TRAINER_ID", "0"))
                worker_endpoints_env = os.getenv("PADDLE_TRAINER_ENDPOINTS")
                current_endpoint = os.getenv("PADDLE_CURRENT_ENDPOINT")
                worker_endpoints = worker_endpoints_env.split(",")
                trainers_num = len(worker_endpoints)
                logging.debug("worker_endpoints:{} trainers_num:{} current_endpoint:{} \
                      trainer_id:{}".format(worker_endpoints, trainers_num,
                                            current_endpoint, trainer_id))
                # prepare nccl2 env.
                config = fluid.DistributeTranspilerConfig()
                config.mode = "nccl2"
                t = fluid.DistributeTranspiler(config=config)
                t.transpile(
                    trainer_id,
                    trainers=worker_endpoints_env,
                    current_endpoint=current_endpoint,
                    program=self.train_program if self.params["is_do_train"] else self.test_program,
                    startup_program=self.startup_program)
                self.num_trainers = trainers_num
                self.trainer_id = trainer_id

    def get_infer_data_meta(self, target_feed_list, fields_dict):
        """
        :param target_feed_list:
        :param fields_dict:
        :return:
        """
        infer_dict = {
            "fields": []
        }
        for name in target_feed_list:
            for k1, v1 in fields_dict.items():  # dict_keys(['text_a', 'label'])
                for k2, v2 in v1.items():
                    if v2:
                        for k3 in v2:
                            # logging.info(k3)
                            if v2[k3] and v2[k3].name == name:
                                field_ele = "%s#%s" % (k1, k3)
                                infer_dict["fields"].append(field_ele)
        return infer_dict

    def save_models(self, save_checkpoints_path, save_inference_path, steps, save_checkpoint=True, save_inference=True):
        """
        :param save_checkpoints_path:
        :param save_inference_path:
        :param steps:
        :return:
        """
        logging.info("start save_models .....")
        if save_checkpoint:
            self.save_checkpoint(self.executor, save_checkpoints_path, self.train_program, steps)
        if save_inference:
            self.save_inference(self.executor, self.feed_target_names, self.inference_output,
                                save_inference_path, self.save_inference_program,
                                self.train_program, steps, self.infer_dict)
        logging.info("end save_models .....")

    def save_checkpoint(self, exe, save_checkpoints_path, program, steps):
        """
        :param exe:
        :param save_checkpoints_path:
        :param program:
        :param steps:
        :return:
        """
        logging.info("start save_checkpoint .....")
        save_path = os.path.join(save_checkpoints_path, "checkpoints_step_" + str(steps))
        if self.is_fleet:
            logging.info("fleet save checkpoints")
            fleet.save_persistables(exe, save_path, program)
        else:
            fluid.io.save_persistables(exe, save_path, program)

        logging.info("end save_checkpoint .....")

    def save_inference(self, exe, feeded_var_names, target_vars, save_inference_path,
                       program, main_program, steps, data_dict):
        """
        :param exe:
        :param feeded_var_names
        :param target_vars
        :param save_inference_path:
        :param program:
        :param steps:
        :param data_dict:
        :return:
        """
        logging.info("start save_inference .....")
        save_path = os.path.join(save_inference_path, "inference_step_" + str(steps))
        if self.is_fleet:
            logging.info("fleet save models")
            fleet.save_inference_model(
                executor=exe,
                dirname=save_path,
                feeded_var_names=feeded_var_names,
                target_vars=target_vars,
                main_program=main_program)
        else:
            fluid.io.save_inference_model(
                save_path,
                feeded_var_names,
                target_vars,
                exe,
                main_program=program,
                model_filename="model",
                params_filename="params")

        logging.info("start save_infer_data_meta .....")
        save_infer_data_meta(data_dict, save_path + '/infer_data_params.json')
        logging.info("end save_infer_data_meta .....")

        logging.info("end save_inference .....")


def linear_warmup_decay(learning_rate, warmup_steps, num_train_steps):
    """ Applies linear warmup of learning rate from 0 and decay to 0."""
    with fluid.default_main_program()._lr_schedule_guard():
        lr = fluid.layers.tensor.create_global_var(
            shape=[1],
            value=0.0,
            dtype='float32',
            persistable=True,
            name="scheduled_learning_rate")

        global_step = fluid.layers.learning_rate_scheduler._decay_step_counter(
        )

        with fluid.layers.control_flow.Switch() as switch:
            with switch.case(global_step < warmup_steps):
                warmup_lr = learning_rate * (global_step / warmup_steps)
                fluid.layers.tensor.assign(warmup_lr, lr)
            with switch.default():
                decayed_lr = fluid.layers.learning_rate_scheduler.polynomial_decay(
                    learning_rate=learning_rate,
                    decay_steps=num_train_steps,
                    end_learning_rate=0.0,
                    power=1.0,
                    cycle=False)
                fluid.layers.tensor.assign(decayed_lr, lr)

        return lr

def append_cast_op(i, o, prog):
    """
    Append a cast op in a given Program to cast input `i` to data type `o.dtype`.
    Args:
        i (Variable): The input Variable.
        o (Variable): The output Variable.
        prog (Program): The Program to append cast op.
    """
    prog.global_block().append_op(
        type="cast",
        inputs={"X": i},
        outputs={"Out": o},
        attrs={"in_dtype": i.dtype,
               "out_dtype": o.dtype})



def copy_to_master_param(p, block):
    """ copy_to_master_param """
    v = block.vars.get(p.name, None)
    if v is None:
        raise ValueError("no param name %s found!" % p.name)
    new_p = fluid.framework.Parameter(
        block=block,
        shape=v.shape,
        dtype=fluid.core.VarDesc.VarType.FP32,
        type=v.type,
        lod_level=v.lod_level,
        stop_gradient=p.stop_gradient,
        trainable=p.trainable,
        optimize_attr=p.optimize_attr,
        regularizer=p.regularizer,
        gradient_clip_attr=p.gradient_clip_attr,
        error_clip=p.error_clip,
        name=v.name + ".master")
    return new_p


def create_master_params_grads(params_grads, main_prog, startup_prog,
                               loss_scaling):
    """ create_master_params_grads """
    master_params_grads = []
    with main_prog._backward_role_guard():
        for p, g in params_grads:
            # create master parameters
            master_param = copy_to_master_param(p, main_prog.global_block())
            startup_master_param = startup_prog.global_block()._clone_variable(
                master_param)
            startup_p = startup_prog.global_block().var(p.name)
            append_cast_op(startup_p, startup_master_param, startup_prog)
            # cast fp16 gradients to fp32 before apply gradients
            # if g.name.find("layer_norm") > -1:
            #     scaled_g = g / loss_scaling
            #     master_params_grads.append([p, scaled_g])
            #     continue
            master_grad = fluid.layers.cast(g, "float32")
            master_grad = master_grad / loss_scaling
            master_params_grads.append([master_param, master_grad])

    return master_params_grads


def update_loss_scaling(is_overall_finite, prev_loss_scaling, num_good_steps,
                        num_bad_steps, incr_every_n_steps,
                        decr_every_n_nan_or_inf, incr_ratio, decr_ratio):
    """
    Update loss scaling according to overall gradients. If all gradients is
    finite after incr_every_n_steps, loss scaling will increase by incr_ratio.
    Otherwisw, loss scaling will decrease by decr_ratio after
    decr_every_n_nan_or_inf steps and each step some gradients are infinite.
    Args:
        is_overall_finite (Variable): A boolean variable indicates whether
                                     all gradients are finite.
        prev_loss_scaling (Variable): Previous loss scaling.
        num_good_steps (Variable): A variable accumulates good steps in which
                                   all gradients are finite.
        num_bad_steps (Variable): A variable accumulates bad steps in which
                                  some gradients are infinite.
        incr_every_n_steps (Variable): A variable represents increasing loss
                                       scaling every n consecutive steps with
                                       finite gradients.
        decr_every_n_nan_or_inf (Variable): A variable represents decreasing
                                            loss scaling every n accumulated
                                            steps with nan or inf gradients.
        incr_ratio(float): The multiplier to use when increasing the loss
                           scaling.
        decr_ratio(float): The less-than-one-multiplier to use when decreasing
                           loss scaling.
    """
    zero_steps = fluid.layers.fill_constant(shape=[1], dtype='int32', value=0)
    with fluid.layers.Switch() as switch:
        with switch.case(is_overall_finite):
            should_incr_loss_scaling = fluid.layers.less_than(
                incr_every_n_steps, num_good_steps + 1)
            with fluid.layers.Switch() as switch1:
                with switch1.case(should_incr_loss_scaling):
                    new_loss_scaling = prev_loss_scaling * incr_ratio
                    loss_scaling_is_finite = fluid.layers.isfinite(
                        new_loss_scaling)
                    with fluid.layers.Switch() as switch2:
                        with switch2.case(loss_scaling_is_finite):
                            fluid.layers.assign(new_loss_scaling,
                                                prev_loss_scaling)
                        with switch2.default():
                            pass
                    fluid.layers.assign(zero_steps, num_good_steps)
                    fluid.layers.assign(zero_steps, num_bad_steps)

                with switch1.default():
                    fluid.layers.increment(num_good_steps)
                    fluid.layers.assign(zero_steps, num_bad_steps)

        with switch.default():
            should_decr_loss_scaling = fluid.layers.less_than(
                decr_every_n_nan_or_inf, num_bad_steps + 1)
            with fluid.layers.Switch() as switch3:
                with switch3.case(should_decr_loss_scaling):
                    new_loss_scaling = prev_loss_scaling * decr_ratio
                    static_loss_scaling = \
                        fluid.layers.fill_constant(shape=[1],
                                                   dtype='float32',
                                                   value=1.0)
                    less_than_one = fluid.layers.less_than(new_loss_scaling,
                                                           static_loss_scaling)
                    with fluid.layers.Switch() as switch4:
                        with switch4.case(less_than_one):
                            fluid.layers.assign(static_loss_scaling,
                                                prev_loss_scaling)
                        with switch4.default():
                            fluid.layers.assign(new_loss_scaling,
                                                prev_loss_scaling)
                    fluid.layers.assign(zero_steps, num_good_steps)
                    fluid.layers.assign(zero_steps, num_bad_steps)
                with switch3.default():
                    fluid.layers.assign(zero_steps, num_good_steps)
                    fluid.layers.increment(num_bad_steps)



def apply_dynamic_loss_scaling(loss_scaling, master_params_grads,
                               incr_every_n_steps, decr_every_n_nan_or_inf,
                               incr_ratio, decr_ratio):
    """ apply_dynamic_loss_scaling """
    _incr_every_n_steps = fluid.layers.fill_constant(
        shape=[1], dtype='int32', value=incr_every_n_steps)
    _decr_every_n_nan_or_inf = fluid.layers.fill_constant(
        shape=[1], dtype='int32', value=decr_every_n_nan_or_inf)

    _num_good_steps = fluid.layers.create_global_var(
        name=fluid.unique_name.generate("num_good_steps"),
        shape=[1],
        value=0,
        dtype='int32',
        persistable=True)
    _num_bad_steps = fluid.layers.create_global_var(
        name=fluid.unique_name.generate("num_bad_steps"),
        shape=[1],
        value=0,
        dtype='int32',
        persistable=True)

    grads = [fluid.layers.reduce_sum(g) for [_, g] in master_params_grads]
    all_grads = fluid.layers.concat(grads)
    all_grads_sum = fluid.layers.reduce_sum(all_grads)
    is_overall_finite = fluid.layers.isfinite(all_grads_sum)

    update_loss_scaling(is_overall_finite, loss_scaling, _num_good_steps,
                        _num_bad_steps, _incr_every_n_steps,
                        _decr_every_n_nan_or_inf, incr_ratio, decr_ratio)

    # apply_gradient append all ops in global block, thus we shouldn't
    # apply gradient in the switch branch.
    with fluid.layers.Switch() as switch:
        with switch.case(is_overall_finite):
            pass
        with switch.default():
            for _, g in master_params_grads:
                fluid.layers.assign(fluid.layers.zeros_like(g), g)

def master_param_to_train_param(master_params_grads, params_grads, main_prog):
    """ master_param_to_train_param """
    for idx, m_p_g in enumerate(master_params_grads):
        train_p, _ = params_grads[idx]
        with main_prog._optimized_guard([m_p_g[0], m_p_g[1]]):
            if train_p.name.find("layer_norm") > -1:
                fluid.layers.assign(m_p_g[0], train_p)
            else:
                append_cast_op(m_p_g[0], train_p, main_prog)

def optimization(loss,
                 warmup_steps,
                 num_train_steps,
                 learning_rate,
                 train_program,
                 startup_prog,
                 weight_decay,
                 scheduler='linear_warmup_decay',
                 use_fp16=False,
                 use_dynamic_loss_scaling=False,
                 init_loss_scaling=1.0,
                 incr_every_n_steps=1000,
                 decr_every_n_nan_or_inf=2,
                 incr_ratio=2.0,
                 decr_ratio=0.8,
                 dist_strategy=None):
    """ optimization """
    if warmup_steps > 0:
        if scheduler == 'noam_decay':
            scheduled_lr = fluid.layers.learning_rate_scheduler \
                .noam_decay(1 / (warmup_steps * (learning_rate ** 2)),
                            warmup_steps)
        elif scheduler == 'linear_warmup_decay':
            scheduled_lr = linear_warmup_decay(learning_rate, warmup_steps,
                                               num_train_steps)
        else:
            raise ValueError("Unkown learning rate scheduler, should be "
                             "'noam_decay' or 'linear_warmup_decay'")
        optimizer = fluid.optimizer.Adam(learning_rate=scheduled_lr, epsilon=1e-06)
        # optimizer = fluid.optimizer.Adam(learning_rate=scheduled_lr)
    else:
        scheduled_lr = fluid.layers.create_global_var(
            name=fluid.unique_name.generate("learning_rate"),
            shape=[1],
            value=learning_rate,
            dtype='float32',
            persistable=True)
        optimizer = fluid.optimizer.Adam(learning_rate=scheduled_lr, epsilon=1e-06)
        # optimizer = fluid.optimizer.Adam(learning_rate=scheduled_lr)
        optimizer._learning_rate_map[fluid.default_main_program(
        )] = scheduled_lr

    fluid.clip.set_gradient_clip(
        clip=fluid.clip.GradientClipByGlobalNorm(clip_norm=1.0))

    def exclude_from_weight_decay(name):
        """ exclude_from_weight_decay """
        if name.find("layer_norm") > -1:
            return True
        bias_suffix = ["_bias", "_b", ".b_0"]
        for suffix in bias_suffix:
            if name.endswith(suffix):
                return True
        return False

    param_list = dict()

    loss_scaling = fluid.layers.create_global_var(
        name=fluid.unique_name.generate("loss_scaling"),
        shape=[1],
        value=init_loss_scaling,
        dtype='float32',
        persistable=True)

    if use_fp16:
        loss *= loss_scaling
        param_grads = optimizer.backward(loss)

        master_param_grads = create_master_params_grads(
            param_grads, train_program, startup_prog, loss_scaling)

        for param, _ in master_param_grads:
            param_list[param.name] = param * 1.0
            param_list[param.name].stop_gradient = True

        if use_dynamic_loss_scaling:
            apply_dynamic_loss_scaling(
                loss_scaling, master_param_grads, incr_every_n_steps,
                decr_every_n_nan_or_inf, incr_ratio, decr_ratio)

        optimizer.apply_gradients(master_param_grads)

        if weight_decay > 0:
            for param, grad in master_param_grads:
                if exclude_from_weight_decay(param.name.rstrip(".master")):
                    continue
                with param.block.program._optimized_guard(
                        [param, grad]), fluid.framework.name_scope("weight_decay"):
                    updated_param = param - param_list[
                        param.name] * weight_decay * scheduled_lr
                    fluid.layers.assign(output=param, input=updated_param)

        master_param_to_train_param(master_param_grads, param_grads,
                                    train_program)

    else:
        for param in train_program.global_block().all_parameters():
            param_list[param.name] = param * 1.0
            param_list[param.name].stop_gradient = True

        if dist_strategy is not None:
            optimizer = fleet.distributed_optimizer(optimizer, strategy=dist_strategy)

        _, param_grads = optimizer.minimize(loss)

        if weight_decay > 0:
            for param, grad in param_grads:
                if exclude_from_weight_decay(param.name):
                    continue
                with param.block.program._optimized_guard(
                        [param, grad]), fluid.framework.name_scope("weight_decay"):
                    updated_param = param - param_list[
                        param.name] * weight_decay * scheduled_lr
                    fluid.layers.assign(output=param, input=updated_param)
    result = collections.OrderedDict()
    result['scheduled_lr'] = scheduled_lr
    if use_fp16:
        result['loss_scaling'] = loss_scaling
    return result

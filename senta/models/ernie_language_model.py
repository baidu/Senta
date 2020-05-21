# -*- coding: utf-8 -*
"""
Erniexx Language Model
"""
import collections
from senta.common.rule import InstanceName
from senta.modules.ernie import ErnieModel
from senta.common.register import RegisterSet
from senta.models.model import Model


@RegisterSet.models.register
class ErnieLM(Model):
    """ErnieLM"""
    def __init__(self, model_params, args, task_group):
        # tricky code because base class Model need dict as first parameter
        model_params.print_config()
        self.task_group = task_group

        super(ErnieSKEPMTLM, self).__init__(model_params)
        self.args = args
        self.scheduled_lr = None
        self.loss_scaling = None

    def forward(self, fields_dict, phase):
        """
        forward calculate
        """
        src_ids = fields_dict['src_ids']
        pos_ids = fields_dict['pos_ids']
        sent_ids = fields_dict['sent_ids']
        task_ids = fields_dict['task_ids']
        input_mask = fields_dict['input_mask']
        mask_label = fields_dict['mask_label']
        mask_pos = fields_dict['mask_pos']
        lm_weight = fields_dict['lm_weight']
        #senti_pos = fields_dict['senti_pos']
        #senti_pol = fields_dict['senti_pol']
        #pair_label = fields_dict['pair_label']

        pretrain_ernie = ErnieModel(
            src_ids=src_ids,
            position_ids=pos_ids,
            sentence_ids=sent_ids,
            task_ids=task_ids,
            input_mask=input_mask,
            config=self.model_params,
            weight_sharing=self.args.weight_sharing,
            use_fp16=self.args.use_fp16)

        result = collections.OrderedDict()

        mask_lm_loss = pretrain_ernie.get_lm_output(mask_label, mask_pos)
        total_loss = mask_lm_loss * lm_weight

        #senti_pol_loss = pretrain_ernie.get_senti_output(senti_pol, senti_pos)
        #pair_loss = pretrain_ernie.get_pair_output(pair_label)
        #total_loss += senti_pol_loss
        #total_loss += pair_loss
        
        result['mask_lm_loss'] = mask_lm_loss
        result['lm_weight'] = lm_weight

        #result['senti_pol_loss'] = senti_pol_loss
        #result['pair_loss'] = pair_loss

        for task in self.task_group:
            task_labels = fields_dict[task['task_name']]
            task_weight = fields_dict[task['task_name'] + '_weight']
            task_loss, task_acc = pretrain_ernie.get_task_output(task, task_labels)
            total_loss += task_loss * task_weight * task["loss_weight"]
            task_acc.persistable = True
            task_weight.persistable = True
            result['acc_' + task['task_name']] = task_acc
            result['task_weight_of_' + task['task_name']] = task_weight

        result[InstanceName.LOSS] = total_loss
        return result

    def optimizer(self, loss, is_fleet=False):
        """
        optimizer
        """
        optimizer_output_dict = collections.OrderedDict()
        optimizer_output_dict['use_ernie_opt'] = True

        opt_args_dict = collections.OrderedDict()
        opt_args_dict["loss"] = loss
        opt_args_dict["warmup_steps"] = self.args.warmup_steps
        opt_args_dict["num_train_steps"] = self.args.num_train_steps
        opt_args_dict["learning_rate"] = self.args.learning_rate
        opt_args_dict["weight_decay"] = self.args.weight_decay
        opt_args_dict["scheduler"] = self.args.lr_scheduler
        opt_args_dict["use_fp16"] = self.args.use_fp16
        opt_args_dict["use_dynamic_loss_scaling"] = self.args.use_dynamic_loss_scaling
        opt_args_dict["init_loss_scaling"] = self.args.init_loss_scaling
        opt_args_dict["incr_every_n_steps"] = self.args.incr_every_n_steps
        opt_args_dict["decr_every_n_nan_or_inf"] = self.args.decr_every_n_nan_or_inf
        opt_args_dict["incr_ratio"] = self.args.incr_ratio
        opt_args_dict["decr_ratio"] = self.args.decr_ratio

        optimizer_output_dict["opt_args"] = opt_args_dict

        return optimizer_output_dict



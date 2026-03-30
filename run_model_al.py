# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Running BERT finetuning & evaluation on hate speech classification datasets.

Integrated with SOC explanation regularization

"""

from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import sys
import json

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset, Subset)
from torch.utils.data.distributed import DistributedSampler
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm, trange

from torch.nn import CrossEntropyLoss, MSELoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score

from bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from bert.modeling import BertForSequenceClassification, BertConfig, MultiTaskBert, MultiTaskBertList
from bert.tokenization import BertTokenizer
from bert.optimization import BertAdam, WarmupLinearSchedule

from loader import GabProcessor, WSProcessor, NytProcessor, convert_examples_to_features, GabAnnotatorProcessor, GabMajorityProcessor, GabAnnotatorListProcessor
from utils.config import configs, combine_args

# for hierarchical explanation algorithms
from hiex import SamplingAndOcclusionExplain

from utils.apply_acquisition import apply_acquisition_function

from torchsampler import ImbalancedDatasetSampler
# import torch.optim.lr_scheduler as lr_scheduler
from mlflow import log_metric, log_param
import mlflow
import socket
logger = logging.getLogger(__name__)

def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels, pred_probs, task_name, pred_heads, vars):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    p, r = precision_score(y_true=labels, y_pred=preds), recall_score(y_true=labels, y_pred=preds)
    roc = 0.
    if task_name != 'gab_anno' and task_name != 'gab_anno_list':
        try:
            roc = roc_auc_score(y_true=labels, y_score=pred_probs[:,1])
        except ValueError:
            roc = 0.
    return {
        "acc": acc,
        "f1": f1,
        "precision": p,
        "recall": r,
        "auc_roc": roc
    }


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def compute_metrics(task_name, preds, labels, pred_probs, pred_heads, vars_gt):
    assert len(preds) == len(labels)
    # create a dict to store the results of the evaluation
    actual_metrics = {}
    actual_metrics = acc_and_f1(preds, labels, pred_probs, task_name, pred_heads, vars_gt)

    if pred_heads is not None:
        if len(pred_heads) > 0:
            #calculate the variance of the predictions of the pred_heads
            pred_var = np.var(pred_heads[0], axis=0)
            # calculate the correlation between the variance and the variance of the predictions
            var_corr = np.corrcoef(pred_var, vars_gt)[0,1]
            actual_metrics['pearson'] = var_corr
        else:
            # calculate the softmax of the pred logits
            # pred_probs = F.softmax(torch.from_numpy(pred_probs), dim=1).numpy()
            # take the predicted one which has the highest probability
            pred_var = np.max(pred_probs, axis=1)
            # calculate the correlation between the variance and the variance of the predictions
            var_corr = np.corrcoef(pred_var, vars_gt)[0,1]
            actual_metrics['pearson'] = var_corr
    else:
        actual_metrics['pearson'] = 0.
    return actual_metrics

class CustomTensorDataset(TensorDataset):
    def filter(self, indices):
        self.tensors = tuple(tensor for tensor in self.tensors if tensor['ID'] not in indices)
    def seletect(self, indices):
        return CustomTensorDataset(*[tensor[indices] for tensor in self.tensors])
def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--negative_weight", default=1., type=float)
    parser.add_argument("--neutral_words_file", default='data/identity.csv')

    # if true, use test data instead of val data
    parser.add_argument("--test", action='store_true')

    # Explanation specific arguments below

    # whether run explanation algorithms
    parser.add_argument("--explain", action='store_true', help='if true, explain test set predictions')
    parser.add_argument("--debug", action='store_true')

    # which algorithm to run
    parser.add_argument("--algo", choices=['soc'])

    # the output filename without postfix
    parser.add_argument("--output_filename", default='temp.tmp')

    # see utils/config.py
    parser.add_argument("--use_padding_variant", action='store_true')
    parser.add_argument("--mask_outside_nb", action='store_true')
    parser.add_argument("--nb_range", type=int)
    parser.add_argument("--sample_n", type=int)

    # whether use explanation regularization
    parser.add_argument("--reg_explanations", action='store_true')
    parser.add_argument("--reg_strength", type=float)
    parser.add_argument("--reg_mse", action='store_true')

    # whether discard other neutral words during regularization. default: False
    parser.add_argument("--discard_other_nw", action='store_false', dest='keep_other_nw')

    # whether remove neutral words when loading datasets
    parser.add_argument("--remove_nw", action='store_true')

    # if true, generate hierarchical explanations instead of word level outputs.
    # Only useful when the --explain flag is also added.
    parser.add_argument("--hiex", action='store_true')
    parser.add_argument("--hiex_tree_height", default=5, type=int)

    # whether add the sentence itself to the sample set in SOC
    parser.add_argument("--hiex_add_itself", action='store_true')

    # the directory where the lm is stored
    parser.add_argument("--lm_dir", default='runs/lm')

    # if configured, only generate explanations for instances with given line numbers
    parser.add_argument("--hiex_idxs", default=None)
    # if true, use absolute values of explanations for hierarchical clustering
    parser.add_argument("--hiex_abs", action='store_true')

    # if either of the two is true, only generate explanations for positive / negative instances
    parser.add_argument("--only_positive", action='store_true')
    parser.add_argument("--only_negative", action='store_true')

    # stop after generating x explanation
    parser.add_argument("--stop", default=100000000, type=int)

    # early stopping with decreasing learning rate. 0: direct exit when validation F1 decreases
    parser.add_argument("--early_stop", default=5, type=int)

    # other external arguments originally here in pytorch_transformers

    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--eval_step', type=int, default=200, help="Number of evaluation steps")
    parser.add_argument('--fold', type=int, default=1, help="evaluation fold")
    parser.add_argument('--eval_mode', type=str, default='majority', help="evaluation mode: majority, annotation")
    parser.add_argument('--over_sampling', action='store_true', help="whether to use over sampling")
    parser.add_argument('--rounds', type=int, default=1, help="round of active learning training")
    parser.add_argument('--query_sample_size', type=int, default=100, help="number of samples to query for each round")
    parser.add_argument('--init_size', type=int, default=100, help="number of initial samples")
    parser.add_argument('--active_strategy', type=str, default='entropy', help="active learning strategy")
    parser.add_argument('--experiment_name', type=str, default='default', help="experiment name")
    parser.add_argument('--run_name', type=str, default='default_runName', help="run name")
    parser.add_argument('--num_heads', type=int, default=1, help="number of heads for classification")
    parser.add_argument('--lr_halv_f1', action='store_true', help="whether to half learning rate when f1 decreases")
    parser.add_argument('--halve_lr_after', type=int, default=3, help="halve learning rate after x epochs")
    parser.add_argument('--num_labels', type=int, default=2, help="number of labels")
    parser.add_argument('--dal_learning_rate', type=float, default=0.01, help="learning rate for DAL")
    parser.add_argument('--dal_epochs', type=int, default=10, help="number of epochs for DAL")
    parser.add_argument('--dal_weight_decay', type=float, default=0.01, help="weight decay for DAL")
    parser.add_argument('--T', type=int, default= 10, help="iteration for BALD")
    parser.add_argument('--sampling_strategy', type=str, default=None, help="sampling strategy")
    parser.add_argument('--class_weight', action='store_true', default=False, help="whether to use class weight")
    parser.add_argument('--error_test',   action='store_true', default=False)
    
    args = parser.parse_args()

    combine_args(configs, args)
    args = configs

    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # if mlflow experiment name has not been created yet, create one
    if mlflow.get_experiment_by_name(args.experiment_name) is None:
        experimentID = mlflow.create_experiment(args.experiment_name)
    else:
        experimentID = mlflow.get_experiment_by_name(args.experiment_name).experiment_id
    mlflow.set_experiment(args.experiment_name)
    mlflow.autolog()
    mlflow.log_params(vars(args))
    # set mlflow run name
    mlflow.set_tag('mlflow.runName', args.run_name)


    processors = {
        'gab': GabProcessor,
        'ws': WSProcessor,
        'nyt':  NytProcessor, 
        'gab_anno': GabAnnotatorProcessor, 
        'gab_maj': GabMajorityProcessor,
        'gab_anno_list': GabAnnotatorListProcessor, 
    }

    output_modes = {
        'gab': 'classification',
        'ws': 'classification',
        'nyt': 'classification',
        'gab_anno': 'classification',
        'gab_maj': 'classification',
        'gab_anno_list': 'multitask_classification',
    }

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    #if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
    #    raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # save configs
    f = open(os.path.join(args.output_dir, 'args.json'), 'w')
    json.dump(args.__dict__, f, indent=4)
    f.close()

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    processor= processors[task_name](configs, tokenizer=tokenizer)
    output_mode = output_modes[task_name]

    label_list = processor.get_labels()
    num_labels = len(label_list)

    train_examples = None
    num_train_optimization_steps = None
    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir, fold=args.fold)

    if args.do_train:
        
        train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer, output_mode, configs)
        
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        if args.sampling_strategy is not None:
             all_text_ids = torch.tensor([f.text_id for f in train_features], dtype=torch.long)

        if task_name == 'gab_anno':
            all_annotator_ids = torch.tensor([f.annotator_id for f in train_features], dtype=torch.long)
        if output_mode == "classification":
            all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        elif output_mode == "multitask_classification":
            all_annotation_list = torch.tensor([f.annotations for f in train_features], dtype=torch.long)
            all_annotator_list = torch.tensor([f.annotators for f in train_features], dtype=torch.long)
        elif output_mode == "regression":
            all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.float)

        if task_name == 'gab_anno':
            train_data = CustomTensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_annotator_ids, all_text_ids)
        elif task_name == "gab_anno_list":
            train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_annotation_list, all_annotator_list) 
        else:
            if args.sampling_strategy is not None:
                train_data = CustomTensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_text_ids)
            else:
                train_data = CustomTensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

        #TODO: make this Dataset class to support query, unlabelled data, labelled data. 
        num_labels = 2#len(set(all_label_ids.tolist()))
        run_active_learning(args, train_data, device, output_mode, n_gpu, processor, tokenizer, num_labels)
    

def train(args, train_data, device, task_name, output_mode, n_gpu, processor, tokenizer, rounds, num_labels):
    # start a mlflow child run
    with mlflow.start_run(run_name='round_{}'.format(rounds), nested=True):
        # Log the parameters used in this round
        mlflow.log_params(args.__dict__)
        # log the rounds number
        mlflow.log_param('round', rounds)

        #TODO: rewrite this part to make it more general
        if task_name == 'gab_anno':
            all_label_ids = torch.tensor([data[-3] for data in train_data])
        elif task_name == "gab_anno_list":
            # concat all the annotations list into a list
            all_label_ids = [data[3] for data in train_data]
            all_label_ids = torch.cat(all_label_ids, dim=0)
        else:
            if args.error_test:
                all_label_ids = torch.tensor([data[-1] for data in train_data])
            else:
                all_label_ids = torch.tensor([data[-2] for data in train_data])
        label_list = processor.get_labels()
        # label_list = all_label_ids.unique().tolist()
            # Prepare model
        cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE),
                                                                    'distributed_{}'.format(args.local_rank))
        num_train_optimization_steps = int(
        len(train_data) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_data))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        

        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

        #TODO: rewrite this part, make it depend on specific strategy, not task name
        if task_name == 'gab_anno':
            model = MultiTaskBert.from_pretrained(args.bert_model, cache_dir=cache_dir, num_labels=num_labels, num_heads=args.num_heads)
        elif task_name == 'gab_anno_list':
            model = MultiTaskBertList.from_pretrained(args.bert_model, cache_dir=cache_dir, num_labels=num_labels, num_heads=args.num_heads)
        else:
            model = BertForSequenceClassification.from_pretrained(args.bert_model,
                                                                cache_dir=cache_dir,
                                                                num_labels=num_labels)
        model.to(device)

        if args.fp16:
            model.half()

        if args.local_rank != -1:
            try:
                from apex.parallel import DistributedDataParallel as DDP
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

            model = DDP(model)

    # Prepare optimizer
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        if args.fp16:
            try:
                from apex.optimizers import FP16_Optimizer
                from apex.optimizers import FusedAdam
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

            optimizer = FusedAdam(optimizer_grouped_parameters,
                                lr=args.learning_rate,
                                bias_correction=False,
                                max_grad_norm=1.0)
            if args.loss_scale == 0:
                optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
            else:
                optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
            warmup_linear = WarmupLinearSchedule(warmup=args.warmup_proportion,
                                                t_total=num_train_optimization_steps)

        else:
            if args.do_train:
                optimizer = BertAdam(optimizer_grouped_parameters,
                                    lr=args.learning_rate,
                                    warmup=args.warmup_proportion,
                                    t_total=num_train_optimization_steps)

        

        if args.do_train and not args.lr_halv_f1:
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1 - (step / num_train_optimization_steps))

        explainer = None

        global_step = 0
        nb_tr_steps = 0
        tr_loss, tr_reg_loss = 0, 0
        tr_reg_cnt = 0
        epoch = -1
        val_best_f1 = -1
        val_best_pearson = -1
        val_best_loss = 1e10
        early_stop_countdown = args.early_stop


        
        

        epoch = 0
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        if args.over_sampling :
            train_dataloader = DataLoader(train_data, sampler=ImbalancedDatasetSampler(train_data, labels=all_label_ids), batch_size=args.train_batch_size, )
        else:
            train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)


        # count the number of positive and negative samples in the all_label_ids list
        pos_cnt = 0
        neg_cnt = 0
        for i in all_label_ids:
            if i == 0:
                neg_cnt += 1
            else:
                pos_cnt += 1
        if args.class_weight:
            class_weight = torch.FloatTensor([ 1 - neg_cnt/len(all_label_ids), 1 - pos_cnt/len(all_label_ids)]).to(device)
        else:
            class_weight = torch.FloatTensor([1, 1]).to(device)
        model.train()
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                if task_name != 'gab_anno' and task_name != 'gab_anno_list':
                    if args.sampling_strategy is not None:
                        input_ids, input_mask, segment_ids, label_ids, text_ids = batch
                    else:
                        input_ids, input_mask, segment_ids, label_ids = batch

                    # define a new function to compute loss values for both output_modes
                    logits = model(input_ids, segment_ids, input_mask, labels=None)

                    if output_mode == "classification":
                        loss_fct = CrossEntropyLoss(class_weight)
                        loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
                    elif output_mode == "regression":
                        loss_fct = MSELoss()
                        loss = loss_fct(logits.view(-1), label_ids.view(-1))
                elif task_name == 'gab_anno_list':
                    input_ids, input_mask, segment_ids, annotation_list, annotator_list = batch
                    loss_fct = CrossEntropyLoss(class_weight)
                    loss = model(input_ids, segment_ids, input_mask, annotations=annotation_list, annotator_ids=annotator_list, loss_fct=loss_fct)
                else:
                    input_ids, input_mask, segment_ids, label_ids, annotator_ids, text_ids = batch
                    loss_fct = CrossEntropyLoss(class_weight)
                    loss = model(input_ids, segment_ids, input_mask, labels=label_ids, annotator_ids=annotator_ids, loss_fct=loss_fct)

                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                tr_loss += loss.item()
                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()



                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used that handles this automatically
                        lr_this_step = args.learning_rate * warmup_linear.get_lr(
                            global_step / num_train_optimization_steps,
                            args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
                    if not args.lr_halv_f1:
                        lr_scheduler.step()

                if (global_step+1) % args.eval_step == 0:
                    val_result = validate(args, model, processor, tokenizer, output_mode, label_list, device,
                                            num_labels,
                                            task_name, tr_loss, global_step, epoch, explainer, rounds)
                    val_acc, val_f1, pearson = val_result['acc'], val_result['f1'], val_result['pearson']
                    if pearson > val_best_pearson:
                        val_best_pearson = pearson

                    if val_f1 > val_best_f1:
                        val_best_f1 = val_f1
                        if args.local_rank == -1 or torch.distributed.get_rank() == 0:
                                save_model(args, model, tokenizer, num_labels)
                    elif args.lr_halv_f1:
                        if epoch > args.halve_lr_after:
                        # halve the learning rate
                            for param_group in optimizer.param_groups:
                                param_group['lr'] *= 0.5
                            early_stop_countdown -= 1
                            logger.info("Reducing learning rate... Early stop countdown %d" % early_stop_countdown)
                                
                            if early_stop_countdown < 0:
                                break
                    log_metric('val_pearson', pearson, step=global_step)
                    log_metric('val_best_pearson', val_best_pearson, step=global_step)
                    log_metric('val_acc', val_acc, step=global_step)
                    log_metric('val_f1', val_f1, step=global_step)
                    log_metric('val_best_f1', val_best_f1, step=global_step)
                    log_metric('val_loss', val_result['eval_loss'], step=epoch)
                log_metric('train_loss', loss, step=global_step)
                log_metric('lr', optimizer.param_groups[0]['lr'], step=global_step)
                log_metric('total_steps', global_step, step=global_step)
            

            val_result = validate(args, model, processor, tokenizer, output_mode, label_list, device,
                                        num_labels,
                                        task_name, tr_loss, global_step, epoch, explainer, rounds)
            val_acc, val_f1, pearson = val_result['acc'], val_result['f1'], val_result['pearson']
            log_metric('val_acc_epoch', val_acc, step=global_step)
            log_metric('val_f1_epoch', val_f1, step=global_step)
            log_metric('val_loss_epoch', val_result['eval_loss'], step=epoch)
            log_metric('train_loss_epoch', tr_loss, step=epoch)
            log_metric('val_pearson_epoch', pearson, step=global_step)
            epoch += 1
        

        if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):

            validate(args, model, processor, tokenizer, output_mode, label_list, device, num_labels,
                            task_name, tr_loss, global_step=global_step, epoch=-1, explainer=explainer, rounds=rounds)

        
    return model


   

def run_active_learning(args, train_data, device, output_mode, n_gpu, processor, tokenizer, num_labels):
    pool_size = len(train_data)
    # set initial training dazta, unlabeled training data
    labeled_idxs_bool = np.zeros(pool_size, dtype=bool)
    # create random indices for initial training data 
    init_idxs = np.arange(0, args.init_size)
    labeled_idxs_bool[init_idxs] = True
    
    
    
    # start active learning loop
    for i in range(args.rounds):
        # create labeled and unlabeled datasets
        labeled_idxs = np.arange(pool_size)[labeled_idxs_bool]
        labeled_dataset = Subset(train_data, labeled_idxs)

        unlabeled_idxs = np.arange(pool_size)[~labeled_idxs_bool]
        unlabeled_dataset = Subset(train_data, unlabeled_idxs)
        
        model = train(args, labeled_dataset, device, args.task_name, output_mode, n_gpu, processor, tokenizer, i, num_labels)

        top_k_samples_idxs, labeled_dataset, unlabeled_dataset= apply_acquisition_function(args, model,unlabeled_idxs, unlabeled_dataset, device, labeled_dataset, labeled_idxs)

        # add top k samples to labeled dataset
        labeled_idxs_bool[top_k_samples_idxs] = True
    return None

def validate(args, model, processor, tokenizer, output_mode, label_list, device, num_labels,
             task_name, tr_loss, global_step, epoch, explainer=None, rounds=0):
    if not args.test:
        eval_examples = processor.get_dev_examples(args.data_dir, fold=args.fold)
    else:
        eval_examples = processor.get_test_examples(args.data_dir)
    eval_features = convert_examples_to_features(
        eval_examples, label_list, args.max_seq_length, tokenizer, output_mode, configs)
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_var = torch.tensor([f.var for f in eval_features], dtype=torch.float)

    if output_mode == "classification" or output_mode == "multitask_classification":
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.float)

    if task_name == 'gab_anno' and args.eval_mode == 'annotation':
        all_annotator_ids = torch.tensor([f.annotator_id for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_annotator_ids)
    else:
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    model.train(False)
    eval_loss, eval_loss_reg = 0, 0
    eval_reg_cnt = 0
    nb_eval_steps = 0
    preds = []
    pred_labels = []
    # for detailed prediction results
    input_seqs = []

    pred_softmax = []
    pred_heads = []

    preds_anno_all = []
    labels_new_all = []
    # create eval loss and other metric required by the task
    if output_mode == "classification" or output_mode == "multitask_classification":
        loss_fct = CrossEntropyLoss()
    elif output_mode == "regression":
        loss_fct = MSELoss()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(device) for t in batch)

        if task_name != 'gab_anno' and task_name != 'gab_anno_list':
            input_ids, input_mask, segment_ids, label_ids = batch
            with torch.no_grad():
                logits = model(input_ids, segment_ids, input_mask, labels=None)
                tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))

        elif task_name == 'gab_anno_list':
            input_ids, input_mask, segment_ids, label_ids = batch
            with torch.no_grad():
                tmp_eval_loss, preds_anno, logits = model(input_ids, segment_ids, input_mask, labels=label_ids,  loss_fct=loss_fct, pred_mode=args.eval_mode, pred=True)
        else: # anno way

            if args.eval_mode == 'annotation':
                input_ids, input_mask, segment_ids, label_ids, annotator_ids = batch
                
                with torch.no_grad():
                    tmp_eval_loss, preds_anno, labels_new = model(input_ids, segment_ids, input_mask, labels=label_ids, annotator_ids=annotator_ids, loss_fct=loss_fct,  pred_mode=args.eval_mode, pred=True)
            
                
                
                

            else:
                input_ids, input_mask, segment_ids, label_ids = batch
                with torch.no_grad():
                    tmp_eval_loss, preds_anno, logits_all, pred_all = model(input_ids, segment_ids, input_mask, labels=label_ids, loss_fct=loss_fct, pred_mode=args.eval_mode, pred=True)


        eval_loss += tmp_eval_loss.mean().item()

        nb_eval_steps += 1

        ### classic ###
        if task_name != 'gab_anno'  and task_name != 'gab_anno_list':
            if len(preds) == 0:
                # logits.shape == [32, 2]
                preds.append(logits.detach().cpu().numpy())

            else:
                preds[0] = np.append(preds[0], logits.detach().cpu().numpy(), axis=0)
        elif args.eval_mode == 'annotation':
            # cumulate the preds_anno and labels_new
            preds_anno_all.append(preds_anno)
            labels_new_all.append(labels_new)
            pred_labels = None
            pred_heads = None
        else:
            if len(pred_labels) == 0:
                pred_labels.append(preds_anno)
                pred_heads.append(pred_all)

            else:
                pred_labels[0] = np.append(pred_labels[0], preds_anno, axis=1)
                pred_heads[0] = np.append(pred_heads[0], pred_all, axis=1)
        ### annotation ###
        for b in range(input_ids.size(0)):
            i = 0
            while i < input_ids.size(1) and input_ids[b,i].item() != 0:
                i += 1
            token_list = tokenizer.convert_ids_to_tokens(input_ids[b,:i].cpu().numpy().tolist())
            input_seqs.append(' '.join(token_list))


    eval_loss = eval_loss / nb_eval_steps
    eval_loss_reg = eval_loss_reg / (eval_reg_cnt + 1e-10)

    ###  classic ###
    if task_name != 'gab_anno' and task_name != 'gab_anno_list':
        preds = preds[0]
        if output_mode == "classification":
            pred_labels = np.argmax(preds, axis=1)
        elif output_mode == "regression":
            pred_labels = np.squeeze(preds)
        pred_prob = F.softmax(torch.from_numpy(preds).float(), -1).numpy()
    elif args.eval_mode == 'annotation':
        # calculate the f1 using pred_anno_all and labels_new_all
        pred_labels = np.concatenate(preds_anno_all, axis=0)
        all_label_ids = torch.tensor(np.concatenate(labels_new_all, axis=0))
        pred_prob = None
        # f1 = f1_score(all_label_ids, pred_labels)
        # log_metric('val_f1', f1, global_step)  
    else:
    ### annotation ###
        pred_prob=None
        pred_labels = pred_labels[0].reshape(-1)
        # pred_heads = pred_heads[0].reshape(-1)
        # pred_labels  = np.append(pred_labels, )
    
    result = compute_metrics(task_name, pred_labels, all_label_ids.numpy(), pred_prob, pred_heads, all_var)
    loss = tr_loss / (global_step + 1e-10) if args.do_train else None

    result['eval_loss'] = eval_loss
    result['eval_loss_reg'] = eval_loss_reg
    result['global_step'] = global_step
    result['loss'] = loss

    
    split = 'dev' if not args.test else 'test'

    output_eval_file = os.path.join(args.output_dir, "eval_results_%d_%s_%s_rounds_%s.txt"
                                    % (global_step, split, args.task_name, rounds))
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results *****")
        logger.info("Epoch %d" % epoch)
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    output_detail_file = os.path.join(args.output_dir, "eval_details_%d_%s_%s_rounds_%s.txt"
                                    % (global_step, split, args.task_name, rounds))

    model.train(True)
    return result



def save_model(args, model, tokenizer, num_labels):
    # Save a trained model, configuration and tokenizer
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

    # If we save using the predefined names, we can load using `from_pretrained`
    output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(args.output_dir)

if __name__ == "__main__":
    main()

import os
import argparse
from pathlib import Path
from ruamel.yaml import YAML
import mlflow
from datetime import datetime

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

now = datetime.now()
current_time = now.strftime("%m_%d_%Y_%R")


def brexit_active_anno(args):
    method = args.active_strategy
    seed = args.seed
    run_name = f'Brexit_anno_act_method_{method}_seed_{seed}_{current_time}'
    single_run('config/anno_act.yaml', args, run_name)


def baselines_maj(args):
    method = args.active_strategy
    seed = args.seed
    run_name = f'gab_{method}_seed_{seed}_{current_time}'
    single_run('config/maj_act.yaml', args, run_name)


def start_expriement(args):
    if args.mode == 'baseline':
        baselines_maj(args)
    elif args.mode == 'anno':
        brexit_active_anno(args)

def single_run(config_path, 
                args,
                run_name):
    yaml = YAML(typ='rt')
    yaml.preserve_quotes = True
    config_path = Path(config_path)
    config = yaml.load(Path(config_path))
    if args.error_test:
        config['--error_test'] = 'None'
    if args.data_path is not None:
        config['--data_dir'] = args.data_path
    if args.l1_half:
        config['--lr_halv_f1'] = 'None'
        config['--halve_lr_after'] = args.halve_lr_after
    

    gpu = args.gpu
    config['--experiment_name'] = args.experiment_name
    experiment_name = args.experiment_name
    if args.seed is not None:
        config['--seed'] = args.seed
    config['--fold'] = args.fold
    config['--run_name'] = run_name

    if args.active_strategy is not None:
        config['--active_strategy'] = args.active_strategy
    if args.over_sampling:
        # add key of oversampling
        config['--over_sampling'] = 'None'
        config['--negative_weight'] = 1
    if args.output_dir is not None:
        config['--output_dir'] = args.output_dir
    else:
        config['--output_dir'] = f'runs/{experiment_name}/' + config['--task_name'] + f'_fold_{args.fold}_seed_{args.seed}'
    if args.num_workers is not None:
        config['--num_workers'] = args.num_workers   
    if args.learning_rate is not None:
        config['--learning_rate'] = args.learning_rate
    if args.train_batch_size is not None:
        config['--train_batch_size'] = args.train_batch_size
    if args.active_learning:
        cmd = f'max_seeds=10 current_seed=0 CUDA_VISIBLE_DEVICES={gpu} python run_model_al.py '
    else:
        cmd = f'max_seeds=10 current_seed=0 CUDA_VISIBLE_DEVICES={gpu} python run_model.py '
    if args.class_weight:
        config['--class_weight'] = 'None'
    if args.sampling_strategy is not None:
        config['--sampling_strategy'] = args.sampling_strategy
    if args.num_heads is not None:
        config['--num_heads'] = args.num_heads
    if args.init_size is not None:
        config['--init_size'] = args.init_size
    if args.num_train_epochs is not None:
        config['--num_train_epochs'] = args.num_train_epochs
    if args.rounds is not None:
        config['--rounds'] = args.rounds
    if args.query_sample_size is not None:
        config['--query_sample_size'] = args.query_sample_size
    if args.eval_mode is not None:
        config['--eval_mode'] = args.eval_mode

    options = []
    for k, v in config.items():   
        if v != 'None':
            options += [ f'{k} {v}']  
        else: options += [ f'{k}']
    cmd += ' '.join(options)

    os.system(cmd)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--active_learning', action='store_true', default=False)
    parser.add_argument('--mode', type=str, default='baseline')
    parser.add_argument('--experiment_name', type=str, default='test6')
    parser.add_argument('--over_sampling', action='store_true', default=False)
    parser.add_argument('--class_weight', action='store_true', default=False)
    parser.add_argument('--sampling_strategy', type=str, default='instance_first')
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--lr_halv_f1', action='store_true', default=False)
    parser.add_argument('--halve_lr_after', type=int, default=1)
    parser.add_argument('--num_heads', type=int, default=6)
    parser.add_argument('--l1_half', action='store_true', default=False)
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--active_strategy', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--error_test', action='store_true', default=False)
    parser.add_argument('--init_size', type=int, default=60)
    parser.add_argument('--query_sample_size', type=int, default=60)
    parser.add_argument('--num_train_epochs', type=int, default=10)
    parser.add_argument('--rounds', type=int, default=100)
    parser.add_argument('--eval_mode', type=str, default='majority')

    args = parser.parse_args()
    start_expriement(args)

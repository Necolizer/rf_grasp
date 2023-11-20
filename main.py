# -*- coding: utf-8 -*-

import os
import numpy as np
import time
import random
import torch
import argparse
import yaml

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from backbone.rt1_extractor import RT1Extractor
from env.elfin_ag145_env import ElfinAG145Env


def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])  # import return model
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

def get_parser():
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(description='')
    
    # processor
    parser.add_argument('--seed', type=int, default=1, help='seed')
    parser.add_argument('--work_dir', default='./work_dir/simple_task', help='the work folder for storing results')
    parser.add_argument('--config', default='./config/simple_task.yaml', help='path to the configuration file')
    parser.add_argument('--run_mode', default='train', help='must be train or test')
    parser.add_argument('--print_log', type=str2bool, default=True, help='print logging or not')

    # env
    parser.add_argument('--n_envs', type=int, default=1, help='')
    parser.add_argument('--max_episode_length', type=int, default=160, help='')
    parser.add_argument('--reward_type', default='dense', help='')
    parser.add_argument('--tolerance', type=float, default=0.02, help='')
    parser.add_argument('--load_object', type=str2bool, default=True, help='')
    parser.add_argument('--bin', type=int, default=64, help='')
    parser.add_argument('--movable_joints', type=int, default=6, help='')
    parser.add_argument('--raw_img_shape', type=int, default=[1280, 720], help='')
    parser.add_argument('--use_depth_img', type=str2bool, default=False, help='')
    parser.add_argument('--resized_img_shape', type=int, default=[224, 224], help='')

    # model
    # parser.add_argument('--model', default=None, help='the model will be used')
    # parser.add_argument('--model_args', default=dict(), help='the arguments of model')
    # parser.add_argument('--weights', default=None, help='the weights for model testing')
    # parser.add_argument('--ignore_weights', type=str, default=[], nargs='+', help='the name of weights which will be ignored in the initialization')

    # cuda
    parser.add_argument('--cuda_visible_device', default='0', help='')
    parser.add_argument('--device', type=int, default=[0], nargs='+', help='the indexes of GPUs for training or testing')

    # rl stable-baseline3
    parser.add_argument('--net_arch', type=int, default=[256,128], help='')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='')
    parser.add_argument('--total_timesteps', type=int, default=8000000, help='the total number of samples (env steps) to train on')
    parser.add_argument('--rollout_steps', type=int, default=3200, help='')
    parser.add_argument('--freq', type=int, default=32000, help='')
    parser.add_argument('--batch_size', type=int, default=32, help='training batch size')
    parser.add_argument('--progress_bar', type=str2bool, default=True, help='')

    return parser


class Processor():
    """ Processor for Reinforcement Learning """

    def __init__(self, arg):
        self.arg = arg
        self.global_step = 0
        self.lr = self.arg.base_lr
        self.best_acc = 0

        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)

        self.init_env()
        self.load_model()

    def init_env(self):
        self.env = ElfinAG145Env(
            max_episode_length = self.arg.max_episode_length,
            reward_type = self.arg.reward_type,
            tolerance = self.arg.tolerance,
            load_object = self.arg.load_object,
            seed = self.arg.seed,
            executable_file = None,
            scene_file = None,
            asset_bundle_file = None,
            assets = [],
            bin = self.arg.bin,
            movable_joints = self.arg.movable_joints,
            raw_img_shape = self.arg.raw_img_shape,
            use_depth_img = self.arg.use_depth_img,
            resized_img_shape = self.arg.resized_img_shape,
        )

        # self.eval_env = ElfinAG145Env(

        # )

    def load_model(self):
        output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        self.output_device = output_device

        policy_kwargs = dict(
            features_extractor_class = RT1Extractor,
            features_extractor_kwargs = {'device': self.output_device}, 
            net_arch = self.arg.net_arch
        )
        
        self.model = PPO(
            "MultiInputPolicy", 
            env = self.env, 
            learning_rate = self.arg.lr, 
            n_steps = self.arg.rollout_steps * self.arg.num_envs, 
            batch_size = self.arg.batch_size,
            n_epochs = 10,
            gamma = self.arg.gamma,
            policy_kwargs = policy_kwargs, 
            tensorboard_log = self.arg.work_dir,
            verbose = 1,
            seed = self.arg.seed,
            device = torch.device("cuda", self.output_device)
        )

    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.strftime('%Y-%m-%d %H:%M', time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        if self.arg.print_log:
            with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                print(str, file=f)

    def train(self):

        # eval_callback = EvalCallback(
        #     eval_env = self.eval_env, 
        #     best_model_save_path = self.arg.work_dir,
        #     log_path = self.arg.work_dir, 
        #     eval_freq = max(self.arg.freq // self.arg.n_envs, 1),
        #     deterministic = True, 
        #     render = False) 

        checkpoint_callback = CheckpointCallback(
            save_freq = max(self.arg.freq // self.arg.n_envs, 1),
            save_path = self.arg.work_dir,
            name_prefix = "rl_model",
            save_replay_buffer = True,
            save_vecnormalize = True,
        )

        # self.model.learn(
        #     total_timesteps = self.arg.total_timesteps, 
        #     callback = [checkpoint_callback, eval_callback], 
        #     progress_bar = self.arg.progress_bar
        # )

        self.model.learn(
            total_timesteps = self.arg.total_timesteps, 
            callback = [checkpoint_callback], 
            progress_bar = self.arg.progress_bar
        )

    def start(self):

        if self.arg.run_mode == 'train':

            for argument, value in vars(self.arg).items():
                self.print_log('{}: {}'.format(argument, value))

            # def count_parameters(model):
            #     return sum(p.numel() for p in model.parameters() if p.requires_grad)
            # self.print_log(f'# Parameters: {count_parameters(self.model)}')

            self.print_log('###***************start training***************###')
            
            self.train()
            self.print_log('Done.\n')

if __name__ == '__main__':
    parser = get_parser()

    # load arg form config file
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f, yaml.FullLoader)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = arg.cuda_visible_device
    init_seed(arg.seed)
    processor = Processor(arg)
    processor.start()

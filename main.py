import multiprocessing

import logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO,
                    datefmt='%m/%d/%Y %I:%M:%S')
log = logging.getLogger(__name__)

import argparse
import os
import sys
import torch
from Models.SDNetTrainer import SDNetTrainer
from Utils.Arguments import Arguments
if __name__ == "__main__":
    # multiprocessing.set_start_method('spawn')
    opt = None

    parser = argparse.ArgumentParser(description='SDNet')
    parser.add_argument('--command', default='train', help='Command: train')
    parser.add_argument('--conf_file', default='conf_stvqa', help='Path to conf file.')
    parser.add_argument('--log_file', default='', help='Path to log file.')

    cmdline_args = parser.parse_args()
    command = cmdline_args.command
    conf_file = cmdline_args.conf_file
    conf_args = Arguments(conf_file)
    opt = conf_args.readArguments()
    opt['cuda'] = torch.cuda.is_available()
    opt['confFile'] = conf_file
    opt['datadir'] = os.path.dirname(conf_file)  # conf_file specifies where the data folder is

    if cmdline_args.log_file != '':
        if not os.path.exists('myLog'):
            os.makedirs('myLog')
        file_handle = logging.FileHandler(os.path.join('myLog', cmdline_args.log_file +'.txt'))
        log.addHandler(file_handle)

    for key,val in cmdline_args.__dict__.items():
        if val is not None and key not in ['command', 'conf_file']:
            opt[key] = val

    model = SDNetTrainer(opt)


    print('Select command: ' + command)
    model.train()

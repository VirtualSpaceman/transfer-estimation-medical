import pandas as pd
import numpy as np
import os
import argparse
import subprocess
import queue
import time
from math import log10
from scipy.stats import qmc

def get_exponent(number):
    return log10(number)

clf_layers = {'resnet18': 'fc',
              'resnet34': 'fc',
              'resnet50': 'fc',
              'vit_small_patch16_224': 'head',
              'efficientnet_b0': 'classifier',
              'mobilenetv2_100': 'classifier',
              'mobilenetv2_050': 'classifier',
              'densenet121': 'classifier',
              'densenet161': 'classifier',
              'densenet169': 'classifier'}

medical_datasets = ['brain_tumor', 'breakhis', 'isic19']
TRAIN_PATH_TEMPLATE = "./data/{}/train_split_01.csv"
VAL_PATH_TEMPLATE = "./data/{}/val_split_01.csv"

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--d', default=2, type=int, help='dimension of hyperparameters')
parser.add_argument('--dataset', type=str, choices=medical_datasets, help="which dataset to use")
parser.add_argument('--n_runs', type=int, default=75)
parser.add_argument('--batch', type=int, default=128)
parser.add_argument('--model', default=clf_layers.keys(), type=str, required=True)
parser.add_argument('--gpus', default='0', type=str)
parser.add_argument('--max_lr', type=float, default=1e-1)
parser.add_argument('--min_lr', type=float, default=1e-4)
parser.add_argument('--max_wd', type=float, default=1e-4)
parser.add_argument('--min_wd', type=float, default=1e-6)

args = parser.parse_args()
n_runs = args.n_runs
gpu_queue = queue.Queue()

which_gpus = args.gpus.split(',')
n_gpus = len(which_gpus)
for gpu_id in which_gpus:
    gpu_queue.put(int(gpu_id))


model = args.model
max_lr, min_lr = args.max_lr, args.min_lr
max_wd, min_wd = args.max_wd, args.min_wd

# Create Halton sequence using 2 dimensions (Learning Rate x Weight decay)
sampler = qmc.Halton(d=args.d, scramble=False)

# Sample from the distribution n_runs samples. 
sample_continued = sampler.random(n=args.n_runs)

# Halton sequence generates log-spaced hyperparameters
l_bounds = [get_exponent(min_lr), get_exponent(min_wd)]
u_bounds = [get_exponent(max_lr) , get_exponent(max_wd)]
scaled = 10**qmc.scale(sample_continued, l_bounds, u_bounds)
print(scaled)

TRAIN_PATH = TRAIN_PATH_TEMPLATE.format(args.dataset)
VAL_PATH = VAL_PATH_TEMPLATE.format(args.dataset)

running = []
for i in range(n_runs):
    while len(running) >= n_gpus:
        for process in running:
            if process.poll() is not None:
                gpu_queue.put(process.gpu_id)
                running.remove(process)
                print(f"Process {i} finished with return code {process.returncode}")
        time.sleep(60)

    gpu_id = gpu_queue.get()
    print('hparams', scaled)
    
    # Start process on that GPU 
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu_id))
    process = subprocess.Popen(["python3", "finetune_models.py", 
                    "--lr", str(scaled[i][0]), 
                    "--wd", str(scaled[i][1]),
                    "--epochs", "100",
                    "--batch_size", f"{args.batch}",  
                    "--model", model, 
                    "--clf_layer", f"{clf_layers[model]}", 
                    "--train_csv", f"{TRAIN_PATH}",
                    "--val_csv", f"{VAL_PATH}",
                    "--pretrained",
                   ], env=env)

    print(f"Process {i} started on GPU {gpu_id}")
    process.gpu_id = gpu_id
    running.append(process)
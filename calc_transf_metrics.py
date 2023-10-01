import os

import torch
from torch.utils.data import DataLoader

import numpy as np
from argparse import ArgumentParser

import utils

from custom_dataset import CSVDataset

from collections import OrderedDict

from transfer_metrics.hscore import h_score,regularized_h_score
from transfer_metrics.leep import log_expected_empirical_prediction, gaussian_log_expected_empirical_prediction
from transfer_metrics.logme import log_maximum_evidence
from transfer_metrics.nce import negative_conditional_entropy
from transfer_metrics.gbc import bhattacharyya_coefficient

from sklearn.metrics import balanced_accuracy_score

from finetune_models import FineTuner

if __name__ == '__main__':
    MODELS = utils.get_model_names()
    parser = ArgumentParser(usage='%(prog)s [options]')
    parser.add_argument("--test_split", type=str, help="which split to run inference")
    
    parser.add_argument("--model", type=str, choices=MODELS, help="Which model to evaluate")
    parser.add_argument("--layer", type=str, help="Before which layer to extract features")
    parser.add_argument("--img_size", type=int, default=224, help="Image size for train/val/test")
    
    parser.add_argument("--ckpt_path", default=None, help="Which checkpoint to evaluate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--workers", type=int, default=16, help="number of workers")

    
    # parser.print_help()    
    args = parser.parse_args()
    task = args.test_split
    
    # get ood test set or iid training set path 
    test_csv_path = utils.get_path_transfer(task)
    
    csv_name = test_csv_path.split('/')[-1]
    csv_name = csv_name.split('.')[0]

    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_transforms = None
    print(f'[{device}] Calc Transferabilities of {args.model} on task {task}. Csv filename: {csv_name}')

    print('Conducting feature extraction')
    data_transforms = utils.get_data_transforms(args.model, args.img_size)
    print("data_transform: ", data_transforms)
    
    # get pre-trained model on imageNet by default or load from checkpoint
    if args.ckpt_path is None:
        model = utils.get_model(args.model, 
                                pretrained=True, 
                                pretrained_checkpoint=None)
    
    else:
        model_class = FineTuner.load_from_checkpoint(args.ckpt_path)
        model = torch.nn.Sequential(OrderedDict([
                                          ('encoder', model_class.encoder),
                                          ('head', model_class.classifier)
        ]))
        
        extract_before_layer = model.head
    
    model = model.to(device)
    # create test dataset
    score_dataset = CSVDataset(imgs_folder='', 
                               labels_csv=test_csv_path,
                                _format = '', sep = ',', 
                                transforms=data_transforms['test'])
    
    # calculate the number of classes
    num_classes = len(score_dataset.class_counts.keys())

    #get score loader
    score_loader = DataLoader(score_dataset, 
                              batch_size=args.batch_size, 
                              shuffle=False, 
                              num_workers=args.workers,
                              pin_memory=True)
    
    
    print(f'Using {len(score_dataset)} samples for ranking')
    
    # extract features before the classification layer. 
    extract_before_layer = eval(f'model.{args.layer}') if args.ckpt_path is None else model.head
    features, predictions, targets = utils.forwarding_dataset(score_loader, 
                                                              model,
                                                              layer=extract_before_layer, 
                                                              device=device)

    exp_name = f"{args.model}_{task}_{csv_name}_set"
    print(f"Experiment Name: {exp_name}")     
            
    print(f"Feat shape: {features.shape} -- Preds shape: {predictions.shape} -- Targets shape: {targets.shape}")
    print('Conducting transferability calculation')
    
    
    metrics_logs = dict()
    metrics_logs[f'hscore_score'] = h_score(features, targets)
    metrics_logs[f'reg_hscore_score'] = regularized_h_score(features, targets)
    metrics_logs[f'leep_score'] = log_expected_empirical_prediction(predictions, targets)
    metrics_logs[f'nleep_score'] = gaussian_log_expected_empirical_prediction(features, targets)
    metrics_logs[f'nce_score'] = negative_conditional_entropy(np.argmax(predictions, axis=1), targets)
    metrics_logs[f'gbc_score'] = bhattacharyya_coefficient(features, targets)
    metrics_logs[f'logme_score'] = log_maximum_evidence(features, targets)
    
    # calculate the test performance
    class_predictions = np.argmax(predictions, axis=1)
    
    # just ignore the balanced acc performance if you are evaluating ImageNet pre-trained models
    bal_acc = 0.0 if args.ckpt_path is None else balanced_accuracy_score(targets, class_predictions)
    
    metrics_logs['bal_acc'] = bal_acc
    for k, v in metrics_logs.items():
        print(f"Metric: {k} = {v}")
        
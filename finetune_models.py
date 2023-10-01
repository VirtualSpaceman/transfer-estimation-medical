import os
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.loggers.base import LoggerCollection
from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor
)

import torchvision
from torchvision import transforms

import utils

from sklearn import metrics

from argparse import ArgumentParser

from custom_dataset import CSVDataset


class FineTuner(pl.LightningModule):
    def __init__(self, params):
        super(FineTuner, self).__init__()

        # save parameters for late testing
        self.save_hyperparameters()

        # save the parameters as arguments
        self.params = params

        # the encoder from simclr network
        self.encoder = self.load_model()

        # build classification head
        self.classifier = self.set_classifier()

    def load_model(self):
        backbone = utils.get_model(
            self.params.model, self.params.pretrained, self.params.ckpt_path
        )

        if getattr(self.params, "freeze_model", False):
            for param in backbone.parameters():
                param.requires_grad = False

        return backbone

    def set_classifier(self):
        # get the feature dimension
        in_features = getattr(self.encoder, self.params.clf_layer).in_features
        # override the current classification head
        setattr(self.encoder, self.params.clf_layer, nn.Identity())
        return nn.Linear(in_features, self.params.classes)

    def forward(self, x):
        # standard forwass pass. Get the latent representation for each input
        # for example: (B, 2048) for resnet 50.
        representations = self.encoder(x)
        return representations

    def training_step(self, train_batch, batch_idx):
        """
        Executes when a batch is sampled from the dataloader in one training step
        
        Args:
        train_batch: the batch itself according to the dataloader (B x DIM)
        batch_idx: index for each sample in the current batch 
        
        """
        # get the imagens and labels
        imgs, labels = train_batch

        # get batch latent representation
        representations = self.forward(imgs)

        # calculate the logits
        logits = self.classifier(representations)

        # calculate the cross-entropy
        loss = F.cross_entropy(logits, labels)

        self.log("train/loss", loss, on_step=False, on_epoch=True, logger=True)

        # return the loss for optimzation. We use labels and cofidences in training_epoch_end
        return {
            "loss": loss,
            "labels": labels,
            "confidences": F.softmax(logits.detach(), dim=1),
        }

    def training_epoch_end(self, outputs):
        """
        Executes after passing through all batches in training step
        
        Args:
        outputs: all outputs from the training_step (list by default)
        
        """

        all_metrics = self.__compute_metrics(outputs)
        for (metric_name, metric_value) in all_metrics:
            self.log(
                f"train/{metric_name}_epoch",
                metric_value,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

    def validation_step(self, batch, batch_idx):
        """
        Executes when a batch is sampled from the dataloader in validation step
        
        Args:
        train_batch: the batch itself according to the dataloader
        batch_idx: index for each sample in the current batch 
        
        """
        # get the images and labels
        imgs, labels = batch

        # get the representations and logits
        representations = self.encoder(imgs)
        logits = self.classifier(representations)

        # calculate the loss
        loss = F.cross_entropy(logits, labels)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        # return the labels and the confidences for calculating all performance metrics
        return {"labels": labels, "confidences": F.softmax(logits.detach(), dim=1)}

    def validation_epoch_end(self, outputs):
        """
        Executes after passing through all batches in validation step
        
        Args:
        outputs: all outputs from the validation_step (list by default)
        
        """
        all_metrics = self.__compute_metrics(outputs)

        for (metric_name, metric_value) in all_metrics:
            self.log(
                f"val/{metric_name}_epoch",
                metric_value,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

    def test_step(self, batch, batch_idx):
        """
        Function to be executed when a batch is sampled from the dataloader test step
        
        Args:
        train_batch: the batch itself according to the dataloader
        batch_idx: index for each sample in the current batch 
        
        """
        # get tha images, labels and name of the imagefile
        imgs, labels = batch

        representations = self.encoder(imgs)
        logits = self.classifier(representations)

        loss = F.cross_entropy(logits, labels)

        # calculate the confidence for each prediction
        confidence = F.softmax(logits, dim=1)

        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return {"labels": labels.detach().data, "confidences": confidence}

    def test_epoch_end(self, outputs):
        """
        Executes after passing through all batches in test step
        
        Args:
        outputs: all outputs from the test_step (list by default)
        
        """
        all_metrics = self.__compute_metrics(outputs)
        for (metric_name, metric_value) in all_metrics:
            self.log(
                f"test/{metric_name}_epoch",
                metric_value,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

    def __get_vectors(self, outputs):
        # get all vectors to calculate the predictions
        confidences = torch.cat([x["confidences"] for x in outputs], dim=0)
        
        # get only the model predictions 
        _, model_predictions = confidences.max(dim=1)
        
        # move to cpu 
        model_predictions = model_predictions.cpu().numpy()
        confidences = confidences.cpu().numpy()

        # store the true labels for all samples in a numpy array
        labels = torch.cat([x["labels"] for x in outputs], dim=0)
        labels = labels.cpu().numpy()

        return model_predictions, confidences, labels

    def __compute_metrics(self, outputs):
        calc_metrics = []
        model_predictions, confidences, labels = self.__get_vectors(outputs)

        # calculate balanced acc
        try:
            balanced_acc = metrics.balanced_accuracy_score(labels, model_predictions)
        except Exception as e:
            balanced_acc = 0.0

        # calculate accuracy
        acc = metrics.accuracy_score(labels, model_predictions)
        
        # return all calculated metrics
        calc_metrics.append(("balanced_ACC", balanced_acc))
        calc_metrics.append(("ACC", acc))
        return calc_metrics

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.params.lr,
            momentum=0.9,
            weight_decay=self.params.wd,
            nesterov=True,
        )

        # use cosine as base scheduler
        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.params.epochs, eta_min=1e-6
            )
        }

        return [optimizer], [lr_scheduler]


if __name__ == "__main__":
    schedulers = ["cosine"]
    MODELS = utils.get_model_names()
    parser = ArgumentParser(usage="%(prog)s [options]")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs.")
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Optimizer Learning Rate."
    )
    parser.add_argument("--wd", type=float, default=1e-3, help="Weight decay.")

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Max number of epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for training and validate",
    )

    # Dataset related args
    parser.add_argument(
        "--img_size", type=int, default=224, help="Image size for train/val/test"
    )
    
    parser.add_argument("--train_csv", type=str, default="/", help="train csv file")
    parser.add_argument("--val_csv", type=str, default="/", help="validation csv file")

    # Model related args
    parser.add_argument("--model", type=str, choices=MODELS, required=True)
    parser.add_argument("--pretrained", action="store_true", help="pretrained models")
    parser.add_argument(
        "--ckpt_path", type=str, default=None, help="path to start model on"
    )
    parser.add_argument(
        "--clf_layer", type=str, default="fc", help=f"Classification layer"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./saved_models/",
        help=f"Path to save the experiments",
    )
    parser.add_argument(
        "--freeze_model", action="store_true", help="if freeze the backbone"
    )

    # Others
    parser.add_argument("--workers", type=int, default=32, help="Number of workers")
    parser.add_argument("--debug", action="store_true", help="Init in debug mode")
    parser.add_argument("--online", action="store_true", help="online logger")
    parser.add_argument(
        "--project_name", type=str, default="uncategorized", help=f"Project to log data on"
    )

    # Parse args
    args = parser.parse_args()

    # these are false and none by default
    loggers = False
    callbacks = None

    print("Setting Up datasets and augmentations...")
    data_transforms = utils.get_data_transforms(
        args.model, args.img_size
    )
    print(data_transforms)

    train_dataset = args.train_csv.split("/")[-2]
    num_classes = None

    print(" ==== Using CSV Datasets ===")
    print(" ==== Creating Training Set ==== ")
    # init train dataset
    train_ds = CSVDataset(
        imgs_folder="",
        labels_csv=args.train_csv,
        _format="",
        sep=",",
        transforms=data_transforms["train"],
    )

    # set the class number
    args.classes = len(train_ds.class_counts.keys())
    print(" ==== Creating Validation Set ==== ")
    # init validation dataset
    val_ds = CSVDataset(
        imgs_folder="",
        labels_csv=args.val_csv,
        _format="",
        sep=",",
        transforms=data_transforms["val"],
    )

    print(" ==== Creating Test Set ==== ")
    # get path for test dataset and init dataset
    test_path_labels = utils.get_path_from_task(train_dataset)
    test_ds = CSVDataset(
        imgs_folder="",
        labels_csv=test_path_labels,
        _format="",
        sep=",",
        transforms=data_transforms["test"],
    )

    

    # set dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=args.workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.workers,
    )

    name_exp = f"{args.model}_bz_{args.batch_size}_lr_{args.lr}_wd_{args.wd}_sched_cossine_size_{args.img_size}_cls_{args.classes}"
    print("Setting Up Model and PL-Trainer...")
    print("Experiment name: ", name_exp)
    
    # Create the model according to PL lightning base class
    model = FineTuner(args)

    if not args.debug:
        print("Setting Up loggers and callbacks...")
        model_path = f"{args.model_path}{args.model}/{train_dataset}/{name_exp}"
        # create CSV logger
        csv = CSVLogger(model_path, name="finetuning")

        # Learning Rate Logger
        lr_logger = LearningRateMonitor()

        # saves checkpoints to 'dirpath' whenever 'val_loss' has a new min
        checkpoint_callback = ModelCheckpoint(
            monitor="val/loss",
            mode="min",
            save_top_k=1,
            dirpath=f"{model_path}/finetuning/{args.model}/version_{csv._get_next_version()}/checkpoints/",
            filename="{epoch:03d}-{val/loss:.3f}-{val/ACC_epoch:.3f}-{val/balanced_ACC_epoch:.3f}",
        )

        callbacks = [lr_logger, checkpoint_callback]
        loggers = [csv]

        # if online logger...
        if args.online:
            tags = [f"{args.classes}classes", args.model, args.scheduler, train_dataset]

            online_logger = WandbLogger(
                project=args.project_name, name=name_exp, tags=tags
            )

            # log some parameters
            online_logger.experiment.log(
                {
                    "model_path": model_path,
                    "train_aug": str(data_transforms["train"]),
                    "val_aug": str(data_transforms["val"]),
                    "test_aug": str(data_transforms["test"]),
                }
            )

            loggers.append(online_logger)

    # setup the trainer parameters
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if torch.cuda.is_available() else None,
        devices=args.gpus,
        logger=loggers,
        callbacks=callbacks,
        fast_dev_run=args.debug,
        num_sanity_val_steps=-1,
        limit_train_batches=10 if args.debug else 1.0,
    )

    if not args.debug and args.online:
        # log gradients, parameter histogram and model topology
        online_logger.watch(model)

    # train the model
    trainer.fit(model, train_loader, val_loader)

    # check if its not in debug mode
    if not args.debug:
        # obtain metrics, logs and parameters path.
        base_path = os.path.join(
            csv._save_dir, csv._name, "version_" + str(csv._version)
        )
        metrics_path = os.path.join(base_path, "metrics.csv")
        param_path = os.path.join(base_path, "hparams.yaml")
        best_model_path = checkpoint_callback.best_model_path

        # Evaluate the best model according to the callback function
        trainer.test(
            dataloaders=test_loader, ckpt_path=checkpoint_callback.best_model_path
        )

        if args.online:
            online_logger.experiment.log(
                {
                    "base_path": base_path,
                    "metrics_path": metrics_path,
                    "params_path": param_path,
                    "best_model_path": best_model_path,
                }
            )
            online_logger.finalize("success")
    else:
        # otherwise, test the model on the last epoch 
        trainer.test(model, dataloaders=test_loader)

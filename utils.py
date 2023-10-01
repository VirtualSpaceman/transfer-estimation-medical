import timm
import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torchvision import transforms

import torchvision.models as models


def get_torchvision_names():
    return sorted(
        name
        for name in models.__dict__
        if name.islower()
        and not name.startswith("__")
        and callable(models.__dict__[name])
    )


def get_timm_names():
    return timm.list_models()


def get_model_names():
    return get_torchvision_names() + get_timm_names()


def forwarding_dataset(score_loader, model, layer, device):
    """
    A forward forcasting on a given score loader

    :params score_loader: the dataloader for scoring transferability
    :params model: the model for scoring transferability
    :params layer: before which layer features are extracted, for registering hooks
    :params device: if use gpu or cpu to extract the features
    returns
        features: extracted features of model
        prediction: probability outputs of model
        targets: ground-truth labels of dataset
    """
    features = []
    outputs = []
    targets = []

    def hook_fn_forward(module, input, output):
        features.append(input[0].detach().cpu())
        outputs.append(output.detach().cpu())

    forward_hook = layer.register_forward_hook(hook_fn_forward)

    model.eval()
    with torch.no_grad():
        for _, (data, target) in enumerate(score_loader):
            targets.append(target)
            data = data.to(device)
            _ = model(data)

    forward_hook.remove()

    features = torch.cat([x for x in features]).numpy()
    outputs = torch.cat([x for x in outputs])
    predictions = F.softmax(outputs, dim=-1).numpy()
    targets = torch.cat([x for x in targets]).numpy()

    return features, predictions, targets


def get_model(model_name, pretrained=True, pretrained_checkpoint=None):
    if model_name in get_timm_names():
        # load models from pytorch-image-models
        backbone = timm.create_model(model_name, pretrained=pretrained)
    elif model_name in get_torchvision_names():
        # load models from common.vision.models
        backbone = models.__dict__[model_name](weights="DEFAULT")
    if pretrained_checkpoint:
        print("=> loading pre-trained model from '{}'".format(pretrained_checkpoint))
        pretrained_dict = torch.load(pretrained_checkpoint)
        msg = backbone.load_state_dict(pretrained_dict, strict=False)
        print(msg)
    return backbone


def adjust_clf_layer(backbone, layer, num_classes):
    in_features = getattr(backbone, layer).in_features
    setattr(backbone, layer, nn.Linear(in_features, num_classes))
    return backbone


def get_path_from_task(task):
    
    dataset_to_test_path = {
        "brain_tumor": "./data/brain_tumor/test.csv",
        "breakhis": "./data/breakhis/test.csv",
        "isic19": "./data/isic19/test.csv",
    }
    
    return dataset_to_test_path[task]


def get_path_transfer(task):
    dataset_to_test_path = {
        "brain_tumor": "./data/brain_tumor/train_split_01.csv",
        "brain_tumor_ood": "./data/brain_tumor/ood/out_test_NINS.csv",
        "breakhis": "./data/breakhis/train_split_01.csv",
        "breakhis_ood": "./data/breakhis/ood/ICIAR_out_distribution.csv",
        "isic19": "./data/isic19/train_split_01.csv",
        "isic19_ood": "./data/isic19/ood/pad-ufes-20.csv"
    }
    
    return dataset_to_test_path[task]


def get_data_transforms(model_name, size=224):
    """
    Get proper data augmentation for the method given as parameter
    
    """

    # mean and std for imagenet
    mean = [0.485, 0.456, 0.406]
    std = [0.228, 0.224, 0.225]

    train_trans = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomResizedCrop(size, scale=(0.75, 1.0)),
        transforms.RandomRotation(45),
        transforms.ColorJitter(hue=0.2),
        transforms.ToTensor(),
    ]

    val_trans = [
        transforms.Resize(size + 32),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
    ]

    test_trans = [
        transforms.Resize(size + 32),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
    ]

    # vit is the only one which was trained without imagenet normalization
    if model_name in timm.list_models("vit_*"):
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

    normalize = transforms.Normalize(mean=mean, std=std)
    train_trans.append(normalize)
    val_trans.append(normalize)
    test_trans.append(normalize)

    # dict to store the transformations for each step
    data_transforms = {
        "train": transforms.Compose(train_trans),
        "val": transforms.Compose(val_trans),
        "test": transforms.Compose(test_trans),
    }

    return data_transforms

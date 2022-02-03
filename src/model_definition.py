import torch.nn as nn
from pathlib import Path
from hyper_param import *
import urllib

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', device='cpu', classes = 1, autoshape = False, pretrained = True)  # load on GPU



def train(model: nn.Module,
          train_loader: torch.utils.data.DataLoader,
          device: torch.device,
          optimizer: torch.optim,
          log_interval: int,
          epoch: int) -> Dict[str, float]:
    """Trains a neural network for one epoch.

    Args:
        model: the model to train.
        train_loader: the data loader containing the training data.
        device: the device to use to train the model.
        optimizer: the optimizer to use to train the model.
        log_interval: the log interval.
        epoch: the number of the current epoch.

    Returns:
        A dictionary containing:
            the sum of classification and regression loss.
            the classification loss.
            the regression loss.
    """
    size_ds_train = len(train_loader.dataset)
    num_batches = len(train_loader)

    model.train()
    for idx_batch, (images, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        loss_class = loss_dict['classification']
        loss_boxes_regr = loss_dict['bbox_regression']
        losses = loss_class + loss_boxes_regr
        losses.backward()
        optimizer.step()

    dict_losses_train = {'bbox_regression': loss_boxes_regr,
                         'classification': loss_class,
                         'sum': losses}
    return dict_losses_train


for k, v in model.named_parameters():
    print(k)

#freeze = ['model.%s.' % x for x in range(10)]  # parameter names to freeze (full or partial)
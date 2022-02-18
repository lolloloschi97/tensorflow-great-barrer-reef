import torch.nn as nn
import os
import torch.utils as utils
import torchvision.models.detection.backbone_utils as ut
from tqdm import tqdm
from time import sleep
from torchvision.ops.feature_pyramid_network import LastLevelP6P7
from torchvision.models.detection import RetinaNet, retinanet_resnet50_fpn
from torchvision.models.detection.anchor_utils import AnchorGenerator
from hyper_param import *
from timeit import default_timer as timer
from PIL import Image
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from validation import validate
import torch
import torch.nn.functional as F


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2,
    reduction: str = "none",
):
    """
    Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py .
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples or -1 for ignore. Default = 0.25
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(
        inputs, targets, reduction="none"
    )
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss

def train(writer: SummaryWriter,
          model: nn.Module,
          train_loader: utils.data.DataLoader,
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
    with tqdm(train_loader, unit="batch") as tepoch:
        for idx_batch, (images, targets) in enumerate(tepoch):
            tepoch.set_description(f"Epoch {epoch}")
            optimizer.zero_grad()
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            loss_class = loss_dict['classification']
            loss_boxes_regr = loss_dict['bbox_regression']
            losses = loss_class + loss_boxes_regr
            losses.backward()
            optimizer.step()

            if log_interval > 0:
                if idx_batch % log_interval == 0:
                    global_step = idx_batch + (epoch * num_batches)
                    writer.add_scalar('Metrics/Loss_Train_IT_Sum', losses, global_step)
                    writer.add_scalar('Metrics/Loss_Train_IT_Boxes', loss_boxes_regr, global_step)
                    writer.add_scalar('Metrics/Loss_Train_IT_Classification', loss_class, global_step)

            tepoch.set_postfix(loss = {'classification' : loss_class, 'bbox_regression': loss_boxes_regr, 'sum': losses}.items())
            sleep(0.1)

    dict_losses_train = {'bbox_regression': loss_boxes_regr,
                         'classification': loss_class,
                         'sum': losses}
    return dict_losses_train

def validate(model: nn.Module,
            val_loader: torch.utils.data.DataLoader,
            device: torch.device) -> Dict[str, float]:
    """Evaluate the model.

    Args:
        model: the model to evalaute.
        val_loader: the data loader containing the validation data.
        device: the device to use to evaluate the model.

   Returns:
        A dictionary containing:
            the sum of classification and regression loss.
            the classification loss.
            the regression loss.
    """
    size_ds_train = len(val_loader.dataset)
    num_batches = len(val_loader)

    samples_val = 0
    regr_val = 0
    loss_class_val = 0.
    loss_regr_val = 0.
    model = model.eval()

    with torch.no_grad():
        for idx_batch, (images, targets) in enumerate(val_loader):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            list_dict = model(images, targets)
            for ind, el in enumerate(list_dict):

                loss_class = sigmoid_focal_loss(el['labels'],targets[ind]['labels'], reduction='sum')
                loss_class_val += loss_class.item() * len(images)
                loss_boxes_regr = torch.nn.functional.l1_loss(el['boxes'],targets[ind]['boxes'], reduction='sum')
                loss_regr_val += loss_boxes_regr.item() * len(targets)

            samples_val += len(images)
            regr_val += len(targets)

    loss_class_val /= samples_val
    loss_regr_val /= regr_val

    loss_sum = loss_class_val + loss_regr_val

    dict_losses_val = {'bbox_regression': loss_regr_val,
                       'classification': loss_class_val,
                       'sum': loss_sum}

    return dict_losses_val


def training_loop(writer: SummaryWriter,
                  num_epochs: int,
                  optimizer: torch.optim,
                  lr_scheduler: torch.optim.lr_scheduler,
                  log_interval: int,
                  model: nn.Module,
                  loader_train: utils.data.DataLoader,
                  loader_val: utils.data.DataLoader,
                  verbose: bool = True) -> Dict[str, List[float]]:
    """Executes the training loop.

        Args:
            writer: the summary writer for tensorboard.
            num_epochs: the number of epochs.
            optimizer: the optimizer to use.
            lr_scheduler: the scheduler for the learning rate.
            log_interval: interval to print on tensorboard.
            model: the model to train.
            loader_train: the data loader containing the training data.
            loader_val: the data loader containing the validation data.
            verbose: if true print the value of loss.

        Returns:
            A dictionary with the statistics computed during the train:
            the values for the train loss for each epoch.
            the values for the train accuracy for each epoch.
            the time of execution in seconds for the entire loop.
            the model trained
    """
    loop_start = timer()

    losses_bb_values_train = []
    losses_class_values_train = []
    losses_sum_values_train = []

    losses_bb_values_val = []
    losses_class_values_val = []
    losses_sum_values_val = []

    for epoch in range(1, num_epochs + 1):
        time_start = timer()
        losses_epoch_train = train(writer, model, loader_train, DEVICE,
                                   optimizer, log_interval, epoch)
        if not os.path.exists(CHECKPOINT_ROOT):
            os.makedirs(CHECKPOINT_ROOT)

        path_checkpoint = os.path.join(CHECKPOINT_ROOT,
                                       f'retina_net_{epoch}_epochs.bin')
        torch.save(model.state_dict(), path_checkpoint)

        losses_epoch_val = validate(model, loader_val, DEVICE)

        loss_bb_train = losses_epoch_train['bbox_regression']
        losses_bb_values_train.append(loss_bb_train)

        loss_bb_val = losses_epoch_val['bbox_regression']
        losses_bb_values_val.append(loss_bb_val)

        loss_class_train = losses_epoch_train['classification']
        losses_class_values_train.append(loss_class_train)

        loss_class_val = losses_epoch_val['classification']
        losses_class_values_val.append(loss_class_val)

        loss_sum_train = losses_epoch_train['sum']
        losses_sum_values_train.append(loss_sum_train)

        loss_sum_val = losses_epoch_val['sum']
        losses_sum_values_val.append(loss_sum_val)

        time_end = timer()

        lr = optimizer.param_groups[0]['lr']

        if verbose:
            print(f'Epoch: {epoch} '
                  f' Lr: {lr:.8f} '
                  f' Losses Train: Sum = [{loss_sum_train:.4f}] Class = [{loss_class_train:.4f}] Boxes = [{loss_bb_train:.4f}]'
                  f' Losses Val: Sum = [{loss_sum_val:.4f}] Class = [{loss_class_val:.4f}] Boxes = [{loss_bb_val:.4f}]'
                  f' Time one epoch (s): {(time_end - time_start):.4f} ')

        # Plot to tensorboard
        writer.add_scalar('Hyperparameters/Learning Rate', lr, epoch)
        writer.add_scalar('Metrics/Losses_Train/Sum', loss_sum_train, epoch)
        writer.add_scalar('Metrics/Losses_Train/Boxes', loss_bb_train, epoch)
        writer.add_scalar('Metrics/Losses_Train/Classification', loss_class_train, epoch)

        writer.add_scalar('Metrics/Losses_Val/Sum', loss_sum_val, epoch)
        writer.add_scalar('Metrics/Losses_Val/Boxes', loss_bb_val, epoch)
        writer.add_scalar('Metrics/Losses_Val/Classification', loss_class_val, epoch)
        writer.flush()

        if lr_scheduler:
            lr_scheduler.step()

    loop_end = timer()
    time_loop = loop_end - loop_start
    if verbose:
        print(f'Time for {num_epochs} epochs (s): {(time_loop):.3f}')

    return {'bbox_regression_train': losses_bb_values_train,
            'classification_train': losses_class_values_train,
            'sum_train': losses_sum_values_train,
            'bbox_regression_val': losses_bb_values_val,
            'classification_val': losses_class_values_val,
            'sum_val': losses_sum_values_val,
            'time': time_loop}

def detect_objects(image: Image,
                   detector: nn.Module,
                   threshold: float,
                   categories: List[str]) -> Tuple[List[List[int]],
                                                   List[float],
                                                   List[str],
                                                   List[int]]:
    """Detects objects in the image using the provided detector.
    This function puts the model in the eval mode.

    Args:
        image: the input image.
        detector: the object detector.
        threshold: the detection confidence score, any detections with scores
        below this value are discarded.
        categories: the names of the categories of the data set used to train the network.

    Returns:
        The bounding boxes of the predicted objects using the [x_min, y_min, x_max, y_max] format,
        with values between 0 and image height and 0 and image width.
        The scores of the predicted objects.
        The categories of the predicted objects.
    """
    detector.eval()
    with torch.no_grad():
        predictions = detector(image)
        predictions = predictions[0]

    # Get scores, boxex, and labels
    scores = predictions['scores'].detach().cpu().numpy()
    boxes = predictions['boxes'].detach().cpu().numpy()
    labels = predictions['labels'].cpu().numpy()

    # Get all the boxes above the threshold
    mask = scores >= threshold
    boxes_filtered = boxes[mask].astype(np.int32)

    # Get the names of the categories above the threshold
    indices_filtered = [idx for idx, score in enumerate(list(scores)) if score >= threshold]
    categories_filtered = [categories[labels[i]] for i in indices_filtered]

    # Get only the scores above the threshold
    labels_filtered = labels[mask]
    scores_filtered = scores[mask]

    return boxes_filtered, scores_filtered, categories_filtered, labels_filtered


def execute(name_train: str,
            model: nn.Module,
            starting_lr: float,
            num_epochs: int,
            data_loader_train: torch.utils.data.DataLoader,
            data_loader_val: torch.utils.data.DataLoader) -> None:
    """Executes the training loop.

    Args:
        name_train: the name for the log subfolder.
        model: the model to train.
        starting_lr: the staring learning rate.
        num_epochs: the number of epochs.
        data_loader_train: the data loader with training data.
        data_loader_val: the data loader with validation data.
    """
    # Visualization
    log_interval = 20
    log_dir = os.path.join('logs', name_train)
    writer = torch.utils.tensorboard.writer.SummaryWriter(log_dir)

    # Optimization
    optimizer = torch.optim.Adam(model.parameters(), lr=starting_lr)

    # Learning Rate schedule
    scheduler = lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)

    statistics = training_loop(writer, num_epochs, optimizer, scheduler,
                               log_interval, model, data_loader_train, data_loader_val,
                               verbose = True)
    writer.close()

    # Save the model
    if not os.path.exists(CHECKPOINT_ROOT):
        os.makedirs(CHECKPOINT_ROOT)

    path_checkpoint = os.path.join(CHECKPOINT_ROOT,
                                   f'{name_train}_{num_epochs}_epochs.bin')


    torch.save(model.state_dict(), path_checkpoint)

def set_requires_grad_for_layer(layer: torch.nn.Module, train: bool) -> None:
    """Sets the attribute requires_grad to True or False for each parameter.

        Args:
            layer: the layer to freeze.
            train: if true train the layer.
    """
    for p in layer.parameters():
        p.requires_grad = train


retina_net = retinanet_resnet50_fpn(pretrained = True,
                                    num_classes = 91,
                                    pretrained_backbone = True,
                                    trainable_backbone_layers = None)


retina_net.head.classification_head.cls_logits = nn.Conv2d(256, 9, kernel_size=(3, 3), stride =(1, 1), padding=(1, 1))
retina_net.head.classification_head.num_classes = 1
set_requires_grad_for_layer(retina_net.backbone, False)
set_requires_grad_for_layer(retina_net.anchor_generator, False)
set_requires_grad_for_layer(retina_net.head.classification_head, True)
set_requires_grad_for_layer(retina_net.head.regression_head, False)

retina_net.to(DEVICE)

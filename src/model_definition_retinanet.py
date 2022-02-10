import torch.nn as nn
import os
import torch.utils as utils
import torchvision.models.detection.backbone_utils as ut
from torchvision.ops.feature_pyramid_network import LastLevelP6P7
from torchvision.models.detection import RetinaNet
from torchvision.models.detection.anchor_utils import AnchorGenerator
from hyper_param import *
from timeit import default_timer as timer
from PIL import Image
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter

def train(writer: utils.tensorboard.writer.SummaryWriter,
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

        if log_interval > 0:
            if idx_batch % log_interval == 0:
                global_step = idx_batch + (epoch * num_batches)
                writer.add_scalar('Metrics/Loss_Train_IT_Sum', losses, global_step)
                writer.add_scalar('Metrics/Loss_Train_IT_Boxes', loss_boxes_regr, global_step)
                writer.add_scalar('Metrics/Loss_Train_IT_Classification', loss_class, global_step)

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

    model = model.eval()

    with torch.no_grad():
        for idx_batch, (images, targets) in enumerate(val_loader):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)

            loss_class = loss_dict['classification']
            loss_boxes_regr = loss_dict['bbox_regression']
            losses = loss_class + loss_boxes_regr

    dict_losses_val = {'bbox_regression': loss_boxes_regr,
                       'classification': loss_class,
                       'sum': losses}
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
                               log_interval, model, data_loader_train,
                               verbose = True)
    writer.close()

    # Save the model
    if not os.path.exists('../checkpoints'):
        os.makedirs('../checkpoints')

    path_checkpoint = os.path.join('../checkpoints',
                                   f'{name_train}_{num_epochs}_epochs.bin')
    torch.save(model.state_dict(), path_checkpoint)


backbone = ut.resnet_fpn_backbone('resnet34',
                                  pretrained=False,
                                  trainable_layers=5,
                                  returned_layers=[2, 3, 4],
                                  extra_blocks=LastLevelP6P7(256, 256))

sizes_anchors = ((8, 10, 12),
                 (16, 20, 25),
                 (32, 40, 50),
                 (64, 80, 101),
                 (128, 161, 203))

ratios_anchors = ((0.5, 1.0, 2.0),) * len(sizes_anchors)


anchor_generator = AnchorGenerator(sizes=sizes_anchors,
                                   aspect_ratios=ratios_anchors)

retina_net = RetinaNet(backbone,
                          2,
                          anchor_generator=anchor_generator,
                          min_size=800,
                          max_size=1333)
retina_net.to(DEVICE)

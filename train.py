import os
import argparse
import logging
from tqdm import tqdm
from typing import Callable, Tuple
from PIL import Image
from functools import partial

import torch
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.transforms import transforms


from augmentation.randaugment import RandAugment
from augmentation.augmentations import get_weak_augmentation, cutout, get_normalizer
from eval import evaluate
from datasets.config import IMG_SIZE
from utils.train import EMA, cosine_lr_decay, get_wd_param_list
from utils.eval import AverageMeterSet
from utils.metrics import write_metrics
from utils.misc import save_state, load_state

MIN_VALIDATION_SIZE = 50

logger = logging.getLogger()


class FixMatchTransform:
    """
    FixMatchTransform implements the augmentation strategies for labeled and unlabeled data used in FixMatch.
    """
    def __init__(self, weak_transform: Callable, strong_transform: Callable = None):
        """
        Initializes a FixMatchTransform object.

        Parameters
        ----------
        weak_transform: Callable
            Weak augmentation strategy used labeled samples or to compute pseudo-labels for unlabeled samples.
        strong_transform: Callable (default: None)
            Strong augmentation strategy, which is used to augment unlabeled samples.
        """
        self.weak = weak_transform
        self.strong = strong_transform

    def __call__(self, img: Image):
        """
        Applies transforms to input image. Returns two transformed images if strong_transform is also specified.

        Parameters
        ----------
        img: PIL.Image
            Input image for which augmented version(s) are computed.

        Returns
        -------
        transformed_image: Optional[PIL.Image, List]
            Returns augmented version(s) of the input image.
        """
        if self.strong is None:
            return self.weak(img)
        else:
            return self.weak(img), self.strong(img)

    @classmethod
    def labeled(cls, dataset: str, img_size: int, padding: int):
        """
        Constructor for the FixMatchTransform class for labeled images. In FixMatch, labeled images are
        only augmented with a weak augmentation strategy consisting of random crops and random horizontal flips.

        Parameters
        ----------
        cls:
            Reference to FixMatchTransform class
        dataset: str
            String specifying dataset to which transform is applied. Important to select correct normalizer.
        img_size: int
            Size of input images (assuming images are squared)
        padding: int
            Number of padding pixels used as input to weak_augmentation transform
        Returns
        -------
        cls: FixMatchTransform
            Function returns instance of FixMatchTransform based on given inputs.
        """
        return cls(
            weak_transform=transforms.Compose([
                get_weak_augmentation(img_size, padding),
                get_normalizer(dataset)
            ])
        )

    @classmethod
    def unlabeled(cls, dataset: str, strong_aug: Callable, img_size: int, padding: int, cutout_mag: float = 0.5):
        """
        Constructor for the FixMatchTransform class for unlabeled images. In FixMatch, unlabeled images are
        only augmented with a weak augmentation strategy consisting of random crops and random horizontal flips.

        Parameters
        ----------
        cls:
            Reference to FixMatchTransform class
        dataset: str
            String specifying dataset to which transform is applied. Important to select correct normalizer.
        strong_aug: Callable
            Callable object implementing the applied strong augmentation strategy, i.e. RandAugment or CTAugment
            (not implemented yet).
        img_size: int
            Size of input images (assuming images are squared)
        padding: int
            Number of padding pixels used as input to weak_augmentation transform
        cutout_mag: float
            Magnitude of cutout operation applied to the images after the weak and strong augmentation.
        Returns
        -------
        cls: FixMatchTransform
            Returns instance of FixMatchTransform based on given inputs.
        """
        return cls(
            weak_transform=transforms.Compose([
                get_weak_augmentation(img_size, padding),
                get_normalizer(dataset)
            ]),
            strong_transform=transforms.Compose([
                get_weak_augmentation(img_size, padding),
                strong_aug,
                partial(cutout, mag=cutout_mag),
                get_normalizer(dataset)
            ])
        )


# --- Transforms ---
def get_transform_dict(args, strong_aug: Callable):
    """
    Generates dictionary with transforms for all datasets

    Parameters
    ----------
    args: argparse.Namespace
        Namespace object that contains all command line arguments with their corresponding values
    strong_aug: Callable
        Callable object implementing the applied strong augmentation strategy, i.e. RandAugment or CTAugment
        (not implemented yet).
    Returns
    -------
    transform_dict: Dict
        Dictionary containing transforms for the labeled train set, unlabeled train set
        and the validation / test set
    """
    img_size = IMG_SIZE[args.dataset]
    padding = int(0.125 * img_size)
    return {
        "train": FixMatchTransform.labeled(args.dataset, img_size, padding),
        "train_unlabeled": FixMatchTransform.unlabeled(args.dataset, strong_aug, img_size, padding),
        "test": get_normalizer(args.dataset),
    }


# --- Optimization & Scheduler---
def get_optimizer(args, model):
    """
    Initialize and return SGD optimizer. The weight decay is applied to all parameters except for BatchNorm parameters,
    which are filtered out by the function get_wd_param_list.

    Parameters
    ----------
    args: argparse.Namespace
        Namespace that contains all command line arguments with their corresponding values
    model: torch.nn.Module
        torch module, i.e. neural net, which is trained using FixMatch
    Returns
    -------
    optim: torch.optim.Optimizer
        Returns SGD optimizer which is used for model training
    """
    optim_params = get_wd_param_list(model)
    return SGD(
        optim_params,
        lr=args.lr,
        momentum=args.beta,
        weight_decay=args.wd,
        nesterov=True,
    )


def get_scheduler(args, optimizer):
    """
    Initialize and return scheduler object. FixMatch uses a learning rate scheduler, which applies a cosine learning
    rate decay over the course of the training process.

    Parameters
    ----------
    args: argparse.Namespace
        Namespace that contains all command line arguments with their corresponding values
    optimizer: torch.optim.Optimizer
        Optimizer which is used for model training and for which learning rate is updated using the scheduler.
    Returns
    -------
    scheduler: torch.optim.lr_scheduler.LambdaLR
        Returns LambdaLR scheduler instance using a cosine learning rate decay.
    """
    return LambdaLR(
        optimizer, lambda x: cosine_lr_decay(x, args.iters_per_epoch * args.epochs)
    )


# --- Training ---
def train(
    args: argparse.Namespace,
    model: torch.nn.Module,
    train_loader_labeled: DataLoader,
    train_loader_unlabeled: DataLoader,
    validation_loader: DataLoader,
    test_loader: DataLoader,
    writer: SummaryWriter,
    save_path: str
):
    """
    Method for FixMatch training of model based on given data loaders and parameters.

    Parameters
    ----------
    args: argparse.Namespace
        Namespace that contains all command line arguments with their corresponding values
    model: torch.nn.Module
        The torch model to train
    train_loader_labeled: DataLoader
        Data loader of labeled dataset
    train_loader_unlabeled: DataLoader
        Data loader of unlabeled dataset
    validation_loader: DataLoader
        Data loader of validation set (usually empty as by default FixMatch does not use a validation dataset)
    test_loader: DataLoader
        Data loader of test set
    writer: SummaryWriter
        SummaryWriter instance which is used to write losses as well as training / evaluation metrics
        to tensorboard summary file.
    save_path: str
        Path to which training data is saved.
    Returns
    -------
    model: torch.nn.Module
        The method returns the trained model
    ema_model: EMA
        The EMA class which maintains an exponential moving average of model parameters. In FixMatch the exponential
        moving average parameters are used for model evaluation and for the reported results.
    writer: SummaryWriter
        SummaryWriter instance which is used to write losses as well as training / evaluation metrics
        to tensorboard summary file.
    """
    model.to(args.device)

    if args.use_ema:
        ema_model = EMA(model, args.ema_decay)
    else:
        ema_model = None

    optimizer = get_optimizer(args, model)
    scheduler = get_scheduler(args, optimizer)

    start_epoch = 0

    if args.resume:
        state_dict = load_state(args.resume)
        model.load_state_dict(state_dict["model_state_dict"])
        if args.use_ema:
            ema_model.shadow = state_dict["ema_model_shadow"]
        optimizer.load_state_dict(state_dict["optimizer"])
        scheduler.load_state_dict(state_dict["scheduler"])
        start_epoch = state_dict["epoch"]

    for epoch in range(start_epoch, args.epochs):
        train_total_loss, train_labeled_loss, train_unlabeled_loss = train_epoch(
            args,
            model,
            ema_model,
            train_loader_labeled,
            train_loader_unlabeled,
            optimizer,
            scheduler,
            epoch,
        )

        if args.use_ema:
            ema_model.assign(model)
            val_metrics = evaluate(args, validation_loader, model, epoch, "Validation")
            test_metrics = evaluate(args, test_loader, model, epoch, "Test")
            ema_model.resume(model)
        else:
            val_metrics = evaluate(args, validation_loader, model, epoch, "Validation")
            test_metrics = evaluate(args, test_loader, model, epoch, "Test")

        writer.add_scalar("Loss/train_total", train_total_loss, epoch)
        writer.add_scalar("Loss/train_labeled", train_labeled_loss, epoch)
        writer.add_scalar("Loss/train_unlabeled", train_unlabeled_loss, epoch)
        write_metrics(writer, epoch, val_metrics, descriptor="val")
        write_metrics(writer, epoch, test_metrics, descriptor="test")
        writer.flush()

        if epoch % args.checkpoint_interval == 0 and args.save:
            save_state(
                epoch,
                model,
                val_metrics.top1,
                optimizer,
                scheduler,
                ema_model,
                save_path,
                filename=f"checkpoint_{epoch}.tar",
            )

    writer.close()
    logger.info(
        "Finished FixMatch training: \* Validation: Acc@1 {val_acc1:.3f}\tAcc@5 {val_acc5:.3f}\t Test: Acc@1 {test_acc1:.3f} Acc@5 {test_acc5:.3f}".format(
            val_acc1=val_metrics.top1,
            val_acc5=val_metrics.top5,
            test_acc1=test_metrics.top1,
            test_acc5=test_metrics.top5,
        )
    )

    save_state(
        epoch,
        model,
        val_metrics.top1,
        optimizer,
        scheduler,
        ema_model,
        save_path,
        filename="last_model.tar",
    )
    return model, ema_model, writer


def train_epoch(
    args: argparse.Namespace,
    model: torch.nn.Module,
    ema_model: EMA,
    train_loader_labeled: DataLoader,
    train_loader_unlabeled: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    epoch,
):
    """
    Method that executes a training epoch, i.e. a pass through all train samples in the training data loaders.

    Parameters
    ----------
    args: argparse.Namespace
        Namespace with command line arguments and corresponding values
    model: torch.nn.Module
        Model, i.e. neural network to train using FixMatch.
    ema_model: EMA
        The EMA class which maintains an exponential moving average of model parameters. In FixMatch the exponential
        moving average parameters are used for model evaluation and for the reported results.
    train_loader_labeled: DataLoader
        Data loader fetching batches from the labeled set of data.
    train_loader_unlabeled: DataLoader
        Data loader fetching batches from the unlabeled set of data.
    optimizer: Optimizer
        Optimizer used for model training. An SGD is used in FixMatch.
    scheduler: torch.optim.lr_scheduler.LambdaLR
        LambdaLR-Scheduler, which controls the learning rate using a cosine learning rate decay.
    epoch: int
        Current epoch
    Returns
    -------
    train_stats: Tuple
        The method returns a tuple containing the total, labeled and unlabeled loss.
    """
    meters = AverageMeterSet()

    model.zero_grad()
    model.train()
    if args.pbar:
        p_bar = tqdm(range(len(train_loader_labeled)))

    for batch_idx, batch in enumerate(
        zip(train_loader_labeled, train_loader_unlabeled)
    ):
        loss = train_step(args, model, batch, meters)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Update EMA model if configured
        if args.use_ema:
            ema_model(model)

        if args.pbar:
            p_bar.set_description(
                "Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.6f}.".format(
                    epoch=epoch + 1,
                    epochs=args.epochs,
                    batch=batch_idx + 1,
                    iter=len(train_loader_labeled),
                    lr=scheduler.get_last_lr()[0],
                )
            )
            p_bar.update()

    if args.pbar:
        p_bar.close()

    return (
        meters["total_loss"].avg,
        meters["labeled_loss"].avg,
        meters["unlabeled_loss"].avg,
    )


def train_step(args: argparse.Namespace, model: torch.nn.Module, batch: Tuple, meters: AverageMeterSet):
    """
    Method that executes a FixMatch training step, i.e. a single training iteration.

    Parameters
    ----------
    args: argparse.Namespace
        Namespace with command line arguments and corresponding values
    model: torch.nn.Module
        Model, i.e. neural network to train using FixMatch.
    batch: Tuple
        Tuple containing loaded objects from both labeled and unlabeled data loaders. Each object is another tuple
        containing samples and targets (only of labeled batch).
    meters: AverageMeterSet
        AverageMeterSet object which is used to track training and testing metrics (loss, accuracy, ...)
        over the entire training process.
    Returns
    -------
    loss: torch.Tensor
        Tensor containing the total FixMatch loss (attached to computational graph) used for optimization
        by backpropagation.
    """
    labeled_batch, unlabeled_batch = batch

    x_weak, labels = labeled_batch
    (u_weak, u_strong), _ = unlabeled_batch

    inputs = torch.cat((x_weak, u_weak, u_strong)).to(args.device)
    labels = labels.to(args.device)

    logits = model(inputs)
    logits_x = logits[:len(x_weak)]
    logits_u_weak, logits_u_strong = logits[len(x_weak):].chunk(2)
    del inputs

    # Compute standard cross entropy loss for labeled samples
    labeled_loss = F.cross_entropy(logits_x, labels, reduction="mean")

    # Compute pseudo-labels for unlabeled samples based on model predictions on weakly augmented samples
    with torch.no_grad():
        pseudo_labels = torch.softmax(logits_u_weak, dim=1)
        max_probs, targets_u = torch.max(pseudo_labels, dim=1)
        mask = max_probs.ge(args.threshold).float()

    # Compute unlabeled loss as cross entropy between strongly augmented (unlabeled) samples and previously computed
    # pseudo-labels.
    unlabeled_loss = (F.cross_entropy(logits_u_strong, targets_u, reduction="none") * mask).mean()

    # Compute total loss
    loss = labeled_loss.mean() + args.wu * unlabeled_loss

    meters.update("total_loss", loss.item(), 1)
    meters.update("labeled_loss", labeled_loss.mean().item(), logits_x.size()[0])
    meters.update("unlabeled_loss", unlabeled_loss.item(), logits_u_strong.size()[0])

    return loss

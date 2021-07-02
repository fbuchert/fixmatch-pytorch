import os
import logging
from tqdm import tqdm
from typing import Callable
from functools import partial

import torch
import torch.nn.functional as F
from PIL import Image
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torchvision import transforms
from torchvision.transforms import transforms


from augmentation.randaugment import RandAugment
from augmentation.augmentations import get_weak_augmentation, cutout, get_normalizer
from eval import evaluate
from datasets.config import IMG_SIZE
from utils.train import EMA, ModelWrapper, cosine_lr_decay, get_wd_param_list
from utils.eval import AverageMeterSet
from utils.metrics import write_metrics
from utils.misc import save_state, load_state

MIN_VALIDATION_SIZE = 50

logger = logging.getLogger()


# --- Transforms ---
def get_transform_dict(args, strong_aug):
    img_size = IMG_SIZE[args.dataset]
    padding = int(0.125 * img_size)
    return {
        "train": FixMatchTransform.labeled(args.dataset, img_size, padding),
        "train_unlabeled": FixMatchTransform.unlabeled(args.dataset, strong_aug, img_size, padding),
        "test": get_normalizer(args.dataset),
    }


# --- Optimization & Scheduler---
def get_optimizer(args, model):
    optim_params = get_wd_param_list(model)
    return SGD(
        optim_params,
        lr=args.lr,
        momentum=args.beta,
        weight_decay=args.wd,
        nesterov=True,
    )


def get_scheduler(args, optimizer):
    return LambdaLR(
        optimizer, lambda x: cosine_lr_decay(x, args.iters_per_epoch * args.epochs)
    )


class FixMatchTransform:
    def __init__(self, weak_transform: Callable, strong_transform: Callable = None):
        self.weak = weak_transform
        self.strong = strong_transform

    def __call__(self, img: Image):
        if self.strong is None:
            return self.weak(img)
        else:
            return self.weak(img), self.strong(img)

    @classmethod
    def labeled(cls, dataset: str, img_size: int, padding: int):
        return cls(
            weak_transform=transforms.Compose(
                [get_weak_augmentation(img_size, padding), get_normalizer(dataset)]
            )
        )

    @classmethod
    def unlabeled(cls, dataset: str, strong_aug: Callable, img_size: int, padding: int, cutout_mag: float = 0.5):
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


# --- Training ---
def train(
    args,
    model,
    train_loader_labeled,
    train_loader_unlabeled,
    validation_loader,
    test_loader,
    writer,
    save_path
):
    model.to(args.device)

    if args.use_ema:
        ema_model = EMA(model, args.ema_decay)
    else:
        ema_model = None

    optimizer = get_optimizer(args, model)
    scheduler = get_scheduler(args, optimizer)

    best_acc = 0
    start_epoch = 0

    if args.resume:
        state_dict = load_state(args.resume)
        model.load_state_dict(state_dict["model_state_dict"])
        if args.use_ema:
            ema_model.shadow = state_dict["ema_model_shadow"]
        optimizer.load_state_dict(state_dict["optimizer"])
        scheduler.load_state_dict(state_dict["scheduler"])
        best_acc = state_dict["acc"]
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

        # Only save best model (based on validation accurcay) if validation set is sufficiently large
        if (
            val_metrics.top1 > best_acc
            and args.save
            and len(validation_loader.dataset) > MIN_VALIDATION_SIZE
        ):
            save_state(
                epoch,
                model,
                val_metrics.top1,
                optimizer,
                scheduler,
                ema_model,
                save_path,
                filename="best_model.tar",
            )
            best_acc = val_metrics.top1

        if epoch % args.checkpoint_interval == 0 and args.save:
            old_checkpoint_files = list(
                filter(lambda x: "checkpoint" in x, os.listdir(save_path))
            )
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

            # Delete old checkpoint files in order to save space
            for file in old_checkpoint_files:
                os.remove(os.path.join(save_path, file))

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
    args,
    model,
    model_ema,
    train_loader_labeled,
    train_loader_unlabeled,
    optimizer,
    scheduler,
    epoch,
):
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
            model_ema(model)

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


def train_step(args, model, batch, meters):
    labeled_batch, unlabeled_batch = batch
    x, labels = labeled_batch
    if isinstance(x, tuple):
        x_weak, x_probe = x
    else:
        x_weak = x

    (u_weak, u_strong), _ = unlabeled_batch

    inputs = torch.cat((x_weak, u_weak, u_strong)).to(args.device)
    labels = labels.to(args.device)

    logits = model(inputs)
    logits_x = logits[:len(x_weak)]
    logits_u_weak, logits_u_strong = logits[len(x_weak):].chunk(2)
    del inputs

    Lx = F.cross_entropy(logits_x, labels, reduction="none")

    with torch.no_grad():
        pseudo_labels = torch.softmax(logits_u_weak, dim=1)
        max_probs, targets_u = torch.max(pseudo_labels, dim=1)
        mask = max_probs.ge(args.threshold).float()

    Lu = (F.cross_entropy(logits_u_strong, targets_u, reduction="none") * mask).mean()

    loss = Lx.mean() + args.wu * Lu

    meters.update("total_loss", loss.item(), 1)
    meters.update("labeled_loss", Lx.mean().item(), logits_x.size()[0])
    meters.update("unlabeled_loss", Lu.item(), logits_u_strong.size()[0])

    return loss

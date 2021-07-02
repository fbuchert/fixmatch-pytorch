import logging
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.eval import AverageMeterSet
from utils.metrics import accuracy, precision, recall, f1, evaluation_metrics


logger = logging.getLogger()


def evaluate(
        args,
        eval_loader: DataLoader,
        model: nn.Module,
        epoch: int,
        descriptor: str = "Test",
):
    """
    Evaluates current model based on the provided evaluation dataloader

    Parameters
    ----------
    args: argparse.Namespace
        Namespace with command line arguments and corresponding values
    eval_loader: torch.utils.data.DataLoader
        DataLoader objects which loads batches of evaluation dataset
    model: nn.Module
        Current model which should be evaluated on prediction task
    epoch: int
        Current epoch which is used for progress bar logging if enabled
    descriptor: str
        Descriptor which is used for progress bar logging if enabled

    Returns
    -------
    eval_tuple: namedtuple
        NamedTuple which holds all evaluation metrics such as accuracy, precision, recall, f1
    """
    meters = AverageMeterSet()

    model.eval()

    pred_labels = []
    true_labels = []

    if args.pbar:
        p_bar = tqdm(range(len(eval_loader)))
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(eval_loader):
            size = len(targets)

            inputs = inputs.to(args.device)
            targets = targets.to(args.device)

            # Output
            logits = model(inputs)
            loss = F.cross_entropy(logits, targets, reduction="mean")

            pred_labels.append(torch.argmax(logits, dim=-1))
            true_labels.append(targets)

            # Compute metrics
            top1, top5 = accuracy(logits, targets, topk=(1, 5))
            meters.update("loss", loss.item(), size)
            meters.update("top1", top1.item(), size)
            meters.update("top5", top5.item(), size)

            if args.pbar:
                p_bar.set_description(
                    "{descriptor}: Epoch: {epoch:4}. Iter: {batch:4}/{iter:4}. Class loss: {cl:4}. Top1: {top1:4}. Top5: {top5:4}".format(
                        descriptor=descriptor,
                        epoch=epoch + 1,
                        batch=i + 1,
                        iter=len(eval_loader),
                        cl=meters["loss"],
                        top1=meters["top1"],
                        top5=meters["top5"],
                    )
                )
                p_bar.update()
    if args.pbar:
        p_bar.close()
    logger.info(
        " * Prec@1 {top1.avg:.3f}\tPrec@5 {top5.avg:.3f}".format(
            top1=meters["top1"], top5=meters["top5"]
        )
    )

    pred_labels = torch.cat(pred_labels)
    true_labels = torch.cat(true_labels)

    eval_tuple = evaluation_metrics(
        loss=meters["loss"].avg,
        top1=meters["top1"].avg,
        top5=meters["top5"].avg,
        prec=precision(pred_labels.cpu(), true_labels.cpu(), average="micro"),
        rec=recall(pred_labels.cpu(), true_labels.cpu(), average="micro"),
        f1=f1(pred_labels.cpu(), true_labels.cpu(), average="micro"),
        prec_macro=precision(pred_labels.cpu(), true_labels.cpu(), average="macro"),
        rec_macro=recall(pred_labels.cpu(), true_labels.cpu(), average="macro"),
        f1_macro=f1(pred_labels.cpu(), true_labels.cpu(), average="macro"),
        prec_weighted=precision(
            pred_labels.cpu(), true_labels.cpu(), average="weighted"
        ),
        rec_weighted=recall(pred_labels.cpu(), true_labels.cpu(), average="weighted"),
        f1_weighted=f1(pred_labels.cpu(), true_labels.cpu(), average="weighted"),
    )
    return eval_tuple

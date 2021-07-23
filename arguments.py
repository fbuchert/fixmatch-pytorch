import argparse


def parse_args():
    """
    parse_args parses command line arguments and returns argparse.Namespace object

    Returns
    -------
    argparse.Namespace
        Namespace that contains all command line arguments with their corresponding values
    """
    parser = argparse.ArgumentParser(description="FixMatch training")

    # General arguments
    parser.add_argument(
        "--model",
        default="wide_resnet28_2",
        choices=[
            "vgg11",
            "vgg13",
            "vgg16",
            "vgg19",
            "resnet18",
            "resnet34",
            "resnet50",
            "wide_resnet28_2",
            "wide_resnet28_10",
            "wide_resnet50_10",
            "densenet121",
            "densenet161",
            "densenet201",
            "preact_resnet18",
            "preact_resnet34",
            "preact_resnet50",
        ],
        type=str,
        help="Used model architecture for classification task",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        type=str,
        choices=["cpu", "cuda"],
        help="Device used for training",
    )
    parser.add_argument(
        "--num-workers",
        default=0,
        type=int,
        help="Number of workers used for data loading",
    )
    parser.add_argument(
        "--out-dir",
        default="./out",
        type=str,
        help="path to which output logs, losses and model checkpoints are saved.",
    )
    parser.add_argument(
        "--dataset",
        default="cifar10",
        choices=[
            "cifar10",
            "cifar100",
            "imagenet",
            "svhn",
            "caltech101",
            "caltech256",
            "stl10",
            "ham10000",
        ],
        type=str,
        help="name of dataset",
    )
    parser.add_argument(
        "--data-dir", default="./data", type=str, help="path to which dataset is saved"
    )
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        help="path to checkpoint from which to resume training",
    )
    parser.add_argument("--epochs", default=1024, type=int, help="number of epochs")
    parser.add_argument(
        "--iters-per-epoch",
        default=1024,
        type=int,
        help="number of iterations per epoch",
    )
    parser.add_argument("--batch-size", default=64, type=int, help="batch_size")
    parser.add_argument("--lr", default=0.03, type=float, help="initial learning rate")
    parser.add_argument("--wd", default=0.0005, type=float, help="weight decay")
    parser.add_argument(
        "--ema-decay",
        default=0.999,
        type=float,
        help="exponential moving average decay",
    )
    parser.add_argument(
        "--pin-memory",
        action="store_true",
        help="Should CPU tensors be directly allocated in Pinned memory for data loading",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=50,
        help="Interval [epoch] in which checkpoints are saved.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=65,
        help="Manually set seed for random number generators",
    )
    parser.add_argument(
        "--trainable-layers",
        type=str,
        nargs='+',
        default=[],
        help='If pretrained flag is set, this specifies the layers for which weights should be frozen'
    )

    # Flags
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        action="store_true",
        default=False,
        help="Flag indicating if models should be loaded as pretrained (if available) or not",
    )
    parser.add_argument(
        "--weighted-sampling",
        dest="weighted_sampling",
        action="store_true",
        default=False,
        help="""Flag indicating if batches selects samples inversely proportional to class distribution,
                i.e. whether on average samples from each class should be selected with equal probability"""
    )
    parser.add_argument(
        "--save",
        dest="save",
        action="store_true",
        default=False,
        help="Flag indicating if models should be saved or not",
    )
    parser.add_argument(
        "--use-ema",
        dest="use_ema",
        action="store_true",
        default=False,
        help="use ema model for plotting",
    )
    parser.add_argument(
        "--random-seed",
        dest="random_seed",
        action="store_true",
        default=False,
        help="if specified seed is randomly set",
    )
    parser.add_argument(
        "--polyaxon",
        dest="polyaxon",
        action="store_true",
        default=False,
        help="Flag indicating if training is run on polyaxon or not",
    )
    parser.add_argument(
        "--pbar", dest="pbar", action="store_true", default=False, help="print progress bar"
    )

    # Dataset split settings
    parser.add_argument("--initial-indices", type=str, help='path to initial indice file to start from')
    parser.add_argument(
        "--num-labeled", default=250, type=float, help="number of labeled samples *per class* for SSL"
    )
    parser.add_argument(
        "--num-validation",
        type=int,
        default=1,
        help="Defines percentage used for training and validation",
    )

    # FixMatch arguments
    parser.add_argument(
        "--mu", default=7, type=int, help="coefficient of unlabeled batch size"
    )
    parser.add_argument(
        "--wu",
        default=1,
        type=float,
        help="Weight of unlabeled loss (all SSL algorithms) ",
    )
    parser.add_argument(
        "--num-augmentations",
        default=2,
        type=int,
        help="number of augmentations used for strong aug.",
    )
    parser.add_argument(
        "--m",
        default=30,
        type=int,
        help="magnitude of randaugment (strong augmentation)",
    )
    parser.add_argument(
        "--threshold",
        default=0.95,
        type=float,
        help="Pseudo-label threshold in FixMatch (tau)",
    )
    parser.add_argument(
        "--beta",
        default=0.9,
        type=float,
        help="beta for momentum of SGD optimizer in FixMatch",
    )
    parser.add_argument(
        "--lr-decay-K",
        default=2 ** 20,
        type=int,
        help="parameter K for cosine LR decay",
    )

    return parser.parse_args()

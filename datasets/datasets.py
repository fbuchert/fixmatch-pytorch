import random
from typing import List, Tuple, Callable

from torchvision import transforms
from torch.utils.data import Dataset

from datasets.config import *
from datasets.custom_datasets import *


DATASET_GETTERS = {
    "cifar10": datasets.CIFAR10,
    "cifar100": datasets.CIFAR100,
    "imagenet": datasets.ImageNet,
    "svhn": datasets.SVHN,
    "stl10": STL10,
    "caltech101": Caltech101,
    "caltech256": Caltech256,
    "ham10000": HAM10000,
}


def get_datasets(
        root_dir: str,
        dataset: str,
        num_labeled: int,
        num_validation: int = 1,
        labeled_train_transform: Callable = None,
        unlabeled_train_transform: Callable = None,
        test_transform: Callable = None,
        download: bool = True,
        dataset_indices: Optional[List] = None
):
    """
    Method that returns all dataset objects required for semi-supervised learning: labeled train set, unlabeled train
    set, validation set and test set. The returned dataset objects can be used as input for data loaders used during
    model training.

    Parameters
    ----------
    root_dir: str
        Path to root data directory to load datasets from or download them to, if not downloaded yet, e.g. `./data`.
    dataset: str
        Name of dataset, e.g. `cifar10`, `imagenet`, etc.
    num_labeled: int
        Number of samples selected for the labeled training set for semi-supervised learning. These samples are
        selected from all training samples.
    num_validation: int
        Number of samples selected for the validation set. These samples are selected from all available
        training samples.
    labeled_train_transform: Callable
        Transform / augmentation strategy applied to the labeled training set.
    unlabeled_train_transform: Callable
        Transform / augmentation strategy applied to the unlabeled training set.
    test_transform: Callable
        Transform / augmentation strategy applied to the validation and test set.
    download: bool
        Boolean indicating whether the dataset should be downloaded or not. If yes, the get_base_sets method will
        download the dataset to the root dir if possible. Automatic downloading is supported for CIFAR-10, CIFAR-100,
        STL-10 and ImageNet.
    dataset_indices: Optional[Dict]
        Dictionary containing indices for the labeled and unlabeled training sets, validation set and test set for
        initialization. This argument should be used if training is resumed, i.e. initializing the dataset splits to
        the same indices as in the previous training run, and dataset indices are loaded. An alternative use case,
        would be to select initial indices in a principled way, e.g. selecting diverse initial samples based on
        representations provided by self-supervised learning.
    Returns
    -------
    dataset_tuple: Tuple[Dict, List, List]
        Returns tuple containing dataset objects of all relevant datasets. The first tuple element is a dictionary
        containing the labeled training dataset at key `labeled` and the unlabeled training dataset at key unlabeled.
        The second and third elements are the validation dataset and the test dataset.
    """
    base_set, test_set = get_base_sets(
        dataset, root_dir, download=download, test_transform=test_transform
    )

    base_indices = list(range(len(base_set)))
    if dataset_indices is None:
        if dataset != 'stl10':
            num_training = len(base_indices) - num_validation
            train_indices, validation_indices = get_uniform_split(base_set.targets, base_indices, split_num=num_training)
            labeled_indices, unlabeled_indices = get_uniform_split(base_set.targets, train_indices, split_num=num_labeled)
        else:
            labeled_indices, unlabeled_indices, validation_indices = sample_stl10_ssl_indices(
                base_set.targets,
                base_set.labeled_indices,
                base_set.unlabeled_indices,
                num_validation,
                num_labeled
            )
    else:
        labeled_indices, unlabeled_indices, validation_indices = (
            dataset_indices["train_labeled"],
            dataset_indices["train_unlabeled"],
            dataset_indices["validation"],
        )

    labeled_train_set = CustomSubset(
        base_set, labeled_indices, transform=labeled_train_transform
    )
    unlabeled_train_set = CustomSubset(
        base_set, unlabeled_indices, transform=unlabeled_train_transform
    )
    validation_set = CustomSubset(
        base_set, validation_indices, transform=test_transform
    )

    return (
        {"labeled": labeled_train_set, "unlabeled": unlabeled_train_set},
        validation_set,
        test_set,
    )


def get_base_sets(dataset, root_dir, download=True, test_transform=None):
    base_set = DATASET_GETTERS[dataset](root_dir, train=True, download=download)
    test_set = DATASET_GETTERS[dataset](
        root_dir, train=False, download=download, transform=test_transform
    )
    return base_set, test_set


def sample_stl10_ssl_indices(
        targets: List,
        stl10_labeled: List,
        stl10_unlabeled: List,
        num_validation: int,
        num_labeled: int
):
    """
    Custom sampling strategy for labeled and unlabeled training indices as well as validation indices for STL-10. STL-10
    is a dataset specifically constructed for semi-supervised learning. Therefore the train set contains both labeled
    and unlabeled samples. As a consequence, only labeled samples can be considered for the selection of the labeled
    training as well as validation indices.

    Parameters
    ----------
    targets: List
        List of targets / class labels corresponding to provided indices of dataset.
    stl10_labeled: List
        List of indices in the STL-10 dataset, which are labeled.
    stl10_unlabeled: List
        List of indices in the STL-10 dataset, which are not labeled.
    num_validation: int
        Number of samples, which are sampled for the validation set.
    num_labeled: int
        Number of samples, which are sampled for the labeled dataset. This does not refer to the labeled part of the
        STL-10 dataset, but the subset of labeled samples for the current semi-supervised learning run, i.e. this
        allows for training only on a subset of labeled samples on STL-10.
    Returns
    ----------
    indices_tuple: Tuple[List, List, List]
        Return selected indices for the labeled and unlabeled train set as well as the validation set.
    """
    base_labeled_targets = np.array(targets)[stl10_labeled].tolist()
    train_indices, validation_indices = get_uniform_split(
        base_labeled_targets,
        stl10_labeled,
        split_num=len(stl10_labeled)-num_validation
    )
    labeled_idx, unlabeled_idx = get_uniform_split(base_labeled_targets, train_indices, split_num=num_labeled)
    validation_indices = np.array(stl10_labeled)[validation_indices].tolist()
    labeled_indices = np.array(stl10_labeled)[labeled_idx].tolist()
    unlabeled_indices = (
            np.array(stl10_labeled)[unlabeled_idx].tolist()
            + stl10_unlabeled
    )
    return labeled_indices, unlabeled_indices, validation_indices


def get_uniform_split(targets: List, indices: List, split_pct: float = None, split_num: int = None):
    """
    Method that splits provided train_indices uniformly according to targets / class labels, i.e. it returns a random
    split of train_indices s.t. indices in both splits are ideally uniformly distributed among classes (as done
    in FixMatch implementation by default).

    Parameters
    ----------
    indices: List
        List of dataset indices on which split should be performed.
    targets: List
        List of targets / class labels corresponding to provided indices of dataset. Based on the provided targets,
        the indices are split s.t. samples in split0 and split1 are uniformly distributed among classes as well as
        possible.
    split_num: int
        Number of total samples selected for first split. Alternatively one can specify a split percentage by providing
        split_pct as input.
    split_pct: float
        Percentage of all indices which are selected for the first split. Should only specified if split_num is not given.
    Returns
    ----------
    split_indices: Tuple[List, List]
        Returns two lists, which contain the indices split according to the parameters split_num or split_pct.
    """
    if split_pct is not None:
        samples_per_class = (split_pct * len(indices)) // len(np.unqiue(targets))
    elif split_num is not None:
        samples_per_class = split_num // len(np.unique(targets))
    else:
        raise ValueError('Expected either split_pct or split_num to be not None.')

    split0_indices, split1_indices = [], []
    for class_label in np.unique(targets):
        class_indices = np.where(np.array(targets)[indices] == class_label)[0]
        np.random.shuffle(class_indices)
        split0_indices += list(class_indices[:samples_per_class])
        split1_indices += list(class_indices[samples_per_class:])
    split0_indices = np.array(indices)[split0_indices].tolist()
    split1_indices = np.array(indices)[split1_indices].tolist()

    # Make sure that the number of selected indices exactly matches split_num if not None
    # If this is not the case, randomly sample indices from split1 and add them to split0
    if split_num is not None and len(split0_indices) < split_num:
        tmp_indices = random.sample(split1_indices, split_num - len(split0_indices))
        split0_indices += tmp_indices
        split1_indices = np.setdiff1d(split1_indices, tmp_indices).tolist()
    return split0_indices, split1_indices

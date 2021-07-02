import os
import random
from functools import partial
from typing import Optional, Callable, List

import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms, datasets

from datasets.datasets import *


class CustomSubset(Dataset):
    """
    CustomSubset is a custom implementation of a torch subset based on a given DataSet object. It is particular helpful,
    when dealing with train/validation/test splits as well as distinguishing
    """
    def __init__(
            self,
            dataset: Dataset,
            indices: List,
            transform: Callable,
            return_index: bool = False,
    ):
        """
        Initializes a new CustomSubset instance.

        Parameters
        ----------
        dataset: Dataset
            Base dataset. The CustomSubset contains only a subset of all sample indices and therefore is a subset of
            the base dataset.
        indices: List
            Indices of the base data set selected to form subset. These indices are a subset of all
            indices in the base dataset. If the base dataset has 10 samples in total for example, indices could be given
            by [1, 4, 9], i.e. forming a Subset of size 3.
        transform: Callable
            Transform / augmentation strategy which is applied to any sample upon call of __getitem__ function
        return_index: bool (default: False)
            Boolean which indicates if a call of __getitem__ also returns the original index of the item
            in the base dataset, i.e. not the index of the item in the current subset but in the original base dataset.
        """
        self.indices = indices
        self.dataset = dataset
        self.transform = transform

        try:
            self.targets = [self.dataset.targets[idx] for idx in self.indices]
        except AttributeError as ex:
            self.targets = []
        try:
            self.classes = dataset.classes
        except AttributeError as ex:
            self.classes = []

        self.return_index = return_index

    def __getitem__(self, idx: int):
        """
        Getter method for CustomSubset. Returns sample and label at given index (with respect to subset).

        Parameters
        ----------
        idx: int
            Get item at index idx of subset (idx references the index in the subset, not in the base set).
        Returns
        ----------
        item: Tuple
            Returns transformed sample and label at index idx of subset. If self.return_index is True, also the index
            in the original base set is returned. This is helpful for active learning query strategies, as
            one can access the index in the original base dataset instead of the index in the subset, which might be
            subject to change.
        """
        img, label = self.dataset[self.indices[idx]]
        if self.return_index:
            return self.transform(img), label, self.indices[idx]
        else:
            return self.transform(img), label

    def __len__(self):
        return len(self.indices)

    def update_subset_indices(self, new_indices: List):
        """
        Updates indices of CustomSubset

        Parameters
        ----------
        new_indices: List
            Update indices the subset references of the original base dataset.
        """
        self.indices = new_indices
        self.targets = [self.dataset.targets[idx] for idx in self.indices]


class Caltech(Dataset):
    """
    Class that implements a torch-compatible dataset object for Caltech101 and Caltech256, as the original
    torchvision.datasets objects for Caltech101 / Caltech256 do not work reliably.
    """
    def __init__(
        self,
        root: str,
        train: bool,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        train_split_pct: float = 9 / 10,
        img_size: int = 224,
        dataset_folder: str = ""
    ):
        """
        Initializes Caltech object instance.

        Parameters
        ----------
        root: str
            Path to root data directory where Caltech dataset is located, e.g. `./data`.
        train: bool
            Boolean indicating if train set or test set is loaded. If train=True the train set is loaded, while if
            train=False the test set is loaded.
        transform: Optional[Callable] (default: None)
            Transform / augmentation strategy that is applied to every image at every call of __getitem__.
        target_transform: Optional[Callable] (default: None)
            Transform / augmentation strategy that is applied to every target / label at every call of __getitem__.
            For numeric labels, such as in image classification, this is usually None.
        download: bool (default: False)
            Boolean indicating if dataset should be downloaded if not available. Automatic download of Caltech datasets
            is not implemented. This argument is added to ensure compatibility with constructors of other torch datasets.
        train_split_pct: float (default: 0.9)
            Float that defines the percentage of the split in training and test data. The training set is sampled
            to be train_split_pct % of all available samples.
        img_size: int (default: 224)
            Image size of loaded images. As images in Caltech datasets have different resolutions, it is common practice
            to resize them to a uniform size, e.g. 224x224 pixels.
        dataset_folder: str (default: "")
            Name of dataset folder in which the Caltech dataset is contained. The dataset folder has to be at the path
            specified by `root`.
        """
        self.root = root
        self.train = train
        self.train_split = train_split_pct
        self.data = []
        self.targets = []
        self.classes = []
        self.classes_to_idx = {}
        self.img_size = img_size
        self.dataset_folder = dataset_folder

        self.load_images()

        self.transform = transform
        self.target_transform = target_transform

    def load_images(self):
        """
        Method that loads paths and targets from the dataset folder of Caltech datasets.
        """
        base_path = os.path.join(self.root, self.dataset_folder)
        if not os.path.exists(base_path):
            raise RuntimeError(
                "Dataset not found! Please place ensure data set is present at {}".format(
                    base_path
                )
            )

        self.classes = sorted(filter(lambda x: not x.startswith('.') and "@ea" not in x, os.listdir(base_path)))
        try:
            # Remove 'BACKGROUND_Google' which is not a proper class if present (Caltech101)
            self.classes.remove("BACKGROUND_Google")
        except Exception as ex:
            pass

        self.classes_to_idx = {
            image_class: idx for idx, image_class in enumerate(self.classes)
        }

        for image_class in self.classes:
            class_paths = list(
                map(
                    lambda x: os.path.join(base_path, image_class, x),
                    filter(
                        lambda x: x.endswith(".jpg"),
                        sorted(os.listdir(os.path.join(base_path, image_class))),
                    )
                )
            )
            split_idx = int(len(class_paths) * self.train_split)
            split_image_paths = class_paths[:split_idx] if self.train else class_paths[split_idx:]
            self.data.extend(split_image_paths)
            self.targets.extend(len(split_image_paths) * [self.classes_to_idx[image_class]])

    def __getitem__(self, index):
        """
        Getter method for Caltech101. Returns sample and label at given index.

        Parameters
        ----------
        idx: int
            Get item at index idx of STL10.
        Returns
        ----------
        item: Tuple
            Returns transformed sample and label at index of STL10.
        """
        img, target = self.data[index], self.targets[index]

        img = transforms.Resize((self.img_size, self.img_size))(Image.open(img).convert("RGB"))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.targets)


Caltech101 = partial(Caltech, dataset_folder="caltech101/101_ObjectCategories")
Caltech256 = partial(Caltech, dataset_folder="caltech256/256_ObjectCategories")


class STL10(datasets.STL10):
    """
    STL10 is a class that implements the STL10 dataset and supplements the original torch implementation
    with class variables and a helper method.
    """
    def __init__(
        self,
        root: str,
        train: bool = True,
        folds: Optional[int] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ):
        """

        Parameters
        ----------
        root: str
            Path to root data directory where STL-10 dataset is located, e.g. `./data`.
        train: bool (default: True)
            Boolean indicating if train set or test set is loaded. If train=True the train set is loaded, while if
            train=False the test set is loaded.
        folds: Optional[int] (default: None)
            STL10 has 10 predefined folds of the dataset. If folds is not None, the specified fold is loaded. Otherwise,
            all 10 folds are loaded.
        transform: Optional[Callable] (default: None)
            Transform / augmentation strategy that is applied to every image at every call of __getitem__.
        target_transform: Optional[Callable] (default: None)
            Transform / augmentation strategy that is applied to every target / label at every call of __getitem__.
            For numeric labels, such as in image classification, this is usually None.
        download: bool (default: False)
            Boolean indicating if dataset should be downloaded if not available. Automatic download of Caltech datasets
            is not implemented. This argument is added to ensure compatibility with constructors of other torch datasets.
        """
        split = "train+unlabeled" if train else "test"
        super().__init__(
            root,
            split=split,
            folds=folds,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )
        self.targets = self.labels
        self.labeled_indices = np.argwhere(self.targets != -1).flatten().tolist()
        self.unlabeled_indices = np.argwhere(self.targets == -1).flatten().tolist()

    def get_random_labeled_indices(self, num_indices):
        return random.sample(self.labeled_indices, num_indices)


class HAM10000(Dataset):
    def __init__(
        self,
        root: str,
        train: bool,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        train_split: float = 9 / 10,
        img_size: int = 224,
    ):
        """
        Parameters
        ----------
        root: str
            Path to root data directory where STL-10 dataset is located, e.g. `./data`.
        train: bool (default: True)
            Boolean indicating if train set or test set is loaded. If train=True the train set is loaded, while if
            train=False the test set is loaded.
        transform: Optional[int] (default: None)
            Transform / augmentation strategy that is applied to every image at every call of __getitem__.
        target_transform: Optional[Callable] (default: None)
            Transform / augmentation strategy that is applied to every target / label at every call of __getitem__.
            For numeric labels, such as in image classification, this is usually None.
        download: bool (default: False)
            Boolean indicating if dataset should be downloaded if not available. Automatic download of Caltech datasets
            is not implemented. This argument is added to ensure compatibility with constructors of other torch datasets.
        train_split: float (default: 0.9)
            Float that defines the percentage of the split in training and test data. The training set is sampled
            to be train_split_pct % of all available samples.
        img_size: int (default: 224)
            Image size of loaded images. As images in HAM10000 datasets have different resolutions, it is common practice
            to resize them to a uniform size, e.g. 224x224 pixels.
        """
        self.root = root
        self.train = train
        self.train_split = train_split
        self.data = []
        self.targets = []
        self.classes = []
        self.classes_to_idx = {}
        self.img_size = img_size

        self.load_images()

        self.transform = transform
        self.target_transform = target_transform

    def load_images(self):
        """
        Method that loads paths and targets from the dataset folder of HAM10000 datasets.
        """
        base_path = os.path.join(self.root, "ham10000", "images")
        if not os.path.exists(base_path):
            raise RuntimeError(
                "Dataset not found! Please place ensure data set is present at {}".format(
                    base_path
                )
            )

        self.classes = sorted(os.listdir(base_path))
        self.classes_to_idx = {
            image_class: idx for idx, image_class in enumerate(self.classes)
        }

        for image_class in self.classes:
            class_paths = list(
                filter(
                    lambda x: x.endswith(".jpg"),
                    sorted(os.listdir(os.path.join(base_path, image_class))),
                )
            )
            np.random.shuffle(class_paths)
            split_idx = int(len(class_paths) * self.train_split)
            split_class_paths = (
                class_paths[:split_idx] if self.train else class_paths[split_idx:]
            )
            for image in split_class_paths:
                image_path = os.path.join(base_path, image_class, image)
                self.data.append(image_path)
                self.targets.append(self.classes_to_idx[image_class])

    def __getitem__(self, idx):
        """
        Getter method for HAM10000. Returns image and label at given index. The image is read from the image path
        saved at the specified index self.data.

        Parameters
        ----------
        idx: int
            Get item at index idx of HAM10000.
        Returns
        ----------
        item: Tuple
            Returns transformed sample and label at index of HAM10000.
        """
        img = cv2.imread(self.data[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = transforms.Resize((self.img_size, self.img_size))(
            Image.fromarray(img)
        )
        target = self.targets[idx]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.targets)

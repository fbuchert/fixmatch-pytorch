# Dict specifying used image sizes for model training
IMG_SIZE = {
    "cifar10": 32,
    "cifar100": 32,
    "imagenet": 256,  # ImageNet images have varying resolutions. It is common to train on subsampled 256x256 images
    "stl10": 96,
    "svhn": 32,
    "caltech101": 224,  # Caltech101 images have varying resolutions. It is common to train on subsampled 224x224 images
    "caltech256": 224,  # Caltech256 images have varying resolutions. It is common to train on subsampled 224x224 images
    "ham10000": 224,
}

# Mean and standard deviation values used for normalizing input images
NORMALIZATION_VARIABLES = {
    "cifar10": {
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2470, 0.2435, 0.2616)
    },
    "cifar100": {
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2470, 0.2435, 0.2616)
    },
    "imagenet": {
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225)
    },
    "svhn": {
        "mean": (0.4377, 0.4438, 0.4728),
        "std": (0.1980, 0.2010, 0.1970)
    },
    "caltech101": {
        "mean": (0.5468, 0.5290, 0.5024),
        "std": (0.3131, 0.3081, 0.3211)
    },
    "caltech256": {
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225)
    },
    "stl10": {
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2470, 0.2435, 0.2616)
    },
    "ham10000": {
        "mean": (0.7798, 0.5437, 0.5645),
        "std": (0.0845, 0.1125, 0.1264)
    }
}

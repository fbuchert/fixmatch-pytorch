# PyTorch Implementation: FixMatch
![License](https://img.shields.io/github/license/fbuchert/fixmatch-pytorch?label=license)

PyTorch implementation of [FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence
](https://arxiv.org/abs/2001.07685) based on the [official tensorflow implementation](https://github.com/google-research/fixmatch).

The implementation supports the following datasets:
- CIFAR-10 / CIFAR-100
- SVHN
- Caltech101 / Caltech256
- STL10
- HAM10000
- ImageNet


## Installation
Required python packages are listed in `requirements.txt`. All dependencies can be installed using pip
```
pip install -r requirements.txt
```
or using conda 
```
conda install --file requirements.txt
```

## Training
FixMatch training is started by running the following command (`--pbar` to show progress bar during training):
```
python main.py --pbar
```
All commandline arguments, which can be used to adapt the configuration of FixMatch are defined and described in `arguments.py`.
By default the following configuration is run:
```
model: 'wide_resnet28_2'
dataset: 'cifar10'
lr: 0.03
wd: 0.0005
num_labeled: 250 (number of labeled samples, i.e. 25 labeled samples per class for cifar10)
iters_per_epoch: 1024
batch_size: 64
epochs: 1024
device: 'cuda'
out_dir: 'fixmatch'
m: 30
threshold: 0.95
beta: 0.9
num_augmentations: 2
mu: 7
wu: 1
```
In addition to these, the following arguments can be used to further configure the FixMatch training process:
* `--device <cuda / cpu>`: Specify whether training should be run on GPU (if available) or CPU
* `--num-workers <num_workers>`: Number of workers used by torch dataloader
* `--resume <path to run_folder>`: Resumes training of training run saved at specified path, e.g. `'out/mixmatch_training/run_0'`. Dataset splits, model state, optimizer state, etc.
  are loaded and training is resumed with specified arguments.
* see `arguments.py` for more
  

Alternatively, the `polyaxon.yaml`-file can be used to start FixMatch training on a polyaxon-cluster:
```
polyaxon run -f polyaxon.yaml -u
```
For a general introduction to polyaxon and its commandline client, please refer to the [official documentation](https://github.com/polyaxon/polyaxon) 
## Monitoring
The training progress (loss, accuracy, etc.) can be monitored using tensorboard as follows:
```
tensorboard --logdir <result_folder>
```
This starts a tensorboard instance at `localhost:6006`, which can be opened in any common browser.

## Evaluation
A trained FixMatch model can be evaluated by running:
```
 python3 eval.py --run-path out/fixmatch_training/run_0 --pbar --device <cuda / cpu>
```
where `--run-path` specifies the path at which the run to be evaluated is saved. Alternatively, one can also check all 
metrics over all epochs using the tensorboard file.

## References
```
@article{sohn2020fixmatch,
    title={FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence},
    author={Kihyuk Sohn and David Berthelot and Chun-Liang Li and Zizhao Zhang and Nicholas Carlini and Ekin D. Cubuk and Alex Kurakin and Han Zhang and Colin Raffel},
    journal={arXiv preprint arXiv:2001.07685},
    year={2020},
}
```

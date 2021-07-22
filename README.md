# FixMatch
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
FixMatch training is started by running the following command:
```
python main.py
```
All commandline arguments, which can be used to adapt the configuration of FixMatch are defined and described in `arguments.py`.


Alternatively, the `polyaxon.yaml`-file can be used to start FixMatch training on a polyaxon-cluster:
```
polyaxon run -f polyaxon_runs/mixmatch_runs/al_cifar10.yaml -u
```
For a general introduction to polyaxon and its commandline client, please refer to the [official documentation](https://github.com/polyaxon/polyaxon) 
## Monitoring
The training progress (loss, accuracy, etc.) can be monitored using tensorboard as follows:
```
tensorboard --logdir <result_folder>
```
This starts a tensorboard instance at `localhost:6006`, which can be opened in any common browser.

## Evaluation


## References
```
@article{sohn2020fixmatch,
    title={FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence},
    author={Kihyuk Sohn and David Berthelot and Chun-Liang Li and Zizhao Zhang and Nicholas Carlini and Ekin D. Cubuk and Alex Kurakin and Han Zhang and Colin Raffel},
    journal={arXiv preprint arXiv:2001.07685},
    year={2020},
}
```
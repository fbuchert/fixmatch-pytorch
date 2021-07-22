import math
from itertools import product
from typing import Tuple, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init


class EMA:
    """
    Class that keeps track of exponential moving average of model parameters of a particular model.
    Also see https://github.com/chrischute/squad/blob/master/util.py#L174-L220.
    """

    def __init__(self, model: torch.nn.Module, decay: float):
        """
        Initialization method for the EMA class.

        Parameters
        ----------
        model: torch.nn.Module
            Torch model for which the EMA instance is used to track the exponential moving average of parameter values
        decay: float
            Decay rate used for exponential moving average of parameters calculation:
            ema_t = decay * p_t + (1-decay) * ema_(t-1)
        """
        self.decay = decay
        self.shadow = {}
        self.original = {}

        # Register model parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.clone().detach()

    def __call__(self, model):
        """
        Implements call method of EMA class

        Parameters
        ----------
        model: torch.nn.Module
            Current model based on which the EMA parameters are updated
        """
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    assert name in self.shadow
                    new_average = (1.0 - self.decay) * param + self.decay * self.shadow[
                        name
                    ]
                    self.shadow[name] = new_average

    def assign(self, model: torch.nn.Module):
        """
        This method assigns the parameter EMAs saved in self.shadow to the given model. The current parameter values
        of the model are saved to self.original. These original parameters can be restored using self.resume.

        Parameters
        ----------
        model: torch.nn.Module
            Model to which the current parameter EMAs are assigned.
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.original[name] = param.clone()
                param.data.copy_(self.shadow[name].data)

    def resume(self, model: torch.nn.Module):
        """
        This method restores the parameters saved in self.original to the given model. It is usually called after
        the `assign` method.

        Parameters
        ----------
        model: torch.nn.Module
            Torch model to which the original parameters are restored
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.original[name].data)


class ModelWrapper:
    """
    ModelWrapper which can be used to extract outputs of intermediate layer of a network.
    """
    def __init__(self, task_model: nn.Module, to_extract: Tuple):
        """
        Initializes a model wrapper for the specified task model and layer names to extract.

        Parameters
        ----------
        task_model: torch.nn.Module
            Torch model to which the original parameters are restored
        to_extract: Tuple
            Tuple that holds names of layers for which intermediate results should be extracted and returned,
            e.g. to_extract=(`avgpool`, `fc`) to extract intermediate results after the avgpool layer and last fully
            connected layer in a ResNet for example.
        """
        self.task_model = task_model
        self.to_extract = to_extract

    def __call__(self, x: torch.Tensor):
        """
        The __call__ method iterates through all modules of the provided `task_model` separately. It extracts and
        returns the intermediate results at layers specified by to_extract

        Parameters
        ----------
        x: torch.Tensor
            Batch of samples, e.g. images, which are passed through the network and for which specified intermediate
            results are extracted
        Returns
        ----------
        results: Optional[torch.Tensor, List[torch.Tensor]]
            Results of forward pass of input batch through the given task model. If len(to_extract) is 1, only the
            single result tensor is returned. Otherwise, a list of tensors is returned, which holds the intermediate
            results of specified layers in the order of occurrence in the network.
        """
        results = []
        for name, child in self.task_model.named_children():
            x = child(x)
            if name == "avgpool":
                x = torch.flatten(x, 1)
            if name in self.to_extract:
                results.append(x)
        return results[-1] if len(results) == 1 else results

    def train(self):
        self.task_model.train()

    def eval(self):
        self.task_model.eval()

    def cuda(self):
        self.task_model.cuda()

    def to(self, device: Union[str, torch.device]):
        self.task_model.to(device)

    def get_embedding_dim(self):
        last_layer = list(self.task_model.modules())[-1]
        return last_layer.in_features


def model_init(m: torch.nn.Module):
    """
    Method that initializes torch modules depending on their type:
        - Convolutional Layers: Xavier Uniform Initialization
        - BatchNorm Layers: Standard initialization
        - Fully connected / linear layers: Xavier Normal Initialization#

    Parameters
    ----------
    m: torch.nn.Module
        Torch module which to be initialized. The specific initialization used depends on the type of module.
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        init.xavier_uniform_(m.weight, gain=math.sqrt(2))
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif classname.find("BatchNorm") != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)
    elif classname.find("Linear") != -1:
        init.xavier_normal_(m.weight, gain=math.sqrt(2))
        if m.bias is not None:
            init.constant_(m.bias, 0)


def wd_check(wd_tuple: Tuple, name: str):
    """
    Method that checks if parameter name matches the key words in wd_tuple. This check is used to filter certain
    types of parameters independent of the layer, which it belongs to, e.g. `conv1.weight`.

    Parameters
    ----------
    wd_tuple: Tuple
        Tuple which contains the phrases which are checked for, e.g. (`conv`, `weight`) or (`fc`, `weight`)
    name: str
        Name of parameter as saved in state dict, e.g. `conv1.weight`
    Returns
    ----------
    wd_check: bool
        Returns a bool indicating whether all strings in wd_tuple are contained in name.
    """
    return all([x in name for x in wd_tuple])


def apply_wd(model: torch.nn.Module, wd: float, param_names: List = ["conv", "fc"], types: List = ["weight"]):
    """
    Method that manually applies weight decay to model parameters that match the specified parameter names and types.

    Parameters
    ----------
    model: torch.nn.Module
        Model to which weight decay is applied
    wd: float
        Float specifying weight decay. Parameters are updated to: param = (1-wd) * param
    param_names: List (default: ["conv", "fc"])
        Parameter names (or substring of names) for which the weight decay is applied.
    types: List (default: ["weight"])
        Parameter types for which weight decay is applied.
    """
    with torch.no_grad():
        for name, param in model.state_dict().items():
            if any(
                    [wd_check(wd_tuple, name) for wd_tuple in product(param_names, types)]
            ):
                param.mul_(1 - wd)


def set_bn_running_updates(model, enable: bool, bn_momentum: float = 0.001):
    """
    Method that enables or disables updates of the running batch norm vars by setting the momentum parameter to 0
    """
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = bn_momentum if enable else 0.0


def cosine_lr_decay(k: int, total_steps: int):
    return max(0.0, math.cos(math.pi * 7 * k / (16 * total_steps)))


def linear_rampup(current: int, rampup_length: int):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)


def set_grads(model: torch.nn.Module, trainable_layers: List[str]):
    """
    Method that enables or disables gradients of model parameters according to specified layers.

    Parameters
    ----------
    model: torch.nn.Module
        Torch model for which parameter gradients should be set
    trainable_layers: List
        List of strings, i.e. layer / parameter names, for which training is enabled. For model parameters, which do not
        match any pattern specified in trainable_layers, training is disable by setting requires_grad to False.
    """
    def is_trainable(x, trainable_layers):
        return any([(layer in x) or ('fc' in x) for layer in trainable_layers])

    for p in model.parameters():
        p.requires_grad = False

    trainable_parameters = [n for n, p in model.named_parameters() if is_trainable(n, trainable_layers)]
    for n, p in model.named_parameters():
        if n in trainable_parameters:
            p.requires_grad = True


def get_wd_param_list(model: torch.nn.Module):
    """
    Get list of model parameters to which weight decay should be applied. The function basically filters out
    all BatchNorm-related parameters to which weight decay should not be applied.

    Parameters
    ----------
    model: torch.nn.Module
        torch model which is trained using weight decay.
    Returns
    -------
    wd_param_list: List
        List containing two dictionaries containing parameters for which weight decay should be applied and parameters
        to which weight decay should not be applied.
    """
    wd_params, no_wd_params = [], []
    for name, param in model.named_parameters():
        # Filter BatchNorm parameters from weight decay parameters
        if "bn" in name:
            no_wd_params.append(param)
        else:
            wd_params.append(param)
    return [{"params": wd_params}, {"params": no_wd_params, "weight_decay": 0}]
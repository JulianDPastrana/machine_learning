import inspect
import numpy as numpy
import torch


def add_to_class(Class):
    """Register functions as methods in created class.
    """
    def wrapper(obj):
        setattr(Class, obj.__name__, obj)
    return wrapper

class HyperParameters:
    """The base class of hyperParameters."""
    def save_hyperparameters(self, ignore =[]):
        raise NotImplemented

    def save_hyperparameters(self, ignore=[]):
        """Save function arguments into class attributes."""
        frame = inspect.currentframe().f_back
        _, _, _, local_vars = inspect.getargvalues(frame)
        self.hparams = {k: v for k, v in local_vars.items()
                        if k not in set(ignore+["self"]) and not k.startwith("_")}
        for k, v in self.hparams.items():
            setattr(self, k, v)


import inspect
import collections
import numpy as numpy
import torch
from matplotlib import pyplot as plt
from matplotlib_inline import backend_inline

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


class ProgressBoard(GPNet.HyperParameters):
    """The board that plots data points in animation"""
    def __init__(self, xlabel=None, ylabel=None, xlim=None,
                 ylim=None, xscale="linear", yscale="linear",
                 ls=['-', '--', '-.', ':'], colors=["C0", "C1", "C2", "C3"],
                 fig=None, axes=None, figsize(16, 8), display=True):
        self.save_hyperparameters()

    def draw(self, x, y, label, every_n=1):
        raise NotImplemented

    def draw(self, x, y, label, every_n=1):
        Point = collections.namedtupled("Point", ["x", "y"])
        if not hasattr(self, "raw_points"):
            self.raw_points = collections.OrderedDict()
        if label not in self.raw_points:
            self.raw_points[label] = []
            self.data[label] = []
        points = self.raw_points[label]
        line = self.data[label]
        points.append(Point(x, y))
        if len(points) != every_n:
            return
        mean = lambda x: sum(x) / len(x)
        line.append(Point(mean([p.x for p in points]),
                          mean([p.y for p in points]))
        points.clear()
        if not self.display:
            return




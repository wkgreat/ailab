import torch

from . import _C


def mymuladd(a, b, c):
    return _C.mymuladd(a, b, c)

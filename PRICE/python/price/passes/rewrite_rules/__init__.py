from .base import *
from .std_rules import *
from .sched_rules import *


def resolve(x):
    if isinstance(x, str):
        x = eval(x)()
    assert isinstance(x, TransformRule)
    return x

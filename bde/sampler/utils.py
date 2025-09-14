"""General utility functions for the module sandbox."""
import inspect
import json
import logging
import operator
import time
from contextlib import contextmanager
from functools import reduce
from json.encoder import JSONEncoder

import jax
import jax.numpy as jnp

from bde.sampler.my_types import ParamTree

logger = logging.getLogger(__name__)

def count_params(params: ParamTree) -> int:
    """Count the number of parameters in the model."""
    return sum([x.size for x in jax.tree.leaves(params)])

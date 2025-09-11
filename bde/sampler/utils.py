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


@contextmanager
def measure_time(name: str):
    """Masure execution time."""
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    logger.info(f'{name} took {end - start:.2f} seconds.')


def flatten_posterior_samples(posterior_samples: dict, prefix: str = ''):
    """Flatten the posterior samples."""
    flat_dict = {}
    for k, v in posterior_samples.items():
        if isinstance(v, dict):
            flat_dict.update(flatten_posterior_samples(v, prefix=f'{prefix}{k}.'))
        else:
            flat_dict[f'{prefix}{k}'] = v
    return flat_dict


def pretty_string_dict(d: dict, indent: int = 3):
    """Pretty print serializable dictionary."""
    return json.dumps(d, indent=indent, cls=CustomJSONEncoder)


def get_flattened_keys(d: dict, sep='.') -> list[str]:
    """Recursively get `sep` delimited path to the leaves of a tree.

    Parameters:
    -----------
    d: dict
        Parameter Tree to get the names of the leaves from.
    sep: str
        Separator for the tree path.

    Returns:
    --------
        list of names of the leaves in the tree.
    """
    keys = []
    for k, v in d.items():
        if isinstance(v, dict):
            keys.extend([f'{k}{sep}{kk}' for kk in get_flattened_keys(v)])
        else:
            keys.append(k)
    return keys


def get_by_path(tree: dict, path: list):
    """Access a nested object in root by item sequence."""
    return reduce(operator.getitem, path, tree)


def set_by_path(tree: dict, path: list, value):
    """Set a value in a pytree by path."""
    reduce(operator.getitem, path[:-1], tree)[path[-1]] = value
    return tree


def count_chains(samples: ParamTree) -> int:
    """Find number of chains in the samples.

    Raises:
        ValueError: If the number of chains is not consistent across layers.
    """
    n = set([x.shape[0] for x in jax.tree.leaves(samples)])
    if len(n) > 1:
        raise ValueError(f'Ambiguous chain dimension across layers. Found {n}')
    return n.pop()


def count_samples(samples: ParamTree) -> int:
    """Find number of samples in the samples.

    Raises:
        ValueError: If the number of samples is not consistent across layers.
    """
    n = set([x.shape[1] for x in jax.tree.leaves(samples)])
    if len(n) > 1:
        raise ValueError(f'Ambiguous sample dimension across layers. Found {n}')
    return n.pop()


def get_mem_size(x: ParamTree) -> int:
    """Get the memory size of the model."""
    return sum([x.nbytes for x in jax.tree_leaves(x)])


def count_params(params: ParamTree) -> int:
    """Count the number of parameters in the model."""
    return sum([x.size for x in jax.tree.leaves(params)])


def count_nan(params: ParamTree) -> ParamTree:
    """Count the number of NaNs in the parameter tree."""
    return jax.tree.map(lambda x: jnp.isnan(x).sum().item(), params)


def impute_nan(params: ParamTree, value: float = 0.0) -> ParamTree:
    """Impute NaNs in the parameter tree with a value."""
    return jax.tree.map(lambda x: jnp.where(jnp.isnan(x), value, x), params)

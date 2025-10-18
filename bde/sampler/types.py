"""Central type aliases shared across the sampler subpackage.

`ParamTree` captures the nested pytree of model parameters, `FileTree` mirrors the
directory layout when persisting samples (kept for potential future use), and
`PRNGKey` annotates JAX pseudo-random keys.
"""

import typing
from pathlib import Path

import jax

ParamTree: typing.TypeAlias = dict[str, typing.Union[jax.Array, "ParamTree"]]
# `FileTree` mirrors the on-disk nesting when persisting checkpoints;
# retained for API stability.
FileTree: typing.TypeAlias = dict[str, typing.Union[Path, "FileTree"]]
# A `PRNGKey` is a uint32 array of shape (2,) produced by `jax.random.PRNGKey`.
PRNGKey = jax.Array

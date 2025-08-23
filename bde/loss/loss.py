"""Loss functions to be used
"""

from abc import ABC, abstractmethod

import jax
import optax.losses
from jax import numpy as jnp

from jax.typing import ArrayLike
from jax import Array
from jax.tree_util import register_pytree_node_class

from typing import (
    Any,
    Optional,
    Sequence,
    Tuple,
)


@register_pytree_node_class
class Loss(ABC):
    """An abstract base class defining an API for loss function classes.
    """

    @abstractmethod
    def __call__(
            self,
            y_true: ArrayLike,
            y_pred: ArrayLike,
            **kwargs,
    ) -> Array:
        """Evaluate the loss.

        Returns an unreduced evaluation of the loss, i.e. the loss is calculated
        separately for each item in the batch.

        Parameters
        ----------
        y_true : ArrayLike
            The ground truth labels.
        y_pred : ArrayLike
            The predictions.

        Returns
        -------
        Array
            The unreduced loss value.
        """
        ...

    @jax.jit
    def apply_reduced(
            self,
            y_true: ArrayLike,
            y_pred: ArrayLike,
            **kwargs,
    ) -> ArrayLike:
        """Evaluate and reduces the loss.

        The loss is evaluated separately for each item in the batch and the loss of
        all batches is reduced by arithmetic mean to a single value.

        Parameters
        ----------
        y_true :  ArrayLike
            The ground truth labels.
        y_pred : ArrayLike
            The predictions.
        **kwargs
            Other keywords that may be passed to the unreduced loss function.

        Returns
        -------
        ArrayLike
            The reduced loss value.
        """
        return self(y_true=y_true, y_pred=y_pred, **kwargs).mean()

    @abstractmethod
    def tree_flatten(self) -> Tuple[Sequence[ArrayLike], Any]:
        """Specify how to serialize module into a JAX PyTree.

        Returns
        -------
        A tuple with 2 elements:
         - The `children`, containing arrays & PyTrees
         - The `aux_data`, containing static and hashable data.
        """
        ...

    @classmethod
    @abstractmethod
    def tree_unflatten(
            cls,
            aux_data: Optional[Tuple],
            children: Tuple,
    ) -> "Loss":
        """Specify how to build a module from a JAX PyTree.

        Parameters
        ----------
        aux_data : Optional[Tuple]
            Contains static, hashable data.
        children : Tuple
            Contain arrays & PyTrees.

        Returns
        -------
        Loss
            Reconstructed loss function.
        """
        ...


class LossMSE(Loss):

    # @jax.jit
    def __call__(
            self,
            y_true: ArrayLike,
            y_pred: ArrayLike,
            **kwargs,
    ) -> Array:
        r"""Evaluate the loss.

        Evaluates an unreduced MSE loss, i.e. the loss is calculated separately for
        each item in the batch.

        Parameters
        ----------
        y_true
            The ground truth labels.
        y_pred
            The predictions.

        Returns
        -------
        Array
            The unreduced loss value.
        """
        res = optax.losses.squared_error(y_pred, y_true)  # shape (N, D, ...)
        return res.mean(axis=tuple(range(1, res.ndim)))  # -> (N,)

    def tree_flatten(self) -> Tuple[Sequence[ArrayLike], Any]:
        """
        #TODO:documentation
        Returns
        -------
        Tuple[Sequence[ArrayLike], Any]

        """
        return tuple(), None  # children=[], aux_data=None

    @classmethod
    def tree_unflatten(
            cls,
            aux_data: Optional[Tuple],
            children: Tuple,
    ) -> "Loss":
        return LossMSE()

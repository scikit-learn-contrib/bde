"""The class DataGen, generates arbitrary data!"""

from typing import Optional

import jax
import jax.numpy as jnp



class TaskType:
    """Holder for task types."""
    REGRESSION = "regression"
    CLASSIFICATION = "classification"


class DataLoader:
    def __init__(
            self,
            *,
            seed: int = 0,
            n_samples: int = 1024,
            n_features: int = 10,
            task: Optional[str] = None,
    ):
        self.seed = int(seed)
        self.n_samples = int(n_samples)
        self.n_features = int(n_features)

        # Generate data by default
        self.data = self._data_gen()
        self.x = self.data["x"]
        self.y = self.data["y"]
        self.w = self.data["w"]

        # self.task = self.infer_task() # TODO: figure out where you are going with this

    def _data_gen(self) -> dict[str, jnp.ndarray]:
        """Generate a simple linear regression dataset: y = X @ w + noise."""
        key = jax.random.PRNGKey(self.seed)
        k_x, k_w, k_eps = jax.random.split(key, 3)
        x = jax.random.normal(k_x, (self.n_samples, self.n_features))
        w = jax.random.normal(k_w, (self.n_features, 1))
        y = x @ w + jax.random.normal(k_eps, (self.n_samples, 1))
        return {"x": x, "w": w, "y": y}

    def infer_task(self) -> "TaskType":
        """This method is responsible for bringing the data in correct format according to the problem,
        is it a regression or classification problem?

        Returns
        -------

        """
        # TODO: figure out where you are going with this
        pass

    @classmethod
    def from_arrays(
            cls,
            X,
            y=None,
            *,
            task: Optional[str] = None,
            kind: str = "user",
    ) -> "DataLoader":
        """
        Construct a loader from user-provided arrays.

        Notes
        -----
        - Implementation should normalize to jax.numpy arrays,
          ensure X shape (N, D) and y shape (N, 1) if provided,
          and store `task` or leave it None for later inference.
        """
        # TODO: implement normalization & shape checks; set fields accordingly
        pass

    def validate(self):
        """This method ensures the shapes and the types of the data


        Returns
        -------

        """

        # TODO: code this method later
        pass

    def keys(self):
        """This method exposes the keys or available fields
        Returns
        -------

        """
        # TODO: figure out where we are going with this
        return list(self.data.keys())

    def load(self):
        """This method loads  the correct type of data
        Returns
        -------

        """
        # TODO: see if we will use sth like this!
        pass

    @property
    def feature_dim(self) -> int:
        return int(self.x.shape[1])

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self) -> int:
        return int(self.x.shape[0])

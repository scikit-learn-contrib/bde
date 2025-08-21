"""this is a bde builder"""

from models.models import Fnn
from training.trainer import FnnTrainer


class BdeBuilder(Fnn, FnnTrainer):
    # TODO: build the BdeBuilderClass
    def __init__(self, sizes, n_members, epochs):
        Fnn.__init__(self, sizes)
        FnnTrainer.__init__(self)
        self.sizes = sizes
        self.n_members = n_members
        self.epochs = epochs

        self.members = []

    def get_model(self, seed: int) -> Fnn:
        """Create a single Fnn model and initialize its parameters

        Parameters
        ----------
        seed : int
            #TODO: complete documentation

        Returns
        -------

        """
        m = Fnn(self.sizes)
        m.init_mlp(seed=seed)
        return m

    def deep_ensemble_creator(self):
        """Create an ensemble of ``n_members`` FNN models.

        Each member is initialized with a different random seed to encourage
        diversity within the ensemble. The initialized models are stored in the
        ``members`` attribute and returned.

        Returns
        -------
        list[Fnn]
            List of initialized FNN models comprising the ensemble.
        """

        self.members = [self.get_model(seed) for seed in range(self.n_members)]
        return self.members

    def store_ensemble_results(self):
        pass

    def fit(self):
        pass

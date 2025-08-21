"""this is a bde builder"""

from models.models import Fnn
from training.trainer import FnnTrainer
import optax


class BdeBuilder(Fnn, FnnTrainer):
    # TODO: build the BdeBuilderClass
    def __init__(self, sizes, n_members, epochs, optimizer):
        Fnn.__init__(self, sizes)
        FnnTrainer.__init__(self)
        self.sizes = sizes
        self.n_members = n_members
        self.epochs = epochs

        self.members = []
        self.optimizer = optimizer or optax.adam(learning_rate=0.01)

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

    def fit(self, model, X, y, optimizer, epochs=100):
        """Train each member of the ensemble

        Parameters
        ---------
        #TODO: documentation
        """
        if not self.members:
            self.deep_ensemble_creator()

        for member in self.members:
            super().fit(model=member, X=X, y=y, optimizer=self.optimizer, epochs=epochs or self.epochs)
        return self.members

    def store_ensemble_results(self):
        pass

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

    def get_model(self, seed):
        m = Fnn(self.sizes)
        return m.init_mlp(seed=seed)

    def deep_ensemble_creator(self):
        pass

    def store_ensemble_results(self):
        pass

    def fit(self):
        pass

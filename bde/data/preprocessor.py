"""This class is for preprocessing data"""

from dataloader import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np

class DataPreProcessor:
    def __init__(self, data: DataLoader):
        self.data = data

    def norm_data(self, data: DataLoader) -> DataLoader:
        # TODO: code [later]

        pass

    def aug_data(self, data: DataLoader) -> DataLoader:
        # TODO: code [later]

        pass

    def add_noise_data(self, data: DataLoader) -> DataLoader:
        # TODO: code [later]

        pass

    def split(self, test_size=0.15, val_size=0.15, random_state=42):
        """Split dataset into train, validation, and test sets.

        Parameters
        ----------
        test_size : float, optional
            Fraction for test set.
        val_size : float, optional
            Fraction for validation set.
        random_state : int, optional
            Random seed.

        Returns
        -------
        tuple
            (train, val, test) as Preprocessor instances.
        """
        indices = np.arange(len(self.data))

        # Split  test set
        train_val_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=random_state)

        # Split train into train + val
        val_relative_size = val_size / (1 - test_size)
        train_idx, val_idx = train_test_split(train_val_idx, test_size=val_relative_size, random_state=random_state)

        # Create new DataPreProcessor instances
        # TODO: figure out how to  code this!
        # def new_inst(idx):
        #     return DataPreProcessor(
        #         self.data)
        #     return new_inst(train_idx), new_inst(val_idx), new_inst(test_idx)
        pass

"""This class is for preprocessing data"""

from dataloader import DataLoader


class DataPreProcessor:
    def __init__(self, data: DataLoader):
        self.data = data
        pass

    def norm_data(self, data: DataLoader) -> DataLoader:
        pass

    def aug_data(self, data: DataLoader) -> DataLoader:
        pass

    def add_noise_data(self, data: DataLoader) -> DataLoader:
        pass
from abc import ABC, abstractmethod

import pandas as pd


class FeatureSet(ABC):
    def __init__(self) -> None:
        self.history = None

    def _update_history(self, new_data: pd.DataFrame):
        if self.history is None:
            self.history = new_data
        else:
            self.history = pd.concat([self.history, new_data], axis=0)

    @abstractmethod
    def update(self, new_data: pd.DataFrame):
        """Updates the current state given a new batch of data"""
        pass

    @abstractmethod
    def get_features(self, X: pd.DataFrame):
        """Returns current values of features to all entities in X"""
        pass

from pandas import DataFrame

from joust.features import FeatureSet

class PlayerPerformance(FeatureSet):
    def __init__(self, player_history: DataFrame) -> None:
        self.player_history = player_history

    def update(self, new_data: DataFrame):
        pass
    
    def get_features(self, X: DataFrame):
        return super().get_features(X)
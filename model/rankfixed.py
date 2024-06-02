from rank import Rank


class RankFixed(Rank):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
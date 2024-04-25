

# standard library imports
# None

# related third-party
import torch


# local application/library specific imports
# None


class BaseNetModel(torch.nn.Module):

    def __init__(self, embedding_model):
        super(BaseNetModel, self).__init__()
        self.model_path = None
        self.embedding_model = embedding_model

    def forward(self, i1, i2):
        pass

    def forward(self, i1, i2, i3=None):
        pass

    def freeze(self):
        self.embedding_model.freeze()
        return

    def unfreeze(self):
        self.embedding_model.unfreeze()
        return

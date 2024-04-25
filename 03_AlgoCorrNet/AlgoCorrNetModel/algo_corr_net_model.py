


# standard library imports
# None

# related third-party
import torch

from AlgoCorrNetModel.base_net_model import BaseNetModel


# local application/library specific imports
# None

class AlgoCorrNetModel(BaseNetModel):

    def __init__(self, embedding_model):
        super().__init__(embedding_model)

    def forward(self, i1, i2):
        text_embeddings1 = self.embedding_model(**i1)
        text_embeddings2 = self.embedding_model(**i2)
        logits = torch.nn.functional.cosine_similarity(text_embeddings1, text_embeddings2)
        return logits

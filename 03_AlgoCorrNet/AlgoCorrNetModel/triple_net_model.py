


# standard library imports
# None

# related third-party
# None

from AlgoCorrNetModel.base_net_model import BaseNetModel


# local application/library specific imports
# None


class TripleNetModel(BaseNetModel):

    def __init__(self, embedding_model):
        super().__init__(embedding_model)

    def forward(self, i1, i2, i3=None):
        text_embeddings1 = self.embedding_model(**i1)
        text_embeddings2 = self.embedding_model(**i2)
        if i3 is None:
            return text_embeddings1, text_embeddings2
        else:
            text_embeddings3 = self.embedding_model(**i3)
            return text_embeddings1, text_embeddings2, text_embeddings3

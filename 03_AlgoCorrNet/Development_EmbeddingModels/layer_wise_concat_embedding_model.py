


# standard library imports
import os

# related third-party
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel

# local application/library specific imports
from APP_CONFIG import Config

# define configuration proxy
working_dir = os.path.dirname(os.getcwd())
configProxy = Config(working_dir)

# get configuration
CONFIG = configProxy.return_config()

MODEL_NAME = CONFIG['BERT_EMBEDDING_MODEL_NAME']


class EmbeddingModel(torch.nn.Module):
    def __init__(self, model_name=MODEL_NAME):
        super(EmbeddingModel, self).__init__()
        config = AutoConfig.from_pretrained(model_name)
        config.update({'output_hidden_states': True})
        self.model = AutoModel.from_pretrained(model_name, config=config)
        self.hidden = F.tanh

    def freeze(self):
        for param in self.model.base_model.parameters():
            param.requires_grad = False
        return

    def unfreeze(self):
        for param in self.model.base_model.parameters():
            param.requires_grad = True
        return

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None, **kwargs):

        # The base bert model do not take labels as input
        if token_type_ids is None:
            model_output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        else:
            model_output = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        all_hidden_states = torch.stack(model_output[2])

        concatenate_pooling = torch.cat(
            (all_hidden_states[-1], all_hidden_states[-2], all_hidden_states[-3]), -1
        )
        output_embeddings = concatenate_pooling[:, 0]

        return output_embeddings

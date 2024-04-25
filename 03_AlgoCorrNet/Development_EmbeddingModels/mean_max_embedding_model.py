


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
        self.hidden = torch.nn.Linear(1536, 768)

    def freeze(self):
        for param in self.model.base_model.parameters():
            param.requires_grad = False
        for param in self.hidden.parameters():
            param.requires_grad = False
        return

    def unfreeze(self):
        for param in self.model.base_model.parameters():
            param.requires_grad = True
        for param in self.hidden.parameters():
            param.requires_grad = True
        return

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None, **kwargs):

        # The base bert model do not take labels as input
        if token_type_ids is None:
            model_output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        else:
            model_output = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        last_hidden_states = model_output[0]

        # Expand Attention Mask from [batch_size, max_len] to [batch_size, max_len, hidden_size]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
        # Sum Embeddings along max_len axis, so now we have [batch_size, hidden_size]
        sum_embeddings = torch.sum(last_hidden_states * input_mask_expanded, 1)
        # Sum Mask along max_len axis. This is done so that we can ignore padding tokens.
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        # Take Average.
        mean_embeddings = sum_embeddings / sum_mask

        # Get the max embeddings
        last_hidden_states[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
        max_embeddings = torch.max(last_hidden_states, 1)[0]

        # Get the mean-max embeddings
        mean_max_embeddings = torch.cat((mean_embeddings, max_embeddings), 1)

        # Squeeze the mean_max_embeddings to a lower size and apply ReLU as activation layer
        output_embeddings = self.hidden(mean_max_embeddings)

        return output_embeddings

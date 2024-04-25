


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
        self.cnn1 = torch.nn.Conv1d(768, 256, kernel_size=2, padding=1)
        self.cnn2 = torch.nn.Conv1d(256, 1, kernel_size=2, padding=1)
        self.hidden = torch.nn.Linear(514, 256)

    def freeze(self):
        for param in self.model.base_model.parameters():
            param.requires_grad = False
        for param in self.cnn1.parameters():
            param.requires_grad = False
        for param in self.cnn2.parameters():
            param.requires_grad = False
        return

    def unfreeze(self):
        for param in self.model.base_model.parameters():
            param.requires_grad = True
        for param in self.cnn1.parameters():
            param.requires_grad = True
        for param in self.cnn2.parameters():
            param.requires_grad = True
        return

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None, **kwargs):

        # The base bert model do not take labels as input
        if token_type_ids is None:
            model_output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        else:
            model_output = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        last_hidden_states = model_output[0]

        last_hidden_states = last_hidden_states.permute(0, 2, 1)
        cnn_embeddings = F.relu(self.cnn1(last_hidden_states))
        cnn_embeddings = self.cnn2(cnn_embeddings)

        output_embeddings = cnn_embeddings.view(cnn_embeddings.size(0), -1)
        output_embeddings = self.hidden(output_embeddings)

        return output_embeddings


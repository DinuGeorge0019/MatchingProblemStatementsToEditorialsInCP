import os
import pandas as pd

from sentence_transformers import SentenceTransformer, LoggingHandler
from sentence_transformers import models, util, datasets, evaluation, losses
from torch.utils.data import DataLoader


# local application/library specific imports
from APP_CONFIG import Config

# define configuration proxy
working_dir = os.path.dirname(os.getcwd())
configProxy = Config(working_dir)

# get configuration
CONFIG = configProxy.return_config()

# get global constants configuration
GLOBAL_CONSTANTS = configProxy.return_global_constants()

BERT_EMBEDDING_MODEL_NAME = CONFIG['BERT_EMBEDDING_MODEL_NAME']


def train_tsdae_model():
    # Define your sentence transformer model using CLS pooling
    model_name = BERT_EMBEDDING_MODEL_NAME
    word_embedding_model = models.Transformer(model_name)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), 'cls')
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    # Read the dataset
    train_dataset = pd.read_csv(CONFIG['PREPROCESSED_DATASET_PATH'])
    train_dataset = train_dataset['statement'].tolist() + train_dataset['editorial'].tolist()
    
    # Create the special denoising dataset that adds noise on-the-fly
    train_dataset = datasets.DenoisingAutoEncoderDataset(train_dataset)

    # DataLoader to batch your data
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    # Use the denoising auto-encoder loss
    train_loss = losses.DenoisingAutoEncoderLoss(model, decoder_name_or_path=model_name, tie_encoder_decoder=True)

    # Call the fit method
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=1,
        weight_decay=0,
        scheduler='constantlr',
        optimizer_params={'lr': 3e-5},
        show_progress_bar=True
    )

    model.save(CONFIG["TSDAE_MODEL_PATH"] + 'tsdae-model')



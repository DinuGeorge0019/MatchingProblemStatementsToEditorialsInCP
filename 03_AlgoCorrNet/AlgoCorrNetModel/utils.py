


# standard library imports
import os
import math

# related third-party
import torch
from transformers import MPNetTokenizer, BertTokenizer, RobertaTokenizer, AutoTokenizer

# local application/library specific imports
from APP_CONFIG import Config

# define configuration proxy
working_dir = os.path.dirname(os.getcwd())
configProxy = Config(working_dir)

# get configuration
CONFIG = configProxy.return_config()
GLOBAL_CONSTANTS = configProxy.return_global_constants()

BERT_EMBEDDING_MODEL_NAME = CONFIG['BERT_EMBEDDING_MODEL_NAME']
BERT_MAX_LENGTH_TENSORS = GLOBAL_CONSTANTS['BERT_MAX_LENGTH_TENSORS']

# TOKENIZER = MPNetTokenizer.from_pretrained(BERT_EMBEDDING_MODEL_NAME)
# TOKENIZER = RobertaTokenizer.from_pretrained(BERT_EMBEDDING_MODEL_NAME)
# TOKENIZER = BertTokenizer.from_pretrained(BERT_EMBEDDING_MODEL_NAME)
TOKENIZER = AutoTokenizer.from_pretrained(BERT_EMBEDDING_MODEL_NAME)


def encode(premise):
    """
    Encode the given premise using a tokenizer.

    Args:
        premise (str): The premise to be encoded.
    Returns:
        dict: A dictionary containing the encoded premise as a PyTorch tensor with shape [BERT_MAX_LENGTH_TENSORS], along with an attention mask.
    """

    encoded_premise = TOKENIZER(
        premise,                                # Encode each sentence
        add_special_tokens=True,                # Compute the CLS token
        truncation=True,                        # Truncate the embeddings to max_length
        max_length=BERT_MAX_LENGTH_TENSORS,     # Pad & truncate all sentences.
        padding='max_length',                   # Pad the embeddings to max_length
        return_attention_mask=True,             # Construct attention masks.
        return_tensors='pt'                     # Return pytorch tensors.
    )

    for item, tensor in encoded_premise.items():
        encoded_premise[item] = torch.squeeze(tensor)

    return encoded_premise


def format_time(elapsed_time):
    """
    Convert elapsed time in seconds to a formatted string in the format hh:mm:ss.

    Args:
        elapsed_time (float): The elapsed time in seconds.
    Returns:
        str: A formatted string representing the elapsed time in the format hh:mm:ss.
    """

    # Convert total seconds to hours, minutes, and seconds
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    hours = int(math.ceil(hours))
    minutes = int(math.ceil(minutes))
    seconds = int(math.ceil(seconds))

    formatted_time = f"{hours:02}:{minutes:02}:{seconds:02}"

    # Format time as hh:mm:ss
    return formatted_time

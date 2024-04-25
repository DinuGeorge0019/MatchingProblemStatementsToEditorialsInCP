


# standard library imports
import os
import re
import string

# related third-party
import pandas as pd
import torch
from torch.utils.data import DataLoader, SequentialSampler
from transformers import BatchEncoding

# local application/library specific imports
from APP_CONFIG import Config
from AlgoCorrNetModel.datasets import EditorialIndexDataset
from AlgoCorrNetModel.datatypes import AlgoNetChatFeature
from AlgoCorrNetModel.utils import encode

# define configuration proxy
working_dir = os.path.dirname(os.getcwd())
configProxy = Config(working_dir)

# get configuration
CONFIG = configProxy.return_config()


class AlgoNetChat:
    def __init__(self, model):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model
        self.model = self.model.to(self.device)
        self.batch_size = 512
        self.__set_editorials_dataloader()

    def __set_editorials_dataloader(self):
        """
        Set the data loaders for creating the index of editorials.

        Returns:
            None
        """

        self.editorials_index_df = pd.read_csv(
            CONFIG['EDITORIALS_INDEX_PATH'],
            usecols=[1]
        )

        editorial_index_dataset = EditorialIndexDataset(self.editorials_index_df)
        self.editorial_index_dataloader = DataLoader(
            editorial_index_dataset,
            sampler=SequentialSampler(editorial_index_dataset),
            batch_size=self.batch_size
        )

    def __preprocess_statement(self, statement_text):
        """
        Preprocesses the statement text by removing unknown symbols, punctuation, and unwanted spaces.

        Args:
            statement_text (str): The statement text to be preprocessed.

        Returns:
            str: The preprocessed statement text with unknown symbols, punctuation, and unwanted spaces removed.
        """
        assert statement_text is not None, "statement_text cannot be None"

        def search_unknown_symbols():
            unknown_symbols_set = set()
            # search for unknown symbols inside the statement_text
            for character in statement_text:
                if character not in string.printable:
                    unknown_symbols_set.add(character)

            # get the full unknown symbols set
            found_unknown_symbols = "".join(unknown_symbols_set)

            # return the full unknown symbols set
            return found_unknown_symbols

        # remove unknown symbols
        unknown_symbols = search_unknown_symbols()
        if unknown_symbols:
            statement_text = re.sub('[%s]' % re.escape(unknown_symbols), ' ', statement_text)
        else:
            pass
        # removing punctuations
        statement_text = re.sub('[%s]' % re.escape(string.punctuation), ' ', statement_text)
        # removing all unwanted spaces
        statement_text = re.sub('\s+', ' ', statement_text)

        return statement_text

    def __create_editorial_index(self, statement_text):
        """
        Creates an editorial index for the given statement text by encoding it, and computing similarity scores with the editorial index data.

        Args:
            statement_text (str): The statement text for which the editorial index is created.

        Returns:
            list: A list of similarity scores and editorial IDs, where each element is a list [score, editorial_id].
        """
        assert statement_text is not None, "statement_text cannot be None"

        # Put the model in evaluation mode - the dropout layers behave differently during evaluation.
        self.model.eval()

        # Encode the statement text
        statement = encode(statement_text)

        # Get the input_ids and attention_mask tensors from the statement dictionary
        statement_input_ids = statement['input_ids']
        statement_attention_mask = statement['attention_mask']

        # Repeat the input_ids and attention_mask tensors along the batch dimension
        statement_input_ids = statement_input_ids.expand(self.batch_size, -1).to(self.device)
        statement_attention_mask = statement_attention_mask.expand(self.batch_size, -1).to(self.device)

        # Create a BatchEncoding object with the repeated tensors
        statement_batch = BatchEncoding({'input_ids': statement_input_ids, 'attention_mask': statement_attention_mask})

        similarity_scores_list = []
        for batch in self.editorial_index_dataloader:
            with torch.no_grad():
                for batch_encoding in batch[AlgoNetChatFeature.EDITORIAL.value]:
                    batch[AlgoNetChatFeature.EDITORIAL.value][batch_encoding] = \
                        batch[AlgoNetChatFeature.EDITORIAL.value][batch_encoding].to(self.device)

                local_batch_size = batch[AlgoNetChatFeature.EDITORIAL_ID.value].shape[0]
                if local_batch_size < self.batch_size:
                    statement_batch['input_ids'] = statement_batch['input_ids'][:local_batch_size]
                    statement_batch['attention_mask'] = statement_batch['attention_mask'][:local_batch_size]

                similarity_scores = self.model(statement_batch, batch[AlgoNetChatFeature.EDITORIAL.value])
                similarity_scores = similarity_scores.cpu().detach().numpy().tolist()
                editorials_ids = batch[AlgoNetChatFeature.EDITORIAL_ID.value].numpy().tolist()
                for score, id in zip(similarity_scores, editorials_ids):
                    similarity_scores_list.append([score, id])

        torch.cuda.empty_cache()

        return similarity_scores_list

    def predict_editorial(self, statement_text):
        """
        Predicts the best editorial for the given statement text by preprocessing the statement text, creating an editorial
        index, and returning the best predicted editorial.

        Args:
            statement_text (str): The statement text for which the best editorial is predicted.

        Returns:
            str: The best predicted editorial.
        """
        assert statement_text is not None, "statement_text cannot be None"

        statement_text = self.__preprocess_statement(statement_text)
        editorials_index = self.__create_editorial_index(statement_text)
        editorials_index.sort(key=lambda x: x[0], reverse=True)

        best_editorial_predicted = self.editorials_index_df.iloc[editorials_index[0][1]]['editorial']
        return best_editorial_predicted

    def compute_correlation_score(self, statement_text, editorial_text):
        statement_text = self.__preprocess_statement(statement_text)
        editorial_text = self.__preprocess_statement(editorial_text)

        # Put the model in evaluation mode - the dropout layers behave differently during evaluation.
        self.model.eval()

        # Encode the statement/editorial text
        statement = encode(statement_text)
        editorial = encode(editorial_text)

        # Get the input_ids and attention_mask tensors from the statement dictionary
        statement_input_ids = statement['input_ids']
        statement_attention_mask = statement['attention_mask']
        editorial_input_ids = editorial['input_ids']
        editorial_attention_mask = editorial['attention_mask']

        # Repeat the input_ids and attention_mask tensors along the batch dimension
        statement_input_ids = statement_input_ids.expand(1, -1).to(self.device)
        statement_attention_mask = statement_attention_mask.expand(1, -1).to(self.device)
        editorial_input_ids = editorial_input_ids.expand(1, -1).to(self.device)
        editorial_attention_mask = editorial_attention_mask.expand(1, -1).to(self.device)

        # Create a BatchEncoding object with the repeated tensors
        statement_batch = BatchEncoding({'input_ids': statement_input_ids, 'attention_mask': statement_attention_mask})
        editorial_batch = BatchEncoding({'input_ids': editorial_input_ids, 'attention_mask': editorial_attention_mask})

        with torch.no_grad():
            similarity_scores = self.model(statement_batch, editorial_batch)
            similarity_scores = similarity_scores.cpu().detach().numpy().tolist()

        torch.cuda.empty_cache()

        return similarity_scores


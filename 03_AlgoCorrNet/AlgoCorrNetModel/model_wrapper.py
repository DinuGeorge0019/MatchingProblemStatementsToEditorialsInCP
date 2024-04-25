


# standard library imports
import os
import random
import gc
from abc import abstractmethod
import pandas as pd

# related third-party
import torch
from sklearn.metrics.pairwise import paired_cosine_distances
from torch.optim import AdamW
from torch.utils.data import DataLoader, SequentialSampler
from datetime import datetime
import numpy as np
from scipy.stats import pearsonr, spearmanr

# local application/library specific imports
from APP_CONFIG import Config
from AlgoCorrNetModel.datasets import CorrelationEvaluationDataset, PKEvaluationDataset
from AlgoCorrNetModel.datatypes import CorrelationFeature, PrecisionAtKFeature

# define configuration proxy
working_dir = os.path.dirname(os.getcwd())
configProxy = Config(working_dir)

# get configuration
CONFIG = configProxy.return_config()


class ModelWrapper:

    def __init__(self, model, freeze_layers=False, rnd_seed=0):
        self.epochs = None
        self.batches = None
        self.seed = rnd_seed
        self.__seed_everything(seed=rnd_seed)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model
        self.model = self.model.to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=2e-5)
        self.criterion = None

        if freeze_layers:
            self.model.freeze()
        else:
            self.model.unfreeze()

        self.train_dataframe = None
        self.train_dataset = None
        self.isTrained = False
        self.start_epoch = 0

        self.model_save_path = CONFIG['NetModel_PATH']
        self.model_save_path_checkpoint = self.model_save_path + f'_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.pth'

    def __seed_everything(self, seed):
        """
        Seed all random number generators and set torch and CUDA environment for reproducibility.

        Args:
            seed (int): The seed value to use for seeding the random number generators.
        Returns:
            None
        """
        random.seed(seed)  # Seed the Python random module
        os.environ['PYTHONHASHSEED'] = str(seed)  # Seed the hash function used in Python
        np.random.seed(seed)  # Seed the NumPy random module
        torch.manual_seed(seed)  # Seed the PyTorch random module for CPU
        torch.cuda.manual_seed(seed)  # Seed the PyTorch random module for CUDA
        torch.cuda.manual_seed_all(seed)  # Seed the random module for all GPUs
        torch.backends.cudnn.deterministic = True  # Set CuDNN to deterministic mode for reproducibility
        torch.backends.cudnn.benchmark = True  # Enable CuDNN benchmarking for faster training

        return None

    @abstractmethod
    def __set_training_data_loaders(self):
        """
        Create data loaders for training dataset.

        This method creates data loaders for the training dataset using the configured batch size, and a linear learning rate scheduler with warm-up steps.
        The data loaders are stored in the class attribute `self.train_dataloader` for later use during training.

        Returns:
            None
        """

        pass

    @abstractmethod
    def train(self, train_dataframe, validation_correlation_dataframe, validation_precision_at_k_dataframe, epochs=1,
              batches=1):
        """
        Train the model using the provided training data.
        This method performs model training using the provided training data and validation data.
        Additionally, it computes and prints the average training loss for each epoch, and computes and stores precision-at-k and correlation scores for validation data after each epoch.

        Args:
            train_dataframe (pandas.DataFrame): The training data as a Pandas DataFrame.
            validation_correlation_dataframe (pandas.DataFrame): The validation data for computing correlation scores as
                a Pandas DataFrame.
            validation_precision_at_k_dataframe (pandas.DataFrame): The validation data for computing precision-at-k
                scores as a Pandas DataFrame.
            epochs (int, optional): The number of epochs to train the model. Defaults to 1.
            batches (int, optional): The number of batches to use during training. Defaults to 1.
        Returns:
            None
        """
        pass

    def compute_correlation_scores(self, eval_dataframe, batches=1):
        """
        Computes correlation scores for the provided evaluation data.

        This method computes correlation scores for the given evaluation data, which is used to evaluate the model's performance during training.
        The evaluation data is provided as a Pandas DataFrame, and the computed correlation scores are stored in the instance variable `self.correlation_scores`.

        Args:
            eval_dataframe (pandas.DataFrame): The evaluation data for computing correlation scores as a Pandas DataFrame.
            batches (int, optional): The number of batches to use for computing correlation scores. Defaults to 1.
        Returns:
            None
        """

        assert eval_dataframe is not None, "eval_dataframe cannot be None"

        eval_dataset = CorrelationEvaluationDataset(eval_dataframe)
        validation_dataloader = DataLoader(
            eval_dataset,
            sampler=SequentialSampler(eval_dataset),
            batch_size=batches
        )

        # Put the model in evaluation mode - the dropout layers behave differently during evaluation.
        self.model.eval()

        # Evaluate data for one epoch
        statement_embeddings_list = []
        editorial_embeddings_list = []
        labels_list = []
        for batch in validation_dataloader:

            with torch.no_grad():
                for feature in CorrelationFeature:
                    for batch_encoding in batch[feature.value]:
                        batch[feature.value][batch_encoding] = batch[feature.value][batch_encoding].to(self.device)

                statement = batch[CorrelationFeature.STATEMENT.value]
                editorial = batch[CorrelationFeature.EDITORIAL.value]
                label = batch[CorrelationFeature.LABEL.value]

                batch_statement_embeddings, batch_editorial_embeddings = self.model(statement, editorial)
                batch_statement_embeddings = batch_statement_embeddings.cpu().detach().numpy()
                batch_editorial_embeddings = batch_editorial_embeddings.cpu().detach().numpy()
                batch_label = label.cpu().detach().numpy()

                for emb1, emb2, l in zip(batch_statement_embeddings, batch_editorial_embeddings, batch_label):
                    statement_embeddings_list.append(emb1)
                    editorial_embeddings_list.append(emb2)
                    labels_list.append(l)

        statement_embeddings_list_np = np.asarray(statement_embeddings_list)
        editorial_embeddings_list_np = np.asarray(editorial_embeddings_list)
        labels_list_np = np.asarray(labels_list)

        cosine_similarity_scores = 1 - paired_cosine_distances(statement_embeddings_list_np,
                                                               editorial_embeddings_list_np)

        eval_pearson_cosine, p_value_pearson_cosine = pearsonr(labels_list_np, cosine_similarity_scores)
        eval_spearman_cosine, p_value_spearman_cosine = spearmanr(labels_list_np, cosine_similarity_scores)

        print(f"Cossine Pearson score: {eval_pearson_cosine} - p_pearson_value: {p_value_pearson_cosine}")
        print(f"Cossine Spearman score: {eval_spearman_cosine} - p_spearman_value: {p_value_spearman_cosine}")

        torch.cuda.empty_cache()

        return None

    def __precision_at_k(self, recommendations, k):
        """
        Computes precision at k for the given recommendations.
        Precision at k is defined as the ratio of relevant items in the top k recommendations to the total number of recommendations.

        Note: The target editorial has index 0

        Args:
            recommendations (list): The list of recommendations with associated scores.
            k (int): The value of k for which to compute the precision.
        Returns:
            int: 1 if the target editorial is in the top k recommendations, otherwise 0
        """

        assert recommendations is not None, "recommendations cannot be None"
        assert k is not None, "k cannot be None"

        # Take the top k recommendations
        top_k = [(recommendations[i], i) for i in range(len(recommendations))]

        # top_k.sort(key=lambda x: x[0])
        top_k.sort(key=lambda x: x[0], reverse=True)

        top_k = top_k[:k]

        # Calculate the number of relevant items in the top k recommendations
        editorial_pos = None
        for idx, editorial in enumerate(top_k):
            if editorial[1] == 0:
                editorial_pos = idx
                break

        if editorial_pos is not None:
            precision_at_k = 1
        else:
            precision_at_k = 0

        return precision_at_k

    def __compute_positional_relevance_score(self, recommendations, k):
        """
        Computes the positional relevance score for the target editorial in a list of k recommendations.

        The positional relevance score is a measure of the target editorial's position in the list of recommendations, expressed as a percentage.
        For example, if the target editorial is at the second position in the list of k recommendations, the score will be 90 (assuming k=10).

        Args:
            recommendations (list): A list of similarity scores of editorials.
            k (int): The value of k for which to compute the positional relevance score.
        Returns:
            float: The positional relevance score for the target editorial.
        """

        assert recommendations is not None, "recommendations cannot be None"
        assert k is not None, "k cannot be None"

        # Take the top k recommendations
        top_k = [(recommendations[i], i) for i in range(len(recommendations))]
        top_k.sort(key=lambda x: x[0], reverse=True)
        top_k = top_k[:k]

        # Get the position of the target editorial based on his score
        editorial_pos = None
        for idx, editorial in enumerate(top_k):
            if editorial[1] == 0:
                editorial_pos = idx
                break

        # Calculate the precision@k score
        precision_at_k = (k - editorial_pos) / k

        return precision_at_k

    def compute_precision_at_k_scores(self, eval_dataframe, validation_call=False):
        """
        Computes the Precision@K scores for a given evaluation dataframe containing a statement and a list of 10 editorials.

        The Precision@K score is a measure of the accuracy of the model in ranking the editorial list based on similarity scores.
        The first editorial in the list is considered the best editorial related to the statement.

        Args:
            eval_dataframe (pandas.DataFrame): A dataframe containing a statement and a list of 10 editorials.
            validation_call : Variable denoting if the api is called from validation or testing
        Returns:
            None
        """

        assert eval_dataframe is not None, "eval_dataframe cannot be None"

        # BATCH SIZE MUST BE EQUAL TO 1
        number_of_editorials = eval_dataframe.shape[1] - 1
        eval_dataset = PKEvaluationDataset(eval_dataframe)
        validation_dataloader = DataLoader(
            eval_dataset,
            sampler=SequentialSampler(eval_dataset),
            batch_size=1  # batch size must be equal to 1
        )

        # Put the model in evaluation mode -- the dropout layers behave differently during evaluation.
        self.model.eval()

        positional_relevance_score = []
        precision_at_5_scores_cosine = []
        precision_at_3_scores_cosine = []
        precision_at_2_scores_cosine = []
        precision_at_1_scores_cosine = []

        # Evaluate data for one epoch
        for batch in validation_dataloader:
            with torch.no_grad():

                for feature in PrecisionAtKFeature:
                    if feature == PrecisionAtKFeature.EDITORIAL_LIST:
                        for editorial_idx in range(number_of_editorials):
                            for batch_encoding in batch[feature.value][editorial_idx]:
                                batch[feature.value][editorial_idx][batch_encoding] = \
                                    batch[feature.value][editorial_idx][batch_encoding].to(self.device)
                    elif feature == PrecisionAtKFeature.STATEMENT:
                        for batch_encoding in batch[feature.value]:
                            batch[feature.value][batch_encoding] = batch[feature.value][batch_encoding].to(self.device)

                statement = batch[PrecisionAtKFeature.STATEMENT.value]
                editorials = batch[PrecisionAtKFeature.EDITORIAL_LIST.value]

                statement_embeddings_list = []
                editorial_embeddings_list = []

                for editorial in editorials:
                    batch_statement_embeddings, batch_editorial_embeddings = self.model(statement, editorial)

                    batch_statement_embeddings = batch_statement_embeddings.cpu().detach().numpy()
                    batch_editorial_embeddings = batch_editorial_embeddings.cpu().detach().numpy()
                    statement_embeddings_list.append(batch_statement_embeddings.reshape(-1))
                    editorial_embeddings_list.append(batch_editorial_embeddings.reshape(-1))

                statement_embeddings_list_np = np.asarray(statement_embeddings_list)
                editorial_embeddings_np = np.asarray(editorial_embeddings_list)

                cosine_similarity_scores = 1 - paired_cosine_distances(statement_embeddings_list_np,
                                                                       editorial_embeddings_np)

                positional_relevance_score.append(
                    self.__compute_positional_relevance_score(cosine_similarity_scores, 10))
                precision_at_5_scores_cosine.append(self.__precision_at_k(cosine_similarity_scores, 5))
                precision_at_3_scores_cosine.append(self.__precision_at_k(cosine_similarity_scores, 3))
                precision_at_2_scores_cosine.append(self.__precision_at_k(cosine_similarity_scores, 2))
                precision_at_1_scores_cosine.append(self.__precision_at_k(cosine_similarity_scores, 1))

        error_analysis_p5_df = pd.DataFrame(precision_at_5_scores_cosine, columns=['positional_error_flag'])
        error_analysis_p5_df.to_csv(CONFIG['ERROR_ANALYSIS_FOLDER_PATH'] + "Error_Analysis_p5_Validation.csv" if validation_call is True
                                    else CONFIG['ERROR_ANALYSIS_FOLDER_PATH'] + "Error_Analysis_p5_Test.csv")
        error_analysis_p3_df = pd.DataFrame(precision_at_3_scores_cosine, columns=['positional_error_flag'])
        error_analysis_p3_df.to_csv(CONFIG['ERROR_ANALYSIS_FOLDER_PATH'] + "Error_Analysis_p3_Validation.csv" if validation_call is True
                                    else CONFIG['ERROR_ANALYSIS_FOLDER_PATH'] + "Error_Analysis_p3_Test.csv")
        error_analysis_p2_df = pd.DataFrame(precision_at_2_scores_cosine, columns=['positional_error_flag'])
        error_analysis_p2_df.to_csv(CONFIG['ERROR_ANALYSIS_FOLDER_PATH'] + "Error_Analysis_p2_Validation.csv" if validation_call is True
                                    else CONFIG['ERROR_ANALYSIS_FOLDER_PATH'] + "Error_Analysis_p2_Test.csv")
        error_analysis_p1_df = pd.DataFrame(precision_at_1_scores_cosine, columns=['positional_error_flag'])
        error_analysis_p1_df.to_csv(CONFIG['ERROR_ANALYSIS_FOLDER_PATH'] + "Error_Analysis_p1_Validation.csv" if validation_call is True
                                    else CONFIG['ERROR_ANALYSIS_FOLDER_PATH'] + "Error_Analysis_p1_Test.csv")

        print(f"Positional relevance score: {sum(positional_relevance_score) / len(positional_relevance_score)}")
        print(f"Mean Precision@5 cosine score: {sum(precision_at_5_scores_cosine) / len(precision_at_5_scores_cosine)}")
        print(f"Mean Precision@3 cosine score: {sum(precision_at_3_scores_cosine) / len(precision_at_3_scores_cosine)}")
        print(f"Mean Precision@2 cosine score: {sum(precision_at_2_scores_cosine) / len(precision_at_2_scores_cosine)}")
        print(f"Mean Precision@1 cosine score: {sum(precision_at_1_scores_cosine) / len(precision_at_1_scores_cosine)}")

        torch.cuda.empty_cache()

        return None

    def save_model(self, epoch):
        """
        Saves the model checkpoint.

        Args:
            epoch (int): The current epoch number.
        Returns:
            None
        """

        assert epoch is not None, "epoch cannot be None"

        self.model_save_path_checkpoint = self.model_save_path + f'_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.pth'

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict()
        }
        torch.save(checkpoint, self.model_save_path_checkpoint)
        print(f"Saved checkpoint {self.model_save_path_checkpoint}")

        gc.collect()
        torch.cuda.empty_cache()
        return None

    def load_model(self, model_path):
        """
        Loads a trained model from a specified file path.

        Args:
            model_path (str): The file path to the saved model checkpoint.
        Returns:
            None
        """

        assert model_path is not None, "model_path cannot be None"

        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {model_path}")

        # Set the model in evaluation mode after loading it
        self.model.eval()

        return None

    def get_trained_model(self):
        """
        Returns the trained model if it is available, otherwise returns None.

        Returns:
            nn.Module or None: The trained model in evaluation mode if available, otherwise None.
        """

        if self.isTrained:
            return self.model.eval()
        return None

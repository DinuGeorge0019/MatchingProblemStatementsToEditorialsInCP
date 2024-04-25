


# standard library imports
import os

# related third-party
import pandas as pd
import numpy as np
import torch
from torch.utils.data import SequentialSampler, DataLoader
from scipy.stats import pearsonr, spearmanr

# local application/library specific imports
from APP_CONFIG import Config
from AlgoCorrNetModel.datasets import CorrelationEvaluationDataset, PKEvaluationDataset
from AlgoCorrNetModel.datatypes import PrecisionAtKFeature, CorrelationFeature

# define configuration proxy
working_dir = os.path.dirname(os.getcwd())
configProxy = Config(working_dir)

# get configuration
CONFIG = configProxy.return_config()


class AlgoCorrNetModelEvaluator:
    def __init__(self, model, testing_pk_dataset_path=None, testing_correlation_dataset_path=None, batches=1):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model
        self.model = self.model.to(self.device)
        self.batches = batches
        self.evaluate_precision_at_k_dataloader = None
        self.evaluate_correlation_dataloader = None
        self.__set_dataloaders(testing_pk_dataset_path, testing_correlation_dataset_path)

    def __set_dataloaders(self, testing_pk_dataset_path, testing_correlation_dataset_path):
        """
        Set the data loaders for testing Precision@K and correlation evaluation.

        Args:
            testing_pk_dataset_path (str): The path to the testing dataset for Precision@K evaluation.
            testing_correlation_dataset_path (str): The path to the testing dataset for correlation evaluation.

        Returns:
            None
        """

        assert testing_pk_dataset_path is not None, "testing_pk_dataset_path cannot be None"
        assert testing_correlation_dataset_path is not None, "testing_correlation_dataset_path cannot be None"

        self.__set_testing_precision_at_k_data_loader(testing_pk_dataset_path)
        self.__set_testing_correlation_data_loader(testing_correlation_dataset_path)

        return None

    def __set_testing_correlation_data_loader(self, testing_correlation_dataset_path):
        """
        Set the data loader for testing correlation evaluation.

        Args:
            testing_correlation_dataset_path (str): The path to the testing dataset for correlation evaluation.

        Returns:
            None
        """
        assert testing_correlation_dataset_path is not None, "testing_correlation_dataset_path cannot be None"

        self.test_correlation_df = pd.read_csv(
            testing_correlation_dataset_path,
            usecols=[1, 2, 3]
        )

        eval_dataset = CorrelationEvaluationDataset(self.test_correlation_df)
        self.evaluate_correlation_dataloader = DataLoader(
            eval_dataset,
            sampler=SequentialSampler(eval_dataset),
            batch_size=self.batches
        )

    def __set_testing_precision_at_k_data_loader(self, testing_pk_dataset_path):
        """
        Set the data loader for testing Precision@K evaluation.

        Args:
            testing_pk_dataset_path (str): The path to the testing dataset for Precision@K evaluation.

        Returns:
            None
        """
        assert testing_pk_dataset_path is not None, "testing_pk_dataset_path cannot be None"

        BATCHES = 1  # batch size must be equal to 1

        self.test_precision_at_k_df = pd.read_csv(
            testing_pk_dataset_path,
            usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        )

        eval_dataset = PKEvaluationDataset(self.test_precision_at_k_df)
        self.evaluate_precision_at_k_dataloader = DataLoader(
            eval_dataset,
            sampler=SequentialSampler(eval_dataset),
            batch_size=BATCHES
        )

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

    def __compute_precision_at_k_scores(self):
        """
        Compute the Precision@K scores for the testing data using the trained model.
        Note: The batch size must be equal to 1 for this computation.

        Returns:
            dict: A dictionary containing the computed Precision@K scores.
                The keys in the dictionary represent different metrics, and the values represent the corresponding scores.
        """

        # BATCH SIZE MUST BE EQUAL TO 1
        number_of_editorials = self.test_precision_at_k_df.shape[1] - 1

        # Put the model in evaluation mode--the dropout layers behave differently during evaluation.
        self.model.eval()

        positional_relevance_score = []
        precision_at_5_scores_cosine = []
        precision_at_3_scores_cosine = []
        precision_at_2_scores_cosine = []
        precision_at_1_scores_cosine = []

        # Evaluate data for one epoch
        for batch in self.evaluate_precision_at_k_dataloader:
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

                similarity_scores_list = []
                for editorial in editorials:
                    similarity_score = self.model(statement, editorial)
                    similarity_scores_list.append(similarity_score)

                positional_relevance_score.append(self.__compute_positional_relevance_score(similarity_scores_list, 10))
                precision_at_5_scores_cosine.append(self.__precision_at_k(similarity_scores_list, 5))
                precision_at_3_scores_cosine.append(self.__precision_at_k(similarity_scores_list, 3))
                precision_at_2_scores_cosine.append(self.__precision_at_k(similarity_scores_list, 2))
                precision_at_1_scores_cosine.append(self.__precision_at_k(similarity_scores_list, 1))

        torch.cuda.empty_cache()

        output_scores = {
            'Mean positional relevance': sum(positional_relevance_score) / len(positional_relevance_score),
            'Mean Precision@5': sum(precision_at_5_scores_cosine) / len(precision_at_5_scores_cosine),
            'Mean Precision@3': sum(precision_at_3_scores_cosine) / len(precision_at_3_scores_cosine),
            'Mean Precision@2': sum(precision_at_2_scores_cosine) / len(precision_at_2_scores_cosine),
            'Mean Precision@1': sum(precision_at_1_scores_cosine) / len(precision_at_1_scores_cosine)
        }

        return output_scores

    def __compute_best_acc_threshold(self, scores, labels, high_score_more_similar=True):
        """
        Compute the best accuracy threshold for a given set of similarity scores and labels.

        Args:
            scores (List[float]): List of similarity scores.
            labels (List[int]): List of binary labels (0 or 1).
            high_score_more_similar (bool, optional): If True, higher score indicates more similarity, else lower score indicates more similarity. Defaults to True.

        Returns:
            Tuple[float, float]: A tuple containing the best accuracy threshold and the corresponding maximum accuracy.
        """

        assert scores is not None, "scores cannot be None"
        assert labels is not None, "labels cannot be None"
        assert len(scores) == len(labels)

        rows = list(zip(scores, labels))
        rows = sorted(rows, key=lambda x: x[0], reverse=high_score_more_similar)

        max_acc = 0
        best_threshold = -1

        positive_so_far = 0
        remaining_negatives = labels.count(0)

        for i in range(len(rows) - 1):
            score, label = rows[i]
            if label == 1:
                positive_so_far += 1
            else:
                remaining_negatives -= 1

            acc = (positive_so_far + remaining_negatives) / len(labels)
            if acc > max_acc:
                max_acc = acc
                best_threshold = (rows[i][0] + rows[i + 1][0]) / 2

        return max_acc, best_threshold

    def __compute_confusion_matrix(self, scores, labels, threshold):
        """
        Compute the confusion matrix for a given set of similarity scores, labels, and threshold.

        Args:
            scores (List[float]): List of similarity scores.
            labels (List[int]): List of binary labels (0 or 1).
            threshold (float): Threshold to classify scores into positive and negative samples.

        Returns:
            Tuple[int, int, int, int]: A tuple containing the counts of true positives, true negatives, false positives, and false negatives.
        """
        assert scores is not None, "scores cannot be None"
        assert labels is not None, "labels cannot be None"
        assert len(scores) == len(labels)

        rows = list(zip(scores, labels))

        true_positives = 0
        true_negatives = 0
        false_positives = 0
        false_negatives = 0
        for i in range(len(rows) - 1):
            scores, label = rows[i]
            if scores > threshold and label == 1:
                true_positives += 1
            elif scores > threshold and label == 0:
                false_positives += 1
            elif scores <= threshold and label == 0:
                true_negatives += 1
            elif scores <= threshold and label == 1:
                false_negatives += 1

        return true_positives, true_negatives, false_positives, false_negatives

    def __compute_accuracy(self, TP, TN, TOTAL):
        """
        Compute accuracy given the counts of true positives (TP), true negatives (TN), and total samples (TOTAL).

        Args:
            TP (int): Count of true positives.
            TN (int): Count of true negatives.
            TOTAL (int): Total count of samples.

        Returns:
            float: Accuracy score.
        """
        return (TP + TN) / TOTAL

    def __compute_precision(self, TP, FP):
        """
        Compute precision given the counts of true positives (TP) and false positives (FP).

        Args:
            TP (int): Count of true positives.
            FP (int): Count of false positives.

        Returns:
            float: Precision score.
        """
        return TP / (TP + FP)

    def __compute_recall(self, TP, FN):
        """
        Compute recall given the counts of true positives (TP) and false negatives (FN).

        Args:
            TP (int): Count of true positives.
            FN (int): Count of false negatives.

        Returns:
            float: Recall score.
        """
        return TP / (TP + FN)

    def __compute_f1_score(self, precision, recall):
        """
        Compute F1 score given the precision and recall values.

        Args:
            precision (float): Precision value.
            recall (float): Recall value.

        Returns:
            float: F1 score.
        """
        return (2 * (precision * recall)) / (precision + recall)

    def __compute_general_scores(self):
        """
        Compute general evaluation scores for the model.

        Returns:
            dict: Dictionary containing evaluation scores, including Pearson and Spearman correlation scores,
                  accuracy, precision, recall, and F1 score.
        """
        # Put the model in evaluation mode - the dropout layers behave differently during evaluation.
        self.model.eval()

        # Evaluate data for one epoch
        scores_list = []
        labels_list = []
        for batch in self.evaluate_correlation_dataloader:

            with torch.no_grad():
                for feature in CorrelationFeature:
                    for batch_encoding in batch[feature.value]:
                        batch[feature.value][batch_encoding] = batch[feature.value][batch_encoding].to(self.device)

                statement = batch[CorrelationFeature.STATEMENT.value]
                editorial = batch[CorrelationFeature.EDITORIAL.value]
                label = batch[CorrelationFeature.LABEL.value]

                batch_scores = self.model(statement, editorial)
                batch_scores = batch_scores.cpu().detach().numpy()
                batch_label = label.cpu().detach().numpy()

                for score, label in zip(batch_scores, batch_label):
                    scores_list.append(score)
                    labels_list.append(label)

        scores_list_np = np.asarray(scores_list)
        labels_list_np = np.asarray(labels_list)

        eval_pearson_cosine, p_value_pearson_cosine = pearsonr(labels_list_np, scores_list_np)
        eval_spearman_cosine, p_value_spearman_cosine = spearmanr(labels_list_np, scores_list_np)

        max_acc, threshold = self.__compute_best_acc_threshold(scores_list, labels_list, True)
        TP, TN, FP, FN = self.__compute_confusion_matrix(scores_list, labels_list, threshold)
        TOTAL = len(labels_list)
        acc = self.__compute_accuracy(TP, TN, TOTAL)
        precision = self.__compute_precision(TP, FP)
        recall = self.__compute_recall(TP, FN)
        f1 = self.__compute_f1_score(precision, recall)

        torch.cuda.empty_cache()

        output_scores = {
            'Cossine Pearson score': eval_pearson_cosine,
            'p_pearson_value': p_value_pearson_cosine,
            'Cossine Spearman score': eval_spearman_cosine,
            'p_spearman_value': p_value_spearman_cosine,
            'max_acc': max_acc,
            'acc': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

        return output_scores

    def compute_metrics(self):
        """
        Computes various evaluation metrics for the model and saves the results to a CSV file.
        This method computes various evaluation metrics for the model, including Pearson and Spearman correlation
        scores, accuracy, precision, recall, and F1 score, as well as precision@k scores. The results are saved
        to a CSV file.

        Returns:
            None
        """

        csv_save_path = CONFIG["ALGOCORRNET_MODEL_EVALUATOR_RESULTS_PATH"]
        precision_at_k_scores = self.__compute_precision_at_k_scores()
        general_scores = self.__compute_general_scores()

        scores = {**general_scores, **precision_at_k_scores}
        scores = {key: str(value) for key, value in scores.items()}

        csv_headers = ['Model Name'] + list(scores.keys())
        output_data = [f'{self.model.model_path}'] + list(scores.values())

        # Create a DataFrame
        df = pd.DataFrame([output_data], columns=csv_headers)

        if not os.path.isfile(csv_save_path):
            # Write the DataFrame to the csv file
            df.to_csv(csv_save_path, index=False)
        else:
            # Append the DataFrame to an existing CSV file
            df.to_csv(csv_save_path, index=False, mode='a', header=False)

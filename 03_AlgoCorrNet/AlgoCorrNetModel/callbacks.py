


# standard library imports
import os

# related third-party
import pandas as pd

# local application/library specific imports
from APP_CONFIG import Config
from AlgoCorrNetModel.algo_corr_net_chat import AlgoNetChat
from AlgoCorrNetModel.algo_corr_net_model_builder import AlgoCorrNetModelBuilder
from AlgoCorrNetModel.algo_corr_net_model_evaluator import AlgoCorrNetModelEvaluator
from AlgoCorrNetModel.siamese_net_model_wrapper import SiameseNetModelWrapper
from AlgoCorrNetModel.triple_net_model_wrapper import TripleNetModelWrapper
from AlgoCorrNetModel.embedding_model import EmbeddingModel
from AlgoCorrNetModel.triple_net_model import TripleNetModel
from AlgoCorrNetModel.siamese_net_model import SiameseNetModel

# define configuration proxy
working_dir = os.path.dirname(os.getcwd())
configProxy = Config(working_dir)

# get configuration
CONFIG = configProxy.return_config()
MODEL_EVALUATION_CONFIG = configProxy.return_model_evaluation_config()
MODEL_TRAINING_CONFIG = configProxy.return_model_training_config()
ACN_MODEL_CONFIG = configProxy.return_acn_model_config()


def train_siamese_net_model(train_dataset_path=CONFIG['CP_SIAMESE_TRAINING_DATASET_PATH']):
    embedding_model = EmbeddingModel()
    model = SiameseNetModel(embedding_model)

    validation_precision_at_k_df = pd.read_csv(CONFIG['CP_VALIDATION_P@K_DATASET_PATH'],
                                               usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    validation_correlation_df = pd.read_csv(CONFIG['CP_VALIDATION_CORRELATION_DATASET_PATH'],
                                            usecols=[1, 2, 3])
    train_df = pd.read_csv(train_dataset_path, usecols=[1, 2, 3])

    modelUtilityWrapper = SiameseNetModelWrapper(model)
    modelUtilityWrapper.train(
        train_dataframe=train_df,
        validation_correlation_dataframe=validation_correlation_df,
        validation_precision_at_k_dataframe=validation_precision_at_k_df,
        epochs=MODEL_TRAINING_CONFIG['EPOCHES'],
        batches=MODEL_TRAINING_CONFIG['BATCHES']
    )


def evaluate_siamese_net_model(model_path=MODEL_EVALUATION_CONFIG['MODEL_PATH']):
    embedding_model = EmbeddingModel()
    model = SiameseNetModel(embedding_model)
    modelUtilityWrapper = SiameseNetModelWrapper(model)
    modelUtilityWrapper.load_model(model_path)

    test_correlation_df = pd.read_csv(CONFIG['CP_TESTING_CORRELATION_DATASET_PATH'],
                                      usecols=[1, 2, 3])
    test_precision_at_k_df = pd.read_csv(CONFIG['CP_TESTING_P@K_DATASET_PATH'],
                                         usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

    modelUtilityWrapper.compute_precision_at_k_scores(test_precision_at_k_df)
    modelUtilityWrapper.compute_correlation_scores(test_correlation_df, batches=MODEL_EVALUATION_CONFIG['BATCHES'])


def get_token_distribution():
    embedding_model = EmbeddingModel()
    model = TripleNetModel(embedding_model)
    train_df = pd.read_csv(CONFIG['CP_MNRL_TRAINING_DATASET_PATH'], usecols=[1, 2, 3])
    modelUtilityWrapper = TripleNetModelWrapper(model)
    modelUtilityWrapper.get_tokens_size(
        train_dataframe=train_df
    )

def train_triple_net_model(train_dataset_path=CONFIG['CP_MNRL_TRAINING_DATASET_PATH']):
    embedding_model = EmbeddingModel()
    model = TripleNetModel(embedding_model)

    validation_precision_at_k_df = pd.read_csv(CONFIG['CP_VALIDATION_P@K_DATASET_PATH'],
                                               usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    validation_correlation_df = pd.read_csv(CONFIG['CP_VALIDATION_CORRELATION_DATASET_PATH'],
                                            usecols=[1, 2, 3])
    train_df = pd.read_csv(train_dataset_path, usecols=[1, 2, 3])

    modelUtilityWrapper = TripleNetModelWrapper(model)
    modelUtilityWrapper.train(
        train_dataframe=train_df,
        validation_correlation_dataframe=validation_correlation_df,
        validation_precision_at_k_dataframe=validation_precision_at_k_df,
        epochs=MODEL_TRAINING_CONFIG['EPOCHES'],
        batches=MODEL_TRAINING_CONFIG['BATCHES']
    )


def evaluate_triple_net_model(model_path=MODEL_EVALUATION_CONFIG['MODEL_PATH']):
    embedding_model = EmbeddingModel()
    model = TripleNetModel(embedding_model)
    modelUtilityWrapper = TripleNetModelWrapper(model)
    modelUtilityWrapper.load_model(model_path)

    test_correlation_df = pd.read_csv(CONFIG['CP_TESTING_CORRELATION_DATASET_PATH'],
                                      usecols=[1, 2, 3])
    test_precision_at_k_df = pd.read_csv(CONFIG['CP_TESTING_P@K_DATASET_PATH'],
                                         usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

    modelUtilityWrapper.compute_precision_at_k_scores(test_precision_at_k_df)
    modelUtilityWrapper.compute_correlation_scores(test_correlation_df, batches=MODEL_EVALUATION_CONFIG['BATCHES'])


def build_untrained_model_wrapper():
    embedding_model = EmbeddingModel()
    model = TripleNetModel(embedding_model)
    modelUtilityWrapper = TripleNetModelWrapper(model)
    modelUtilityWrapper.save_model(epoch=0)

    
def build_algo_corr_net_model(model_path=ACN_MODEL_CONFIG['PRETRAINED_MODEL_PATH']):
    algoCorrNetModelBuilder = AlgoCorrNetModelBuilder()
    algoCorrNetModelBuilder.buildAlgoCorrNetModel(model_path, save_model=True)
    algoCorrNetModel = algoCorrNetModelBuilder.getAlgoCorrNetModel()
    return algoCorrNetModel


def evaluate_algo_corr_net_model(model_path=ACN_MODEL_CONFIG['MODEL_PATH']):
    algoCorrNetModelBuilder = AlgoCorrNetModelBuilder()
    algoCorrNetModel = algoCorrNetModelBuilder.loadAlgoCorrNetModel(model_path)

    algoNetModelEvaluator = AlgoCorrNetModelEvaluator(
        model=algoCorrNetModel,
        testing_correlation_dataset_path=CONFIG['CP_TESTING_CORRELATION_DATASET_PATH'],
        testing_pk_dataset_path=CONFIG['CP_TESTING_P@K_DATASET_PATH'],
        batches=ACN_MODEL_CONFIG['BATCHES']
    )

    algoNetModelEvaluator.compute_metrics()


def predict_editorial(statement_text, print_editorial=True):
    algoCorrNetModelBuilder = AlgoCorrNetModelBuilder()
    algoCorrNetModel = algoCorrNetModelBuilder.loadAlgoCorrNetModel(ACN_MODEL_CONFIG['MODEL_PATH'])

    algoNetChat = AlgoNetChat(algoCorrNetModel)
    predicted_editorial = algoNetChat.predict_editorial(statement_text)

    if print_editorial is True:
        print(predicted_editorial)

    return predicted_editorial


def compute_correlation_score(statement_text, editorial_text, print_score=True):
    algoCorrNetModelBuilder = AlgoCorrNetModelBuilder()
    algoCorrNetModel = algoCorrNetModelBuilder.loadAlgoCorrNetModel(ACN_MODEL_CONFIG['MODEL_PATH'])

    algoNetChat = AlgoNetChat(algoCorrNetModel)
    correlation_score = algoNetChat.compute_correlation_score(statement_text, editorial_text)

    if print_score is True:
        print((correlation_score[0] + 1)/2)

    return correlation_score




class Config(object):
    """
    Config with all the paths and flags needed.
    """

    def __init__(self, WORKING_DIR):
        self.model_evaluation_config = {
            'MODEL_PATH': "Output/Train-NetModels/TripleNetModel_Pytorch_2024-01-12_02-32-34.pth",
            'BATCHES': 16
        }

        self.model_training_config = {
            'BATCHES': 16,
            'EPOCHES': 1
        }

        self.acn_model_config = {
            'PRETRAINED_MODEL_PATH': "Output/Train-NetModels/TripleNetModel_Pytorch_2024-01-12_02-32-34.pth",
            'MODEL_PATH': "Output/AlgoCorrNetModels/AlgoCorrNet-Pytorch_2024-01-12_02-39-10.pth",
            'BATCHES': 32
        }

        self.global_constants = {
            'BERT_MAX_LENGTH_TENSORS': 512,
            'CREATE_PREPROCESSED_DATASET': False,
            'PREPROCESS_DEVELOPMENT_MODE': False,
            'DISPLAY_CLUSTERS_PLOTS': False,
            'GENERATE_VALIDATION_DATASET': False,
            'GENERATE_TESTING_DATASET': False,
            'KMEANS_NO_CLUSTERS': 20,
            'NO_OF_NEGATIVE_EXAMPLES': 1,
            'RANDOM_STATE': 8
        }

        self.config = {
            'BERT_EMBEDDING_MODEL_NAME': 'sentence-transformers/all-mpnet-base-v2',
            # 'BERT_EMBEDDING_MODEL_NAME': 'sentence-transformers/multi-qa-mpnet-base-dot-v1',
            # 'BERT_EMBEDDING_MODEL_NAME': 'sentence-transformers/multi-qa-distilbert-cos-v1',
            # 'BERT_EMBEDDING_MODEL_NAME': 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1',
            # 'BERT_EMBEDDING_MODEL_NAME': 'sentence-transformers/all-distilroberta-v1',
            # 'BERT_EMBEDDING_MODEL_NAME': 'sentence-transformers/all-MiniLM-L12-v2',
            # 'BERT_EMBEDDING_MODEL_NAME': 'sentence-transformers/all-MiniLM-L6-v2',
            # 'BERT_EMBEDDING_MODEL_NAME': 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
            # 'BERT_EMBEDDING_MODEL_NAME': 'sentence-transformers/paraphrase-albert-small-v2',
            # 'BERT_EMBEDDING_MODEL_NAME': 'allenai/scibert_scivocab_uncased',
            # 'BERT_EMBEDDING_MODEL_NAME': 'microsoft/mpnet-base',
            # 'BERT_EMBEDDING_MODEL_NAME': 'roberta-base',
            # 'BERT_EMBEDDING_MODEL_NAME': 'bert-base-uncased',

            "LOG_OVERALL_CORPUS_INFO_PATH": "Output/Logs/Log_OverallCorpusInfo.txt",
            "LOG_TRAINING_CORPUS_CLUSTERS_PATH": "Output/Logs/Log_TrainingCorpusClusters.txt",
            "SILHOUETTE_PLOT_PATH": "Output/Plots/SilhouettePlot",
            "ELBOW_PLOT_PATH": "Output/Plots/ElbowPlot.png",
            "CLUSTERS_SCATTER_PLOT_PATH": "Output/Plots/ClustersScatterPlot.png",
            "DATASET_HISTOGRAM": "Output/Plots/DatasetHistogram.png",

            "RAW_DATASET_PATH": "Output/Dataset_output/CompetitiveProgrammingDataset.csv",
            "PREPROCESSED_DATASET_PATH": "Output/Dataset_output/Preprocessed_CompetitiveProgrammingDataset.csv",
            "PREPROCESSED_TRAINING_DATASET_PATH": "Output/Dataset_output/Preprocessed_TrainingDatasetPath.csv",
            "PREPROCESSED_TESTING_DATASET_PATH": "Output/Dataset_output/Preprocessed_TestingDatasetPath.csv",
            "PREPROCESSED_VALIDATION_DATASET_PATH": "Output/Dataset_output/Preprocessed_ValidationDatasetPath.csv",
            "CP_MNRL_TRAINING_DATASET_PATH": "Output/Dataset_output/CP_MNRLTrainingDataset.csv",
            "CP_MNRL_RANDOM_TRAINING_DATASET_PATH": "Output/Dataset_output/CP_MNRLRandomTrainingDataset.csv",
            "CP_SIAMESE_TRAINING_DATASET_PATH": "Output/Dataset_output/CP_SiameseTrainingDataset.csv",
            "CP_SIAMESE_RANDOM_TRAINING_DATASET_PATH": "Output/Dataset_output/CP_SiameseRandomTrainingDataset.csv",
            "CP_TESTING_CORRELATION_DATASET_PATH": "Output/Dataset_output/CP_TestingCorrelationDataset.csv",
            "CP_VALIDATION_CORRELATION_DATASET_PATH": "Output/Dataset_output/CP_ValidationCorrelationDataset.csv",
            "CP_TESTING_P@K_DATASET_PATH": "Output/Dataset_output/CP_TestingP@KDataset.csv",
            "CP_VALIDATION_P@K_DATASET_PATH": "Output/Dataset_output/CP_ValidationP@KDataset.csv",
            "EDITORIALS_INDEX_PATH": "Output/Dataset_output/EditorialsIndex.csv",
            "ERROR_ANALYSIS_FOLDER_PATH": "ErrorAnalysis/Output/",
            "LOG_TEST_ERROR_ANALYSIS_PATH": "ErrorAnalysis/Output/Log_TestErrorAnalysisReport.txt",
            "LOG_VALIDATION_ERROR_ANALYSIS_PATH": "ErrorAnalysis/Output/Log_ValidationErrorAnalysisReport.txt",
            "LOG_TEST_DETAILED_ERROR_ANALYSIS_PATH": "ErrorAnalysis/Output/Log_DetailedTestErrorAnalysisReport.txt",
            "LOG_VALIDATION_DETAILED_ERROR_ANALYSIS_PATH": "ErrorAnalysis/Output/Log_DetailedValidationErrorAnalysisReport.txt",

            "NetModel_PATH": "Output/Train-NetModels/",
            "AlgoCorrNetModel_PATH": "Output/AlgoCorrNetModels/",

            "ALGOCORRNET_MODEL_EVALUATOR_RESULTS_PATH": "Output/AlgoCorrNetEvaluator/AlgoCorrNetModelEvaluator_results.csv",

            "WORKING_DIR": f"{WORKING_DIR}",
            "CODEFORCES_DATASET_PATH": f"{WORKING_DIR}//01_CODEFORCES_DATASET"
        }

    def return_acn_model_config(self):
        """
        Return acn model configuration

        Returns:
            dict
        """
        return self.acn_model_config

    def return_model_evaluation_config(self):
        """
        Return net model evaluation configuration

        Returns:
            dict
        """
        return self.model_evaluation_config

    def return_model_training_config(self):
        """
        Return net model training configuration

        Returns:
            dict
        """
        return self.model_training_config

    def return_config(self):
        """
        Return entire config dictionary

        Returns:
            dict
        """
        return self.config

    def return_global_constants(self):
        """
        Return the headers used to request the GET operation

        Returns:
            dict
        """
        return self.global_constants

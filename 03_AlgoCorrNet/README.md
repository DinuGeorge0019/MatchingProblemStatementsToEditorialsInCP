1. **Hardware requirements**
-  To train the model presented in this paper, we used a virtual machine
with an NVIDIA A100 GPU (40 GB GPU), 80 GB RAM size, and an 80 GB HDD.

2. **Software Requirements**
-  Python 3.10 or higher
-  Install the libraries from requirements.txt.
-  There are two options to recreate the experiments of the paper:
    -  Use the notebook from ../ColabEnvironment/ColabEnvironment.ipynb
    -  Use the main.py interface of the framework with the required arguments
-  To access helpful information, you can execute the main.py script with the "--help" attribute. This will provide you with relevant details on how to use the framework and its available options.

3. **Note:** If the notebook from ../ColabEnvironment/ColabEnvironment.ipynb the update of __init__.py file from APP_CONFIG module can be skipped, as the path of the trained modules / provided datasets can be updated in the notebook

4. **Training the model using the Multiple Negative Ranking Loss architecture**
-  APP_CONFIG module -> __init__.py file shall be updated with base model path you want to finetune (config -> BERT_EMBEDDING_MODEL_NAME)
- train_triple_net_model shall be called to finetune the embedding model using the Multiple Negative Ranking Loss architecture
- The finetuned model shall be saved at Output\Train-NetModels
- APP_CONFIG module -> __init__.py file shall be updated with the path of the finetuned embedding model (acn_model_config -> PRETRAINED_MODEL_PATH)
- build_algo_corr_net_model shall be called to build the AlgoCorrNetModel
- The finetuned model shall be saved at Output\AlgoCorrNetModels

5. **Training the model using the Softmax architecture**
-  APP_CONFIG module -> __init__.py file shall be updated with base model path you want to finetune (config -> BERT_EMBEDDING_MODEL_NAME)
- train_siamese_net_model shall be called to finetune the embedding model using the Siamese Network Architecture
- The finetuned model shall be saved at Output\Train-NetModels
- APP_CONFIG module -> __init__.py file shall be updated with the path of the finetuned embedding model (acn_model_config -> PRETRAINED_MODEL_PATH)
- build_algo_corr_net_model shall be called to build the AlgoCorrNetModel
- The finetuned model shall be saved at Output\AlgoCorrNetModels

6. **Untrained base model evaluation**
-  APP_CONFIG module -> __init__.py file shall be updated with base model path you want to evaluate (config -> BERT_EMBEDDING_MODEL_NAME)
- build a wrapper over the base model by calling build_untrained_model_wrapper
- The model shall be saved at Output\Train-NetModels
- APP_CONFIG module -> __init__.py file shall be updated with the path of the embedding model (acn_model_config -> PRETRAINED_MODEL_PATH)
- build_algo_corr_net_model shall be called to build the AlgoCorrNetModel
- The model shall be saved at Output\AlgoCorrNetModels
- APP_CONFIG module -> __init__.py file shall be updated with the path of the AlgoCorrNetModel (acn_model_config -> MODEL_PATH)
- evaluate_algo_corr_net_model shall be called to evaluate the model on the test dataset
- The evaluation results shall be saved at ..\Output\AlgoCorrNetEvaluator\AlgoCorrNetModelEvaluator_results.csv

7. **Evaluate the AlgoCorrNet model**
- APP_CONFIG module -> __init__.py file shall be updated with the path of the AlgoCorrNetModel (acn_model_config -> MODEL_PATH)
- evaluate_algo_corr_net_model shall be called to evaluate the model on the test dataset
- The evaluation results shall be saved at ..\Output\AlgoCorrNetEvaluator\AlgoCorrNetModelEvaluator_results.csv

8. **Evaluate the gpt-3.5-turbo-1106 and gpt-4-1106-preview on the test dataset**
- ..\ChatGPTExperiments\ directory contains the notebooks used to evaluate the models performance on the test dataset
- update the API-KEY string to an OPENAI api key to be able to make the api calls 
- results will be saved in chat_gpt3_5_results.xlsx and chat_gpt4_results.xlsx
- To calculate the P@K run the cells from P@K Score chapter

9. **Error analisys**
- create_error_analysis shall be called after the model has been evaluated on the test dataset in order to reproduce the error analisys presented in the paper
- The reports shall be saved at ..\ErrorAnalysis\Output

10. **Programming Challenge - Editorial relevance score calculation**
- compute_correlation_score shall be called to generate the matching score of an editorial and problem statement
- The result shall be printed in console

Additional notes:
- To update the pooling architecture of the embedding model update  ..\AlgoCorrNetModel\embedding_model.py file with the content of a file from Development_EmbeddingModels (the default loaded configuration is the mean pooling as described in the paper)
-  Output\Dataset_output contains the datasets used for training / evaluation 
    - CompetitiveProgrammingDataset.csv -> Overall dataset of 1550 competitive programming challanges
    - Preprocessed_CompetitiveProgrammingDataset.csv -> Preprocessed version of the CompetitiveProgrammingDataset
    - Preprocessed_TestingDatasetPath.csv -> Testing Dataset
    - Preprocessed_TrainingDatasetPath.csv -> Training Datset
    - Preprocessed_ValidationDatasetPath.csv -> Validation Dataset
    - CP_MNRLRandomTrainingDataset.csv -> Random partitioned training dataset of the Triple Network Architecture
    - CP_MNRLTrainingDataset.csv -> Unsupervised preprocessed training dataset of the Triple Network Architecture
    - CP_SiameseRandomTrainingDataset.csv -> Random partitioned training dataset of the Siamese Network Architecture
    - CP_SiameseTrainingDataset.csv -> Unsupervised preprocessed training dataset of the Siamese Network Architecture
    - CP_TestingCorrelationDataset.csv -> Pearson / Spearman correlation testing dataset
    - CP_ValidationCorrelationDataset.csv -> Pearson / Spearman correlation validation dataset
    - CP_TestingP@KDataset.csv -> Editorial ranking testing dataset
    - CP_ValidationP@KDataset.csv -> Editorial ranking validation dataset
    - EditorialsIndex.csv -> Collection of 1550 competitive programming editorials

- If regeneration of the datasets is needed:
    - build_raw_dataset shall be called to transform the dataset provided in 01_CODEFORCES_DATASET to a Csv format
    - build_dataset shall be called to regenerate the preprocessed datasets

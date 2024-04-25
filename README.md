# Project Folder Structure

- The project has the following folder structure:

    - `00_CODEFORCES_DATA/`: Codeforces competitive programming contests collection
    - `01_CODEFORCES_DATASET/`: JSON format of the Codeforces competitive programming problem statements-editorials dataset.
    - `02_CODEFORCES_WEBSCRAPPER/`: Web scrapper used to fetch all the problem statements and editorials from Codeforces
    - `03_AlgoCorrNet/`: Framework used to train/test the proposed system

- Note: To replicate the experiments proposed in the paper 'Matching Problem Statements to Editorials in Competitive Programming' use the framework from 03_AlgoCorrNet

- The datasets referenced in the paper can be found in 03_AlgoCorrNet\Output\Dataset_output
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
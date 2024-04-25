

# standard library imports
import os
import json
import string
import regex as re
import random

from matplotlib import pyplot as plt
from tqdm.auto import tqdm

# related third-party
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# local application/library specific imports
from APP_CONFIG import Config
from DatasetBuilder.k_means_clustering import KMeansClustering

# define configuration proxy
working_dir = os.path.dirname(os.getcwd())
configProxy = Config(working_dir)

# get configuration
CONFIG = configProxy.return_config()

# get global constants configuration
GLOBAL_CONSTANTS = configProxy.return_global_constants()

CREATE_PREPROCESSED_DATASET = GLOBAL_CONSTANTS['CREATE_PREPROCESSED_DATASET']
PREPROCESS_DEVELOPMENT_MODE = GLOBAL_CONSTANTS['PREPROCESS_DEVELOPMENT_MODE']
DISPLAY_CLUSTERS_PLOTS = GLOBAL_CONSTANTS['DISPLAY_CLUSTERS_PLOTS']
GENERATE_VALIDATION_DATASET = GLOBAL_CONSTANTS['GENERATE_VALIDATION_DATASET']
GENERATE_TESTING_DATASET = GLOBAL_CONSTANTS['GENERATE_TESTING_DATASET']
NO_OF_NEGATIVE_EXAMPLES = GLOBAL_CONSTANTS['NO_OF_NEGATIVE_EXAMPLES']
RANDOM_STATE = GLOBAL_CONSTANTS['RANDOM_STATE']

random.seed(RANDOM_STATE)

class DatasetBuilder:
    def __init__(self):
        self.df_test = None
        self.df_train = None
        self.df_val = None
        self.df = None
        self.clustered_editorials = None
        self.training_corpus_final_form = []
        self.training_random_corpus_final_form = []
        self.training_siamese_corpus_final_form = []
        self.training_random_siamese_corpus_final_form = []
        self.testing_corpus_correlation_final_form = []
        self.validation_corpus_correlation_final_form = []
        self.testing_corpus_precision_at_k_final_form = []
        self.validation_corpus_precision_at_k_final_form = []
        self.full_raw_dataset = []

    def build_raw_dataset(self):
        """
        Reads input files from the dataset destination and creates a dataframe of the full raw competitive programming dataset.

        Returns:
            None
        """

        print("\nReading corpus")
        raw_dataset = []
        for folder_name in os.listdir(CONFIG['CODEFORCES_DATASET_PATH']):
            folder_path = os.path.join(CONFIG['CODEFORCES_DATASET_PATH'], folder_name)
            for file in tqdm(os.listdir(folder_path)):
                file_path = os.path.join(folder_path, file)
                input_file = open(file_path)
                json_data = json.load(input_file)

                raw_dataset.append([json_data['link'],
                                    json_data['name'],
                                    json_data['statement'],
                                    json_data['input'],
                                    json_data['output'],
                                    json_data['tutorial'],
                                    json_data['solution']])

                input_file.close()

        # Create the raw dataset df of the
        raw_dataset_df = pd.DataFrame(
            raw_dataset,
            columns=['problem_link',
                     'problem_name',
                     'problem_statement',
                     'input',
                     'output',
                     'editorial',
                     'coding_solution'])
        raw_dataset_df.to_csv(CONFIG['RAW_DATASET_PATH'])

    def __read_input_files(self):
        """
        Reads input files from the dataset destination and creates a dataframe.

        This method reads input files from the dataset destination directory, which contains JSON files. Each JSON file
        represents a statement-editorial pair. The method iterates through all the files, reads the JSON data, and
        extracts the statement and editorial text.

        Returns:
            None
        """

        print("\nReading corpus")
        statement_editorial_pairs = []
        for folder_name in os.listdir(CONFIG['CODEFORCES_DATASET_PATH']):
            folder_path = os.path.join(CONFIG['CODEFORCES_DATASET_PATH'], folder_name)
            for file in tqdm(os.listdir(folder_path)):
                file_path = os.path.join(folder_path, file)
                input_file = open(file_path)
                json_data = json.load(input_file)

                """DEBUGGING CODE"""
                # if '\n' == json_data['tutorial'] or None == json_data['tutorial'] or "" == json_data['tutorial']:
                #     print(json_data['link'] + "  empty")
                # elif json_data['tutorial'] in temp:
                #     print(json_data['link'] + "  duplicate")
                # temp.append(json_data['tutorial'])
                """DEBUGGING CODE"""

                statement_editorial_pairs.append([json_data['statement'], json_data['tutorial']])

                input_file.close()

        # Create the initial dataframe of the input dataset
        self.df = pd.DataFrame(statement_editorial_pairs, columns=['statement', 'editorial'])

    def __search_unknown_symbols(self, print_symbols=False):
        """
        Searches for unknown symbols in the editorials and statements.

        Unknown symbols are characters that are not part of the printable ASCII characters.
        The found unknown symbols are stored in a set. Finally, the method returns the found unknown symbols as a string.

        Args:
            print_symbols (bool, optional): If True, the found unknown symbols are printed to the console. Defaults to
                False.

        Returns:
            str: A string containing the found unknown symbols.
        """

        # create a cluster of all editorials and statements
        statement_editorial_data = np.concatenate((self.df['editorial'].values, self.df['statement'].values))
        unknown_symbols_set = set()
        # search for unknown symbols inside the editorials + statements
        for text in statement_editorial_data:
            for character in text:
                if character not in string.printable:
                    unknown_symbols_set.add(character)
        # get the full unknown symbols set
        found_unknown_symbols = "".join(unknown_symbols_set)

        if print_symbols:
            print(found_unknown_symbols)

        # return the full unknown symbols set
        return found_unknown_symbols

    def __preprocess_corpus(self):
        """
        Preprocesses the initial corpus data.

        Note:
            - The 'GENERATE_VALIDATION_DATASET', 'PREPROCESS_DEVELOPMENT_MODE', and 'RANDOM_STATE' flags are used to control
              the behavior of the preprocessing steps.
        """

        print("\nPreprocessing initial corpus")

        # remove unknown symbols
        unknown_symbols = self.__search_unknown_symbols()
        if unknown_symbols:
            self.df['statement'] = [re.sub('[%s]' % re.escape(unknown_symbols), ' ', text) for text in
                                    self.df['statement']]
            self.df['editorial'] = [re.sub('[%s]' % re.escape(unknown_symbols), ' ', text) for text in
                                    self.df['editorial']]
        else:
            pass

        # removing punctuations
        self.df['statement'] = [re.sub('[%s]' % re.escape(string.punctuation), ' ', text) for text in
                                self.df['statement']]
        self.df['editorial'] = [re.sub('[%s]' % re.escape(string.punctuation), ' ', text) for text in
                                self.df['editorial']]

        # removing all unwanted spaces
        self.df['statement'] = [re.sub('\s+', ' ', text) for text in self.df['statement']]
        self.df['editorial'] = [re.sub('\s+', ' ', text) for text in self.df['editorial']]

        # Remove all duplicated sequences
        duplicated_rows_bool = self.df['statement'].duplicated() & self.df['editorial'].duplicated()
        self.df = self.df[~duplicated_rows_bool]

        # Remove non relevant editorials / statements
        # self.df = self.df.drop(
        #     index=[5, 24, 36, 61, 65, 74, 325, 506, 670, 685, 688, 722, 730, 736, 798, 857, 877, 879, 911,
        #            1108, 1378])
        
        self.df = self.df.drop(index=[325, 506, 722, 730, 736, 911])
        
        self.df.to_csv(CONFIG['PREPROCESSED_DATASET_PATH'], index=False)

    def __split_dataset(self):
        print("Split preprocessed corpus")
        
        # Load the preprocessed dataset
        self.df = pd.read_csv(CONFIG['PREPROCESSED_DATASET_PATH'])

        # Split dataset between train and test
        self.df_train, self.df_test = train_test_split(self.df, test_size=0.2, random_state=RANDOM_STATE)

        if GENERATE_VALIDATION_DATASET:
            # Split the train set into train and validation sets
            self.df_train, self.df_val = train_test_split(self.df_train, test_size=0.1, random_state=RANDOM_STATE)
            self.df_val.to_csv(CONFIG['PREPROCESSED_VALIDATION_DATASET_PATH'], index=False)
        
        self.df_train.to_csv(CONFIG['PREPROCESSED_TRAINING_DATASET_PATH'], index=False)
        self.df_test.to_csv(CONFIG['PREPROCESSED_TESTING_DATASET_PATH'], index=False)

    def __finish_datasets_preprocessing(self):
        # Load the preprocessed datasets
        self.df = pd.read_csv(CONFIG['PREPROCESSED_DATASET_PATH'])
        self.df_train = pd.read_csv(CONFIG['PREPROCESSED_TRAINING_DATASET_PATH'])
        if GENERATE_TESTING_DATASET:
            self.df_test = pd.read_csv(CONFIG['PREPROCESSED_TESTING_DATASET_PATH'])
        if GENERATE_VALIDATION_DATASET:
            self.df_val = pd.read_csv(CONFIG['PREPROCESSED_VALIDATION_DATASET_PATH'])
        
        # Cluster editorials based on similarity
        if PREPROCESS_DEVELOPMENT_MODE:
            self.__print_overall_info()
            self.__create_dataset_histogram()
            self.__cluster_editorials(display_elbow_plot=True, display_silhouette_plot=True,
                                      display_cluster_results=True, display_cluster_scatter=True)
        else:
            self.__cluster_editorials(display_cluster_results=True, display_cluster_scatter=True)

    def __create_dataset_histogram(self):
        # Get the data for editorials and statements
        editorials_data = self.df['editorial'].values
        statements_data = self.df['statement'].values

        # Compute the length of each editorial and statement
        editorials_lengths = [len(editorial) for editorial in editorials_data]
        statements_lengths = [len(statement) for statement in statements_data]

        # Compute the mean, minimum, and maximum lengths of each
        editorials_mean_length = np.mean(editorials_lengths)
        editorials_min_length = np.min(editorials_lengths)
        editorials_max_length = np.max(editorials_lengths)
        statements_mean_length = np.mean(statements_lengths)
        statements_min_length = np.min(statements_lengths)
        statements_max_length = np.max(statements_lengths)

        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(120, 50))

        # Create a histogram of the editorial lengths
        ax1.bar(range(len(editorials_lengths)), editorials_lengths, width=10)

        # Add horizontal lines at the mean, minimum, and maximum lengths
        ax1.axhline(editorials_mean_length, color='r', linestyle='dashed', linewidth=10, label='Mean')
        ax1.axhline(editorials_min_length, color='g', linestyle='dotted', linewidth=10, label='Minimum')
        ax1.axhline(editorials_max_length, color='b', linestyle='dotted', linewidth=10, label='Maximum')

        # Add labels and title for the first subplot
        ax1.set_xlabel('Editorial index', fontsize=120)
        ax1.set_ylabel('Number of characters', fontsize=120)
        ax1.set_title('Histogram of Editorial Lengths', fontsize=120)

        # Set the font size of the x and y axis ticks
        ax1.tick_params(axis='both', which='major', labelsize=120)

        # Create a histogram of the statement lengths
        ax2.bar(range(len(statements_lengths)), statements_lengths, width=10)

        # Add horizontal lines at the mean, minimum, and maximum lengths
        ax2.axhline(statements_mean_length, color='r', linestyle='dashed', linewidth=10)
        ax2.axhline(statements_min_length, color='g', linestyle='dotted', linewidth=10)
        ax2.axhline(statements_max_length, color='b', linestyle='dotted', linewidth=10)

        # Add labels and title for the second subplot
        ax2.set_xlabel('Statement index', fontsize=120)
        ax2.set_ylabel('Number of characters', fontsize=120)
        ax2.set_title('Histogram of Statement Lengths', fontsize=120)

        # Set the font size of the x and y axis ticks
        ax2.tick_params(axis='both', which='major', labelsize=120)

        # Add a legend to the whole figure
        fig.legend(loc='upper right', fontsize=120)

        # Save the figure to a file
        fig.savefig(CONFIG["DATASET_HISTOGRAM"], bbox_inches='tight')

        # Close the figure
        plt.close(fig)

    def __print_overall_info(self):
        """
        Print overall corpus details and save them to a log file.

        This method calculates and prints various statistics about the overall corpus, including the size of the initial
        corpus, the number of training, testing, and validation sentence-editorial pairs, and the maximum, minimum, and
        average lengths of editorials, statements, and overall text data. The calculated statistics are saved to a log file.

        Returns:
            None
        """

        print("\nPrinting overall corpus details")
        editorials_data = self.df['editorial'].values
        statements_data = self.df['statement'].values
        overall_data = np.concatenate((editorials_data, statements_data))

        editorials_length = [len(text) for text in editorials_data]
        statements_length = [len(text) for text in statements_data]
        overall_length = [len(text) for text in overall_data]

        with open(CONFIG["LOG_OVERALL_CORPUS_INFO_PATH"], 'w') as log_file:
            log_file.write(f'Size of initial corpus: {len(self.df)}\n')
            log_file.write(f'Number of training sentence-editorial pairs: {len(self.df_train)}\n')
            if GENERATE_TESTING_DATASET:
                log_file.write(f'Number of testing sentence-editorial pairs: {len(self.df_test)}\n')
            if GENERATE_VALIDATION_DATASET:
                log_file.write(f'Number of validation sentence-editorial pairs: {len(self.df_val)}\n')

            log_file.write(f'Maximum editorial length: {max(editorials_length)}\n')
            log_file.write(f'Minimum editorial length: {min(editorials_length)}\n')
            log_file.write(f'Average editorial length: {int(sum(editorials_length) / len(editorials_length))}\n')
            log_file.write(f'Standard deviation of editorial length: {np.std(editorials_length):.2f}\n')

            log_file.write(f'Maximum statement length: {max(statements_length)}\n')
            log_file.write(f'Minimum statement length: {min(statements_length)}\n')
            log_file.write(f'Average statement length: {int(sum(statements_length) / len(statements_length))}\n')
            log_file.write(f'Standard deviation of statement length: {np.std(statements_length):.2f}\n')

            log_file.write(f'Maximum overall length: {max(overall_length)}\n')
            log_file.write(f'Minimum overall length: {min(overall_length)}\n')
            log_file.write(f'Average overall length: {int(sum(overall_length) / len(overall_length))}\n')
            log_file.write(f'Standard deviation of overall length: {np.std(overall_length):.2f}\n')

        print(f"""\nOverall corpus info were saved at {CONFIG["LOG_OVERALL_CORPUS_INFO_PATH"]}""")

    def __cluster_editorials(self, display_elbow_plot=False, display_silhouette_plot=False,
                             display_cluster_results=False, display_cluster_scatter=False):
        """
        Cluster the editorials using a clustering algorithm.

        Args:
            display_elbow_plot (bool): Whether to display the elbow plot for cluster evaluation. Defaults to False.
            display_silhouette_plot (bool): Whether to display the silhouette plot for cluster evaluation.
                Defaults to False.
            display_cluster_results (bool): Whether to display the cluster results. Defaults to False.
            display_cluster_scatter (bool): Whether to display the cluster scatter plot. Defaults to False.

        Returns:
            None
        """

        print("\nClustering editorial corpus")
        cluster_builder = KMeansClustering(self.df_train['editorial'].values)
        self.clustered_editorials = cluster_builder.fit_predict(display_elbow_plot, display_silhouette_plot,
                                                                display_cluster_results, display_cluster_scatter)

    def __create_training_corpus(self):
        """
        Create a training corpus for training the model.

        This method creates a training corpus by sampling negative examples from clustered editorials. For each editorial
        in the training dataset, the method finds the cluster it belongs to and samples negative examples from the same
        cluster by randomly selecting other editorials from the cluster. The number of negative examples to sample is
        controlled by the constant NO_OF_NEGATIVE_EXAMPLES.

        Returns:
            None
        """
        print("\nCreating training corpus")

        for index, statement, editorial in tqdm(self.df_train.itertuples()):
            # for each editorial find the cluster which he belongs to
            for cluster_id, cluster_of_editorials in self.clustered_editorials.items():
                if editorial in cluster_of_editorials:
                    shuffled_cluster_of_editorials = random.sample(cluster_of_editorials, len(cluster_of_editorials))
                    negative_editorials_counter = 0
                    for editorial_id, editorial_from_cluster in enumerate(shuffled_cluster_of_editorials):
                        if editorial != editorial_from_cluster:
                            self.training_corpus_final_form.append([statement, editorial, editorial_from_cluster])
                            negative_editorials_counter += 1
                        # when N editorials have been marked as bad example stop
                        if negative_editorials_counter == NO_OF_NEGATIVE_EXAMPLES:
                            break
                    break

    def __create_random_training_corpus(self):
        """
        Create a random training corpus for training the model.

        This method creates a training corpus by sampling negative examples from random editorials. For each editorial
        in the training dataset, the method randomly selects a negative examples. The number of negative examples to sample is
        controlled by the constant NO_OF_NEGATIVE_EXAMPLES.

        Returns:
            None
        """
        print("\nCreating random training corpus")

        for index, statement, editorial in tqdm(self.df_train.itertuples()):
            negative_editorials_counter = 0
            editorials = [local_editorial for local_index, local_statement, local_editorial in
                          self.df_train.itertuples()
                          if local_editorial != editorial]
            shuffled_editorials = random.sample(editorials, len(editorials))
            for editorial_id, editorial_from_cluster in enumerate(shuffled_editorials):
                if editorial != editorial_from_cluster:
                    self.training_random_corpus_final_form.append(
                        [statement, editorial, editorial_from_cluster])
                    negative_editorials_counter += 1
                # when N editorials have been marked as bad example stop
                if negative_editorials_counter == NO_OF_NEGATIVE_EXAMPLES:
                    break

    def __create_random_siamese_training_corpus(self):
        """
        Create a random siamese network training corpus.

        This method creates a training corpus by sampling negative and positive examples from editorials.

        Returns:
            None
        """
        print("\nCreating random siamese training corpus")

        for index, statement, editorial in tqdm(self.df_train.itertuples()):
            self.training_random_siamese_corpus_final_form.append([statement, editorial, 1])
            negative_editorials_counter = 0
            editorials = [local_editorial for local_index, local_statement, local_editorial in
                          self.df_train.itertuples()
                          if local_editorial != editorial]
            shuffled_editorials = random.sample(editorials, len(editorials))
            for editorial_id, editorial_from_cluster in enumerate(shuffled_editorials):
                if editorial != editorial_from_cluster:
                    self.training_random_siamese_corpus_final_form.append([statement, editorial_from_cluster, 0])
                    negative_editorials_counter += 1
                # when N editorials have been marked as bad example stop
                if negative_editorials_counter == NO_OF_NEGATIVE_EXAMPLES:
                    break
        self.training_random_siamese_corpus_final_form = random.sample(
            self.training_random_siamese_corpus_final_form,
            len(self.training_random_siamese_corpus_final_form)
        )

    def __create_siamese_training_corpus(self):
        """
        Create a siamese network training corpus.

        This method creates a training corpus by sampling negative and positive examples from editorials.

        Returns:
            None
        """
        print("\nCreating siamese training corpus")

        for index, statement, editorial in tqdm(self.df_train.itertuples()):
            self.training_siamese_corpus_final_form.append([statement, editorial, 1])
            # for each editorial find the cluster which he belongs to
            for cluster_id, cluster_of_editorials in self.clustered_editorials.items():
                if editorial in cluster_of_editorials:
                    shuffled_cluster_of_editorials = random.sample(cluster_of_editorials, len(cluster_of_editorials))
                    negative_editorials_counter = 0
                    for editorial_id, editorial_from_cluster in enumerate(shuffled_cluster_of_editorials):
                        if editorial != editorial_from_cluster:
                            self.training_siamese_corpus_final_form.append([statement, editorial_from_cluster, 0])
                            negative_editorials_counter += 1
                        # when N editorials have been marked as bad example stop
                        if negative_editorials_counter == NO_OF_NEGATIVE_EXAMPLES:
                            break
                    break
        self.training_siamese_corpus_final_form = random.sample(
            self.training_siamese_corpus_final_form,
            len(self.training_siamese_corpus_final_form)
        )

    def __generate_test_precision_at_k_final_corpus(self, input_df):
        """
        Generate a test corpus for calculating precision at k metric.

        This method generates a test corpus for calculating precision at k metric for evaluating the performance of the model.
        The test corpus is generated by randomly selecting k negative examples for each editorial in the input dataframe,
        where k is set to 9. The generated test corpus is returned as a list of lists consisting of the statemenet, the target editorial and some randomly selected k negative examples

        Args:
            input_df (DataFrame): Input dataframe containing statement and editorial pairs.

        Returns:
            List of Lists: Test corpus in final form for calculating precision at k metric.
        """

        testing_precision_at_k_final_form_corpus = []
        for index, statement, editorial in tqdm(input_df.itertuples()):
            dataset_row = [statement, editorial]
            for i in range(9):
                random_row = input_df.sample()
                while random_row.iloc[0]['editorial'] == editorial:
                    random_row = input_df.sample()
                dataset_row.append(random_row.iloc[0]['editorial'])
            testing_precision_at_k_final_form_corpus.append(dataset_row)
        return testing_precision_at_k_final_form_corpus

    def __create_testing_precision_at_k_corpus(self):
        """
        Create the test and validation corpus for calculating precision at k metric.

        Returns:
            None
        """
        if GENERATE_TESTING_DATASET:
            print("\nCreating P@K testing corpus")
            self.testing_corpus_precision_at_k_final_form = self.__generate_test_precision_at_k_final_corpus(self.df_test)
        if GENERATE_VALIDATION_DATASET:
            print("\nCreating P@K validation corpus")
            self.validation_corpus_precision_at_k_final_form = self.__generate_test_precision_at_k_final_corpus(
                self.df_val)

    def __generate_test_correlation_final_corpus(self, input_df):
        """
        Generate the test corpus for calculating correlation coefficient.

        This method generates the test corpus for calculating the correlation coefficient, which is used
        to evaluate the performance of the model. For each editorial in the input dataframe, two rows are added
        to the test corpus. The first row contains the statement and the editorial with a label of 1, indicating
        that they are positively correlated. The second row contains the same statement with a randomly selected
        editorial (different from the actual editorial) with a label of 0, indicating that they are not correlated.
        The generated test corpus is shuffled before returning.

        Args:
            input_df (DataFrame): Input dataframe containing the statements and editorials.

        Returns:
            list: The generated test corpus in the final form with statement, editorial, and correlation label.
        """
        testing_correlation_final_form_corpus = []
        for index, statement, editorial in tqdm(input_df.itertuples()):
            testing_correlation_final_form_corpus.append([statement, editorial, 1])
            random_row = input_df.sample()
            while random_row.iloc[0]['editorial'] == editorial:
                random_row = input_df.sample()
            testing_correlation_final_form_corpus.append([statement, random_row.iloc[0]['editorial'], 0])
        random.shuffle(testing_correlation_final_form_corpus)
        return testing_correlation_final_form_corpus

    def __create_testing_correlation_corpus(self):
        """
        Create the testing and validation corpus for calculating correlation coefficient.

        Returns:
            None
        """
        if GENERATE_TESTING_DATASET:
            print("\nCreating correlation testing corpus")
            self.testing_corpus_correlation_final_form = self.__generate_test_correlation_final_corpus(self.df_test)
        if GENERATE_VALIDATION_DATASET:
            print("\nCreating correlation validation corpus")
            self.validation_corpus_correlation_final_form = self.__generate_test_correlation_final_corpus(self.df_val)

    def __save_corpus_final_version(self):
        """
        Save the final version of the generated datasets.

        Returns:
            None
        """

        print("\nSaving the datasets")

        train_df = pd.DataFrame(self.training_corpus_final_form, columns=['anchor', 'positive', 'negative'])
        train_df.to_csv(CONFIG['CP_MNRL_TRAINING_DATASET_PATH'])

        random_train_df = pd.DataFrame(self.training_random_corpus_final_form,
                                       columns=['anchor', 'positive', 'negative'])
        random_train_df.to_csv(CONFIG['CP_MNRL_RANDOM_TRAINING_DATASET_PATH'])

        siamese_train_df = pd.DataFrame(self.training_siamese_corpus_final_form,
                                        columns=['premise', 'hypothesis', 'label'])
        siamese_train_df.to_csv(CONFIG['CP_SIAMESE_TRAINING_DATASET_PATH'])

        random_siamese_train_df = pd.DataFrame(self.training_random_siamese_corpus_final_form,
                                        columns=['premise', 'hypothesis', 'label'])
        random_siamese_train_df.to_csv(CONFIG['CP_SIAMESE_RANDOM_TRAINING_DATASET_PATH'])

        if GENERATE_TESTING_DATASET:
            test_correlation_df = pd.DataFrame(self.testing_corpus_correlation_final_form,
                                               columns=['statement', 'editorial', 'label'])
            test_correlation_df.to_csv(path_or_buf=CONFIG['CP_TESTING_CORRELATION_DATASET_PATH'])
            test_pk_df = pd.DataFrame(self.testing_corpus_precision_at_k_final_form)  # 1.statement, 2...11 editorials
            test_pk_df.to_csv(path_or_buf=CONFIG['CP_TESTING_P@K_DATASET_PATH'])
        if GENERATE_VALIDATION_DATASET:
            validation_correlation_df = pd.DataFrame(self.validation_corpus_correlation_final_form,
                                                     columns=['statement', 'editorial', 'label'])
            validation_correlation_df.to_csv(
                path_or_buf=CONFIG['CP_VALIDATION_CORRELATION_DATASET_PATH'])
            validation_pk_df = pd.DataFrame(
                self.validation_corpus_precision_at_k_final_form)  # 1.statement, 2...11 editorials
            validation_pk_df.to_csv(path_or_buf=CONFIG['CP_VALIDATION_P@K_DATASET_PATH'])

    def __create_editorial_index(self):
        """
        Create an index of editorials.

        Returns:
            None
        """

        print("\nCreating editorial index")
        editorial_index_df = self.df['editorial']
        editorial_index_df.to_csv(path_or_buf=CONFIG['EDITORIALS_INDEX_PATH'])

    def build_dataset(self):
        """
        Build all testing and validation datasets.

        Returns:
            None
        """
        if CREATE_PREPROCESSED_DATASET:
            self.__read_input_files()
            self.__preprocess_corpus()
            self.__create_editorial_index()
            self.__split_dataset()
        self.__finish_datasets_preprocessing()
        if not PREPROCESS_DEVELOPMENT_MODE:
            self.__create_training_corpus()
            self.__create_random_training_corpus()
            self.__create_siamese_training_corpus()
            self.__create_random_siamese_training_corpus()
            if GENERATE_TESTING_DATASET:
                self.__create_testing_correlation_corpus()
                self.__create_testing_precision_at_k_corpus()
            self.__save_corpus_final_version()

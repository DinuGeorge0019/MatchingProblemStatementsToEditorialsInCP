


# standard library imports
import os
import re
import string
import pandas as pd

# local application/library specific imports
from APP_CONFIG import Config

# define configuration proxy
working_dir = os.path.dirname(os.getcwd())
configProxy = Config(working_dir)

# get configuration
CONFIG = configProxy.return_config()


class ErrorAnalysisModule:

    def __init__(self):
        self.diff_analysis = {
            'A': [0, 0],
            'B': [0, 0],
            'C': [0, 0],
            'D': [0, 0],
            'E': [0, 0],
            'F': [0, 0],
            'G': [0, 0]
        }

        self.test_precision_at_k_df = pd.read_csv(CONFIG['CP_TESTING_P@K_DATASET_PATH'],
                                             usecols=[1, 2])
        self.raw_dataset = pd.read_csv(CONFIG['RAW_DATASET_PATH'], usecols=[1, 3, 6])
        self.raw_dataset['problem_statement'] = self.raw_dataset['problem_statement'].apply(self.__preprocess_statement)
        self.raw_dataset['editorial'] = self.raw_dataset['editorial'].apply(self.__preprocess_statement)

    def __init_diff_analysis(self):
        for key in self.diff_analysis:
            self.diff_analysis[key] = [0, 0]

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

    def __create_problem_statement_collection(self, p_error_analysis_df):

        error_analysis_collection = []

        for index, (test_row, error_row) in enumerate(
                zip(self.test_precision_at_k_df.iterrows(), p_error_analysis_df.iterrows())):
            test_index, test_row = test_row
            error_index, error_row = error_row
            error_analysis_collection.append([test_row[0], test_row[1], error_row['positional_error_flag'], 'X'])

        for idx, problem in enumerate(error_analysis_collection):
            # Check the position of the problem in the raw_dataset DataFrame
            problem_position = self.raw_dataset.index[self.raw_dataset['problem_statement'] == problem[0]].tolist()

            if len(problem_position) > 0:
                problem_difficulty = self.raw_dataset.iloc[problem_position[0], 0].split("/")[-1]
                error_analysis_collection[idx][3] = problem_difficulty[-2 if len(problem_difficulty) > 1 else -1]

        return error_analysis_collection

    def __write_error_analysis_log(self, p_error_analysis_collection, log_path, analysis_title):

        self.__init_diff_analysis()

        for problem in p_error_analysis_collection:
            difficulty = problem[3]
            score = problem[2]

            if difficulty in self.diff_analysis:
                self.diff_analysis[difficulty][score] += 1

        with open(log_path, 'a') as log_file:
            log_file.write(f'{analysis_title} error analysis\n')
            for difficulty, data in self.diff_analysis.items():
                log_file.write(f"Problem difficulty {difficulty}: Bad match: {data[0]}, Good match: {data[1]}\n")
            log_file.write('\n')

    def __write_detailed_error_analysis_log(self, error_analysis_collection, log_path):
        with open(log_path, 'w') as log_file:
            for problem_statement, problem_editorial, _, problem_difficulty in error_analysis_collection:
                log_file.write(f'Problem Difficulty: {problem_difficulty}\n')
                log_file.write(f'Problem Statement:\n {problem_statement}\n')
                log_file.write(f'Problem Editorial:\n {problem_editorial}\n')
                log_file.write('\n')

    def create_error_analysis(self, file_name, log_path, analysis_title):
        error_analysis_df = pd.read_csv(CONFIG['ERROR_ANALYSIS_FOLDER_PATH'] + file_name, usecols=[1])
        error_analysis_collection = self.__create_problem_statement_collection(error_analysis_df)
        self.__write_error_analysis_log(error_analysis_collection, log_path, analysis_title)
        if analysis_title == 'P@1':
            if 'Validation' in file_name:
                filtered_collection = [data for data in error_analysis_collection if data[2] == 0]
                self.__write_detailed_error_analysis_log(filtered_collection,
                                                         CONFIG["LOG_VALIDATION_DETAILED_ERROR_ANALYSIS_PATH"])
            elif 'Test' in file_name:
                filtered_collection = [data for data in error_analysis_collection if data[2] == 0]
                self.__write_detailed_error_analysis_log(filtered_collection,
                                                         CONFIG["LOG_TEST_DETAILED_ERROR_ANALYSIS_PATH"])

    def create_pk_test_error_analysis(self):
        with open(CONFIG["LOG_TEST_ERROR_ANALYSIS_PATH"], 'w') as file:
            file.truncate(0)

        self.create_error_analysis("Error_Analysis_p1_Test.csv", CONFIG["LOG_TEST_ERROR_ANALYSIS_PATH"], 'P@1')
        self.create_error_analysis("Error_Analysis_p2_Test.csv", CONFIG["LOG_TEST_ERROR_ANALYSIS_PATH"], 'P@2')
        self.create_error_analysis("Error_Analysis_p3_Test.csv", CONFIG["LOG_TEST_ERROR_ANALYSIS_PATH"], 'P@3')
        self.create_error_analysis("Error_Analysis_p5_Test.csv", CONFIG["LOG_TEST_ERROR_ANALYSIS_PATH"], 'P@5')

    def create_pk_validation_error_analysis(self):
        with open(CONFIG["LOG_VALIDATION_ERROR_ANALYSIS_PATH"], 'w') as file:
            file.truncate(0)

        self.create_error_analysis("Error_Analysis_p1_Validation.csv", CONFIG["LOG_VALIDATION_ERROR_ANALYSIS_PATH"], 'P@1')
        self.create_error_analysis("Error_Analysis_p2_Validation.csv", CONFIG["LOG_VALIDATION_ERROR_ANALYSIS_PATH"], 'P@2')
        self.create_error_analysis("Error_Analysis_p3_Validation.csv", CONFIG["LOG_VALIDATION_ERROR_ANALYSIS_PATH"], 'P@3')
        self.create_error_analysis("Error_Analysis_p5_Validation.csv", CONFIG["LOG_VALIDATION_ERROR_ANALYSIS_PATH"], 'P@5')

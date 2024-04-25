

class Config(object):
    """
    Config with all the paths and flags needed.
    """

    def __init__(self, WORKING_DIR):
        self.request_headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Max-Age': '3600',
        }

        self.config = {
            "CODEFORCES_GET_CONTEST_LIST_REQUEST_LINK": "http://codeforces.com/api/contest.list",
            "CODEFORCES_BASE_CONSTEST_LINK": "http://codeforces.com/contest/",
            "CHROME_DRIVER_PATH": "C:\\Program Files\\Google\\Chrome\\Application\\chromedriver.exe",
            "CODEFORCES_LINK": "https://codeforces.com/",

            "WORKING_DIR": f"{WORKING_DIR}",
            "DATASET_DESTINATION": f"{WORKING_DIR}\\01_CODEFORCES_DATASET",
            "CODEFORCES_EDUCATIONAL_FILE": f"{WORKING_DIR}\\00_CODEFORCES_DATA\\EDUCATIONAL_CONTESTS.in",
            "CODEFORCES_DIV1_FILE": f"{WORKING_DIR}\\00_CODEFORCES_DATA\\DIV1_CONTESTS.in",
            "CODEFORCES_DIV2_FILE": f"{WORKING_DIR}\\00_CODEFORCES_DATA\\DIV2_CONTESTS.in",
            "CODEFORCES_DIV1&2_FILE": f"{WORKING_DIR}\\00_CODEFORCES_DATA\\DIV1&2_CONTESTS.in",
            "CODEFORCES_DIV3_FILE": f"{WORKING_DIR}\\00_CODEFORCES_DATA\\DIV3_CONTESTS.in",
            "CODEFORCES_DIV4_FILE": f"{WORKING_DIR}\\00_CODEFORCES_DATA\\DIV4_CONTESTS.in"
        }

    def return_config(self):
        """
        Return entire config dictionary
        Returns:
            None
        """
        return self.config

    def return_request_headers(self):
        """
        Return the headers used to request the GET operation
        Returns:
            None
        """
        return self.request_headers

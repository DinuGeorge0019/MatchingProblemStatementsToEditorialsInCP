

# standard library imports
import os
import time
import requests
import json

# related third-party
from bs4 import BeautifulSoup
from selenium import webdriver

# local application/library specific imports
from APP_CONFIG import Config
from AlgoProblem import AlgoProblem

# define configuration proxy
working_dir = os.path.dirname(os.getcwd())
configProxy = Config(working_dir)

# get configuration
CONFIG = configProxy.return_config()

# get headers configuration
HEADERS = configProxy.return_request_headers()


class CodeforcesWebScrapper:
    def __init__(self):
        self.contests_list = []
        self.VARIANT1 = 0
        self.VARIANT2 = 0
        self.VARIANT3 = 0
        self.VARIANT4 = 0
        self.VARIANT5 = 0

    # Function to remove tags
    def __remove_page_element_tags(self, html_content, content_list):
        """
        Remove HTML tags from the specified list of page elements in the given HTML content.
        Args:
            html_content (bs4.BeautifulSoup): The BeautifulSoup object representing the HTML content.
            content_list (list): A list of page elements to remove from the HTML content.
        Returns:
            bs4.BeautifulSoup: The modified BeautifulSoup object with the specified page elements removed.
        """
        for data in html_content(content_list):
            # Remove tags
            data.decompose()

        # return data by retrieving the tag content
        return html_content

    def fetch_contests(self):
        """
        Create a request to the codeforces api to retrieve all contests links and saves to the specified file afterwards.
        Args:
            None
        Returns:
            True if the operation succeeded, False otherwise
        """
        codeforces_data_path = os.path.join(CONFIG["WORKING_DIR"], "00_CODEFORCES_DATA")
        if not os.path.isdir(codeforces_data_path):
            os.mkdir(codeforces_data_path)

        def write_line(file, contest_id) -> None:
            file.write(CONFIG["CODEFORCES_BASE_CONSTEST_LINK"] + str(contest_id) + "\n")

        request_answer = requests.get(CONFIG["CODEFORCES_GET_CONTEST_LIST_REQUEST_LINK"])
        request_answer = request_answer.json()
        if request_answer['status'] == 'OK':

            answer_result_data = request_answer['result']

            div1_contests_file = open(CONFIG["CODEFORCES_DIV1_FILE"], "w")
            div1_2_contests_file = open(CONFIG["CODEFORCES_DIV1&2_FILE"], "w")
            div2_contests_file = open(CONFIG["CODEFORCES_DIV2_FILE"], "w")
            div3_contests_file = open(CONFIG["CODEFORCES_DIV3_FILE"], "w")
            div4_contests_file = open(CONFIG["CODEFORCES_DIV4_FILE"], "w")
            educational_contests_file = open(CONFIG["CODEFORCES_EDUCATIONAL_FILE"], "w")

            for contest_data in answer_result_data:
                if 'Educational' in contest_data['name']:
                    write_line(educational_contests_file, contest_data['id'])
                elif 'Div. 1' in contest_data['name'] and "Div. 2" in contest_data['name']:
                    write_line(div1_2_contests_file, contest_data['id'])
                elif 'Div. 1' in contest_data['name']:
                    write_line(div1_contests_file, contest_data['id'])
                elif 'Div. 2' in contest_data['name']:
                    write_line(div2_contests_file, contest_data['id'])
                elif 'Div. 3' in contest_data['name']:
                    write_line(div3_contests_file, contest_data['id'])
                elif 'Div. 4' in contest_data['name']:
                    write_line(div4_contests_file, contest_data['id'])

            div1_contests_file.close()
            div1_2_contests_file.close()
            div2_contests_file.close()
            div3_contests_file.close()
            div4_contests_file.close()
            educational_contests_file.close()
            return True
        else:
            print('Codeforces api servers are down')
            return False

    def __read_contests_list(self, contest_file_path) -> None:
        """
        Read the list of contests links from the specified file
        Args:
            contest_file_path (str): The file path to the contest file.
        Returns:
            None
        """
        with open(contest_file_path) as file:
            lines = file.readlines()
            self.contests_list = [line.rstrip() for line in lines]

    def __get_contest_blog_link(self, contest_page_link):
        """
        Return the link of the editorial for the given contest page.
        Args:
            contest_page_link (str): The link of the contest page, expected to be a valid URL.
        Returns:
            None
        """
        req = requests.get(contest_page_link, HEADERS)
        soup = BeautifulSoup(req.content, 'html5lib')
        contests_materials = soup.find_all("a", href=lambda href: href and "/blog/entry/" in href)

        tutorial_content_list = []
        for content in contests_materials:
            if "Tutorial" in content.text or "Editorial" in content.text or "T (en)" in content.text or "E (en)" in content.text:
                if "codeforces.com" not in content.get("href"):
                    tutorial_content_list.append(CONFIG["CODEFORCES_LINK"] + content.get("href"))
                else:
                    tutorial_content_list.append(content.get("href"))

        if len(tutorial_content_list) >= 1:
            return tutorial_content_list[0]

        return None

    def __get_all_problems_from_contest(self, contest_page_link):
        """
        Returns the links of the problems from the given contest page.
        Args:
            contest_page_link (str): The link to the contest page, expected to be a valid URL.
        Returns:
            list: A list of strings, where each string represents a link to a problem page.
        """
        req = requests.get(contest_page_link, HEADERS)
        soup = BeautifulSoup(req.content, 'html5lib')
        problems_table = soup.find("table", class_="problems")
        problems = problems_table.find_all("a", href=lambda href: href and href.startswith(
            "/contest/") and "/problem/" in href)
        list_of_problems = []
        for problem_link in problems:
            full_link = CONFIG["CODEFORCES_LINK"] + problem_link.get("href")
            if full_link not in list_of_problems:
                list_of_problems.append(full_link)
        return list(list_of_problems)

    def __get_contest_blog_content(self, contest_blog_link):
        """
        Retrieves relevant content from a contest blog page identified by the given link.

        Args:
            contest_blog_link (str): The link to the contest blog page, expected to be a valid URL.

        Returns:
            tuple: A tuple containing two values:
                - blog_relevant_content (bs4.element.Tag or None): The relevant content extracted from the blog page,
                  which could be a BeautifulSoup Tag or None if no relevant content is found.
                - variant (int): An integer representing the type of variant identified based on the content of the blog page.
        """
        driver_options = webdriver.ChromeOptions()
        driver_options.add_experimental_option('excludeSwitches', ['enable-logging'])
        driver = webdriver.Chrome(executable_path=CONFIG["CHROME_DRIVER_PATH"], options=driver_options)
        driver.get(contest_blog_link)
        time.sleep(1)
        soup = BeautifulSoup(driver.page_source, 'html5lib')
        driver.quit()

        blog_relevant_content = soup.find("div", class_="content")

        unofficial_contest_content = soup.find("div", class_="title")
        if unofficial_contest_content is not None and "unofficial" in unofficial_contest_content.text.lower():
            print("VARIANT 5 blog contest: " + contest_blog_link)
            self.VARIANT5 += 1
            return blog_relevant_content, 5

        variant_spoiler_title = blog_relevant_content.find("b", class_="spoiler-title")
        if variant_spoiler_title is not None and \
                ("tutorial" in variant_spoiler_title.text.lower() or "editorial" in variant_spoiler_title.text.lower()):
            print("VARIANT 1 blog contest: " + contest_blog_link)
            self.VARIANT1 += 1
            return blog_relevant_content, 1

        if variant_spoiler_title is not None and \
                ("++ solution" in variant_spoiler_title.text
                 or "Code" in variant_spoiler_title.text):
            print("VARIANT 3 blog contest: " + contest_blog_link)
            self.VARIANT3 += 1
            return blog_relevant_content, 3

        variant_link_to_other_page = blog_relevant_content.find_all("a", href=lambda
            href: href and "http://pastebin.com/" in href)
        if variant_link_to_other_page is not None and len(variant_link_to_other_page) >= 2:
            print("VARIANT 4 blog contest: " + contest_blog_link)
            self.VARIANT4 += 1
            return blog_relevant_content, 4

        variant_problem_statement = blog_relevant_content.find("div", class_="problem-statement")
        variant_topography = blog_relevant_content.find("div", class_="ttypography")
        if variant_problem_statement is not None or variant_topography is not None:
            print("VARIANT 2 blog contest: " + contest_blog_link)
            self.VARIANT2 += 1
            return blog_relevant_content, 2

        print("Unexpected variant: " + contest_blog_link)
        return None, -1

    def __proces_contests_list(self, contest_category):
        """
        Process the list of contests for a given contest category and generates the dataset file for each problem.

        Args:
            contest_category (str): The contest category for which the contests list is to be processed.

        Returns:
            None
        """
        saved_problem_counter = 0
        EXCLUDED_CONTESTS = ["598", "762", "938", "630"]

        dataset_path = os.path.join(CONFIG["DATASET_DESTINATION"], contest_category)
        if not os.path.isdir(dataset_path):
            os.mkdir(dataset_path)

        for contest_link in self.contests_list:
            contest_id = contest_link.split("/")[-1]
            if contest_id in EXCLUDED_CONTESTS:
                continue
            contest_blog_link = self.__get_contest_blog_link(contest_link)
            if contest_blog_link is None:
                continue
            contest_content, variant_type = self.__get_contest_blog_content(contest_blog_link)
            if contest_content is None:
                continue

            contest_problems = self.__get_all_problems_from_contest(contest_link)
            for problem_link in contest_problems:
                saved_problem_counter += 1
                problem = AlgoProblem(problem_link, contest_content, variant_type)
                if not problem.interactive and not problem.noSolution and not problem.noTutorial:
                    file_path = os.path.join(CONFIG["DATASET_DESTINATION"], contest_category, problem.name + ".json")
                    if os.path.isfile(file_path):
                        file_path = os.path.join(CONFIG["DATASET_DESTINATION"], contest_category, problem.name + str(saved_problem_counter) + ".json")
                    output_file = open(file_path, "w")
                    output_file.write(json.dumps(problem.__dict__))
                    output_file.close()

    def build_dataset(self):
        """
        Fetch all algorithmic problems from the contests retrieved.
        Args:
            None
        Returns:
            None
        """
        print("Processing EDUCATIONAL contests")
        self.__read_contests_list(CONFIG["CODEFORCES_EDUCATIONAL_FILE"])
        self.__proces_contests_list("EDUCATIONAL")
        print("Processing DIV3 contests")
        self.__read_contests_list(CONFIG["CODEFORCES_DIV3_FILE"])
        self.__proces_contests_list("DIV3")
        print("Processing DIV4 contests")
        self.__read_contests_list(CONFIG["CODEFORCES_DIV4_FILE"])
        self.__proces_contests_list("DIV4")

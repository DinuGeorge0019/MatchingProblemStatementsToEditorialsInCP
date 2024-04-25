

# standard library imports
import os
import time
import requests

# related third-party
from bs4 import BeautifulSoup
from selenium import webdriver

# local application/library specific imports
from APP_CONFIG import Config

# define configuration proxy
working_dir = os.path.dirname(os.getcwd())
configProxy = Config(working_dir)

# get configuration
CONFIG = configProxy.return_config()

# get headers configuration
HEADERS = configProxy.return_request_headers()


class AlgoProblem:
    def __init__(self, problem_link: str, contest_blog_content: BeautifulSoup, variant_type: int):
        self.link = problem_link
        self.problemId = ""
        split_link = problem_link.split("/")
        self.shortId = split_link[-3] + split_link[-1]
        self.contest_number = split_link[-3]
        self.name = ""
        self.statement = ""
        self.input = ""
        self.output = ""
        self.tutorial = ""
        self.solution = ""
        self.interactive = False
        self.noSolution = False
        self.noTutorial = False
        self.__get_problem_initial_data()
        self.__get_problem_solution(contest_blog_content, variant_type)
        if self.tutorial == "":
            self.noTutorial = True
        if self.solution == "":
            self.noSolution = True

    def __handle_variant_one(self, contest_blog_content: BeautifulSoup):
        """
        Handle variant one of contest blog content.
        Official editorial
        __ HIGHEST PRIO __
        The number of editorials is high, as this is the main editorial template.
        This method extracts tutorial and solution information from the contest blog content.
        Args:
            contest_blog_content (BeautifulSoup): BeautifulSoup object representing the contest blog content.
        Returns:
            None
        """
        print(self.link)
        problems = contest_blog_content.find_all("p")
        for problem in problems:
            if self.shortId in problem.text:
                self.tutorial = problem.find_next("div", class_="problem-statement")
                if not self.tutorial:
                    self.tutorial = problem.find_next("div", class_="spoiler-content")
                    if not self.tutorial:
                        print("Unexpected problem with tutorial at: " + self.name + " " + self.link)
                    else:
                        self.tutorial = self.__remove_page_element_tags(self.tutorial,
                                                                        ['style', 'span', 'script', 'meta'])
                        self.tutorial = self.tutorial.text
                else:
                    self.tutorial = self.__remove_page_element_tags(self.tutorial, ['style', 'span', 'script', 'meta'])
                    self.tutorial = self.tutorial.text

                self.solution = problem.find_next("code", class_="prettyprint prettyprinted")
                if not self.solution:
                    print("Unexpected problem with tutorial at: " + self.name + " " + self.link)
                else:
                    self.solution = self.solution.text

                break

    def __handle_variant_two(self, contest_blog_content):
        """
        Handle variant two of contest blog content.
        __ MID PRIO __
        __ NO SOLUTION IN EDITORIAL __
        The number of editorials is relative high, as this is the last editorial template
        This method extracts tutorial and solution information from the contest blog content.
        Args:
            contest_blog_content (BeautifulSoup): BeautifulSoup object representing the contest blog content.
        Returns:
            None
        """
        print(self.link)
        problems = contest_blog_content.find_all("h3")
        if len(problems) == 0:
            problems = contest_blog_content.find_all("p")
            for problem in problems:
                problem = self.__remove_page_element_tags(problem, ['style', 'span', 'script', 'meta'])
                if self.shortId in problem.text:
                    next_paragraphs = problem.find_next_siblings()
                    for paragraph in next_paragraphs:
                        paragraph_children = paragraph.findChildren("a", href=lambda href: href and href.startswith(
                            "/contest/"))
                        if paragraph_children:
                            break
                        self.tutorial += paragraph.text + "\n"
                    break
        else:
            for problem in problems:
                if self.shortId in problem.text:
                    self.tutorial = problem.find_next("div", class_="problem-statement")
                    if not self.tutorial:
                        print("Unexpected problem with tutorial at: " + self.name + " " + self.link)
                    else:
                        self.tutorial = self.__remove_page_element_tags(self.tutorial,
                                                                        ['style', 'span', 'script', 'meta'])
                        self.tutorial = self.tutorial.text
                    break

        standing_page_link = "https://codeforces.com/contest/{0}/standings".format(self.contest_number)
        req = requests.get(standing_page_link, HEADERS)
        standing_soup = BeautifulSoup(req.content, 'html5lib')
        standing = standing_soup.find("table", class_="standings").findChildren()
        winner_table_line = None
        line_count = 0
        for line in standing:
            if line.name == 'tr':
                line_count += 1
            if line_count == 2:
                winner_table_line = line
                break

        submission_id = winner_table_line.find("td", {"problemid": self.problemId})
        cell_status = submission_id.findChildren()[0]["class"]
        if cell_status[0] != 'cell-accepted':
            self.noSolution = True
            return
        else:
            submission_id = submission_id['acceptedsubmissionid']

        submission_link = "https://codeforces.com/contest/{0}/submission/{1}".format(self.contest_number, submission_id)
        while not self.solution:
            req = requests.get(submission_link, HEADERS)
            solution_page_soup = BeautifulSoup(req.content, 'html5lib')
            self.solution = solution_page_soup.find("pre", {"id": "program-source-text"})
            if not self.solution:
                print("Too many requests at " + self.name + " " + self.link)
                time.sleep(10)
        self.solution = self.solution.text

    def __handle_variant_three(self, contest_blog_content):
        """
        Handle variant three of contest blog content.
        __ MID PRIO __
        __ SOLUTION IN EDITORIAL with dynamic template __
        The number of editorials is not high but relevant.
        This method extracts tutorial and solution information from the contest blog content.
        Args:
            contest_blog_content (BeautifulSoup): BeautifulSoup object representing the contest blog content.
        Returns:
            None
        """
        print(self.link)
        problems = contest_blog_content.find_all("h3")
        for problem in problems:
            if self.shortId in problem.text:
                next_paragraphs = problem.find_next_siblings()
                seen_first_paragraph = False
                if next_paragraphs[0].find('a') is None:
                    for paragraph in next_paragraphs:
                        if paragraph.name == 'div' and seen_first_paragraph:
                            break
                        paragraph = self.__remove_page_element_tags(paragraph, ['style', 'span', 'script', 'meta'])
                        self.tutorial += paragraph.text + "\n"
                        seen_first_paragraph = True
                elif "/profile/" in next_paragraphs[0].find('a')['href']:
                    for paragraph in next_paragraphs[1:]:
                        if paragraph.name == 'div':
                            break
                        paragraph = self.__remove_page_element_tags(paragraph, ['style', 'span', 'script', 'meta'])
                        self.tutorial += paragraph.text + "\n"

                self.solution = problem.find_next("code", class_="prettyprint prettyprinted").text
                if not self.solution:
                    print("A solution does not exists for this variant: " + self.name + " " + self.link)
                break

    def __handle_variant_four(self, contest_blog_content):
        """
        Handle variant four of contest blog content.
        __ MID PRIO __
        __ SOLUTION IN EDITORIAL with link to another website __
        The number of editorials is not high but relevant.
        This method extracts tutorial and solution information from the contest blog content.
        Args:
            contest_blog_content (BeautifulSoup): BeautifulSoup object representing the contest blog content.
        Returns:
            None
        """
        print(self.link)
        problems = contest_blog_content.find_all("h3")
        for problem in problems:
            if self.shortId in problem.text:
                solution_link = problem.find_next("a", href=lambda href: href and (
                        "http://pastebin.com/" in href or "http://ideone.com/" in href))
                end_of_tutorial = solution_link.parent
                next_paragraphs = problem.find_next_siblings()

                seen_first_paragraph = False
                if next_paragraphs[0].find('a') is None:
                    for paragraph in next_paragraphs:
                        if paragraph == end_of_tutorial and seen_first_paragraph:
                            break
                        paragraph = self.__remove_page_element_tags(paragraph, ['style', 'span', 'script', 'meta'])
                        self.tutorial += paragraph.text + "\n"
                        seen_first_paragraph = True
                elif "/profile/" in next_paragraphs[0].find('a')['href']:
                    for paragraph in next_paragraphs[1:]:
                        if paragraph == end_of_tutorial:
                            break
                        paragraph = self.__remove_page_element_tags(paragraph, ['style', 'span', 'script', 'meta'])
                        self.tutorial += paragraph.text + "\n"

                solution_link = solution_link.get("href")
                req = requests.get(solution_link, HEADERS)
                solution_page_soup = BeautifulSoup(req.content, 'html5lib')
                if "http://pastebin.com/" in solution_link:
                    self.solution = solution_page_soup.find("div", class_="source cpp").text
                elif "http://ideone.com/" in solution_link:
                    self.solution = solution_page_soup.find("pre", class_="cpp").text

                if not self.solution:
                    print("A solution does not exists for this variant: " + self.name + " " + self.link)

                break

    def __handle_variant_five(self, contest_blog_content):
        """
        Handle variant five of contest blog content.
        Unofficial editorial
        __ LOW PRIO __
        __ SOLUTION IN EDITORIAL with link to another codeforces page __
        The number of unofficial tutorials is low
        This method extracts tutorial and solution information from the contest blog content.
        Args:
            contest_blog_content (BeautifulSoup): BeautifulSoup object representing the contest blog content.
        Returns:
            None
        """
        print(self.link)
        problems = contest_blog_content.find_all("h4")
        for problem in problems:
            if self.shortId in problem.text:
                end_of_tutorial = problem.find_next("h4")
                next_paragraphs = problem.find_next_siblings()
                for paragraph in next_paragraphs:
                    if paragraph == end_of_tutorial:
                        break
                    paragraph = self.__remove_page_element_tags(paragraph, ['style', 'span', 'script', 'meta'])
                    self.tutorial += paragraph.text + "\n"

                solution_link = problem.find_next("a", href=lambda href: href and "/submission/" in href).get("href")
                req = requests.get(CONFIG["CODEFORCES_LINK"] + solution_link, HEADERS)
                solution_soup = BeautifulSoup(req.content, 'html5lib')
                self.solution = solution_soup.find("pre", id="program-source-text").text

                if not self.solution:
                    print("A solution does not exists for this variant: " + self.name + " " + self.link)

                break

    def __get_problem_solution(self, contest_blog_content, variant_type):
        """
        Get problem solution from contest blog content based on the specified variant type.
        Args:
            contest_blog_content (BeautifulSoup): BeautifulSoup object representing the contest blog content.
            variant_type (int): Variant type of the contest blog content.
        Returns:
            None
        """
        if variant_type == 1:
            self.__handle_variant_one(contest_blog_content)
        elif variant_type == 2:
            self.__handle_variant_two(contest_blog_content)
        elif variant_type == 3:
            self.__handle_variant_three(contest_blog_content)
        elif variant_type == 4:
            self.__handle_variant_four(contest_blog_content)
        elif variant_type == 5:
            self.__handle_variant_five(contest_blog_content)

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

    def __get_problem_initial_data(self):
        """
        Get the problem initial data: name, statement, input, output
        Args:
            None
        Returns:
            None
        """
        driver_options = webdriver.ChromeOptions()
        driver_options.add_experimental_option('excludeSwitches', ['enable-logging'])
        driver = webdriver.Chrome(executable_path=CONFIG["CHROME_DRIVER_PATH"], options=driver_options)
        driver.get(self.link)
        time.sleep(1)
        soup = BeautifulSoup(driver.page_source, 'html5lib')
        driver.quit()

        soup = self.__remove_page_element_tags(soup, ['style', 'span', 'script', 'meta'])

        self.problemId = soup.find("input", {"name": "problemId"})['value']

        problem_statement = soup.find("div", class_="problem-statement")

        self.name = problem_statement.findChild("div", class_="title").getText()
        special_chars = "!#$%^&*()/?:"
        for char in special_chars:
            self.name = self.name.replace(char, ' ')

        self.statement = problem_statement.findChild("div", class_=None).getText()

        input_paragraphs = problem_statement.findChild("div", class_="input-specification")
        if input_paragraphs:
            input_paragraphs = input_paragraphs.contents[1:]
            self.input = ' '.join([data.text for data in input_paragraphs])
        else:
            self.input = None

        output_paragraphs = problem_statement.findChild("div", class_="output-specification")
        if output_paragraphs:
            output_paragraphs = output_paragraphs.contents[1:]
            self.output = ' '.join([data.text for data in output_paragraphs])
        else:
            self.output = None

        if not (self.statement and self.input and self.output):
            self.interactive = True

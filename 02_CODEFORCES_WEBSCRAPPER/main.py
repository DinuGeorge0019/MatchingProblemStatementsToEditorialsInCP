

# standard library imports
import argparse

# local application/library specific imports
from CodeforcesWebScrapper import CodeforcesWebScrapper


def main():
    parser = argparse.ArgumentParser()
    webScrapper = CodeforcesWebScrapper()
    arguments = [
        ("fetch_codeforces_data", webScrapper.fetch_contests, "Fetch codeforces contests links."),
        ("build_codeforces_dataset", webScrapper.build_dataset, "Build codeforces dataset."),
    ]

    for arg, _, description in arguments:
        parser.add_argument(f'--{arg}', action='store_true', help=description)

    params = parser.parse_args()
    for arg, fun, _ in arguments:
        if hasattr(params, arg) and getattr(params, arg):
            print(f"Executing {arg}")
            fun()


if __name__ == '__main__':
    main()


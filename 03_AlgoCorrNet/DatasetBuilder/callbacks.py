


# standard library imports
# None

# related third-party
# None

# local application/library specific imports
from DatasetBuilder.dataset_builder import DatasetBuilder


def build_dataset():
    datasetBuilder = DatasetBuilder()
    datasetBuilder.build_dataset()


def build_raw_competitive_programming_dataset():
    datasetBuilder = DatasetBuilder()
    datasetBuilder.build_raw_dataset()

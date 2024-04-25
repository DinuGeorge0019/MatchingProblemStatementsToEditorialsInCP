


# standard library imports
from enum import Enum


# related third-party
# None

# local application/library specific imports
# None

class TrainingSiameseFeature(Enum):
    PREMISE = 0
    HYPOTHESIS = 1
    LABEL = 2


class TrainingMNRLFeature(Enum):
    ANCHOR = 0
    POSITIVE = 1
    NEGATIVE = 2


class CorrelationFeature(Enum):
    STATEMENT = 0
    EDITORIAL = 1
    LABEL = 2


class PrecisionAtKFeature(Enum):
    STATEMENT = 0
    EDITORIAL_LIST = 1


class AlgoNetChatFeature(Enum):
    EDITORIAL = 0
    EDITORIAL_ID = 1

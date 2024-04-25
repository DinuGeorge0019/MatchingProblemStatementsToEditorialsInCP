


# standard library imports
# None

# related third-party
# None

# local application/library specific imports
from ErrorAnalysis.error_analysis_module import ErrorAnalysisModule


def create_error_analysis():
    errorAnalysisModule = ErrorAnalysisModule()
    errorAnalysisModule.create_pk_test_error_analysis()
    errorAnalysisModule.create_pk_validation_error_analysis()


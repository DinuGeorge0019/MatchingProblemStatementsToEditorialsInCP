


# standard library imports
import argparse

# local application/library specific imports
from DatasetBuilder.callbacks import build_dataset, build_raw_competitive_programming_dataset
from AlgoCorrNetModel.callbacks import train_triple_net_model, evaluate_triple_net_model, \
    evaluate_algo_corr_net_model, predict_editorial, build_algo_corr_net_model, evaluate_siamese_net_model, \
    train_siamese_net_model, compute_correlation_score, build_untrained_model_wrapper
from ErrorAnalysis.callbacks import create_error_analysis


def main():
    parser = argparse.ArgumentParser()

    arguments = [
        ("build_raw_dataset", build_raw_competitive_programming_dataset, "Build raw competitive programming dataset"),
        ("build_dataset", build_dataset, "Build preprocessed dataset"),
        ("train_triple_net_model", train_triple_net_model, "Train Triple Net embedding model"),
        ("evaluate_triple_net_model", evaluate_triple_net_model, "Evaluate Triple Net embedding model"),
        ("train_siamese_net_model", train_siamese_net_model, "Train Siamese Net embedding model"),
        ("evaluate_siamese_net_model", evaluate_siamese_net_model, "Evaluate Siamese Net embedding model"),
        ("build_untrained_model_wrapper", build_untrained_model_wrapper, "Build untrained base model wrapper"),
        ("build_algo_corr_net_model", build_algo_corr_net_model, "Build and return Algo Correlation Net model"),
        ("evaluate_algo_corr_net_model", evaluate_algo_corr_net_model, "Evaluate Algo Correlation Net model"),
        ("create_error_analysis", create_error_analysis, "Create an error analysis based on model evaluation"),
        ("predict_editorial", predict_editorial,
         "Get the best correlated editorial from the index for a given statement using AlgoCorrNet"),
        ("compute_correlation_score", compute_correlation_score,
         "Return the correlation score between a problem statement and an editorial")
    ]

    for arg, _, description in arguments:
        if arg == "predict_editorial":
            parser.add_argument(f'--{arg}', type=str, help=description, nargs=1, metavar='PROBLEM_STATEMENT')
        elif arg == "compute_correlation_score":
            parser.add_argument(f'--{arg}', type=str, help=description, nargs=2, metavar=('PROBLEM_STATEMENT', 'EDITORIAL'))
        else:
            parser.add_argument(f'--{arg}', action='store_true', help=description)

    params = parser.parse_args()
    for arg, fun, _ in arguments:
        if hasattr(params, arg) and getattr(params, arg):
            print(f"Executing {arg}")
            if arg == "predict_editorial":
                problem_statement = params.predict_editorial
                fun(problem_statement)
            elif arg == "compute_correlation_score":
                problem_statement, editorial = params.compute_correlation_score
                fun(problem_statement, editorial)
            else:
                fun()


if __name__ == '__main__':
    main()

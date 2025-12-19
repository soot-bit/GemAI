import argparse
import torch.optim as optim
from .models import tabnet, autogluon
from . import data
from .config import config


def main():
    """
    Main CLI entrypoint for the GemAI project.
    """
    parser = argparse.ArgumentParser(description="GemAI Project CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- Data Command ---
    process_data_parser = subparsers.add_parser("process-data", help="Run the data preprocessing pipeline")


    # --- Training Command ---
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument(
        "model", choices=["tabnet", "autogluon"], help="Model to train"
    )

    # --- Tuning Command ---
    tune_parser = subparsers.add_parser("tune", help="Hyperparameter tune a model")
    tune_parser.add_argument("model", choices=["tabnet"], help="Model to tune")

    args = parser.parse_args()

    if args.command == "process-data":
        data.run_preprocessing()

    elif args.command == "train":
        if args.model == "tabnet":
            constructor_params = config.tabnet.initial_params.model_dump(
                exclude={"scheduler_params"}
            )
            constructor_params["optimizer_fn"] = optim.Adam
            tabnet.run_training(constructor_params)
        elif args.model == "autogluon":
            autogluon.run_autogluon_training()

    elif args.command == "tune":
        if args.model == "tabnet":
            tabnet.run_tuning()


if __name__ == "__main__":
    main()

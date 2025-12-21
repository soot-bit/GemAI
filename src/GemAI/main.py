import argparse
import torch.optim as optim
from .models import tabnet, autogluon
from . import data
from .config import config


def _train(args):
    if args.model == "tabnet":
        params = config.tabnet.params.model_dump(
            exclude={"sched_params"}
        )
        params["optimizer_fn"] = optim.Adam
        tabnet.train_tn(params)
    elif args.model == "autogluon":
        autogluon.train_ag()


def _tune(args):
    if args.model == "tabnet":
        tabnet.tune_tn()


def main():
    """
    Main CLI entrypoint for the GemAI project.
    """
    parser = argparse.ArgumentParser(description="GemAI Project CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- Command Parsers ---
    subparsers.add_parser("prep", help="Run the data preprocessing pipeline")
    trn_parser = subparsers.add_parser("train", help="Train a model")
    trn_parser.add_argument(
        "model", choices=["tabnet", "autogluon"], help="Model to train"
    )
    tune_parser = subparsers.add_parser("tune", help="Hyperparameter tune a model")
    tune_parser.add_argument("model", choices=["tabnet"], help="Model to tune")

    args = parser.parse_args()

    # --- Command-Function Mapping ---
    commands = {
        "prep": data.preprocess,
        "train": _train,
        "tune": _tune,
    }

    cmd_func = commands.get(args.command)

    # For commands that don't have sub-arguments
    if args.command in ["prep"]:
        cmd_func()
    else:
        cmd_func(args)




if __name__ == "__main__":
    main()

import argparse
import torch.optim as optim
from .models import tabnet, autogluon
from . import data
from .config import settings

def main():
    """
    Main CLI entrypoint for the GemAI project.
    """
    parser = argparse.ArgumentParser(description="GemAI Project CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- Data Command ---
    data_parser = subparsers.add_parser("process-data", help="Run data preprocessing pipeline")

    # --- Training Command ---
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("model", choices=["tabnet", "autogluon"], help="Model to train")

    # --- Tuning Command ---
    tune_parser = subparsers.add_parser("tune", help="Hyperparameter tune a model")
    tune_parser.add_argument("model", choices=["tabnet"], help="Model to tune")

    # --- Serving Command ---
    serve_parser = subparsers.add_parser("serve", help="Serve a model with FastAPI")
    # No arguments needed for serve yet, but could add --host, --port etc.

    args = parser.parse_args()

    if args.command == "process-data":
        data.run_preprocessing()

    elif args.command == "train":
        if args.model == "tabnet":
            constructor_params = settings.tabnet.initial_params.model_dump(exclude={'scheduler_params'})
            constructor_params['optimizer_fn'] = optim.Adam
            tabnet.run_training(constructor_params)
        elif args.model == "autogluon":
            autogluon.run_autogluon_training()

    elif args.command == "tune":
        if args.model == "tabnet":
            tabnet.run_tuning()

    elif args.command == "serve":
        # To avoid making uvicorn a direct dependency, we run it as a shell command.
        # This is more flexible for a CLI tool.
        import os
        from .config import get_project_root
        
        app_dir = get_project_root() / "app"
        os.chdir(app_dir) # uvicorn needs to be run from the app's directory
        
        print("Starting FastAPI server... (uvicorn main:app --reload)")
        os.system("uvicorn main:app --reload --host 0.0.0.0 --port 8000")

if __name__ == "__main__":
    main()

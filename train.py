import argparse
from src.config import settings
from src.tuner import run_tuning
from src.trainer import run_training

def main():
    p = argparse.ArgumentParser(description="Train or tune TabNet on Diamonds")
    p.add_argument(
        "--tune",
        action="store_true",
        help="Run hyperparameter tuning before final training"
    )
    p.add_argument(
        "--trials", type=int, default=50,
        help="Number of tuning trials (if --tune)"
    )
    p.add_argument(
        "--timeout", type=int, default=3600,
        help="Tuning timeout in seconds (if --tune)"
    )
    args = p.parse_args()

    if args.tune:
        print("🔍 Starting hyperparameter tuning…")
        run_tuning(n_trials=args.trials, timeout=args.timeout)
 

    run_training(settings)

if __name__ == "__main__":
    main()

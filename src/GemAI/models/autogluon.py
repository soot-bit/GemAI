from autogluon.tabular import TabularPredictor, TabularDataset

from ..config import settings, get_project_root
from ..data import load_split_data
from .. import utils

def run_autogluon_training():
    """
    Trains an AutoGluon model based on the project configuration.
    """
    utils.logger.info("Starting AutoGluon training...")
    
    utils.logger.info("Loading and preparing data for AutoGluon...")
    train_df, val_df = load_split_data()
    train_data = TabularDataset(train_df)
    val_data = TabularDataset(val_df)
    utils.logger.info("Data prepared.")

    model_dir = get_project_root() / settings.paths.autogluon_dir

    predictor = TabularPredictor(
        label=settings.data.target,
        eval_metric=settings.autogluon.eval_metric,
        path=str(model_dir)
    ).fit(
        train_data,
        presets=settings.autogluon.preset,
        time_limit=settings.autogluon.time_limit,
    )

    utils.logger.info("--- AutoGluon Leaderboard (Model Ranking on Validation Data) ---")
    leaderboard = predictor.leaderboard(val_data, silent=True)
    utils.logger.info(f"\n{leaderboard.to_string()}")

    # You might want to save the leaderboard to a file as well
    leaderboard_path = model_dir / "leaderboard.csv"
    leaderboard.to_csv(leaderboard_path)
    utils.logger.info(f"Leaderboard saved to {leaderboard_path}")

    return predictor
